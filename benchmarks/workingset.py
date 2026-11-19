"""Step 9 working-set sweep: does the KV-cache strategy keep an agent's contexts alive?

An agent that pauses and resumes many long conversations needs each conversation's KV cache to survive
between turns. Here S sessions, each with its own distinct long context of C tokens, are filled once
(turn 0) and then revisited with a short distinct follow-up question (turn 1, 2, ...). Requests are sent one
at a time, so caching is not confused with batching (Step 8).

The working set is about S x C tokens. The GPU cache pool holds 309,248 tokens, so with C = 4096 about 74
sessions fit. A revisit finds its context only if the strategy still holds it:

* ``baseline``           no prefix caching: every revisit prefills the whole context again.
* ``prefix_cache``       GPU-resident prefix caching: contexts survive until the pool fills, then the
                         least recently used blocks are evicted.
* ``kv_offload_native``  prefix caching plus vLLM's native CPU offload: evicted blocks are kept in an
                         8 GiB CPU buffer (about as many tokens again) and loaded back on a hit.

Revisiting every session in the same order each round is the worst case for a least-recently-used cache:
once the working set exceeds the capacity, every revisit misses. The optional ``shuffled`` order revisits
sessions in a seeded random order, which gives a smoother, less adversarial curve.

Each cell (one working-set size) uses its own block of sessions, offset by the sizes before it in the
canonical ladder, so cells never reuse each other's contexts and every strategy sees the same texts.
Cached blocks of earlier cells stay in the cache but are the least recently used, so they are evicted first.

Per-request prefix-cache hits come from the engine's counters read before and after each request (with a
short settle time because counters can lag; measurement contract limit 11). They count the engine's own
hits, whichever tier served them.

Command line:
  python -m benchmarks.workingset --workload-dir <dir> --strategy prefix_cache --sessions 48 96 --out-dir <dir>
  python -m benchmarks.workingset --table results/<run> [results/<run> ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Callable, Optional

from benchmarks import metrics as mt
from benchmarks import runner, sampling
from benchmarks import vllm_engine as engine

RECORD_SCHEMA = "kv-bench-request/4"  # /3 fields are not used; adds strategy, session, turn, phase, cache hits
CONFIG_SCHEMA = "kv-bench-run-config/3"
LADDER = (48, 96, 144, 192)
OUTPUT_TOKENS = 16
QUESTION = "\n\nFollow-up {turn}: reply with the single word: ok."


def session_offset(size: int, ladder: tuple = LADDER) -> int:
    if size not in ladder:
        raise ValueError(f"working-set size {size} is not on the ladder {ladder}")
    return sum(x for x in ladder if x < size)


def turn_text(context: str, turn: int) -> str:
    return context + QUESTION.format(turn=turn)


def visit_order(size: int, round_index: int, order: str, seed: int) -> list:
    ids = list(range(size))
    if order == "same":
        return ids
    if order == "shuffled":
        random.Random(f"{seed}-{size}-{round_index}").shuffle(ids)
        return ids
    raise ValueError(f"unknown revisit order {order!r}")


def counter_snapshot(host: str, port: int, metrics_fn: Callable) -> dict:
    try:
        m = metrics_fn(host, port)
    except Exception:
        return {}
    return {k: m.get(k) for k in ("prefix_cache_queries_total", "prefix_cache_hits_total",
                                  "external_prefix_cache_hits_total")}


def _delta(before: dict, after: dict, key: str) -> Optional[float]:
    a, b = before.get(key), after.get(key)
    return None if a is None or b is None else b - a


def run_workingset(*, workload_dir: pathlib.Path, host: str, port: int, model: str, revision: str,
                   strategy: str, sizes: list, rounds: int, revisit_order: str, seed: int, timeout_s: float,
                   out_dir: pathlib.Path, server_settings: Optional[dict], stop_on_error: bool = True,
                   settle_s: float = 0.5, warmup: bool = True, ladder: tuple = LADDER,
                   server_log: Optional[str] = None, monitor: Optional[sampling.MemoryMonitor] = None,
                   baseline_s: float = 0.0, metrics_fn: Callable = sampling.read_engine_metrics,
                   clock: Callable[[], float] = time.monotonic) -> tuple:
    wl_manifest, prompts = runner.load_workload(workload_dir)
    context_tokens = wl_manifest["levels"][0]
    contexts = [p for p in prompts if p["level"] == context_tokens]
    needed = sum(ladder) + (1 if warmup else 0)
    if len(contexts) < needed:
        raise ValueError(f"workload has {len(contexts)} contexts of {context_tokens} tokens, need {needed} "
                         f"(sum of the ladder {ladder} plus a spare for the warmup)")
    plan_sizes = sorted(sizes)
    for s in plan_sizes:
        session_offset(s, ladder)

    config = {
        "schema": CONFIG_SCHEMA, "engine": "vllm-openai-chat-stream", "model": model, "revision": revision,
        "sampling": {"temperature": 0, "max_tokens": OUTPUT_TOKENS, "ignore_eos": True},
        "strategy": strategy, "workload_id": wl_manifest["workload_id"], "context_tokens": context_tokens,
        "ladder": list(ladder), "sizes": plan_sizes, "rounds": rounds, "revisit_order": revisit_order,
        "seed": seed, "settle_s": settle_s, "question": QUESTION, "warmup": warmup, "timeout_s": timeout_s,
        "stop_on_error": stop_on_error, "load_policy": "one request at a time (concurrency 1)",
        "server": server_settings or "external server; settings not controlled by this run",
    }
    cfg_hash = runner.config_hash(config)
    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{cfg_hash[:8]}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "config_hash": cfg_hash, "config": config, "started_utc": started.isoformat(),
                "git_commit": runner._git_commit(),
                "environment_json_sha256": runner._file_sha256(pathlib.Path("configs/environment.json")),
                "python": platform.python_version(), "package_versions": runner.installed_versions(),
                "lock_mismatches": runner._lock_mismatches(), "platform": platform.platform(),
                "server_log": server_log, "finished_utc": None, "counts": None}
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    try:
        (run_dir / "workload_manifest.json").write_text(json.dumps(wl_manifest, indent=2, sort_keys=True))
    except OSError:
        pass
    if monitor is not None:
        monitor.start()
        if baseline_s > 0:
            monitor.record_baseline(baseline_s)

    records, counts = [], {"requests": 0, "ok": 0, "errors": 0, "warmup": 0, "usable": 0}
    last = counter_snapshot(host, port, metrics_fn)
    seq = 0
    stop = False

    def send(out, ctx: dict, session: Optional[int], turn: int, phase: str, size: Optional[int],
             is_warmup: bool) -> dict:
        nonlocal seq, last
        text = turn_text(ctx["text"], turn)
        prompt = {"prompt_id": f"{ctx['prompt_id']}-T{turn}", "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "target_input_tokens": context_tokens, "actual_input_tokens": None}
        payload = engine.build_payload(model, text, OUTPUT_TOKENS, ignore_eos=True)
        result = engine.stream_chat(host, port, payload, timeout_s=timeout_s, clock=clock)
        memory = None
        if monitor is not None:
            end = result.t_complete if result.t_complete is not None else clock()
            memory = monitor.request_block(result.t_start, end)
        time.sleep(settle_s)
        after = counter_snapshot(host, port, metrics_fn)
        queries, hits = _delta(last, after, "prefix_cache_queries_total"), _delta(last, after, "prefix_cache_hits_total")
        external = _delta(last, after, "external_prefix_cache_hits_total")
        last = after
        rec = runner.build_record(run_id=run_id, cfg_hash=cfg_hash, request_id=f"{run_id}-{seq:05d}", prompt=prompt,
                                  is_warmup=is_warmup, trial_index=turn, result=result, expected_model=model,
                                  max_tokens=OUTPUT_TOKENS, ignore_eos=True, memory=memory)
        rec["schema"] = RECORD_SCHEMA
        rec.update({"strategy": strategy, "working_set_sessions": size, "session": session, "turn": turn,
                    "phase": phase, "context_tokens": context_tokens, "context_id": ctx["prompt_id"],
                    "prefix_cache_queries_delta": queries, "prefix_cache_hits_delta": hits,
                    "external_prefix_cache_hits_delta": external,
                    "hit_fraction": (hits / queries) if queries else None})
        out.write(json.dumps(rec, sort_keys=True) + "\n")
        out.flush()
        seq += 1
        records.append(rec)
        counts["requests"] += 1
        counts["warmup"] += is_warmup
        counts["ok"] += rec["status"] == "ok"
        counts["errors"] += rec["status"] != "ok"
        counts["usable"] += rec["usable_for_analysis"]
        return rec

    with open(run_dir / "requests.jsonl", "w", encoding="utf-8") as out:
        if warmup:
            r = send(out, contexts[sum(ladder)], None, 0, "warmup", None, True)
            stop = stop_on_error and r["status"] != "ok"
        for size in plan_sizes:
            if stop:
                break
            base = session_offset(size, ladder)
            for turn in range(rounds + 1):
                order = list(range(size)) if turn == 0 else visit_order(size, turn, revisit_order, seed)
                for session in order:
                    r = send(out, contexts[base + session], session, turn, "fill" if turn == 0 else "revisit",
                             size, False)
                    if stop_on_error and r["status"] != "ok":
                        stop = True
                        break
                if stop:
                    break

    if monitor is not None:
        monitor.stop()
        monitor.write(run_dir)
        manifest["memory_monitoring"] = monitor.describe()
    if server_log:
        try:
            manifest["engine_reported_startup"] = sampling.parse_vllm_startup_log(
                pathlib.Path(server_log).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            manifest["engine_reported_startup"] = None
    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["counts"] = counts
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return run_dir, records


# ----------------------------------------------------------------------------- analysis
def summarize(records: list) -> dict:
    """Per working-set size and phase: TTFT, hit share, and memory peaks, over usable records only."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in mt.usable(records):
        if r.get("phase") in ("fill", "revisit"):
            cells[r["working_set_sessions"]][r["phase"]].append(r)
    out = {}
    for size in sorted(cells):
        entry = {"sessions": size}
        for phase, rs in cells[size].items():
            q = sum(r["prefix_cache_queries_delta"] or 0 for r in rs)
            h = sum(r["prefix_cache_hits_delta"] or 0 for r in rs)
            served = [r for r in rs if (r["hit_fraction"] or 0) > 0.5]
            shm = [(r.get("memory") or {}).get("cpu", {}).get("container_shmem_peak_mib") for r in rs]
            fil = [(r.get("memory") or {}).get("cpu", {}).get("container_file_peak_mib") for r in rs]
            kv = [(r.get("memory") or {}).get("engine", {}).get("kv_cache_usage_peak") for r in rs]
            vals = lambda xs: [x for x in xs if x is not None]  # noqa: E731
            entry[phase] = {
                "n": len(rs), "ttft_s": mt.describe(r["ttft_s"] for r in rs),
                "hit_token_share": (h / q) if q else None,
                "requests_mostly_cached": len(served), "share_mostly_cached": len(served) / len(rs),
                "container_file_peak_mib": max(vals(fil), default=None),
                "container_shmem_peak_mib": max(vals(shm), default=None),
                "kv_cache_usage_peak": max(vals(kv), default=None)}
        out[str(size)] = entry
    return out


def format_table(records: list) -> str:
    cells = summarize(records)
    strategy = next((r.get("strategy") for r in records if r.get("strategy")), "?")
    head = (f"{strategy}\n{'sessions':>8} {'phase':>8} {'n':>4} {'ttft med [min-max] s':>26} {'hit tok share':>14} "
            f"{'mostly cached':>14} {'kv peak':>8} {'file peak MiB':>14}")
    lines = [head]
    for entry in cells.values():
        for phase in ("fill", "revisit"):
            e = entry.get(phase)
            if not e:
                continue
            t = e["ttft_s"]
            hit = f"{e['hit_token_share']:.3f}" if e["hit_token_share"] is not None else "n/a"
            kv = f"{e['kv_cache_usage_peak']:.3f}" if e["kv_cache_usage_peak"] is not None else "n/a"
            fp = f"{e['container_file_peak_mib']:.0f}" if e["container_file_peak_mib"] is not None else "n/a"
            ttft = f"{t['median']:.3f} [{t['min']:.3f}-{t['max']:.3f}]"
            cached = f"{e['requests_mostly_cached']}/{e['n']}"
            lines.append(f"{entry['sessions']:>8} {phase:>8} {e['n']:>4} {ttft:>26} {hit:>14} {cached:>14} "
                         f"{kv:>8} {fp:>14}")
    return "\n".join(lines)


def offload_evidence(log_text: str) -> dict:
    """What the engine's own log says about the CPU offload tier (None-safe; empty log gives zeros).

    vLLM prints a ``KV Transfer metrics`` line and an ``External prefix cache hit rate`` every ~10 s. Totals
    here are sums over those intervals. ``external_hit_rate_max_pct`` above 0 is the engine saying a
    prefix hit was supplied by a tier other than the GPU; ``load_bytes_total`` above 0 means data was read
    back from the CPU buffer. Neither is proof of speed-up (the request timings are).
    """
    import re
    out = {"intervals": 0, "store_bytes_total": 0.0, "load_bytes_total": 0.0, "cpu_cache_usage_max": None,
           "cpu_cache_read_usage_max": None, "external_hit_rate_max_pct": None, "prefix_hit_rate_max_pct": None,
           "block_bytes": None}
    for line in log_text.splitlines():
        m = re.search(r"row=(\d+)", line)
        if m and out["block_bytes"] is None and "CPU offload region" in line:
            out["block_bytes"] = int(m.group(1))
        m = re.search(r"External prefix cache hit rate: ([\d.]+)%", line)
        if m:
            v = float(m.group(1))
            out["external_hit_rate_max_pct"] = v if out["external_hit_rate_max_pct"] is None else max(out["external_hit_rate_max_pct"], v)
        m = re.search(r"Prefix cache hit rate: ([\d.]+)%", line)
        if m:
            v = float(m.group(1))
            out["prefix_hit_rate_max_pct"] = v if out["prefix_hit_rate_max_pct"] is None else max(out["prefix_hit_rate_max_pct"], v)
        if "KV Transfer metrics:" not in line:
            continue
        out["intervals"] += 1
        for key, val in re.findall(r"vllm:(kv_offload_[a-z_]+)=([0-9.eE+-]+)", line):
            v = float(val)
            if key == "kv_offload_store_bytes":
                out["store_bytes_total"] += v
            elif key == "kv_offload_load_bytes":
                out["load_bytes_total"] += v
            elif key == "kv_offload_cpu_cache_usage_perc":
                out["cpu_cache_usage_max"] = v if out["cpu_cache_usage_max"] is None else max(out["cpu_cache_usage_max"], v)
            elif key == "kv_offload_cpu_cache_read_usage_perc":
                out["cpu_cache_read_usage_max"] = v if out["cpu_cache_read_usage_max"] is None else max(out["cpu_cache_read_usage_max"], v)
    if out["block_bytes"]:
        out["store_blocks_total"] = out["store_bytes_total"] / out["block_bytes"]
        out["load_blocks_total"] = out["load_bytes_total"] / out["block_bytes"]
    return out


# ----------------------------------------------------------------------------- command line
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 9 working-set sweep (one strategy per launch)")
    ap.add_argument("--table", nargs="+", type=pathlib.Path, help="print the summary of run directories and exit")
    ap.add_argument("--workload-dir", type=pathlib.Path)
    ap.add_argument("--strategy", choices=["baseline", "prefix_cache", "kv_offload_native"])
    ap.add_argument("--sessions", type=int, nargs="+", help="working-set sizes, from the ladder")
    ap.add_argument("--rounds", type=int, default=1, help="revisit rounds after the fill")
    ap.add_argument("--revisit-order", choices=["same", "shuffled"], default="same")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--settle", type=float, default=0.5, help="seconds after each request before reading counters")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--sample-interval", type=float, default=0.5)
    ap.add_argument("--baseline-seconds", type=float, default=3.0)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    if args.table:
        for d in args.table:
            problems = mt.validate_run_dir(d)
            print(f"{d.name}: validation {'OK' if not problems else problems[:5]}")
            print(format_table(mt.load_records(d / "requests.jsonl")))
        return
    if not (args.workload_dir and args.strategy and args.sessions):
        ap.error("--workload-dir, --strategy and --sessions are required")

    from benchmarks import probe_strategies as probe

    wl_manifest, _ = runner.load_workload(args.workload_dir)
    prefix, extra, _ = probe.CONFIGS[args.strategy]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(args.out_dir / f"server_{args.strategy}.log")
    server = engine.VLLMServer(wl_manifest["model_id"], wl_manifest["revision"], port=args.port,
                               max_model_len=args.max_model_len, enable_prefix_caching=prefix,
                               enforce_eager=False, log_path=log_path, extra_args=extra)
    probe_args = argparse.Namespace(sample_memory=True, gpu_index=args.gpu_index, host="127.0.0.1",
                                    port=args.port, sample_interval=args.sample_interval)
    with server:
        monitor = runner.make_monitor(probe_args, server.pid)
        run_dir, records = run_workingset(
            workload_dir=args.workload_dir, host="127.0.0.1", port=args.port, model=wl_manifest["model_id"],
            revision=wl_manifest["revision"], strategy=args.strategy, sizes=args.sessions, rounds=args.rounds,
            revisit_order=args.revisit_order, seed=args.seed, timeout_s=args.timeout, out_dir=args.out_dir,
            server_settings={**server.settings, "strategy": args.strategy}, settle_s=args.settle,
            server_log=log_path, monitor=monitor, baseline_s=args.baseline_seconds)
    try:
        (run_dir / "server.log").write_bytes(pathlib.Path(log_path).read_bytes())
    except OSError:
        pass
    print(f"\nrun directory: {run_dir}")
    print(format_table(records))
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad[:5]:
        print(f"  FAILED {r['request_id']}: {r['error_category']}: {str(r['error_message'])[:160]}")
    try:
        text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
        if "KV Transfer metrics:" in text:
            print("offload evidence from the server log:", json.dumps(offload_evidence(text)))
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
