"""Benchmark runner (Step 5): one streamed request at a time, one JSONL record each.

Each run writes ``results/<run_id>/``:

* ``run_manifest.json``  resolved configuration, config hash, git commit. Written
                         BEFORE the first request, rewritten with counts at the end.
* ``requests.jsonl``     one record per request, flushed as it completes, including
                         warmups (``is_warmup: true``) and failures.
* ``server.log``         vLLM output when the runner launched the server.

Every request produces a record, even on failure. A record only counts toward
analysis when ``usable_for_analysis`` is true: status ok, not a warmup, and no
validation errors. Token counts come from engine usage; they are never inferred
from chunk counts. Concurrency above 1 belongs to Step 8 and is rejected here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import platform
import subprocess
import time
from typing import Callable, Optional

from benchmarks import sampling
from benchmarks import vllm_engine as engine

# /2 adds the optional per-request ``memory`` block. /1 records (no memory block)
# remain valid; see docs/measurement_contract.md.
SCHEMA = "kv-bench-request/2"
CLOCK_NAME = "time.monotonic"
TTFT_BASIS = ("client: request write start -> first non-empty content event; "
              "includes TCP connect, server queueing, prefill, and transport")
DECODE_BASIS = ("engine usage completion_tokens minus 1, over first-to-last content-event "
                "time; a chunk-timing proxy because one event can carry several tokens")
WARMUP_PROMPT = {"prompt_id": "warmup", "text": "Reply with the single word: ready.",
                 "target_input_tokens": None, "actual_input_tokens": None,
                 "prompt_sha256": hashlib.sha256(b"Reply with the single word: ready.").hexdigest()}
WARMUP_MAX_TOKENS = 8


# ----------------------------------------------------------------------------- records
def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def build_record(*, run_id: str, cfg_hash: str, request_id: str, prompt: dict, is_warmup: bool,
                 trial_index: int, result: engine.StreamResult, expected_model: str,
                 max_tokens: int, ignore_eos: bool, memory: Optional[dict] = None) -> dict:
    r = result
    ttft = (r.t_first_content - r.t_start) if r.t_first_content is not None else None
    e2e = (r.t_complete - r.t_start) if r.t_complete is not None else None
    decode_time = ((r.t_last_content - r.t_first_content)
                   if r.t_first_content is not None and r.t_last_content is not None else None)
    decode_tps = None
    if (r.completion_tokens is not None and r.completion_tokens >= 2
            and decode_time is not None and decode_time > 0):
        decode_tps = (r.completion_tokens - 1) / decode_time

    rec = {
        "schema": SCHEMA,
        "run_id": run_id,
        "request_id": request_id,
        "config_hash": cfg_hash,
        "wall_start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "is_warmup": is_warmup,
        "trial_index": trial_index,
        "prompt_id": prompt["prompt_id"],
        "prompt_sha256": prompt["prompt_sha256"],
        "target_input_tokens": prompt.get("target_input_tokens"),
        "input_tokens_manifest": prompt.get("actual_input_tokens"),
        "prompt_tokens": r.prompt_tokens,
        "prompt_tokens_source": "engine_usage" if r.prompt_tokens is not None else "unavailable",
        "output_tokens": r.completion_tokens,
        "output_tokens_source": "engine_usage" if r.completion_tokens is not None else "unavailable",
        "output_target_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "finish_reason": r.finish_reason,
        "status": r.status,
        "error_category": r.error_category,
        "error_message": r.error_message,
        "http_status": r.http_status,
        "expected_model": expected_model,
        "response_model": r.response_model,
        "clock": CLOCK_NAME,
        "t_start_s": r.t_start,
        "t_first_content_s": r.t_first_content,
        "t_last_content_s": r.t_last_content,
        "t_complete_s": r.t_complete,
        "ttft_s": ttft,
        "ttft_basis": TTFT_BASIS,
        "e2e_latency_s": e2e,
        "decode_time_s": decode_time,
        "decode_tokens_per_s": decode_tps,
        "decode_basis": DECODE_BASIS,
        "content_chunks": r.content_chunks,
        "content_chunks_note": "event count, not a token count",
        "output_text": r.text,
        "output_sha256": hashlib.sha256(r.text.encode()).hexdigest(),
    }
    if memory is not None:
        rec["memory"] = memory
    rec["validation_errors"] = validate_record(rec)
    rec["usable_for_analysis"] = (rec["status"] == "ok" and not is_warmup
                                  and not rec["validation_errors"])
    return rec


def validate_record(rec: dict) -> list:
    """Return a list of problems; empty means the record is internally consistent."""
    errs = []
    if "memory" in rec:
        errs += sampling.validate_memory_block(rec["memory"])
    if rec["status"] != "ok":
        if not rec.get("error_category"):
            errs.append("error_without_category")
        return errs
    stamps = [rec["t_start_s"], rec["t_first_content_s"], rec["t_last_content_s"], rec["t_complete_s"]]
    if any(t is None for t in stamps):
        errs.append("missing_timestamp")
    elif not (stamps[0] <= stamps[1] <= stamps[2] <= stamps[3]):
        errs.append("timestamps_out_of_order")
    if rec["output_tokens"] is None or rec["prompt_tokens"] is None:
        errs.append("missing_engine_usage")
    elif rec["output_tokens"] < 1:
        errs.append("zero_output_tokens")
    elif rec["output_tokens"] > rec["output_target_tokens"]:
        errs.append("output_exceeds_max_tokens")
    if rec["finish_reason"] is None:
        errs.append("missing_finish_reason")
    if rec["response_model"] != rec["expected_model"]:
        errs.append("model_mismatch")
    if (rec["input_tokens_manifest"] is not None and rec["prompt_tokens"] is not None
            and rec["input_tokens_manifest"] != rec["prompt_tokens"]):
        errs.append("input_token_mismatch")
    return errs


# ----------------------------------------------------------------------------- run
TRACKED_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "huggingface_hub", "numpy")


def installed_versions(names=TRACKED_PACKAGES) -> dict:
    """Versions actually installed in this interpreter (None when absent). Recorded in every
    manifest because an unpinned install once silently moved vLLM from 0.33.0 to 0.34.0."""
    from importlib import metadata
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _lock_mismatches():
    """Installed-vs-lock differences, or None when no lock file is present (e.g. in tests)."""
    from benchmarks import envcheck
    try:
        return envcheck.check()
    except Exception:  # never let a version check break a run
        return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _file_sha256(path: pathlib.Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def level_order(levels: list, trials: int, order: str) -> list:
    """Level order for each trial. ``rotate`` shifts the list by one per trial so no level
    is always measured first or last (the allocator keeps memory between requests)."""
    if order not in ("ascending", "rotate"):
        raise ValueError(f"unknown order {order!r}")
    levels = list(levels)
    if order == "ascending" or not levels:
        return [levels for _ in range(trials)]
    return [levels[t % len(levels):] + levels[:t % len(levels)] for t in range(trials)]


def load_workload(workload_dir: pathlib.Path) -> tuple:
    manifest = json.loads((workload_dir / "manifest.json").read_text())
    prompts = [json.loads(line) for line in
               (workload_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return manifest, prompts


def run_benchmark(*, workload_dir: pathlib.Path, host: str, port: int, model: str, revision: str,
                  levels: Optional[list], limit_per_level: int, trials: int, warmup: int,
                  timeout_s: float, ignore_eos: bool, out_dir: pathlib.Path,
                  server_settings: Optional[dict], stop_on_error: bool = True, concurrency: int = 1,
                  clock: Callable[[], float] = time.monotonic,
                  server_log: Optional[str] = None,
                  monitor: Optional[sampling.MemoryMonitor] = None,
                  baseline_s: float = 0.0,
                  warmup_per_level: bool = False, order: str = "ascending") -> tuple:
    if concurrency != 1:
        raise ValueError("concurrency above 1 is implemented in Step 8, not Step 5")

    wl_manifest, prompts = load_workload(workload_dir)
    level_list = list(levels or wl_manifest["levels"])
    by_level, warm_by_level = {}, {}
    for level in level_list:
        at_level = [p for p in prompts if p["level"] == level]
        by_level[level] = at_level[:limit_per_level]
        if warmup_per_level and by_level[level]:
            if len(at_level) <= limit_per_level:
                raise ValueError(f"level {level}: per-level warmup needs a spare prompt; "
                                 f"generate per_level >= {limit_per_level + 1}")
            warm_by_level[level] = at_level[limit_per_level]  # never measured
    selected = [p for level in level_list for p in by_level[level]]
    if not selected:
        raise ValueError("no prompts selected; check --levels against the workload manifest")
    level_list = [lv for lv in level_list if by_level[lv]]
    max_tokens = wl_manifest["output_target_tokens"]

    config = {
        "schema": "kv-bench-run-config/1",
        "engine": "vllm-openai-chat-stream",
        "model": model,
        "revision": revision,
        "sampling": {"temperature": 0, "max_tokens": max_tokens, "ignore_eos": ignore_eos},
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "trials": trials,
        "workload_id": wl_manifest["workload_id"],
        "selected_prompt_ids": [p["prompt_id"] for p in selected],
        "warmup_per_level": warmup_per_level,
        "warmup_prompt_ids": [p["prompt_id"] for p in warm_by_level.values()],
        "level_order": order,
        "timeout_s": timeout_s,
        "stop_on_error": stop_on_error,
        "server": server_settings or "external server; settings not controlled by this run",
    }
    cfg_hash = config_hash(config)
    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{cfg_hash[:8]}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "config": config,
        "started_utc": started.isoformat(),
        "git_commit": _git_commit(),
        "environment_json_sha256": _file_sha256(pathlib.Path("configs/environment.json")),
        "python": platform.python_version(),
        "package_versions": installed_versions(),
        "lock_mismatches": _lock_mismatches(),
        "platform": platform.platform(),
        "server_log": server_log,
        "finished_utc": None,
        "counts": None,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))  # before any request

    try:  # keep the exact workload definition with the results
        (run_dir / "workload_manifest.json").write_text(json.dumps(wl_manifest, indent=2, sort_keys=True))
    except OSError:
        pass

    if monitor is not None:
        monitor.start()
        if baseline_s > 0:
            monitor.record_baseline(baseline_s)  # idle: no request in flight yet

    orders = level_order(level_list, trials, order)
    plan = ([(WARMUP_PROMPT, True, 0, WARMUP_MAX_TOKENS)] * warmup
            + [(warm_by_level[lv], True, 0, max_tokens) for lv in level_list if lv in warm_by_level]
            + [(p, False, t, max_tokens) for t in range(trials) for lv in orders[t]
               for p in by_level[lv]])
    records, counts = [], {"requests": 0, "ok": 0, "errors": 0, "warmup": 0, "usable": 0}
    with open(run_dir / "requests.jsonl", "w", encoding="utf-8") as out:
        for n, (prompt, is_warmup, trial, req_max) in enumerate(plan):
            payload = engine.build_payload(model, prompt["text"], req_max, ignore_eos)
            result = engine.stream_chat(host, port, payload, timeout_s=timeout_s, clock=clock)
            memory = None
            if monitor is not None:
                window_end = result.t_complete if result.t_complete is not None else clock()
                memory = monitor.request_block(result.t_start, window_end)
            rec = build_record(run_id=run_id, cfg_hash=cfg_hash, request_id=f"{run_id}-{n:04d}",
                               prompt=prompt, is_warmup=is_warmup, trial_index=trial,
                               result=result, expected_model=model, max_tokens=req_max,
                               ignore_eos=ignore_eos, memory=memory)
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            out.flush()
            records.append(rec)
            counts["requests"] += 1
            counts["warmup"] += is_warmup
            counts["ok"] += rec["status"] == "ok"
            counts["errors"] += rec["status"] != "ok"
            counts["usable"] += rec["usable_for_analysis"]
            if stop_on_error and rec["status"] != "ok":
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


def print_summary(run_dir: pathlib.Path, records: list) -> None:
    print(f"\nrun directory: {run_dir}")
    print(f"{'prompt':<12}{'warm':<6}{'status':<8}{'ttft_s':>9}{'e2e_s':>9}{'in':>7}{'out':>5}"
          f"{'tok/s':>8}  {'finish':<8}{'usable'}")
    for r in records:
        f = lambda v, w, p=3: f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        print(f"{r['prompt_id']:<12}{str(r['is_warmup']):<6}{r['status']:<8}{f(r['ttft_s'], 9)}"
              f"{f(r['e2e_latency_s'], 9)}{str(r['prompt_tokens'] or '-'):>7}"
              f"{str(r['output_tokens'] or '-'):>5}{f(r['decode_tokens_per_s'], 8, 1)}  "
              f"{str(r['finish_reason']):<8}{r['usable_for_analysis']}")
        if r["status"] != "ok":
            print(f"    error: {r['error_category']}: {r['error_message']}")
        if r["validation_errors"]:
            print(f"    validation: {r['validation_errors']}")
    first = next((r for r in records if r["status"] == "ok"), None)
    if first:
        print(f"\nsample output ({first['prompt_id']}): {first['output_text'][:200]!r}")


def make_monitor(args, server_pid: Optional[int]) -> Optional[sampling.MemoryMonitor]:
    """Build the memory monitor, or None when sampling is off or the GPU is unreadable."""
    if not args.sample_memory:
        return None
    try:
        gpu_reader = sampling.make_gpu_reader(args.gpu_index)
        gpu_reader()  # fail here with a clear message, not silently on the sampler thread
    except Exception as exc:
        print(f"warning: GPU memory sampling disabled ({exc!r})")
        gpu_reader = None
    monitor = sampling.MemoryMonitor(
        gpu_reader=gpu_reader,
        engine_reader=lambda: sampling.read_engine_metrics(args.host, args.port),
        cpu_reader=sampling.make_cpu_reader(server_pid),
        interval_s=args.sample_interval)
    try:
        monitor.engine_catalog = sampling.fetch_engine_catalog(args.host, args.port)
    except Exception as exc:
        print(f"warning: could not read the engine metrics catalog ({exc!r})")
    return monitor


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 5 benchmark runner (concurrency 1).")
    ap.add_argument("--workload-dir", type=pathlib.Path, required=True,
                    help="directory with manifest.json and prompts.jsonl from workloads.generate")
    ap.add_argument("--model", help="defaults to the workload manifest model_id")
    ap.add_argument("--revision", help="defaults to the workload manifest revision")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--launch-server", action="store_true", help="start vllm serve for this run")
    ap.add_argument("--max-model-len", type=int, help="default: largest selected input + output + 64")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--levels", type=int, nargs="+")
    ap.add_argument("--limit-per-level", type=int, default=1)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--warmup-per-level", action="store_true",
                    help="one unmeasured full-length warmup per level (uses a spare prompt)")
    ap.add_argument("--order", choices=["ascending", "rotate"], default="ascending",
                    help="level order per trial; rotate shifts it by one each trial")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="disable natural stopping; recorded in the config and every record")
    ap.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--sample-memory", action=argparse.BooleanOptionalAction, default=True,
                    help="sample GPU memory, engine /metrics, and CPU memory during the run")
    ap.add_argument("--sample-interval", type=float, default=0.5, help="seconds between samples")
    ap.add_argument("--baseline-seconds", type=float, default=3.0,
                    help="idle seconds sampled before the first request (memory baseline)")
    ap.add_argument("--gpu-index", type=int, default=0, help="physical GPU index to sample")
    args = ap.parse_args()

    wl_manifest, _ = load_workload(args.workload_dir)
    model = args.model or wl_manifest["model_id"]
    revision = args.revision or wl_manifest["revision"]
    if revision == "main":
        ap.error("revision must be a pinned commit hash, not 'main'")
    levels = args.levels or wl_manifest["levels"]
    kwargs = dict(workload_dir=args.workload_dir, host=args.host, port=args.port, model=model,
                  revision=revision, levels=levels, limit_per_level=args.limit_per_level,
                  trials=args.trials, warmup=args.warmup, timeout_s=args.timeout,
                  ignore_eos=args.ignore_eos, out_dir=args.out_dir,
                  stop_on_error=args.stop_on_error, concurrency=args.concurrency,
                  warmup_per_level=args.warmup_per_level, order=args.order)

    if args.launch_server:
        max_len = args.max_model_len or (max(levels) + wl_manifest["output_target_tokens"] + 64)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(args.out_dir / "server.log")
        server = engine.VLLMServer(model, revision, port=args.port, host=args.host,
                                   max_model_len=max_len,
                                   gpu_memory_utilization=args.gpu_memory_utilization,
                                   enable_prefix_caching=args.enable_prefix_caching,
                                   enforce_eager=args.enforce_eager, log_path=log_path)
        with server:
            monitor = make_monitor(args, server.pid)
            run_dir, records = run_benchmark(server_settings=server.settings, server_log=log_path,
                                             monitor=monitor, baseline_s=args.baseline_seconds,
                                             **kwargs)
        # keep the server log with the run it belongs to
        try:
            (run_dir / "server.log").write_bytes(pathlib.Path(log_path).read_bytes())
        except OSError:
            pass
    else:
        run_dir, records = run_benchmark(server_settings=None, monitor=make_monitor(args, None),
                                         baseline_s=args.baseline_seconds, **kwargs)
    print_summary(run_dir, records)


if __name__ == "__main__":
    main()
