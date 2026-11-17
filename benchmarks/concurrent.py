"""Concurrent-request runner (Step 8): fixed-concurrency waves against one server.

Load policy (docs/experiment_plan.md: "fixed-concurrency batches"): a *wave* of N requests is
submitted together (threads released by a barrier), each with its own distinct prompt of the same
input length, and the next wave starts only after every request in the wave has finished. A *cell*
is one (input length, N) pair; every cell is run for several waves, and the cell order is rotated
from wave to wave so no cell is always measured first or last.

What this can and cannot tell you (see docs/measurement_contract.md, limit 16):

* Requests overlap in time because the client sends them together. Whether the engine actually
  batches them is a separate fact. It is recorded from the engine's own ``running`` and
  ``waiting`` gauges (sampled, so a lower bound) and from the timestamps (``co_decode_s``).
* TTFT includes queueing behind the other requests' prefill, so TTFT rises with N by design. It is
  a client-side figure, never split into queue and prefill time.
* Memory blocks describe the whole wave window, not one request, so every record in a wave carries
  the same block.

Every request still produces one record (failures included). Command line: see runner.py
(``--concurrency-levels``).
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import platform
import threading
import time
from typing import Callable, Optional

from benchmarks import runner, sampling, waves
from benchmarks import vllm_engine as engine

RECORD_SCHEMA = "kv-bench-request/3"  # /2 plus concurrency, wave_id, wave_index, slot, wave_size
CONFIG_SCHEMA = "kv-bench-run-config/2"
LOAD_POLICY = ("closed waves: N distinct prompts of one length are submitted together; the next wave "
               "starts after all N have finished")


def run_wave(host: str, port: int, model: str, prompts: list, max_tokens: int, ignore_eos: bool,
             timeout_s: float, clock: Callable[[], float] = time.monotonic) -> list:
    """Send one prompt per thread, released together; returns StreamResults in prompt order."""
    n = len(prompts)
    barrier = threading.Barrier(n)
    results: list = [None] * n

    def work(i: int) -> None:
        payload = engine.build_payload(model, prompts[i]["text"], max_tokens, ignore_eos)
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            pass
        try:
            results[i] = engine.stream_chat(host, port, payload, timeout_s=timeout_s, clock=clock)
        except Exception as exc:  # the client never raises for request failures; this is a bug guard
            res = engine.StreamResult(t_start=clock(), error_category="runner_exception",
                                      error_message=f"{type(exc).__name__}: {exc}")
            results[i] = res

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def plan_cells(levels: list, concurrencies: list) -> list:
    return [(lv, n) for lv in levels for n in concurrencies]


def select_prompts(prompts: list, levels: list, concurrencies: list) -> tuple:
    """Per level: the first N prompts are measured, the next N (starting after the largest N) are the
    warmup wave's, so a warmup never reuses a measured prompt."""
    top = max(concurrencies)
    measured, warm = {}, {}
    for lv in levels:
        at_level = [p for p in prompts if p["level"] == lv]
        if len(at_level) < 2 * top:
            raise ValueError(f"level {lv}: {len(at_level)} prompts, need {2 * top} "
                             f"(max concurrency {top} measured + {top} spare for warmup waves); "
                             f"generate per_level >= {2 * top}")
        measured[lv] = at_level[:top]
        warm[lv] = at_level[top:2 * top]
    return measured, warm


def run_concurrent_benchmark(*, workload_dir: pathlib.Path, host: str, port: int, model: str, revision: str,
                             levels: Optional[list], concurrencies: list, waves_per_cell: int,
                             warmup: int, timeout_s: float, ignore_eos: bool, out_dir: pathlib.Path,
                             server_settings: Optional[dict], stop_on_error: bool = True,
                             order: str = "rotate", clock: Callable[[], float] = time.monotonic,
                             server_log: Optional[str] = None,
                             monitor: Optional[sampling.MemoryMonitor] = None,
                             baseline_s: float = 0.0) -> tuple:
    if not concurrencies or min(concurrencies) < 1:
        raise ValueError("concurrencies must be positive integers")
    wl_manifest, prompts = runner.load_workload(workload_dir)
    level_list = list(levels or wl_manifest["levels"])
    measured_by_level, warm_by_level = select_prompts(prompts, level_list, concurrencies)
    max_tokens = wl_manifest["output_target_tokens"]
    cells = plan_cells(level_list, concurrencies)

    config = {
        "schema": CONFIG_SCHEMA,
        "engine": "vllm-openai-chat-stream",
        "model": model,
        "revision": revision,
        "sampling": {"temperature": 0, "max_tokens": max_tokens, "ignore_eos": ignore_eos},
        "load_policy": LOAD_POLICY,
        "levels": level_list,
        "concurrency_levels": list(concurrencies),
        "waves_per_cell": waves_per_cell,
        "cell_order": order,
        "warmup_requests": warmup,
        "warmup_wave_per_cell": True,
        "workload_id": wl_manifest["workload_id"],
        "measured_prompt_ids": {str(lv): [p["prompt_id"] for p in measured_by_level[lv][:max(concurrencies)]]
                                for lv in level_list},
        "warmup_prompt_ids": {str(lv): [p["prompt_id"] for p in warm_by_level[lv]] for lv in level_list},
        "timeout_s": timeout_s,
        "stop_on_error": stop_on_error,
        "server": server_settings or "external server; settings not controlled by this run",
    }
    cfg_hash = runner.config_hash(config)
    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{cfg_hash[:8]}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "config_hash": cfg_hash, "config": config,
        "started_utc": started.isoformat(), "git_commit": runner._git_commit(),
        "environment_json_sha256": runner._file_sha256(pathlib.Path("configs/environment.json")),
        "python": platform.python_version(), "package_versions": runner.installed_versions(),
        "lock_mismatches": runner._lock_mismatches(), "platform": platform.platform(),
        "server_log": server_log, "finished_utc": None, "counts": None,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))  # before any request
    try:
        (run_dir / "workload_manifest.json").write_text(json.dumps(wl_manifest, indent=2, sort_keys=True))
    except OSError:
        pass

    if monitor is not None:
        monitor.start()
        if baseline_s > 0:
            monitor.record_baseline(baseline_s)

    # (cell, wave index, is_warmup): short warmup is handled separately below
    orders = runner.level_order(cells, waves_per_cell, order)
    schedule = ([(c, 0, True) for c in cells]
                + [(c, w, False) for w in range(waves_per_cell) for c in orders[w]])

    records, counts = [], {"requests": 0, "ok": 0, "errors": 0, "warmup": 0, "usable": 0}
    seq = 0

    def emit(out, prompt, is_warmup, trial, result, req_max, memory, extra) -> dict:
        nonlocal seq
        rec = runner.build_record(run_id=run_id, cfg_hash=cfg_hash, request_id=f"{run_id}-{seq:04d}",
                                  prompt=prompt, is_warmup=is_warmup, trial_index=trial, result=result,
                                  expected_model=model, max_tokens=req_max, ignore_eos=ignore_eos,
                                  memory=memory)
        rec["schema"] = RECORD_SCHEMA
        rec.update(extra)
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
        stop = False
        for _ in range(warmup):  # short, single, sequential: opens connections and loads kernels
            payload = engine.build_payload(model, runner.WARMUP_PROMPT["text"], runner.WARMUP_MAX_TOKENS, ignore_eos)
            res = engine.stream_chat(host, port, payload, timeout_s=timeout_s, clock=clock)
            emit(out, runner.WARMUP_PROMPT, True, 0, res, runner.WARMUP_MAX_TOKENS, None,
                 {"concurrency": 1, "wave_id": "warmup", "wave_index": 0, "slot": 0, "wave_size": 1,
                  "level": None})
            stop = stop_on_error and res.status != "ok"
        for (level, n), w, is_warmup in schedule:
            if stop:
                break
            chosen = (warm_by_level if is_warmup else measured_by_level)[level][:n]
            wave_id = f"L{level}-N{n}-{'warm' if is_warmup else f'w{w}'}"
            results = run_wave(host, port, model, chosen, max_tokens, ignore_eos, timeout_s, clock)
            t0 = min(r.t_start for r in results)
            t1 = max((r.t_complete if r.t_complete is not None else clock()) for r in results)
            memory = None
            if monitor is not None:
                memory = dict(monitor.request_block(t0, t1))
                memory["window"] = ("wave: earliest request submission to latest completion in the wave, "
                                    "monotonic clock; shared by every request in the wave")
            for slot, (prompt, res) in enumerate(zip(chosen, results)):
                rec = emit(out, prompt, is_warmup, w, res, max_tokens, memory,
                           {"concurrency": n, "wave_id": wave_id, "wave_index": w, "slot": slot,
                            "wave_size": n, "level": level})
                if stop_on_error and rec["status"] != "ok":
                    stop = True  # finish recording this wave, then stop (docs/experiment_plan.md section 7)

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
    print(waves.format_cell_table(records))
    bad = [r for r in records if r["status"] != "ok"]
    for r in bad[:10]:
        print(f"  FAILED {r['wave_id']} slot {r['slot']}: {r['error_category']}: {str(r['error_message'])[:160]}")
    print("batch invariance:", json.dumps(waves.batch_invariance(records)))
