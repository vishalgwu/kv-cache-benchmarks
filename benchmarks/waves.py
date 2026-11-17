"""Per-wave and per-cell statistics for concurrent runs (Step 8), recomputed from request records.

Definitions (docs/measurement_contract.md, limit 16):

* ``span_s``            earliest request start to latest request completion in the wave.
* ``agg_output_tok_s``  output tokens of the wave / span_s. It includes every request's prefill, so it
                        is an end-to-end rate for the wave, not a decode rate.
* ``max_in_flight``     largest number of requests whose [start, complete] intervals overlap, from the
                        client timestamps. This is OVERLAP: what the client asked for.
* ``co_decode_s``       time in which every stream in the wave was past its first token and none had
                        finished: min(last content) - max(first content), floored at 0. A positive
                        value means all N were decoding together.
* ``engine_running_peak`` / ``engine_waiting_peak``  the engine's own gauges, sampled (lower bounds).
                        This is BATCHING: what the engine scheduled together. Overlap without
                        ``running`` above 1 would mean the engine serialised the requests.

A wave counts only if every request in it is usable (a partly failed wave would bias the aggregate).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Optional

from benchmarks import metrics as mt


def _engine(rec: dict) -> dict:
    return (rec.get("memory") or {}).get("engine") or {}


def _gpu(rec: dict) -> dict:
    return (rec.get("memory") or {}).get("gpu") or {}


def max_in_flight(recs: list) -> int:
    events = sorted([(r["t_start_s"], 1) for r in recs] + [(r["t_complete_s"], -1) for r in recs],
                    key=lambda e: (e[0], e[1]))  # an end at the same instant as a start frees first
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def wave_stats(recs: list) -> dict:
    t0 = min(r["t_start_s"] for r in recs)
    t1 = max(r["t_complete_s"] for r in recs)
    span = t1 - t0
    tokens = sum(r["output_tokens"] for r in recs)
    ttft = [r["ttft_s"] for r in recs]
    first_max = max(r["t_first_content_s"] for r in recs)
    last_min = min(r["t_last_content_s"] for r in recs)
    run_peak = [_engine(r).get("requests_running_peak") for r in recs]
    wait_peak = [_engine(r).get("requests_waiting_peak") for r in recs]
    kv = [_engine(r).get("kv_cache_usage_peak") for r in recs]
    gmem = [_gpu(r).get("peak_used_mib_in_window") for r in recs]
    pick = lambda xs: max((x for x in xs if x is not None), default=None)  # noqa: E731
    return {"wave_id": recs[0]["wave_id"], "n": len(recs), "span_s": span, "output_tokens": tokens,
            "agg_output_tok_s": tokens / span if span > 0 else None,
            "start_skew_s": max(r["t_start_s"] for r in recs) - t0,
            "max_in_flight": max_in_flight(recs),
            "ttft_first_s": min(ttft), "ttft_last_s": max(ttft),
            "co_decode_s": max(0.0, last_min - first_max),
            "engine_running_peak": pick(run_peak), "engine_waiting_peak": pick(wait_peak),
            "kv_cache_usage_peak": pick(kv), "gpu_peak_used_mib": pick(gmem)}


def complete_waves(records: list) -> dict:
    """wave_id -> usable records, for measured waves in which every request is usable."""
    groups = defaultdict(list)
    for r in records:
        if r.get("wave_id") and not r.get("is_warmup"):
            groups[r["wave_id"]].append(r)
    return {k: v for k, v in groups.items()
            if len(v) == v[0]["wave_size"] and all(x["usable_for_analysis"] for x in v)}


def summarize_cells(records: list) -> dict:
    ws = complete_waves(records)
    cells = defaultdict(list)
    for recs in ws.values():
        cells[(recs[0]["level"], recs[0]["concurrency"])].append(recs)
    out = {}
    for (level, n) in sorted(cells):
        waves = cells[(level, n)]
        flat = [r for recs in waves for r in recs]
        stats = [wave_stats(recs) for recs in waves]
        out[f"{level}x{n}"] = {
            "level": level, "concurrency": n, "n_waves": len(waves), "n_requests": len(flat),
            "ttft_s": mt.describe(r["ttft_s"] for r in flat),
            "ttft_first_s": mt.describe(s["ttft_first_s"] for s in stats),
            "ttft_last_s": mt.describe(s["ttft_last_s"] for s in stats),
            "e2e_latency_s": mt.describe(r["e2e_latency_s"] for r in flat),
            "decode_tokens_per_s_per_request": mt.describe(r["decode_tokens_per_s"] for r in flat),
            "agg_output_tok_s": mt.describe(s["agg_output_tok_s"] for s in stats),
            "span_s": mt.describe(s["span_s"] for s in stats),
            "co_decode_s": mt.describe(s["co_decode_s"] for s in stats),
            "max_in_flight": max(s["max_in_flight"] for s in stats),
            "start_skew_s_max": max(s["start_skew_s"] for s in stats),
            "engine_running_peak": max((s["engine_running_peak"] for s in stats
                                        if s["engine_running_peak"] is not None), default=None),
            "engine_waiting_peak": max((s["engine_waiting_peak"] for s in stats
                                        if s["engine_waiting_peak"] is not None), default=None),
            "kv_cache_usage_peak": max((s["kv_cache_usage_peak"] for s in stats
                                        if s["kv_cache_usage_peak"] is not None), default=None),
            "gpu_peak_used_mib": max((s["gpu_peak_used_mib"] for s in stats
                                      if s["gpu_peak_used_mib"] is not None), default=None),
        }
    return out


def batch_invariance(records: list) -> dict:
    """Does the same prompt give the same greedy output at every concurrency? Compared by output hash.
    Batching changes the shapes of the GPU kernels, so a difference is possible and is recorded."""
    by_prompt = defaultdict(lambda: defaultdict(set))
    for r in mt.usable(records):
        if r.get("concurrency") is not None:
            by_prompt[r["prompt_id"]][r["concurrency"]].add(r["output_sha256"])
    compared = {p: v for p, v in by_prompt.items() if len(v) >= 2}
    differing = sorted(p for p, v in compared.items() if len({h for hs in v.values() for h in hs}) > 1)
    within = sorted(p for p, v in by_prompt.items() if any(len(hs) > 1 for hs in v.values()))
    return {"prompts_compared_across_concurrency": len(compared), "differing_prompt_ids": differing,
            "prompts_differing_within_one_concurrency": within,
            "all_identical": not differing and not within}


def validate_waves(records: list) -> list:
    """Wave bookkeeping checks; empty list means consistent."""
    problems = []
    groups = defaultdict(list)
    for r in records:
        if "wave_id" in r:
            groups[r["wave_id"]].append(r)
    for wid, recs in groups.items():
        if wid == "warmup":
            continue
        size = recs[0]["wave_size"]
        if len(recs) != size:
            problems.append(f"{wid}: {len(recs)} records but wave_size {size}")
        if sorted(r["slot"] for r in recs) != list(range(len(recs))):
            problems.append(f"{wid}: slots are not 0..{len(recs) - 1}")
        if len({r["prompt_id"] for r in recs}) != len(recs):
            problems.append(f"{wid}: a prompt appears twice in one wave")
        if any(r["concurrency"] != size for r in recs):
            problems.append(f"{wid}: concurrency differs from wave_size")
        if len({r["is_warmup"] for r in recs}) != 1:
            problems.append(f"{wid}: mixes warmup and measured requests")
    return problems


def format_cell_table(records: list) -> str:
    cells = summarize_cells(records)
    head = (f"{'input':>6} {'N':>3} {'waves':>5} {'ttft med [min-max] s':>24} {'first/last tok s':>19} "
            f"{'dec/req tok/s':>13} {'agg tok/s med [min-max]':>25} {'run/wait':>9} {'inflt':>5} "
            f"{'codec s':>8} {'kv':>6}")
    lines = [head]
    for c in cells.values():
        t, f, la = c["ttft_s"], c["ttft_first_s"], c["ttft_last_s"]
        a, d = c["agg_output_tok_s"], c["decode_tokens_per_s_per_request"]
        ttft = f"{t['median']:.2f} [{t['min']:.2f}-{t['max']:.2f}]"
        fl = f"{f['median']:.2f}/{la['median']:.2f}"
        agg = f"{a['median']:.1f} [{a['min']:.1f}-{a['max']:.1f}]"
        rw = (f"{c['engine_running_peak']:.0f}/{c['engine_waiting_peak']:.0f}"
              if c["engine_running_peak"] is not None else "n/a")
        kv = f"{c['kv_cache_usage_peak']:.3f}" if c["kv_cache_usage_peak"] is not None else "n/a"
        lines.append(f"{c['level']:>6} {c['concurrency']:>3} {c['n_waves']:>5} {ttft:>24} {fl:>19} "
                     f"{d['median']:>13.1f} {agg:>25} {rw:>9} {c['max_in_flight']:>5} "
                     f"{c['co_decode_s']['median']:>8.2f} {kv:>6}")
    return "\n".join(lines)
