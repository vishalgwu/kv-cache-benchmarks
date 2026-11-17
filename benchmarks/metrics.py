"""Metric calculations and run validation (Step 6).

Everything here is recomputed from stored request records, never from live state,
so any reported number can be audited against ``requests.jsonl``.

Definitions are in docs/measurement_contract.md. In short:

* Only records with ``usable_for_analysis`` true enter a statistic: status ok, not
  a warmup, no validation errors. Failures are counted and reported separately.
* Percentiles use linear interpolation and are reported only when the sample is
  large enough: p95 needs n >= 20 and p99 needs n >= 100. Below that the value is
  None with a note, because a small trial set cannot establish tail latency.

Command line:  python -m benchmarks.metrics results/<run_id>
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
from collections import Counter
from typing import Iterable, Optional

from benchmarks import runner, sampling

TAIL_MIN_SAMPLES = {"p95": 20, "p99": 100}
REQUEST_SCHEMAS = {"kv-bench-request/1", "kv-bench-request/2"}


def load_records(path: pathlib.Path) -> list:
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile (p in 0..100); None for an empty input."""
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def describe(values: Iterable[Optional[float]]) -> dict:
    """Summary with an explicit sample count and tail-percentile gating."""
    xs = [v for v in values if v is not None and math.isfinite(v)]
    n = len(xs)
    out = {"n": n, "min": min(xs) if xs else None, "median": statistics.median(xs) if xs else None,
           "mean": statistics.fmean(xs) if xs else None,
           "stdev": statistics.stdev(xs) if n >= 2 else None, "max": max(xs) if xs else None}
    notes = []
    for name, p in (("p95", 95), ("p99", 99)):
        if n >= TAIL_MIN_SAMPLES[name]:
            out[name] = percentile(xs, p)
        else:
            out[name] = None
            notes.append(f"{name} needs n>={TAIL_MIN_SAMPLES[name]} (n={n})")
    out["tail_note"] = "; ".join(notes) or None
    return out


def usable(records: list) -> list:
    return [r for r in records if r.get("usable_for_analysis")]


def aggregate_throughput(records: list) -> dict:
    """Aggregate output throughput over the included (usable) records.

    ``span`` runs from the first submission to the last completion, so it includes
    any idle gaps between sequential requests. ``busy`` sums per-request end-to-end
    latency. For overlapping requests (Step 8) use ``span``; the two coincide only
    for back-to-back requests. Failures are never included.
    """
    ok = usable(records)
    if not ok:
        return {"label": "unavailable", "reason": "no usable records", "n_requests": 0}
    span = max(r["t_complete_s"] for r in ok) - min(r["t_start_s"] for r in ok)
    busy = sum(r["e2e_latency_s"] for r in ok)
    tokens = sum(r["output_tokens"] for r in ok)
    return {"label": "measured", "n_requests": len(ok), "output_tokens_total": tokens,
            "span_s": span, "busy_s": busy,
            "output_tokens_per_s_over_span": tokens / span if span > 0 else None,
            "output_tokens_per_s_over_busy_time": tokens / busy if busy > 0 else None,
            "token_source": "engine usage completion_tokens"}


def summarize_run(records: list) -> dict:
    ok = usable(records)
    measured = [r for r in records if not r.get("is_warmup")]
    failures = Counter(r["error_category"] for r in measured if r["status"] != "ok")
    invalid = [r["request_id"] for r in measured if r["status"] == "ok" and not r["usable_for_analysis"]]
    return {
        "counts": {"total": len(records), "warmup": sum(bool(r.get("is_warmup")) for r in records),
                   "measured": len(measured), "usable": len(ok),
                   "failed": sum(failures.values()), "ok_but_invalid": len(invalid)},
        "failures_by_category": dict(failures),
        "ok_but_invalid_request_ids": invalid,
        "ttft_s": describe(r["ttft_s"] for r in ok),
        "e2e_latency_s": describe(r["e2e_latency_s"] for r in ok),
        "decode_tokens_per_s": describe(r["decode_tokens_per_s"] for r in ok),
        "output_tokens": describe(r["output_tokens"] for r in ok),
        "aggregate": aggregate_throughput(records),
        "by_level": summarize_by_level(records),
        "determinism": determinism(records),
    }


def summarize_by_level(records: list) -> dict:
    """Per input-length statistics over usable records, with n always shown."""
    levels = sorted({r["target_input_tokens"] for r in usable(records)
                     if r.get("target_input_tokens") is not None})
    out = {}
    for lv in levels:
        rs = [r for r in usable(records) if r["target_input_tokens"] == lv]
        out[str(lv)] = {"n": len(rs), "prompts": len({r["prompt_id"] for r in rs}),
                        "ttft_s": describe(r["ttft_s"] for r in rs),
                        "decode_tokens_per_s": describe(r["decode_tokens_per_s"] for r in rs),
                        "e2e_latency_s": describe(r["e2e_latency_s"] for r in rs),
                        "output_tokens": describe(r["output_tokens"] for r in rs)}
    return out


def determinism(records: list) -> dict:
    """Do repeats of the same prompt give identical greedy output? Compared by output hash."""
    groups: dict = {}
    for r in usable(records):
        groups.setdefault(r["prompt_id"], []).append(r["output_sha256"])
    rep = {pid: {"n": len(h), "identical": len(set(h)) == 1} for pid, h in groups.items() if len(h) >= 2}
    return {"prompts_with_repeats": len(rep), "all_identical": all(v["identical"] for v in rep.values()),
            "differing_prompt_ids": sorted(k for k, v in rep.items() if not v["identical"]),
            "note": ("greedy decoding is not guaranteed bit-identical on a GPU; a difference is "
                     "recorded, not hidden") if rep else "no prompt was repeated"}


def format_level_table(records: list) -> str:
    """Compact per-level table (medians with min-max, n always shown). Prefill speed is the input
    length over the median TTFT, so it includes queueing and transport; it is a client-side figure."""
    by = summarize_by_level(records)
    rows = {int(k): v for k, v in by.items()}
    ok = usable(records)
    head = (f"{'input':>7} {'n':>3} {'ttft med [min-max] s':>26} {'prefill tok/s':>14} "
            f"{'decode med [min-max] tok/s':>28} {'e2e med s':>10} {'kv peak':>8}")
    lines = [head]
    for lv in sorted(rows):
        v = rows[lv]
        t, d, e = v["ttft_s"], v["decode_tokens_per_s"], v["e2e_latency_s"]
        kv = [r["memory"]["engine"].get("kv_cache_usage_peak") for r in ok
              if r["target_input_tokens"] == lv and "memory" in r]
        kv = [x for x in kv if x is not None]
        ttft = f"{t['median']:.3f} [{t['min']:.3f}-{t['max']:.3f}]"
        dec = f"{d['median']:.1f} [{d['min']:.1f}-{d['max']:.1f}]"
        kvs = f"{max(kv):.4f}" if kv else "n/a"
        lines.append(f"{lv:>7} {v['n']:>3} {ttft:>26} {lv / t['median']:>14.0f} {dec:>28} "
                     f"{e['median']:>10.2f} {kvs:>8}")
    return "\n".join(lines)


def validate_run_dir(run_dir: pathlib.Path) -> list:
    """Cross-check a run directory. Returns problems; an empty list means consistent."""
    run_dir = pathlib.Path(run_dir)
    problems = []
    try:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        records = load_records(run_dir / "requests.jsonl")
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read run files: {exc}"]

    if runner.config_hash(manifest["config"]) != manifest["config_hash"]:
        problems.append("manifest config_hash does not match its config")
    counts = manifest.get("counts") or {}
    recount = {"requests": len(records), "ok": sum(r["status"] == "ok" for r in records),
               "errors": sum(r["status"] != "ok" for r in records),
               "warmup": sum(bool(r["is_warmup"]) for r in records),
               "usable": sum(bool(r["usable_for_analysis"]) for r in records)}
    for key, value in recount.items():
        if counts.get(key) != value:
            problems.append(f"manifest counts[{key}]={counts.get(key)} but records give {value}")

    ids = [r["request_id"] for r in records]
    if len(set(ids)) != len(ids):
        problems.append("duplicate request_id")
    for r in records:
        rid = r["request_id"]
        if r.get("schema") not in REQUEST_SCHEMAS:
            problems.append(f"{rid}: unknown schema {r.get('schema')!r}")
        if r["run_id"] != manifest["run_id"]:
            problems.append(f"{rid}: run_id differs from manifest")
        if r["config_hash"] != manifest["config_hash"]:
            problems.append(f"{rid}: config_hash differs from manifest")
        if r["validation_errors"] != runner.validate_record(r):
            problems.append(f"{rid}: stored validation_errors do not match a recomputation")
        expect_usable = r["status"] == "ok" and not r["is_warmup"] and not r["validation_errors"]
        if r["usable_for_analysis"] != expect_usable:
            problems.append(f"{rid}: usable_for_analysis={r['usable_for_analysis']} should be {expect_usable}")
        if r["status"] != "ok" and r["usable_for_analysis"]:
            problems.append(f"{rid}: failed request marked usable")
        if r["status"] == "ok" and r["ttft_s"] is not None:
            if abs((r["t_first_content_s"] - r["t_start_s"]) - r["ttft_s"]) > 1e-6:
                problems.append(f"{rid}: ttft_s cannot be recomputed from timestamps")
        if "memory" in r:
            problems += [f"{rid}: {e}" for e in sampling.validate_memory_block(r["memory"])]
    return problems


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--table"]
    if len(args) != 1:
        sys.exit("usage: python -m benchmarks.metrics [--table] results/<run_id>")
    run_dir = pathlib.Path(args[0])
    problems = validate_run_dir(run_dir)
    print("validation:", "OK" if not problems else f"{len(problems)} problem(s)")
    for p in problems:
        print("  -", p)
    records = load_records(run_dir / "requests.jsonl")
    if "--table" in sys.argv:
        print(format_level_table(records))
        print("determinism:", json.dumps(determinism(records)))
    else:
        print(json.dumps(summarize_run(records), indent=2))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
