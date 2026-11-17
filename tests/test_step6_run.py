"""Checks against the committed real Step 6 run (Kaggle T4, vLLM 0.33.0, 2026-11-16).

This is REAL measured data. These tests assert what the data shows and guard against
the data or the analysis code drifting. The run is a sampler-validation run, not a
baseline: eager mode, one trial, two prompts per context level.
"""

import json
import math
import pathlib
import shutil
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261116T223433Z-53df2b80"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="committed Step 6 run not present")


def records():
    return mt.load_records(RUN / "requests.jsonl")


def manifest():
    return json.loads((RUN / "run_manifest.json").read_text())


def test_run_validates_and_every_memory_section_is_measured():
    assert mt.validate_run_dir(RUN) == []
    recs = records()
    assert len(recs) == 5 and [r["is_warmup"] for r in recs] == [True, False, False, False, False]
    for r in recs:
        assert r["schema"] == "kv-bench-request/2" and r["memory"]["gpu"]["source"] == "nvml"
        assert all(r["memory"][s]["label"] == "measured" for s in ("gpu", "engine", "cpu"))
    assert [r["usable_for_analysis"] for r in recs] == [False, True, True, True, True]


def test_engine_cache_usage_matches_block_arithmetic():
    """vLLM's own occupancy figure reproduces the cache-size formula to six digits.

    Inferred from the data and reproduced for all four measured requests: 16-token
    blocks, one block reserved by the engine, and the final generated token not yet
    cached. If this fails, one of those inferences or the estimator is wrong.
    """
    startup = manifest()["engine_reported_startup"]
    blocks_total = startup["kv_cache_tokens"] // 16
    assert blocks_total * 16 == startup["kv_cache_tokens"]
    for r in records():
        if r["is_warmup"]:
            continue
        cached_tokens = r["prompt_tokens"] + r["output_tokens"] - 1
        expected = math.ceil(cached_tokens / 16) / (blocks_total - 1)
        got = r["memory"]["engine"]["kv_cache_usage_peak"]
        assert got == pytest.approx(expected, rel=1e-6), r["prompt_id"]


def test_device_memory_is_flat_and_grows_only_on_the_first_request_of_each_size():
    recs = records()
    base = recs[1]["memory"]["gpu"]["baseline_used_mib"]
    assert base == 13043.1875
    peaks = {r["prompt_id"]: r["memory"]["gpu"]["peak_used_mib_in_window"] for r in recs}
    assert peaks["L1024-000"] - base == 56 and peaks["L4096-000"] - base == 162
    assert peaks["L1024-001"] == peaks["L1024-000"] and peaks["L4096-001"] == peaks["L4096-000"]
    # Tiny next to the 8.92 GiB cache pool: device memory does not track cache occupancy.
    assert (peaks["L4096-001"] - base) < 0.02 * (8.92 * 1024)


def test_engine_metric_names_and_scale_were_confirmed_on_the_real_engine():
    catalog = (RUN / "engine_metrics_catalog.tsv").read_text()
    for name in ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
                 "vllm:num_preemptions_total", "vllm:prefix_cache_hits_total",
                 "vllm:prefix_cache_queries_total"):
        assert name in catalog
    assert "1 means 100 percent" in catalog  # a fraction, not a percentage
    assert records()[1]["memory"]["engine"]["metric_names"]["kv_cache_usage"] == "vllm:kv_cache_usage_perc"


def test_single_request_run_never_queued_or_preempted():
    for r in records()[1:]:
        e = r["memory"]["engine"]
        assert e["requests_running_peak"] == 1.0 and e["requests_waiting_peak"] == 0.0
        assert e["preemptions_in_window"] == 0.0


def test_sampler_cadence_was_tight_with_no_read_errors():
    mm = manifest()["memory_monitoring"]
    for name in ("gpu", "engine", "cpu"):
        assert mm[name]["read_errors"] == 0 and mm[name]["n_samples"] == 113
        assert abs(mm[name]["mean_gap_s"] - 0.25) < 0.001 and mm[name]["max_gap_s"] < 0.26


def test_generation_config_fix_removed_the_sampling_override():
    log = (RUN / "server.log").read_text(encoding="utf-8")
    assert "overridden by the model's `generation_config.json`" not in log
    assert "'generation_config': 'vllm'" in log
    assert manifest()["config"]["server"]["generation_config"] == "vllm"


def test_same_prompt_across_sessions_and_the_effect_of_the_sampling_fix():
    """L1024-000 has the same prompt hash as in the first run, so the generator is reproducible
    across sessions. Its output differs because the first run used the model's repetition
    penalty (1.1); the fix changed the sampling, as intended."""
    first = mt.load_records(RUN.parent / "20261116T210839Z-d6a00143" / "requests.jsonl")[1]
    new = records()[1]
    assert new["prompt_id"] == first["prompt_id"] == "L1024-000"
    assert new["prompt_sha256"] == first["prompt_sha256"]
    assert new["output_sha256"] != first["output_sha256"]


def test_validation_catches_a_tampered_memory_block(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RUN, run)
    recs = mt.load_records(run / "requests.jsonl")
    recs[1]["memory"]["gpu"]["label"] = "guess"
    (run / "requests.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    problems = mt.validate_run_dir(run)
    assert any("bad label" in p for p in problems)
    assert any("do not match a recomputation" in p for p in problems)
