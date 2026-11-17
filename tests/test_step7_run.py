"""Checks against the committed Step 7 pilot run (real Kaggle T4 data, enforce_eager OFF).

Run 20261117T032543Z-acb4745b is a pilot: n = 6 per level, two prompts per level. These tests
check that the stored data is internally consistent and that the figures quoted in the docs
are the ones in the records. They do not turn the pilot into a finding.
"""

import json
import math
import pathlib
import statistics
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import runner as rn  # noqa: E402
from benchmarks import sampling as sp  # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261117T032543Z-acb4745b"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="pilot run not present")


@pytest.fixture(scope="module")
def records():
    return mt.load_records(RUN / "requests.jsonl")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))


def test_run_validates(records):
    assert mt.validate_run_dir(RUN) == []
    assert len(records) == 22 and sum(r["usable_for_analysis"] for r in records) == 18


def test_forced_output_length_and_stop_reason(records):
    assert {r["output_tokens"] for r in records if r["prompt_id"] != "warmup"} == {128}
    assert {r["finish_reason"] for r in records} == {"length"}
    assert all(r["ignore_eos"] for r in records)


def test_level_order_rotates_and_warmups_are_spare_prompts(records, manifest):
    measured = [(r["trial_index"], r["target_input_tokens"]) for r in records if not r["is_warmup"]]
    per_trial = {}
    for trial, level in measured:
        per_trial.setdefault(trial, [])
        if level not in per_trial[trial]:
            per_trial[trial].append(level)
    assert per_trial == {0: [1024, 4096, 8192], 1: [4096, 8192, 1024], 2: [8192, 1024, 4096]}
    cfg = manifest["config"]
    assert not set(cfg["warmup_prompt_ids"]) & set(cfg["selected_prompt_ids"])


def test_repeats_are_bit_identical(records):
    d = mt.determinism(records)
    assert d["prompts_with_repeats"] == 6 and d["all_identical"]


def test_per_level_medians_match_the_quoted_figures(records):
    by = mt.summarize_by_level(records)
    assert by["1024"]["ttft_s"]["median"] == pytest.approx(0.322, abs=0.001)
    assert by["4096"]["ttft_s"]["median"] == pytest.approx(3.250, abs=0.001)
    assert by["8192"]["ttft_s"]["median"] == pytest.approx(11.862, abs=0.001)
    assert by["1024"]["decode_tokens_per_s"]["median"] == pytest.approx(50.0, abs=0.1)
    assert by["8192"]["decode_tokens_per_s"]["median"] == pytest.approx(41.6, abs=0.1)
    assert all(v["n"] == 6 and v["ttft_s"]["p95"] is None for v in by.values())  # n < 20


def test_ttft_grows_faster_than_input_length(records):
    med = {k: v["ttft_s"]["median"] for k, v in mt.summarize_by_level(records).items()}
    for a, b in (("1024", "4096"), ("4096", "8192")):
        growth = med[b] / med[a]
        assert growth > int(b) / int(a)  # super-linear in these medians (pilot, n = 6)


def test_engine_cache_usage_follows_the_block_formula(records, manifest):
    assert manifest["engine_reported_startup"]["kv_cache_tokens"] == 309248
    parsed = sp.parse_vllm_startup_log((RUN / "server.log").read_text(encoding="utf-8", errors="replace"))
    assert parsed["vllm_version_in_log"] == "0.34.0"
    assert parsed["kv_cache_blocks"] == 4832 and parsed["kv_block_size_tokens"] == 64
    for r in records:
        if r["prompt_id"] == "warmup":
            continue
        used = math.ceil((r["prompt_tokens"] + r["output_tokens"] - 1) / 64)
        expect = used / (parsed["kv_cache_blocks"] - 1)
        assert r["memory"]["engine"]["kv_cache_usage_peak"] == pytest.approx(expect, rel=1e-9)


def test_device_memory_grows_only_at_first_use_of_a_new_size(records):
    growth = [(r["prompt_id"], r["memory"]["gpu"]["peak_minus_window_start_mib"]) for r in records]
    first = {pid: g for pid, g in growth[:4]}
    assert first == {"warmup": 2.0, "L1024-002": 54.0, "L4096-002": 106.0, "L8192-002": 0.0}
    assert all(g == 0.0 for _, g in growth[4:])  # retained high-water mark


def test_prefix_cache_and_preemptions_stayed_flat(manifest):
    d = manifest["memory_monitoring"]["engine_counter_deltas_over_run"]
    assert d == {"preemptions_total": 0.0, "prefix_cache_hits_total": 0.0,
                 "prefix_cache_queries_total": 0.0}


def test_gpu_was_saturated_in_decode_with_cuda_graphs(records):
    gpu = [json.loads(l) for l in (RUN / "samples_gpu.jsonl").read_text().splitlines()]
    utils = []
    for r in records:
        if r["is_warmup"]:
            continue
        utils += [g["gpu_util_pct"] for g in gpu
                  if r["t_first_content_s"] <= g["t_s"] <= r["t_last_content_s"]]
    assert statistics.fmean(utils) > 95  # eager mode showed only 49-61 percent


def test_runner_records_installed_versions():
    v = rn.installed_versions(("pytest", "no_such_package_xyz"))
    assert v["pytest"] and v["no_such_package_xyz"] is None
    assert set(rn.TRACKED_PACKAGES) >= {"vllm", "torch", "transformers"}


def test_log_parser_reads_version_and_blocks_from_text():
    text = ("Initializing a V1 LLM engine (v0.34.0) with config\n"
            "GPU KV cache size: 309,248 tokens, Maximum concurrency for 32,768 tokens per request: 9.44x\n"
            "Extended KV cache to 4832 blocks (8.26 GiB).\n")
    p = sp.parse_vllm_startup_log(text)
    assert (p["vllm_version_in_log"], p["kv_cache_blocks"], p["kv_block_size_tokens"]) == ("0.34.0", 4832, 64)
    q = sp.parse_vllm_startup_log("GPU KV cache size: 333,904 tokens\n")  # older log: no block line
    assert q["kv_cache_blocks"] is None and q["kv_block_size_tokens"] is None
