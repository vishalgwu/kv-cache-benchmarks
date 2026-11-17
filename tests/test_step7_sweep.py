"""Step 7 context-length sweep: real Kaggle T4 data, vLLM 0.34.0, CUDA graphs on.

Run 20261117T053513Z-615f165f: levels 1024/4096/8192/16384/32000, 2 prompts per level, 3 trials with the
level order rotated (n = 6 per level), 128 output tokens forced, one warmup per level.
These tests pin what the stored data says; they do not generalise beyond this device, model and engine.
"""

import json
import math
import pathlib
import statistics
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261117T053513Z-615f165f"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="sweep run not present")
LEVELS = (1024, 4096, 8192, 16384, 32000)
BLOCKS, BLOCK_TOKENS = 4832, 64


def recs():
    return mt.load_records(RUN / "requests.jsonl")


def measured(level):
    return [r for r in recs() if not r["is_warmup"] and r["target_input_tokens"] == level]


def med(level, field):
    return statistics.median(r[field] for r in measured(level))


def test_run_validates_with_no_failures_and_n6_per_level():
    assert mt.validate_run_dir(RUN) == []
    s = mt.summarize_run(recs())
    assert s["counts"] == {"total": 36, "warmup": 6, "measured": 30, "usable": 30, "failed": 0, "ok_but_invalid": 0}
    assert all(s["by_level"][str(lv)]["n"] == 6 for lv in LEVELS)
    assert s["determinism"]["all_identical"] and s["determinism"]["prompts_with_repeats"] == 10


def test_pinned_stack_and_engine_pool():
    m = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["package_versions"]["vllm"] == "0.34.0" and m["lock_mismatches"] == []
    e = m["engine_reported_startup"]
    assert (e["kv_cache_blocks"], e["kv_block_size_tokens"], e["kv_cache_tokens"]) == (4832, 64, 309248)
    assert m["config"]["level_order"] == "rotate" and m["config"]["warmup_per_level"] is True


def test_one_warmup_per_level_at_the_measured_shape_and_levels_rotate():
    r = recs()
    warm = [x for x in r if x["is_warmup"] and x["prompt_id"] != "warmup"]
    assert [w["target_input_tokens"] for w in warm] == list(LEVELS)
    first = [x["target_input_tokens"] for x in r if not x["is_warmup"] and x["trial_index"] == 0][::2]
    assert first == list(LEVELS)                            # trial 0 ascending
    second = [x["target_input_tokens"] for x in r if not x["is_warmup"] and x["trial_index"] == 1][::2]
    assert second != first and sorted(second) == sorted(first)


def test_prompt_token_counts_match_the_manifest_and_one_prompt_is_off_by_one_token():
    off = {(x["prompt_id"], x["prompt_tokens"]) for x in recs()
           if x["prompt_tokens"] != x["target_input_tokens"] and x["prompt_id"] != "warmup"}
    assert off == {("L4096-002", 4095), ("L8192-002", 8191), ("L16384-001", 16383)}
    assert all(x["prompt_tokens"] == x["input_tokens_manifest"] for x in recs() if x["prompt_id"] != "warmup")


def test_cache_usage_peak_follows_the_block_formula_at_every_level():
    for x in recs():
        if x["prompt_id"] == "warmup":
            continue
        want = math.ceil((x["prompt_tokens"] + x["output_tokens"] - 1) / BLOCK_TOKENS) / (BLOCKS - 1)
        assert x["memory"]["engine"]["kv_cache_usage_peak"] == pytest.approx(want, abs=1e-6), x["request_id"]
    # 32K context uses about a tenth of the pool; the pool is not the limit at concurrency 1
    assert max(x["memory"]["engine"]["kv_cache_usage_peak"] for x in recs()) < 0.11


def test_prefill_is_superlinear_and_slope_approaches_two():
    t = [med(lv, "ttft_s") for lv in LEVELS]
    slopes = [(math.log(t[i + 1]) - math.log(t[i])) / (math.log(LEVELS[i + 1]) - math.log(LEVELS[i]))
              for i in range(4)]
    assert all(1.6 < s < 2.0 for s in slopes) and slopes == sorted(slopes)
    assert 160 < t[4] < 175 and 44 < t[3] < 46             # 32K takes about 2.8 minutes


def test_quadratic_through_1k_4k_8k_predicted_16k_and_32k_within_two_percent():
    t = {lv: med(lv, "ttft_s") for lv in LEVELS}
    p = np.polyfit(LEVELS[:3], [t[lv] for lv in LEVELS[:3]], 2)
    for lv in (16384, 32000):
        assert abs(np.polyval(p, lv) - t[lv]) / t[lv] < 0.02, lv


def test_ttft_is_stable_across_trials_and_trial_order_does_not_move_it():
    for lv in LEVELS:
        xs = [r["ttft_s"] for r in measured(lv)]
        assert (max(xs) - min(xs)) / statistics.median(xs) < 0.06
        by_trial = [statistics.mean(r["ttft_s"] for r in measured(lv) if r["trial_index"] == k) for k in (0, 1, 2)]
        assert (max(by_trial) - min(by_trial)) / statistics.median(by_trial) < 0.02


def test_decode_slows_steadily_with_context_with_graphs_on():
    d = [med(lv, "decode_tokens_per_s") for lv in LEVELS]
    assert d == sorted(d, reverse=True)
    assert d[0] / d[4] > 2.2 and 22 < d[4] < 23.5           # 32K decodes at about 22.8 tok/s


def test_gpu_ran_at_the_software_power_cap_throughout():
    g = [json.loads(line) for line in (RUN / "samples_gpu.jsonl").read_text().splitlines()]
    masks = [x["throttle_reasons_mask"] for x in g]
    assert set(masks) == {0, 4} and masks.count(4) / len(masks) > 0.99
    clocks = [x["sm_clock_mhz"] for x in g]
    assert min(clocks) < 700 and max(clocks) > 1500         # clock is not constant, so it is a confounder
    assert max(x["temperature_c"] for x in g) <= 85


def test_no_preemption_and_no_prefix_hits():
    m = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    d = m["memory_monitoring"]["engine_counter_deltas_over_run"]
    assert d["preemptions_total"] == 0 and d["prefix_cache_hits_total"] == 0
