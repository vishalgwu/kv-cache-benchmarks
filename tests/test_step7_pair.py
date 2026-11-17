"""Step 7 pilot pair (same prompts, same pinned stack): enforce_eager ON vs OFF, plus a repeat of OFF.

Runs (real Kaggle T4 data, vLLM 0.34.0):
  OFF  20261117T032543Z-acb4745b  and its repeat  20261117T041433Z-acb4745b
  ON   20261117T040952Z-6f8b82fb
n = 6 per level per run (2 prompts x 3 trials): these tests pin what the stored data says and do
not turn the pilot into a finding.
"""

import json
import pathlib
import statistics
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import sampling as sp  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
OFF1, OFF2, ON = (RES / "20261117T032543Z-acb4745b", RES / "20261117T041433Z-acb4745b",
                  RES / "20261117T040952Z-6f8b82fb")
pytestmark = pytest.mark.skipif(not all(p.exists() for p in (OFF1, OFF2, ON)), reason="pilot runs not present")
BYTES_PER_TOKEN = 28672


def recs(d):
    return mt.load_records(d / "requests.jsonl")


def med(d, level, field):
    return statistics.median(r[field] for r in recs(d) if not r["is_warmup"]
                             and r["target_input_tokens"] == level)


@pytest.mark.parametrize("d", [OFF1, OFF2, ON])
def test_every_run_validates_and_is_deterministic_within_itself(d):
    assert mt.validate_run_dir(d) == []
    assert mt.determinism(recs(d))["all_identical"]


def test_new_runs_record_the_pinned_stack_with_no_lock_mismatch():
    for d in (OFF2, ON):
        m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        assert m["package_versions"]["vllm"] == "0.34.0" and m["lock_mismatches"] == []


def test_repeat_of_eager_off_agrees_with_the_first_within_one_percent():
    for level in (1024, 4096, 8192):
        for field in ("ttft_s", "decode_tokens_per_s"):
            a, b = med(OFF1, level, field), med(OFF2, level, field)
            assert abs(b - a) / a < 0.01, (level, field, a, b)


def test_cuda_graphs_speed_up_decode_not_prefill():
    for level, lo, hi in ((1024, 1.8, 2.0), (4096, 1.8, 2.0), (8192, 1.5, 1.65)):
        speedup = med(OFF2, level, "decode_tokens_per_s") / med(ON, level, "decode_tokens_per_s")
        assert lo < speedup < hi, (level, speedup)
        ttft_change = abs(med(OFF2, level, "ttft_s") - med(ON, level, "ttft_s")) / med(ON, level, "ttft_s")
        assert ttft_change < 0.04  # prefill is compute-bound; graphs do not change it much


def test_eager_decode_is_flat_across_context_but_graph_decode_is_not():
    on = [med(ON, lv, "decode_tokens_per_s") for lv in (1024, 4096, 8192)]
    off = [med(OFF2, lv, "decode_tokens_per_s") for lv in (1024, 4096, 8192)]
    assert max(on) / min(on) < 1.02          # launch-bound: hides attention cost
    assert off[2] / off[0] < 0.85            # with graphs, 8K decode is clearly slower


def test_decode_gpu_use_eager_vs_graphs():
    def decode_util(d):
        gpu = [json.loads(l) for l in (d / "samples_gpu.jsonl").read_text().splitlines()]
        u = []
        for r in recs(d):
            if not r["is_warmup"]:
                u += [g["gpu_util_pct"] for g in gpu if r["t_first_content_s"] <= g["t_s"] <= r["t_last_content_s"]]
        return statistics.fmean(u)
    assert decode_util(ON) < 60 and decode_util(OFF2) > 95


def test_outputs_match_across_modes_except_late_tokens_at_8k():
    h = {}
    for name, d in (("off1", OFF1), ("off2", OFF2), ("on", ON)):
        for r in recs(d):
            h.setdefault(r["prompt_id"], {}).setdefault(name, set()).add(r["output_sha256"])
    for pid, by_run in h.items():
        assert by_run["off1"] == by_run["off2"]            # graph run repeats exactly
        same_as_eager = by_run["on"] == by_run["off2"]
        assert same_as_eager or pid in {"L8192-000", "L8192-001"}, pid
    assert h["L8192-000"]["on"] != h["L8192-000"]["off2"]  # recorded, not hidden


def test_final_cache_pool_matches_the_size_formula_and_the_log_estimate_does_not():
    for d, tokens, gib in ((ON, 325632, 8.70), (OFF2, 309248, 8.26)):
        p = sp.parse_vllm_startup_log((d / "server.log").read_text(encoding="utf-8", errors="replace"))
        assert p["kv_cache_tokens"] == tokens and p["kv_cache_final_gib"] == gib
        assert tokens * BYTES_PER_TOKEN / 2 ** 30 == pytest.approx(gib, abs=0.01)  # estimator vs engine
    # the profiling line says 8.92 GiB for both, which is not the pool vLLM built with graphs on
    p = sp.parse_vllm_startup_log((OFF2 / "server.log").read_text(encoding="utf-8", errors="replace"))
    assert p["available_kv_cache_gib"] == 8.92 and p["kv_cache_final_gib"] != 8.92


def test_graph_capture_costs_five_percent_of_cache_capacity():
    on = json.loads((ON / "run_manifest.json").read_text())["engine_reported_startup"]["kv_cache_tokens"]
    off = json.loads((OFF2 / "run_manifest.json").read_text())["engine_reported_startup"]["kv_cache_tokens"]
    assert (on - off) / on == pytest.approx(0.0503, abs=0.001)


def test_first_short_request_without_a_sample_is_unavailable_not_zero():
    first = recs(OFF2)[0]
    assert first["memory"]["gpu"]["label"] == "unavailable" or first["memory"]["gpu"].get("reason")
