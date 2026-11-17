"""Step 8 concurrency grid: real Kaggle T4 data, vLLM 0.34.0, CUDA graphs on.

Run 20261117T062402Z-a80c526b: input 1024/4096/8192 x concurrency 1/2/4/8/16, 3 measured waves per cell
after one unmeasured warmup wave per cell, cell order rotated per wave. These tests pin what the stored data
says for this device, model and engine; there are 3 waves per cell, so nothing here is a tail estimate.
"""

import json
import math
import pathlib
import statistics
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import waves as wv  # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261117T062402Z-a80c526b"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="Step 8 run not present")
LEVELS, NS = (1024, 4096, 8192), (1, 2, 4, 8, 16)
BLOCKS, BLOCK_TOKENS = 4832, 64


@pytest.fixture(scope="module")
def recs():
    return mt.load_records(RUN / "requests.jsonl")


@pytest.fixture(scope="module")
def cells(recs):
    return wv.summarize_cells(recs)


def waves_of(recs, level, n):
    return [rs for rs in wv.complete_waves(recs).values()
            if rs[0]["level"] == level and rs[0]["concurrency"] == n]


def test_run_validates_with_no_failures_and_3_waves_per_cell(recs, cells):
    assert mt.validate_run_dir(RUN) == []
    m = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["counts"] == {"requests": 373, "ok": 373, "errors": 0, "warmup": 94, "usable": 279}
    assert m["package_versions"]["vllm"] == "0.34.0" and m["lock_mismatches"] == []
    assert len(cells) == 15
    assert all(c["n_waves"] == 3 and c["n_requests"] == 3 * c["concurrency"] for c in cells.values())
    assert {r["schema"] for r in recs} == {"kv-bench-request/3"}


def test_requests_really_overlapped_and_the_engine_ran_them_together(cells):
    for c in cells.values():
        assert c["max_in_flight"] == c["concurrency"]               # overlap, from client timestamps
        assert c["engine_running_peak"] == c["concurrency"]         # batching, from the engine gauge
        assert c["start_skew_s_max"] < 0.02                         # the barrier released them together
    assert all(cells[f"{lv}x1"]["engine_waiting_peak"] == 0 for lv in LEVELS)
    assert cells["8192x16"]["engine_waiting_peak"] >= 14


def test_prefill_is_serial_so_the_last_first_token_waits_for_all_the_others(recs):
    for lv in (4096, 8192):
        single = statistics.median(r["ttft_s"] for rs in waves_of(recs, lv, 1) for r in rs)
        for n in (2, 4, 8, 16):
            gaps = []
            for rs in waves_of(recs, lv, n):
                t = sorted(r["ttft_s"] for r in rs)
                gaps += [b - a for a, b in zip(t, t[1:])]
            assert statistics.median(gaps) / single == pytest.approx(1.0, abs=0.02), (lv, n)
            assert min(gaps) > 0.5 * single                        # no two prefills finished together
    # at 1K a few prefills do finish in the same step (zero gap), which long prompts never do
    gaps_1k = []
    for rs in waves_of(recs, 1024, 16):
        t = sorted(r["ttft_s"] for r in rs)
        gaps_1k += [b - a for a, b in zip(t, t[1:])]
    assert min(gaps_1k) < 0.05


def test_first_token_order_is_not_slot_order(recs):
    waves = [rs for rs in wv.complete_waves(recs).values() if rs[0]["concurrency"] >= 4]
    same = sum([r["slot"] for r in sorted(rs, key=lambda r: r["ttft_s"])] == sorted(r["slot"] for r in rs)
               for rs in waves)
    assert same <= 2 and len(waves) == 27                          # arrival order at the server is thread timing


def test_aggregate_throughput_gain_depends_on_how_much_of_the_wave_is_prefill(cells):
    def agg(lv, n):
        return cells[f"{lv}x{n}"]["agg_output_tok_s"]["median"]
    assert 4.8 < agg(1024, 16) / agg(1024, 1) < 5.6               # short prompts: decode batching pays
    assert 1.4 < agg(4096, 16) / agg(4096, 1) < 1.7
    assert 1.1 < agg(8192, 16) / agg(8192, 1) < 1.25              # long prompts: serial prefill dominates
    for lv in LEVELS:
        a = [agg(lv, n) for n in NS]
        assert a == sorted(a)                                      # never falls as N rises, here
        assert a[-1] / a[-2] < 1.3                                 # flattening toward a limit


def test_latency_of_the_last_request_grows_about_linearly_with_n_for_long_prompts(cells):
    for lv in (4096, 8192):
        last = [cells[f"{lv}x{n}"]["ttft_last_s"]["median"] for n in NS]
        for n, t in zip(NS, last):
            assert t / (n * last[0]) == pytest.approx(1.0, abs=0.04)


def test_per_stream_decode_rate_falls_with_n_and_is_confounded_by_other_prefills(cells):
    for lv in LEVELS:
        d = [cells[f"{lv}x{n}"]["decode_tokens_per_s_per_request"]["median"] for n in NS]
        assert d[-1] < d[0]
    assert cells["8192x16"]["decode_tokens_per_s_per_request"]["median"] < 2.0   # about 1.3 tok/s per stream
    assert cells["1024x16"]["decode_tokens_per_s_per_request"]["median"] > 20


def test_cache_usage_peak_is_the_sum_of_the_requests_within_one_sampled_block(recs):
    for rs in wv.complete_waves(recs).values():
        want = sum(math.ceil((r["prompt_tokens"] + r["output_tokens"] - 1) / BLOCK_TOKENS)
                   for r in rs) / (BLOCKS - 1)
        got = wv.wave_stats(rs)["kv_cache_usage_peak"]
        assert want - 1.5 / (BLOCKS - 1) <= got <= want + 1e-6, (rs[0]["wave_id"], got, want)
    top = max(c["kv_cache_usage_peak"] for c in wv.summarize_cells(recs).values())
    assert top == pytest.approx(0.4306, abs=0.001)


def test_device_memory_is_flat_across_concurrency_while_engine_cache_usage_moves(cells):
    gpu = {c["gpu_peak_used_mib"] for c in cells.values()}
    assert len(gpu) <= 2 and max(gpu) == 12953.1875               # preallocated pool: device memory cannot show demand
    assert cells["8192x16"]["kv_cache_usage_peak"] > 50 * cells["1024x1"]["kv_cache_usage_peak"]


def test_no_preemption_and_no_prefix_hits():
    m = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    d = m["memory_monitoring"]["engine_counter_deltas_over_run"]
    assert d["preemptions_total"] == 0 and d["prefix_cache_hits_total"] == 0


def test_greedy_output_is_stable_at_low_concurrency_and_varies_at_high_concurrency(recs):
    meas = [r for r in recs if not r["is_warmup"] and r.get("wave_id")]
    by = {}
    for r in meas:
        by.setdefault((r["concurrency"], r["prompt_id"]), set()).add(r["output_sha256"])
    assert all(len(v) == 1 for (n, _), v in by.items() if n <= 2)
    assert {n for (n, _), v in by.items() if len(v) > 1} == {4, 8, 16}   # recorded, not hidden
    inv = wv.batch_invariance(recs)
    assert not inv["all_identical"] and inv["prompts_compared_across_concurrency"] == 24
    ref = {r["prompt_id"]: r["output_text"] for r in meas if r["concurrency"] == 1}
    pos = {next(i for i, (a, b) in enumerate(zip(ref[r["prompt_id"]], r["output_text"])) if a != b)
           for r in meas if r["concurrency"] > 1 and r["prompt_id"] in ref
           and r["output_text"] != ref[r["prompt_id"]]}
    assert pos == {222}                                            # one divergence point in the 1K prompt's text
