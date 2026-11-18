"""Step 8 extension, 32K concurrency: real Kaggle T4 data (2026-11-18 session), vLLM 0.34.0, CUDA graphs on.

Run 20261118T212830Z-c47924e0: input 32,000 x concurrency 1/4/8/10, one unmeasured warmup wave at N=1, then ONE
measured wave per cell (a single observation per cell, not a distribution). KV pool 309,248 tokens: one 32K request
needs about 502 of the 4,831 usable blocks, so 9 fit and 10 do not.
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

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261118T212830Z-c47924e0"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="32K run not present")
NS = (1, 4, 8, 10)
BLOCKS, BLOCK_TOKENS = 4832, 64


@pytest.fixture(scope="module")
def recs():
    return mt.load_records(RUN / "requests.jsonl")


@pytest.fixture(scope="module")
def cells(recs):
    return wv.summarize_cells(recs)


def wave(recs, n):
    (rs,) = [x for x in wv.complete_waves(recs).values() if x[0]["concurrency"] == n]
    return rs


def test_run_validates_on_the_pinned_stack_with_the_same_pool(cells):
    assert mt.validate_run_dir(RUN) == []
    m = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["counts"] == {"requests": 25, "ok": 25, "errors": 0, "warmup": 2, "usable": 23}
    assert m["package_versions"]["vllm"] == "0.34.0" and m["lock_mismatches"] == []
    e = m["engine_reported_startup"]
    assert (e["kv_cache_blocks"], e["kv_cache_tokens"], e["max_concurrency_reported"]) == (4832, 309248, 9.44)
    assert m["config"]["timeout_s"] == 7200.0 and m["config"]["waves_per_cell"] == 1
    assert [c["concurrency"] for c in cells.values()] == list(NS) and all(c["n_waves"] == 1 for c in cells.values())
    d = m["memory_monitoring"]["engine_counter_deltas_over_run"]
    assert d["preemptions_total"] == 0 and d["prefix_cache_hits_total"] == 0


def test_the_engine_never_ran_more_than_nine_32k_requests_at_once(cells):
    for n in (1, 4, 8):
        assert cells[f"32000x{n}"]["engine_running_peak"] == n
    c = cells["32000x10"]
    assert c["max_in_flight"] == 10                     # the client had all ten in flight
    assert c["engine_running_peak"] == 9                # the engine never had more than nine running
    assert c["engine_waiting_peak"] == 9
    assert c["kv_cache_usage_peak"] == pytest.approx(0.934, abs=0.002)   # nine requests worth of cache


def test_the_tenth_request_was_admitted_when_the_first_one_freed_its_blocks(recs):
    rs = wave(recs, 10)
    t0 = min(r["t_start_s"] for r in rs)
    first_done = min(r["t_complete_s"] for r in rs) - t0
    engine = [json.loads(line) for line in (RUN / "samples_engine.jsonl").read_text().splitlines()]
    series = [(e["t_s"] - t0, e["requests_running"], e["requests_waiting"]) for e in engine
              if t0 <= e["t_s"] <= t0 + 1700]
    assert max(r for _, r, _ in series) == 9
    waiting_before = [w for t, r, w in series if t < first_done - 1.0 and r == 9]
    assert waiting_before and min(waiting_before) == 1  # a tenth request sat in the queue with nine running
    after = [(t, r, w) for t, r, w in series if first_done - 0.01 <= t <= first_done + 2.0]
    assert any(r == 9 and w == 0 for _, r, w in after)  # admitted within two seconds of the first completion
    # the 32K wave is serial prefill anyway: the capacity wait and the serial queue release at the same moment,
    # so this cell cannot separate their costs (recorded as such)
    tenth_ttft = max(r["ttft_s"] for r in rs)
    single = wave(recs, 1)[0]["ttft_s"]
    assert tenth_ttft / (10 * single) == pytest.approx(1.0, abs=0.02)


def test_prefill_is_serial_at_32k(recs):
    single = wave(recs, 1)[0]["ttft_s"]
    assert 160 < single < 166
    for n in (4, 8, 10):
        t = sorted(r["ttft_s"] for r in wave(recs, n))
        gaps = [b - a for a, b in zip(t, t[1:])]
        assert statistics.median(gaps) / single == pytest.approx(1.0, abs=0.02), n
        assert t[-1] / (n * single) == pytest.approx(1.0, abs=0.02), n
    assert max(r["ttft_s"] for r in wave(recs, 10)) > 1600      # the last request waits about 27 minutes


def test_aggregate_throughput_is_flat_at_32k(cells):
    agg = [cells[f"32000x{n}"]["agg_output_tok_s"]["median"] for n in NS]
    assert all(0.74 < a < 0.80 for a in agg) and max(agg) / min(agg) < 1.03


def test_running_streams_advance_one_token_per_prefill_step(recs):
    # The earliest request at N=10 decodes at about 0.1 tok/s, i.e. one token every ~10 s. One 32K prefill is 16
    # chunks of 2,048 tokens if the scheduler budget is 2,048 tokens per step, so a step is ~10.2 s. This arithmetic
    # matches (it is NOT read from the server log, which does not record the budget; see open items).
    single = wave(recs, 1)[0]["ttft_s"]
    rs = sorted(wave(recs, 10), key=lambda r: r["t_first_content_s"])
    interval = 1 / rs[0]["decode_tokens_per_s"]
    assert interval == pytest.approx(single / 16, rel=0.1)
    assert rs[0]["decode_tokens_per_s"] < 0.15 and rs[-1]["decode_tokens_per_s"] > 8     # the last one runs freely
    assert wave(recs, 1)[0]["decode_tokens_per_s"] == pytest.approx(23.5, abs=0.5)


def test_cache_usage_is_within_the_prompt_and_final_block_bounds_until_the_pool_is_full(recs):
    for n in (1, 4, 8):
        rs = wave(recs, n)
        low = sum(math.ceil(r["prompt_tokens"] / BLOCK_TOKENS) for r in rs) / (BLOCKS - 1)
        high = sum(math.ceil((r["prompt_tokens"] + r["output_tokens"] - 1) / BLOCK_TOKENS) for r in rs) / (BLOCKS - 1)
        assert low - 1e-6 <= wv.wave_stats(rs)["kv_cache_usage_peak"] <= high + 1e-6, n
    rs = wave(recs, 10)                                                 # ten requests would need more than the pool
    assert sum(math.ceil(r["prompt_tokens"] / BLOCK_TOKENS) for r in rs) / (BLOCKS - 1) > 1.0
    assert wv.wave_stats(rs)["kv_cache_usage_peak"] < 1.0


def test_n1_output_differs_from_the_batched_output_at_one_text_position(recs):
    meas = [r for r in recs if not r["is_warmup"] and r.get("wave_id")]
    ref = next(r["output_text"] for r in meas if r["concurrency"] == 1)
    batched = {n: next(r["output_text"] for r in meas if r["concurrency"] == n and r["prompt_id"] == "L32000-000")
               for n in (4, 8, 10)}
    assert len(set(batched.values())) == 1 and batched[4] != ref
    assert next(i for i, (a, b) in enumerate(zip(ref, batched[4])) if a != b) == 227
    inv = wv.batch_invariance(recs)
    assert inv["differing_prompt_ids"] == ["L32000-000"] and inv["prompts_differing_within_one_concurrency"] == []


def test_device_memory_flat_and_this_session_throttled_less_than_the_previous_one():
    g = [json.loads(line) for line in (RUN / "samples_gpu.jsonl").read_text().splitlines()]
    assert max(x["memory_used_mib"] for x in g) == 12955.1875
    masks = [x["throttle_reasons_mask"] for x in g]
    assert set(masks) == {0, 4} and masks.count(0) / len(masks) > 0.5    # 0.57 here; 0.002 in the 2026-11-17 sweep
