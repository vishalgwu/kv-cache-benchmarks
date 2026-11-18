"""Step 8 extension, 16K concurrency: real Kaggle T4 data (new session), vLLM 0.34.0, CUDA graphs on.

Run 20261118T030348Z-fb375efd: input 16,384 x concurrency 1/2/4/8/16, one unmeasured warmup wave at N=1, then
2 measured waves per cell, KV pool 309,248 tokens. n = 2 waves per cell: these tests pin what the stored data
says and are not distributions.
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

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261118T030348Z-fb375efd"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="16K run not present")
NS = (1, 2, 4, 8, 16)
BLOCKS, BLOCK_TOKENS = 4832, 64


@pytest.fixture(scope="module")
def recs():
    return mt.load_records(RUN / "requests.jsonl")


@pytest.fixture(scope="module")
def cells(recs):
    return wv.summarize_cells(recs)


def waves_of(recs, n):
    return [rs for rs in wv.complete_waves(recs).values() if rs[0]["concurrency"] == n]


def test_run_validates_and_is_the_pinned_stack_with_the_same_cache_pool(cells):
    assert mt.validate_run_dir(RUN) == []
    m = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["counts"] == {"requests": 64, "ok": 64, "errors": 0, "warmup": 2, "usable": 62}
    assert m["package_versions"]["vllm"] == "0.34.0" and m["lock_mismatches"] == []
    e = m["engine_reported_startup"]
    assert (e["kv_cache_blocks"], e["kv_cache_tokens"], e["kv_block_size_tokens"]) == (4832, 309248, 64)
    assert m["config"]["warmup_cells"] == "smallest" and m["config"]["waves_per_cell"] == 2
    assert [c["concurrency"] for c in cells.values()] == list(NS) and all(c["n_waves"] == 2 for c in cells.values())


def test_every_request_overlapped_and_the_engine_ran_them_together(cells):
    for c in cells.values():
        assert c["max_in_flight"] == c["concurrency"] and c["engine_running_peak"] == c["concurrency"]
    assert cells["16384x16"]["engine_waiting_peak"] == 15


def test_prefill_is_serial_at_16k(recs):
    single = statistics.median(r["ttft_s"] for rs in waves_of(recs, 1) for r in rs)
    assert 44.0 < single < 45.5
    for n in (2, 4, 8, 16):
        gaps = []
        for rs in waves_of(recs, n):
            t = sorted(r["ttft_s"] for r in rs)
            gaps += [b - a for a, b in zip(t, t[1:])]
        assert statistics.median(gaps) / single == pytest.approx(1.0, abs=0.02), n
        assert min(gaps) > 0.9 * single                           # never two prefills in the same step
        last = statistics.median(max(r["ttft_s"] for r in rs) for rs in waves_of(recs, n))
        assert last / (n * single) == pytest.approx(1.0, abs=0.02), n
    assert cells_last(recs, 16) > 700                              # the last of 16 waits about 12 minutes


def cells_last(recs, n):
    return statistics.median(max(r["ttft_s"] for r in rs) for rs in waves_of(recs, n))


def test_concurrency_does_not_raise_aggregate_throughput_at_16k(cells):
    agg = [cells[f"16384x{n}"]["agg_output_tok_s"]["median"] for n in NS]
    assert agg == sorted(agg) and agg[-1] / agg[0] < 1.15          # 2.6 -> 2.8 tok/s: prefill is the whole wave
    assert all(2.4 < a < 3.0 for a in agg)


def test_running_streams_are_starved_while_later_requests_prefill(recs):
    early, late = {}, {}
    for n in NS:
        e, l = [], []
        for rs in waves_of(recs, n):
            s = sorted(rs, key=lambda r: r["t_first_content_s"])
            e.append(s[0]["decode_tokens_per_s"])
            l.append(s[-1]["decode_tokens_per_s"])
        early[n], late[n] = statistics.median(e), statistics.median(l)
    assert early[1] == pytest.approx(32.9, abs=0.5)
    assert early[16] < 0.3 and early[8] < 0.5 and early[4] < 1.0   # about one token every 5 s at N=16
    assert early[2] < early[1] / 10
    assert late[16] > 8 and late[16] < late[1]                     # the last one still decodes, but slower than alone


def test_cache_usage_peak_is_between_prompt_blocks_and_final_blocks_and_never_near_the_pool_limit(recs):
    for rs in wv.complete_waves(recs).values():
        low = sum(math.ceil(r["prompt_tokens"] / BLOCK_TOKENS) for r in rs) / (BLOCKS - 1)
        high = sum(math.ceil((r["prompt_tokens"] + r["output_tokens"] - 1) / BLOCK_TOKENS) for r in rs) / (BLOCKS - 1)
        got = wv.wave_stats(rs)["kv_cache_usage_peak"]
        assert low - 1e-6 <= got <= high + 1e-6, rs[0]["wave_id"]
    top = max(wv.wave_stats(rs)["kv_cache_usage_peak"] for rs in wv.complete_waves(recs).values())
    assert top == pytest.approx(0.853, abs=0.002)                  # 16 x 16K uses 85 percent of the pool


def test_no_preemption_no_prefix_hits_and_only_the_known_shutdown_error_in_the_log():
    m = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    d = m["memory_monitoring"]["engine_counter_deltas_over_run"]
    assert d["preemptions_total"] == 0 and d["prefix_cache_hits_total"] == 0
    log = (RUN / "server.log").read_text(encoding="utf-8", errors="replace").splitlines()
    shutdown = next(i for i, line in enumerate(log) if "EngineCore: trigger received signal=SIGTERM" in line)
    errors = [i for i, line in enumerate(log) if "EngineDeadError" in line]
    assert errors and min(errors) > shutdown                       # the traceback came after our SIGTERM, not before


def test_greedy_output_varied_between_waves_for_some_prompts_at_n8_and_n16(recs):
    meas = [r for r in recs if not r["is_warmup"] and r.get("wave_id")]
    by = {}
    for r in meas:
        by.setdefault((r["concurrency"], r["prompt_id"]), set()).add(r["output_sha256"])
    assert {(n, p) for (n, p), v in by.items() if len(v) > 1} == {(8, "L16384-004"), (8, "L16384-006"), (16, "L16384-004")}
    assert all(len(v) == 1 for (n, _), v in by.items() if n <= 4)
    assert not wv.batch_invariance(recs)["all_identical"]


def test_short_prompt_token_counts_are_recorded_per_record(recs):
    off = {(r["prompt_id"], r["prompt_tokens"]) for r in recs
           if r["prompt_id"] != "warmup" and r["prompt_tokens"] != r["target_input_tokens"]}
    assert off == {("L16384-001", 16383), ("L16384-009", 16383)}
