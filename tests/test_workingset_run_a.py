"""Step 9 working-set sweep, part A: real Kaggle T4 data (2026-11-19), vLLM 0.34.0, concurrency 1.

baseline  20261119T021951Z-3cbc939b : no prefix cache, 48 sessions (the no-reuse reference)
prefix    20261119T022723Z-402f880f : GPU prefix cache only, 48 / 96 / 144 / 192 sessions
Each session has a distinct 4,096-token context, filled once and revisited once in the same order with a
distinct short question. One observation per cell: these tests pin what was recorded, not a distribution.
"""

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import workingset as ws  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
BASE, PREFIX = RES / "20261119T021951Z-3cbc939b", RES / "20261119T022723Z-402f880f"
pytestmark = pytest.mark.skipif(not (BASE.exists() and PREFIX.exists()), reason="part A runs not present")
POOL_BLOCKS, BLOCK_TOKENS = 4831, 64


def recs(d):
    return mt.load_records(d / "requests.jsonl")


def cell(d, size, phase):
    return [r for r in recs(d) if r.get("working_set_sessions") == size and r["phase"] == phase]


@pytest.mark.parametrize("d", [BASE, PREFIX])
def test_runs_validate_on_the_pinned_stack_with_the_same_pool(d):
    assert mt.validate_run_dir(d) == []
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["package_versions"]["vllm"] == "0.34.0" and m["lock_mismatches"] == []
    assert m["engine_reported_startup"]["kv_cache_tokens"] == 309248
    assert m["counts"]["errors"] == 0
    c = m["config"]
    assert (c["context_tokens"], c["rounds"], c["revisit_order"], c["settle_s"]) == (4096, 1, "same", 0.5)
    d_ = m["memory_monitoring"]["engine_counter_deltas_over_run"]
    assert d_["preemptions_total"] == 0


def test_the_baseline_never_reuses_a_context():
    m = json.loads((BASE / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["config"]["strategy"] == "baseline" and m["config"]["sizes"] == [48]
    assert m["memory_monitoring"]["engine_counter_deltas_over_run"]["prefix_cache_queries_total"] == 0
    fill, revisit = cell(BASE, 48, "fill"), cell(BASE, 48, "revisit")
    assert len(fill) == len(revisit) == 48
    med = lambda rs: sorted(r["ttft_s"] for r in rs)[len(rs) // 2]  # noqa: E731
    assert med(revisit) == pytest.approx(med(fill), rel=0.03) and 3.1 < med(revisit) < 3.4


def test_the_prefix_cache_keeps_48_sessions_alive_and_loses_every_larger_working_set():
    cells = ws.summarize(recs(PREFIX))
    assert sorted(cells, key=int) == ["48", "96", "144", "192"]
    assert cells["48"]["revisit"]["requests_mostly_cached"] == 48
    assert cells["48"]["revisit"]["ttft_s"]["median"] < 0.2
    assert cells["48"]["revisit"]["hit_token_share"] == pytest.approx(0.98, abs=0.005)
    for size in ("96", "144", "192"):
        assert cells[size]["revisit"]["requests_mostly_cached"] == 0
        assert cells[size]["revisit"]["hit_token_share"] == 0
        assert cells[size]["revisit"]["ttft_s"]["median"] == pytest.approx(3.25, abs=0.1)
    for size in ("48", "96", "144", "192"):
        assert cells[size]["fill"]["requests_mostly_cached"] == 0               # every first visit is a miss


def test_every_hit_in_the_whole_run_is_one_cell_with_exactly_63_cached_blocks_per_request():
    rs = [r for r in recs(PREFIX) if r["phase"] in ("fill", "revisit")]
    hit_requests = [r for r in rs if (r["prefix_cache_hits_delta"] or 0) > 0]
    assert len(hit_requests) == 48 and {r["working_set_sessions"] for r in hit_requests} == {48}
    assert {r["prefix_cache_hits_delta"] for r in hit_requests} == {63 * BLOCK_TOKENS}   # 4,032 of ~4,109 tokens
    m = json.loads((PREFIX / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["memory_monitoring"]["engine_counter_deltas_over_run"]["prefix_cache_hits_total"] == 48 * 4032
    assert sum(r["prefix_cache_hits_delta"] or 0 for r in recs(PREFIX)) == 48 * 4032


def test_the_cliff_is_bracketed_by_capacity_arithmetic_not_measured():
    rs = [r for r in cell(PREFIX, 48, "fill")]
    blocks_per_session = math.ceil((max(r["prompt_tokens"] for r in rs) + 16) / BLOCK_TOKENS)
    assert blocks_per_session == 65
    assert POOL_BLOCKS // blocks_per_session == 74                               # about 74 sessions fit in the pool
    assert 48 * blocks_per_session < POOL_BLOCKS < 96 * blocks_per_session       # 48 fits, 96 does not
    # no cell lies between 48 and 96, so the observed edge is only known to be inside that gap
    assert {r["working_set_sessions"] for r in recs(PREFIX) if r.get("working_set_sessions")} == {48, 96, 144, 192}


def test_paired_sessions_are_the_same_texts_in_both_strategies():
    a = {r["context_id"]: r["prompt_sha256"] for r in cell(BASE, 48, "fill")}
    b = {r["context_id"]: r["prompt_sha256"] for r in cell(PREFIX, 48, "fill")}
    assert len(a) == 48 and a == b                                               # same contexts, same turn-0 texts
    assert len({r["context_id"] for r in recs(PREFIX) if r.get("working_set_sessions")}) == 480


def test_gpu_resident_caching_costs_no_cpu_memory():
    for d in (BASE, PREFIX):
        cpu = [(r["memory"] or {}).get("cpu", {}) for r in recs(d) if r["phase"] in ("fill", "revisit")]
        files = [c["container_file_peak_mib"] for c in cpu if c.get("container_file_peak_mib") is not None]
        anons = [c["container_anon_peak_mib"] for c in cpu if c.get("container_anon_peak_mib") is not None]
        assert max(files) - min(files) < 100 and max(anons) - min(anons) < 100   # flat across the whole sweep
        assert max(c.get("container_mem_current_peak_mib") or 0 for c in cpu) < 22 * 1024
    gpu = [r["memory"]["gpu"]["peak_used_mib_in_window"] for r in recs(PREFIX) if r["phase"] in ("fill", "revisit")]
    assert max(g for g in gpu if g is not None) == 12955.1875                    # device memory is the same preallocation
