"""Step 9 capability probe: real Kaggle T4 data (2026-11-19 session), vLLM 0.34.0.

results/20261119T014800Z-step9-probe holds one JSON and one server log per configuration. Each configuration
saw two identical 8K requests: these tests pin what the probe recorded (support and evidence), not performance.
"""

import json
import pathlib

import pytest

from benchmarks import probe_strategies as pr

RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261119T014800Z-step9-probe"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="Step 9 probe not present")
GIB = 1024.0


@pytest.fixture(scope="module")
def probes():
    return {n: json.loads((RUN / f"probe_{n}.json").read_text(encoding="utf-8")) for n in pr.CONFIGS}


def log(name):
    return (RUN / f"probe_{name}.server.log").read_text(encoding="utf-8", errors="replace")


def test_baseline_and_prefix_cache_start_and_share_the_same_pool(probes):
    for n in ("baseline", "prefix_cache", "kv_offload_native"):
        assert probes[n]["started"]
        e = probes[n]["engine_reported_startup"]
        assert (e["kv_cache_tokens"], e["kv_cache_blocks"], e["vllm_version_in_log"]) == (309248, 4832, "0.34.0")
    assert "--no-enable-prefix-caching" in probes["baseline"]["server_argv"]
    assert "--enable-prefix-caching" in probes["prefix_cache"]["server_argv"]


def test_baseline_never_hits_the_prefix_cache_and_prefix_caching_does(probes):
    b1, b2 = probes["baseline"]["request_1"], probes["baseline"]["request_2"]
    assert b1["prefix_cache_queries_delta"] == 0 and b2["prefix_cache_hits_delta"] == 0
    assert b2["ttft_s"] > 0.8 * b1["ttft_s"]                       # no caching: the repeat costs the same
    p1, p2 = probes["prefix_cache"]["request_1"], probes["prefix_cache"]["request_2"]
    assert (p1["prefix_cache_hits_delta"], p2["prefix_cache_hits_delta"]) == (0, 8128)
    assert p2["prefix_cache_queries_delta"] == 8192                # 127 of 128 blocks hit: the last block is recomputed
    assert p2["ttft_s"] < 0.05 * p1["ttft_s"]
    assert all(probes[n]["same_output"] for n in ("baseline", "prefix_cache", "kv_offload_native"))


def test_fp8_kv_cache_is_unsupported_on_the_t4_and_the_engine_says_so(probes):
    p = probes["fp8_kv"]
    assert not p["started"] and "--kv-cache-dtype" in p["server_argv"] and "fp8" in p["server_argv"]
    text = log("fp8_kv")
    assert "FP8 KV cache is not supported by the Triton attention backend on Tesla T4" in text
    assert "compute capability 7.5" in text and "SM89" in text
    assert pr.classify([p])[0]["verdict"].startswith("did not start")


def test_native_cpu_offload_started_with_the_connector_and_an_8gib_shared_memory_buffer(probes):
    p = probes["kv_offload_native"]
    assert "--kv-offloading-backend" in p["server_argv"] and "--kv-offloading-size" in p["server_argv"]
    text = log("kv_offload_native")
    assert "Creating v1 connector with name: OffloadingConnector" in text
    assert "Creating offloading spec with name: CPUOffloadingSpec" in text
    assert "Created mmap file /dev/shm/vllm_offload_" in text and "(8.59 GB)" in text
    assert "OffloadingConnector" not in log("prefix_cache") and "OffloadingConnector" not in log("baseline")
    r2 = p["request_2"]
    assert r2["prefix_cache_hits_delta"] == 8128 and r2["ttft_s"] < 0.2


def test_offload_buffer_shows_in_rss_and_page_cache_but_not_in_anonymous_memory(probes):
    base, off = probes["baseline"]["memory_before"], probes["kv_offload_native"]["memory_before"]
    assert 7.9 * GIB < off["server_tree_rss_mib"] - base["server_tree_rss_mib"] < 8.4 * GIB
    assert 7.9 * GIB < off["container_file_mib"] - base["container_file_mib"] < 8.4 * GIB
    assert abs(off["container_anon_mib"] - base["container_anon_mib"]) < 100        # anon did not move
    assert off["container_mem_limit_mib"] == 30720.0 and off["container_mem_current_mib"] > 27 * GIB
    assert off["container_mem_current_mib"] < off["container_mem_limit_mib"]       # 27.8 of 30 GiB: little headroom


def test_the_log_estimate_is_unchanged_by_offloading(probes):
    for n in ("baseline", "prefix_cache", "kv_offload_native"):
        assert probes[n]["engine_reported_startup"]["available_kv_cache_gib"] == 8.92
        assert probes[n]["engine_reported_startup"]["kv_cache_final_gib"] == 8.26


def test_summary_verdicts_match_the_evidence_rules(probes):
    rows = {r["name"]: r for r in pr.classify([probes[n] for n in pr.CONFIGS])}
    assert rows["baseline"]["verdict"].startswith("control")
    assert rows["prefix_cache"]["verdict"].startswith("active")
    assert rows["fp8_kv"]["verdict"].startswith("did not start")
    assert rows["kv_offload_native"]["verdict"].startswith("started and the log mentions offloading")
