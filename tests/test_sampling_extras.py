"""Tests for window-start/retention, container memory, and prefix-cache counters.

Sample series, cgroup files, and counters below are SYNTHETIC TEST FIXTURES and must
never be written to results/ or reported as measurements.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import sampling as sp  # noqa: E402


def make_monitor():
    dummy = lambda: {"memory_used_mib": 0.0}  # noqa: E731 - never started in these tests
    return sp.MemoryMonitor(gpu_reader=dummy, engine_reader=dummy, cpu_reader=dummy, interval_s=1.0)


def gpu_sample(t, used):
    return {"t_s": t, "memory_used_mib": used, "source": "fake", "scope": "device"}


# ----------------------------------------------------------------------------- retention / window start
def test_gpu_block_separates_inherited_memory_from_growth_during_the_request():
    m = make_monitor()
    m.samplers["gpu"].samples += [gpu_sample(1.0, 3000), gpu_sample(2.0, 3054), gpu_sample(3.0, 3054)]
    first = m.request_block(1.5, 2.5)["gpu"]   # allocator grows during this request
    second = m.request_block(2.6, 3.5)["gpu"]  # same peak, but inherited from the first
    assert first["used_mib_at_window_start"] == 3000 and first["peak_used_mib_in_window"] == 3054
    assert first["peak_minus_window_start_mib"] == 54
    assert second["used_mib_at_window_start"] == 3054 and second["peak_used_mib_in_window"] == 3054
    assert second["peak_minus_window_start_mib"] == 0
    assert "keeps memory after a request ends" in second["retention_note"]


def test_window_start_is_none_without_an_earlier_sample():
    m = make_monitor()
    m.samplers["gpu"].samples += [gpu_sample(5.0, 3000)]
    block = m.request_block(1.0, 6.0)
    assert block["gpu"]["used_mib_at_window_start"] is None
    assert block["gpu"]["peak_minus_window_start_mib"] is None
    assert sp.validate_memory_block(block) == []


def test_unavailable_gpu_block_still_reports_window_start():
    m = make_monitor()
    m.samplers["gpu"].samples += [gpu_sample(1.0, 3000), gpu_sample(5.0, 3000)]
    gpu = m.request_block(2.0, 3.0)["gpu"]
    assert gpu["label"] == "unavailable" and gpu["used_mib_at_window_start"] == 3000
    assert gpu["peak_minus_window_start_mib"] is None


# ----------------------------------------------------------------------------- cgroup memory
def test_cgroup_v2_limit_and_current_are_read_in_mib(tmp_path):
    (tmp_path / "memory.max").write_text("4294967296\n")
    (tmp_path / "memory.current").write_text("1073741824\n")
    assert sp._cgroup_mem_mib(str(tmp_path)) == {"container_mem_limit_mib": 4096.0,
                                                 "container_mem_current_mib": 1024.0}


def test_cgroup_v2_unlimited_is_none_not_a_huge_number(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "memory.current").write_text("2097152\n")
    r = sp._cgroup_mem_mib(str(tmp_path))
    assert r["container_mem_limit_mib"] is None and r["container_mem_current_mib"] == 2.0


def test_cgroup_v1_is_used_when_v2_is_absent_and_unlimited_is_none(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (tmp_path / "memory" / "memory.usage_in_bytes").write_text("5242880\n")
    r = sp._cgroup_mem_mib(str(tmp_path))
    assert r["container_mem_limit_mib"] is None and r["container_mem_current_mib"] == 5.0


def test_cgroup_missing_files_give_none(tmp_path):
    assert sp._cgroup_mem_mib(str(tmp_path)) == {"container_mem_limit_mib": None,
                                                 "container_mem_current_mib": None}


def test_cpu_reader_includes_container_memory(monkeypatch):
    monkeypatch.setattr(sp, "_host_mem_mib", lambda: {"host_mem_available_mib": 27000.0,
                                                      "host_mem_total_mib": 32000.0})
    monkeypatch.setattr(sp, "_cgroup_mem_mib", lambda: {"container_mem_limit_mib": 4096.0,
                                                        "container_mem_current_mib": 3000.0})
    r = sp.make_cpu_reader(None)()
    assert r["container_mem_limit_mib"] == 4096.0 and r["host_mem_available_mib"] == 27000.0


def test_cpu_block_reports_container_peak_and_limit():
    m = make_monitor()
    row = lambda t, cur: {"t_s": t, "server_tree_rss_mib": 900.0, "host_mem_available_mib": 5000.0,  # noqa: E731
                          "container_mem_current_mib": cur, "container_mem_limit_mib": 4096.0,
                          "scope": "s", "source": "x"}
    m.samplers["cpu"].samples += [row(1.0, 2000.0), row(2.0, 3500.0), row(9.0, 100.0)]
    cpu = m.request_block(0.5, 2.5)["cpu"]
    assert cpu["container_mem_current_peak_mib"] == 3500.0 and cpu["container_mem_limit_mib"] == 4096.0


# ----------------------------------------------------------------------------- prefix-cache counters
def test_prefix_cache_counters_are_extracted_and_default_to_none():
    samples, _ = sp.parse_prometheus('vllm:prefix_cache_queries_total{engine="0"} 40.0\n'
                                     'vllm:prefix_cache_hits_total{engine="0"} 0.0\n')
    values = sp.extract_engine_values(samples)
    assert values["prefix_cache_queries_total"] == 40.0 and values["prefix_cache_hits_total"] == 0.0
    assert sp.extract_engine_values([])["prefix_cache_hits_total"] is None


def test_counter_delta_uses_the_sample_before_the_window():
    s = [{"t_s": 1.0, "c": 10}, {"t_s": 2.0, "c": 14}, {"t_s": 3.0, "c": 19}, {"t_s": 9.0, "c": 99}]
    assert sp.counter_delta(s, "c", 1.5, 3.5) == 19 - 10
    assert sp.counter_delta(s, "c", 0.0, 2.5) == 14 - 10  # nothing before: first in-window value
    assert sp.counter_delta(s, "c", 4.0, 5.0) is None
    assert sp.counter_delta(s, "missing", 0.0, 10.0) is None


def test_engine_block_reports_prefix_cache_activity_in_the_window():
    m = make_monitor()
    row = lambda t, hits, queries: {  # noqa: E731
        "t_s": t, "kv_cache_usage": 0.1, "requests_running": 1, "requests_waiting": 0,
        "preemptions_total": 0, "prefix_cache_hits_total": hits, "prefix_cache_queries_total": queries,
        "metric_names": {}}
    m.samplers["engine"].samples += [row(1.0, 0, 100), row(2.0, 0, 100), row(3.0, 50, 4196)]
    block = m.request_block(1.5, 3.5)["engine"]
    assert block["prefix_cache_hits_in_window"] == 50
    assert block["prefix_cache_queries_in_window"] == 4096
    assert "counter_note" in block


def test_unavailable_engine_block_has_prefix_fields_as_none():
    m = make_monitor()
    m.samplers["engine"].samples += [{"t_s": 1.0, "kv_cache_usage": 0.1, "metric_names": {}}]
    block = m.request_block(5.0, 6.0)["engine"]
    assert block["label"] == "unavailable"
    assert block["prefix_cache_hits_in_window"] is None and block["prefix_cache_queries_in_window"] is None


def test_run_level_counter_deltas_are_in_the_description():
    m = make_monitor()
    row = lambda t, hits: {"t_s": t, "preemptions_total": 0.0, "prefix_cache_hits_total": hits,  # noqa: E731
                           "prefix_cache_queries_total": 10.0 * t}
    m.samplers["engine"].samples += [row(1.0, 0.0), row(2.0, 0.0), row(3.0, 0.0)]
    assert m.describe()["engine_counter_deltas_over_run"] == {
        "preemptions_total": 0.0, "prefix_cache_hits_total": 0.0, "prefix_cache_queries_total": 20.0}
    empty = sp.MemoryMonitor(interval_s=1.0).describe()["engine_counter_deltas_over_run"]
    assert empty["preemptions_total"] is None
