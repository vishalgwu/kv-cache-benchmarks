"""Sampler, parser, and memory-block tests.

Prometheus text, /proc trees, and sample series below are SYNTHETIC TEST FIXTURES.
The one exception is the committed real vLLM server log. Nothing here may be
written to results/ or reported as a measurement.
"""

import pathlib
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import sampling as sp  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
REAL_LOG = REPO / "results" / "20261116T210839Z-d6a00143" / "server.log"

PROM = """\
# HELP vllm:kv_cache_usage_perc Test help text: 1 means 100 percent.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.40
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} 2.0
vllm:num_requests_running{engine="1"} 1.0
vllm:num_requests_waiting{engine="0"} 5.0
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0"} 3.0
vllm:some_histogram_bucket{le="0.5"} NaN
not a metric line
"""


# ----------------------------------------------------------------------------- prometheus
def test_parse_prometheus_and_extract_known_values():
    samples, meta = sp.parse_prometheus(PROM)
    assert meta["vllm:kv_cache_usage_perc"]["type"] == "gauge"
    assert "100 percent" in meta["vllm:kv_cache_usage_perc"]["help"]
    assert all(v == v for _, _, v in samples)  # NaN lines are dropped
    values = sp.extract_engine_values(samples)
    assert values["kv_cache_usage"] == 0.40  # gauge across engines: max
    assert values["requests_running"] == 3.0  # summed
    assert values["requests_waiting"] == 5.0
    assert values["preemptions_total"] == 3.0
    assert values["metric_names"]["kv_cache_usage"] == "vllm:kv_cache_usage_perc"


def test_older_metric_name_is_accepted():
    samples, _ = sp.parse_prometheus('vllm:gpu_cache_usage_perc{engine="0"} 0.5\n')
    values = sp.extract_engine_values(samples)
    assert values["kv_cache_usage"] == 0.5
    assert values["metric_names"]["kv_cache_usage"] == "vllm:gpu_cache_usage_perc"


def test_missing_engine_metrics_are_none_not_zero():
    values = sp.extract_engine_values([])
    assert values["kv_cache_usage"] is None and values["requests_running"] is None
    assert values["metric_names"]["kv_cache_usage"] is None


# ----------------------------------------------------------------------------- windows
SERIES = [{"t_s": t, "memory_used_mib": m} for t, m in [(1.0, 100), (2.0, 300), (3.0, 200), (4.0, 900)]]


def test_window_stat_includes_boundaries_and_counts_samples():
    assert sp.window_stat(SERIES, 2.0, 3.0, "memory_used_mib", max) == (300, 2)
    assert sp.window_stat(SERIES, 0.0, 10.0, "memory_used_mib", max) == (900, 4)


def test_window_with_no_samples_is_none_not_zero():
    assert sp.window_stat(SERIES, 1.2, 1.8, "memory_used_mib", max) == (None, 0)


def test_interval_stats():
    stats = sp.interval_stats(SERIES, 1.0)
    assert stats["n_samples"] == 4 and stats["max_gap_s"] == 1.0 and stats["mean_gap_s"] == 1.0
    assert sp.interval_stats([], 1.0)["max_gap_s"] is None


# ----------------------------------------------------------------------------- sampler
def test_periodic_sampler_collects_timestamped_samples():
    counter = {"n": 0}

    def read():
        counter["n"] += 1
        return {"value": counter["n"]}

    s = sp.PeriodicSampler("t", read, interval_s=0.02).start()
    time.sleep(0.2)
    s.stop()
    assert len(s.samples) >= 3
    for sample in s.samples:
        assert sample["t_read_start_s"] <= sample["t_s"] <= sample["t_read_end_s"]
    ts = [x["t_s"] for x in s.samples]
    assert ts == sorted(ts)
    n = len(s.samples)
    time.sleep(0.1)
    assert len(s.samples) == n  # stopped for good
    s.stop()  # idempotent


def test_reader_errors_are_counted_and_do_not_stop_sampling():
    calls = {"n": 0}

    def read():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"value": 1}

    s = sp.PeriodicSampler("t", read, interval_s=0.02).start()
    time.sleep(0.15)
    s.stop()
    assert s.read_errors == 1 and "boom" in s.last_error
    assert len(s.samples) >= 2


def test_reader_returning_none_produces_no_sample():
    s = sp.PeriodicSampler("t", lambda: None, interval_s=0.02).start()
    time.sleep(0.08)
    s.stop()
    assert s.samples == []


# ----------------------------------------------------------------------------- GPU reader
def fake_run(stdout="", returncode=0, stderr=""):
    return lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


def test_nvidia_smi_output_is_parsed(monkeypatch):
    monkeypatch.setattr(sp.subprocess, "run", fake_run("5120, 15360, 37\n"))
    r = sp.read_gpu_nvidia_smi(0)
    assert r == {"memory_used_mib": 5120.0, "memory_total_mib": 15360.0, "gpu_util_pct": 37.0,
                 "source": "nvidia-smi", "scope": "device"}


def test_nvidia_smi_na_utilization_is_none(monkeypatch):
    monkeypatch.setattr(sp.subprocess, "run", fake_run("5120, 15360, [N/A]\n"))
    assert sp.read_gpu_nvidia_smi(0)["gpu_util_pct"] is None


def test_nvidia_smi_failure_raises(monkeypatch):
    monkeypatch.setattr(sp.subprocess, "run", fake_run(returncode=9, stderr="no devices"))
    with pytest.raises(RuntimeError, match="no devices"):
        sp.read_gpu_nvidia_smi(0)


# ----------------------------------------------------------------------------- CPU readers
def make_proc(root, pid, ppid, rss_kb):
    d = root / str(pid)
    d.mkdir()
    (d / "status").write_text(f"Name:\tx\nPPid:\t{ppid}\nVmRSS:\t{rss_kb} kB\n")


def test_process_tree_rss_sums_only_the_server_tree(tmp_path):
    make_proc(tmp_path, 100, 1, 1024)   # server
    make_proc(tmp_path, 101, 100, 2048)  # child
    make_proc(tmp_path, 102, 101, 1024)  # grandchild
    make_proc(tmp_path, 200, 1, 99999)   # unrelated process
    assert sp._proc_tree_rss_mib(100, str(tmp_path)) == pytest.approx(4096 / 1024)


def test_process_tree_rss_unknown_pid_is_none(tmp_path):
    make_proc(tmp_path, 100, 1, 1024)
    assert sp._proc_tree_rss_mib(555, str(tmp_path)) is None
    assert sp._proc_tree_rss_mib(100, str(tmp_path / "missing")) is None


def test_host_meminfo_is_parsed_in_mib(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:  16384000 kB\nMemFree: 1 kB\nMemAvailable:   6291456 kB\n")
    r = sp._host_mem_mib(str(f))
    assert r["host_mem_available_mib"] == pytest.approx(6144.0)
    assert r["host_mem_total_mib"] == pytest.approx(16000.0)


def test_cpu_reader_without_pid_reports_host_only(monkeypatch):
    monkeypatch.setattr(sp, "_host_mem_mib", lambda: {"host_mem_available_mib": 6000.0,
                                                      "host_mem_total_mib": 12000.0})
    r = sp.make_cpu_reader(None)()
    assert r["server_tree_rss_mib"] is None and r["host_mem_available_mib"] == 6000.0
    assert "double count" in r["scope"]


# ----------------------------------------------------------------------------- real server log
@pytest.mark.skipif(not REAL_LOG.exists(), reason="committed real run not present")
def test_startup_log_parser_on_the_real_vllm_log():
    info = sp.parse_vllm_startup_log(REAL_LOG.read_text(encoding="utf-8"))
    assert info["kv_cache_tokens"] == 333_904
    assert info["available_kv_cache_gib"] == 8.92
    assert info["max_concurrency_reported"] == 274.59
    assert info["max_concurrency_tokens_per_request"] == 1216
    assert info["model_loading_gib"] == 2.98
    assert info["consumed_weights_plus_non_torch_gib"] == 3.23
    assert info["peak_activation_gib"] == 0.23
    assert info["cuda_graph_gib"] == 0.0
    assert info["free_memory_gib_at_startup"] == 14.41 and info["device_total_gib"] == 14.56
    assert info["attention_backend"] == "TRITON_ATTN"
    assert "engine-reported" in info["source"]


def test_startup_log_parser_returns_none_for_missing_values():
    info = sp.parse_vllm_startup_log("nothing useful here")
    assert info["kv_cache_tokens"] is None and info["attention_backend"] is None


# ----------------------------------------------------------------------------- monitor blocks
def make_monitor():
    gpu = lambda: {"memory_used_mib": 0.0}  # noqa: E731 - never started in these tests
    return sp.MemoryMonitor(gpu_reader=gpu, engine_reader=gpu, cpu_reader=gpu, interval_s=1.0)


def gpu_sample(t, used):
    return {"t_s": t, "memory_used_mib": used, "source": "fake", "scope": "device"}


def test_gpu_block_reports_sampled_peak_and_idle_baseline():
    m = make_monitor()
    m.samplers["gpu"].samples += [gpu_sample(0.5, 3000), gpu_sample(1.0, 3002), gpu_sample(1.5, 3004),
                                  gpu_sample(5.0, 6000), gpu_sample(5.5, 9000), gpu_sample(9.0, 3100)]
    m.baseline_window = (0.0, 2.0)
    gpu = m.request_block(4.0, 6.0)["gpu"]
    assert gpu["label"] == "measured" and gpu["unit"] == "MiB"
    assert gpu["peak_used_mib_in_window"] == 9000 and gpu["n_samples_in_window"] == 2
    assert gpu["baseline_used_mib"] == 3002 and gpu["baseline_n_samples"] == 3  # median
    assert "device" in gpu["scope"] and "lower bound" in gpu["peak_note"]
    assert sp.validate_memory_block(m.request_block(4.0, 6.0)) == []


def test_request_shorter_than_the_interval_is_unavailable_not_invented():
    m = make_monitor()
    m.samplers["gpu"].samples += [gpu_sample(1.0, 3000), gpu_sample(2.0, 3000)]
    block = m.request_block(1.2, 1.7)  # falls between two samples
    assert block["gpu"]["label"] == "unavailable"
    assert block["gpu"]["peak_used_mib_in_window"] is None
    assert "no GPU sample" in block["gpu"]["reason"]
    assert sp.validate_memory_block(block) == []  # a valid, explicit gap


def test_baseline_is_unavailable_without_a_baseline_window():
    m = make_monitor()
    m.samplers["gpu"].samples += [gpu_sample(1.0, 3000)]
    gpu = m.request_block(0.5, 1.5)["gpu"]
    assert gpu["baseline_used_mib"] is None and gpu["baseline_n_samples"] == 0


def test_engine_block_peaks_and_preemption_delta():
    m = make_monitor()
    eng = lambda t, kv, run, wait, pre: {  # noqa: E731
        "t_s": t, "kv_cache_usage": kv, "requests_running": run, "requests_waiting": wait,
        "preemptions_total": pre, "metric_names": {"kv_cache_usage": "vllm:kv_cache_usage_perc"}}
    m.samplers["engine"].samples += [eng(1.0, 0.1, 1, 0, 2), eng(2.0, 0.5, 4, 2, 3),
                                     eng(3.0, 0.3, 3, 1, 6), eng(9.0, 0.0, 0, 0, 6)]
    block = m.request_block(1.5, 3.5)["engine"]
    assert block["label"] == "measured"
    assert block["kv_cache_usage_peak"] == 0.5 and block["requests_running_peak"] == 4
    assert block["requests_waiting_peak"] == 2
    assert block["preemptions_in_window"] == 6 - 2  # last in window minus last before it
    assert block["metric_names"]["kv_cache_usage"] == "vllm:kv_cache_usage_perc"
    assert "scale" in block["kv_cache_usage_scale_note"]


def test_engine_block_without_expected_metrics_is_unavailable():
    m = make_monitor()
    m.samplers["engine"].samples += [{"t_s": 1.0, "kv_cache_usage": None, "requests_running": None,
                                      "requests_waiting": None, "preemptions_total": None,
                                      "metric_names": {}}]
    block = m.request_block(0.5, 1.5)["engine"]
    assert block["label"] == "unavailable" and "none of the expected" in block["reason"]


def test_unconfigured_samplers_are_unavailable_with_a_reason():
    m = sp.MemoryMonitor(interval_s=1.0)
    block = m.request_block(0.0, 1.0)
    for section in ("gpu", "engine", "cpu"):
        assert block[section]["label"] == "unavailable" and "not configured" in block[section]["reason"]
    assert sp.validate_memory_block(block) == []


def test_cpu_block_peak_rss_and_minimum_available_memory():
    m = make_monitor()
    m.samplers["cpu"].samples += [
        {"t_s": 1.0, "server_tree_rss_mib": 900.0, "host_mem_available_mib": 5000.0, "scope": "s", "source": "x"},
        {"t_s": 2.0, "server_tree_rss_mib": 1200.0, "host_mem_available_mib": 4200.0, "scope": "s", "source": "x"}]
    cpu = m.request_block(0.5, 2.5)["cpu"]
    assert cpu["label"] == "measured" and cpu["unit"] == "MiB"
    assert cpu["server_tree_rss_peak_mib"] == 1200.0 and cpu["host_mem_available_min_mib"] == 4200.0


def test_validate_memory_block_rejects_bad_blocks():
    good_gpu = {"label": "measured", "n_samples_in_window": 1, "unit": "MiB", "scope": "device",
                "peak_used_mib_in_window": 5.0}
    ok = {"gpu": good_gpu, "engine": {"label": "unavailable", "reason": "r"},
          "cpu": {"label": "unavailable", "reason": "r"}}
    assert sp.validate_memory_block(ok) == []

    def broken(**gpu_changes):
        return {**ok, "gpu": {**good_gpu, **gpu_changes}}

    assert any("bad label" in e for e in sp.validate_memory_block(broken(label="guess")))
    assert any("no samples" in e for e in sp.validate_memory_block(broken(n_samples_in_window=0)))
    assert any("without a unit" in e for e in sp.validate_memory_block(broken(unit=None)))
    assert any("without a scope" in e for e in sp.validate_memory_block(broken(scope=None)))
    assert any("peak is None" in e for e in sp.validate_memory_block(broken(peak_used_mib_in_window=None)))
    no_reason = {**ok, "engine": {"label": "unavailable"}}
    assert any("without a reason" in e for e in sp.validate_memory_block(no_reason))
    assert any("missing" in e for e in sp.validate_memory_block({"gpu": good_gpu}))


def test_monitor_write_creates_sample_files_and_catalog(tmp_path):
    m = make_monitor()
    m.samplers["gpu"].samples += [gpu_sample(1.0, 3000)]
    m.engine_catalog = [("vllm:kv_cache_usage_perc", "gauge", "help text")]
    m.write(tmp_path)
    assert (tmp_path / "samples_gpu.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "samples_cpu.jsonl").exists() and (tmp_path / "samples_engine.jsonl").exists()
    assert "vllm:kv_cache_usage_perc\tgauge\thelp text" in (tmp_path / "engine_metrics_catalog.tsv").read_text()
