"""Tools added for the Step 7 sweep: compact level table, GPU clock/temperature/throttle sampling."""

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import sampling as sp  # noqa: E402

REAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261117T041433Z-acb4745b"


def fake_nvml(**broken):
    m = types.SimpleNamespace()
    m.NVML_TEMPERATURE_GPU, m.NVML_CLOCK_SM = 0, 1
    m.nvmlInit = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: "h"
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=5120 * 2 ** 20, total=15360 * 2 ** 20)
    m.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=37)
    m.nvmlDeviceGetTemperature = lambda h, k: 61
    m.nvmlDeviceGetClockInfo = lambda h, k: 1590
    m.nvmlDeviceGetPowerUsage = lambda h: 68500
    m.nvmlDeviceGetCurrentClocksEventReasons = lambda h: 4
    for name in broken:
        def boom(*a, **k):
            raise RuntimeError("not supported")
        setattr(m, name, boom)
    return m


def test_nvml_reader_adds_clock_temperature_power_and_throttle(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml())
    r = sp.make_gpu_reader(0)()
    assert r["memory_used_mib"] == 5120.0 and r["gpu_util_pct"] == 37.0
    assert (r["temperature_c"], r["sm_clock_mhz"], r["power_w"], r["throttle_reasons_mask"]) == (61.0, 1590.0, 68.5, 4)


def test_missing_extras_do_not_break_the_memory_reading(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(nvmlDeviceGetPowerUsage=1, nvmlDeviceGetClockInfo=1))
    r = sp.make_gpu_reader(0)()
    assert r["memory_used_mib"] == 5120.0 and r["power_w"] is None and r["sm_clock_mhz"] is None
    assert r["temperature_c"] == 61.0


def test_throttle_mask_falls_back_to_the_older_call_name():
    ns = types.SimpleNamespace(nvmlDeviceGetCurrentClocksThrottleReasons=lambda h: 8)
    assert sp._throttle_mask(ns, "h") == 8
    with pytest.raises(RuntimeError):
        sp._throttle_mask(types.SimpleNamespace(), "h")


def test_window_block_summarises_clock_temperature_and_throttle():
    samples = [{"t_s": t, "memory_used_mib": 1000.0, "source": "nvml", "sm_clock_mhz": clk,
                "temperature_c": temp, "power_w": 60.0 + t, "throttle_reasons_mask": mask}
               for t, clk, temp, mask in ((1, 1590, 60, 0), (2, 1200, 70, 4), (3, 1590, 65, 0))]
    mon = sp.MemoryMonitor(gpu_reader=lambda: samples[0], interval_s=1.0)
    mon._samples = lambda name: samples
    block = mon._gpu_section(0.5, 3.5)
    assert block["sm_clock_mhz_min_in_window"] == 1200 and block["temperature_c_max_in_window"] == 70
    assert block["power_w_max_in_window"] == 63.0 and block["throttle_reasons_seen_in_window"] == [0, 4]
    assert sp.validate_memory_block({"gpu": block, "engine": {"label": "unavailable", "reason": "x"},
                                     "cpu": {"label": "unavailable", "reason": "x"}}) == []


def test_old_records_without_clock_fields_still_work():
    samples = [{"t_s": 1, "memory_used_mib": 1000.0, "source": "nvml"}]
    mon = sp.MemoryMonitor(gpu_reader=lambda: samples[0], interval_s=1.0)
    mon._samples = lambda name: samples
    block = mon._gpu_section(0.5, 1.5)
    assert block["sm_clock_mhz_min_in_window"] is None and block["throttle_reasons_seen_in_window"] is None


@pytest.mark.skipif(not REAL.exists(), reason="pilot run not present")
def test_level_table_on_the_real_pilot_run():
    text = mt.format_level_table(mt.load_records(REAL / "requests.jsonl"))
    lines = text.splitlines()
    assert len(lines) == 4 and lines[0].split()[0] == "input"
    assert lines[1].split()[0] == "1024" and lines[3].split()[0] == "8192"
    assert "0.324" in lines[1] and "11.872" in lines[3]
