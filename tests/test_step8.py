"""Step 8: fixed-concurrency waves. Fake servers and synthetic records only; nothing is written under results/."""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmarks import concurrent as cc  # noqa: E402
from benchmarks import metrics as mt  # noqa: E402
from benchmarks import waves as wv  # noqa: E402
from workloads import generate as g  # noqa: E402
from test_runner import (MODEL, DONE, content_event, echoing_scenario, fake_counter,  # noqa: E402
                         fake_server, finish_event, role_event, usage_event)


@pytest.fixture()
def wl(tmp_path):
    manifest, prompts = g.generate_workload(
        base_seed=5, levels=[64, 128], per_level=8, output_tokens=16, max_context=1024,
        count_tokens=fake_counter, model_id=MODEL, revision="abc123")
    return g.write_workload(tmp_path / "wl8", manifest, prompts)


def slow_scenario(payload):
    # every request takes about 0.4 s end to end, so N simultaneous requests visibly overlap
    n = fake_counter(payload["messages"][0]["content"])
    return [(0, role_event()), (0.15, content_event("a")), (0.1, content_event("b")),
            (0.1, content_event("c")), (0, finish_event()), (0, usage_event(n, 3)), (0, DONE)]


def run(wl, tmp_path, host, port, **over):
    kw = dict(workload_dir=wl, host=host, port=port, model=MODEL, revision="abc123", levels=[64, 128],
              concurrencies=[1, 2, 4], waves_per_cell=2, warmup=1, timeout_s=10.0, ignore_eos=False,
              out_dir=tmp_path / "results", server_settings=None, order="rotate")
    kw.update(over)
    return cc.run_concurrent_benchmark(**kw)


def test_spare_prompts_are_required_for_the_warmup_waves(wl):
    from benchmarks import runner
    _, ps = runner.load_workload(wl)
    meas, warm = cc.select_prompts(ps, [64], [1, 2, 4])
    assert len(meas[64]) == 4 and len(warm[64]) == 4
    assert not {p["prompt_id"] for p in meas[64]} & {p["prompt_id"] for p in warm[64]}
    with pytest.raises(ValueError, match="need 32"):
        cc.select_prompts(ps, [64], [1, 16])


def test_waves_run_together_and_the_run_validates(wl, tmp_path):
    with fake_server(slow_scenario) as (host, port, _):
        run_dir, records = run(wl, tmp_path, host, port)
    assert mt.validate_run_dir(run_dir) == []
    meas = [r for r in records if not r["is_warmup"]]
    warm = [r for r in records if r["is_warmup"] and r["wave_id"] != "warmup"]
    # 2 levels x (1+2+4) requests per wave, 2 measured waves; one warmup wave per cell
    assert len(meas) == 2 * 7 * 2 and len(warm) == 2 * 7
    assert not {r["prompt_id"] for r in meas} & {r["prompt_id"] for r in warm}
    assert {r["schema"] for r in records} == {"kv-bench-request/3"}
    cells = wv.summarize_cells(records)
    assert set(cells) == {f"{lv}x{n}" for lv in (64, 128) for n in (1, 2, 4)}
    for key, c in cells.items():
        assert c["n_waves"] == 2 and c["n_requests"] == 2 * c["concurrency"]
        assert c["max_in_flight"] == c["concurrency"]          # overlap really happened
        assert c["start_skew_s_max"] < 0.1                      # barrier released them together
    assert cells["64x4"]["co_decode_s"]["median"] > 0.05        # all four were decoding at once


def test_cell_order_is_rotated_between_waves(wl, tmp_path):
    with fake_server(echoing_scenario) as (host, port, _):
        _, records = run(wl, tmp_path, host, port, concurrencies=[1, 2])
    by_wave = {}
    for r in records:
        if not r["is_warmup"] and r["slot"] == 0:
            by_wave.setdefault(r["wave_index"], []).append(r["wave_id"].rsplit("-w", 1)[0])
    assert by_wave[0] == ["L64-N1", "L64-N2", "L128-N1", "L128-N2"]
    assert by_wave[1] == ["L64-N2", "L128-N1", "L128-N2", "L64-N1"]


def test_a_failed_request_is_recorded_and_stops_after_its_wave(wl, tmp_path):
    def flaky(payload):
        if "wave-killer" in payload["messages"][0]["content"]:
            return [(0, role_event()), (0, finish_event()), (0, usage_event(5, 0)), (0, DONE)]
        return echoing_scenario(payload)

    from benchmarks import runner
    with fake_server(lambda p: flaky(p)) as (host, port, _):
        # the second prompt of the first level is the one that fails: patch its text on the fly
        orig = runner.load_workload
        def patched(d):
            m, p = orig(d)
            for q in p:
                if q["prompt_id"] == "L64-001":
                    q["text"] = "wave-killer " + q["text"]
            return m, p
        runner.load_workload = patched
        try:
            run_dir, records = run(wl, tmp_path, host, port, concurrencies=[2], waves_per_cell=3,
                                   levels=[64, 128])
        finally:
            runner.load_workload = orig
    bad = [r for r in records if r["status"] != "ok"]
    assert bad and bad[0]["error_category"] == "empty_output" and not bad[0]["usable_for_analysis"]
    first_bad_wave = bad[0]["wave_id"]
    in_wave = [r for r in records if r["wave_id"] == first_bad_wave]
    assert len(in_wave) == 2                                    # the whole wave was still recorded
    assert records[-1]["wave_id"] == first_bad_wave             # and nothing ran after it
    assert first_bad_wave not in wv.complete_waves(records)     # a partly failed wave is not aggregated
    assert mt.validate_run_dir(run_dir) == []


def rec(**kw):
    base = {"wave_id": "L8-N2-w0", "wave_size": 2, "level": 8, "concurrency": 2, "is_warmup": False,
            "usable_for_analysis": True, "output_tokens": 100, "slot": 0, "prompt_id": "a",
            "output_sha256": "h", "t_start_s": 0.0, "t_first_content_s": 1.0, "t_last_content_s": 5.0,
            "t_complete_s": 5.0, "ttft_s": 1.0, "e2e_latency_s": 5.0, "decode_tokens_per_s": 25.0,
            "memory": {"engine": {"requests_running_peak": 2.0, "requests_waiting_peak": 0.0,
                                  "kv_cache_usage_peak": 0.02}, "gpu": {"peak_used_mib_in_window": 9000.0}}}
    base.update(kw)
    return base


def test_wave_arithmetic_on_known_numbers():
    a = rec()
    b = rec(slot=1, prompt_id="b", t_start_s=0.0, t_first_content_s=3.0, t_last_content_s=9.0,
            t_complete_s=10.0, ttft_s=3.0, e2e_latency_s=10.0)
    s = wv.wave_stats([a, b])
    assert s["span_s"] == 10.0 and s["agg_output_tok_s"] == 20.0     # 200 tokens over 10 s
    assert s["ttft_first_s"] == 1.0 and s["ttft_last_s"] == 3.0
    assert s["co_decode_s"] == 2.0                                   # 5 (first to finish) - 3 (last to start)
    assert s["max_in_flight"] == 2 and s["engine_running_peak"] == 2.0


def test_serialised_requests_show_no_overlap_and_no_co_decode():
    a = rec(t_complete_s=5.0)
    b = rec(slot=1, prompt_id="b", t_start_s=5.0, t_first_content_s=6.0, t_last_content_s=10.0, t_complete_s=10.0)
    s = wv.wave_stats([a, b])
    assert s["max_in_flight"] == 1 and s["co_decode_s"] == 0.0


def test_end_at_the_same_instant_as_a_start_does_not_count_as_overlap():
    assert wv.max_in_flight([rec(t_start_s=0.0, t_complete_s=1.0), rec(t_start_s=1.0, t_complete_s=2.0)]) == 1


def test_incomplete_waves_are_excluded_and_bookkeeping_errors_are_found():
    only = [rec()]                                                    # wave_size 2 but one record
    assert wv.complete_waves(only) == {}
    assert any("wave_size" in p for p in wv.validate_waves(only))
    dup = [rec(), rec(slot=1)]                                        # same prompt twice in a wave
    assert any("twice" in p for p in wv.validate_waves(dup))


def test_batch_invariance_flags_a_prompt_that_changes_with_concurrency():
    r1 = rec(concurrency=1, output_sha256="x")
    r2 = rec(concurrency=4, output_sha256="y")
    r3 = rec(prompt_id="b", concurrency=1, output_sha256="z")
    r4 = rec(prompt_id="b", concurrency=4, output_sha256="z")
    out = wv.batch_invariance([r1, r2, r3, r4])
    assert out["differing_prompt_ids"] == ["a"] and not out["all_identical"]
    assert out["prompts_compared_across_concurrency"] == 2


def test_old_sequential_records_are_unaffected():
    seq = {"usable_for_analysis": True, "is_warmup": False, "prompt_id": "p", "output_sha256": "h"}
    assert wv.summarize_cells([seq]) == {} and wv.validate_waves([seq]) == []
    assert "kv-bench-request/3" in mt.REQUEST_SCHEMAS and "kv-bench-request/2" in mt.REQUEST_SCHEMAS


def test_every_request_in_a_wave_carries_the_same_wave_window_memory_block(wl, tmp_path):
    from benchmarks import sampling as sp
    mon = sp.MemoryMonitor(gpu_reader=lambda: {"memory_used_mib": 1000.0, "memory_total_mib": 15360.0,
                                               "source": "nvml", "scope": "device"}, interval_s=0.05)
    with fake_server(slow_scenario) as (host, port, _):
        run_dir, records = run(wl, tmp_path, host, port, levels=[64], concurrencies=[2], waves_per_cell=1,
                               monitor=mon, baseline_s=0.1)
    wave = [r for r in records if r["wave_id"] == "L64-N2-w0"]
    assert len(wave) == 2 and wave[0]["memory"] == wave[1]["memory"]
    assert "wave" in wave[0]["memory"]["window"] and wave[0]["memory"]["gpu"]["label"] == "measured"
    assert mt.validate_run_dir(run_dir) == []


def test_cli_path_and_the_kaggle_cell_per_wave_printer_work_end_to_end(wl, tmp_path):
    import re
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[1]
    with fake_server(slow_scenario) as (host, port, _):
        p = subprocess.run([sys.executable, "-m", "benchmarks.runner", "--workload-dir", str(wl), "--port", str(port),
                            "--levels", "64", "--concurrency-levels", "1", "2", "--waves", "2", "--warmup", "1",
                            "--no-sample-memory", "--baseline-seconds", "0", "--timeout", "10",
                            "--out-dir", str(tmp_path / "cli")], cwd=root, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    assert "waves" in p.stdout and "batch invariance" in p.stdout
    run_dir = next((tmp_path / "cli").glob("2*"))
    assert mt.validate_run_dir(run_dir) == []
    cell = (root / "docs" / "kaggle_step8_conc.py").read_text(encoding="utf-8")
    code = re.search(r"code = r'''(.*?)'''", cell, re.S).group(1)
    script = tmp_path / "wave_print.py"
    script.write_text(code, encoding="utf-8")
    q = subprocess.run([sys.executable, str(script), str(run_dir)], cwd=root, capture_output=True, text=True)
    assert q.returncode == 0, q.stderr[-800:]
    assert "L64-N2-w1" in q.stdout and "ok 2/2" in q.stdout
