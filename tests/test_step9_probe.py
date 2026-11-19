"""Step 9 probe and cgroup memory reading. Fake servers only; nothing is written under results/."""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmarks import probe_strategies as pr  # noqa: E402
from benchmarks import sampling as sp  # noqa: E402
from benchmarks import vllm_engine as eng  # noqa: E402
from test_runner import MODEL, echoing_scenario, fake_server  # noqa: E402

POOL_LINE = ("(EngineCore pid=1) INFO 11-17 05:35:10 [kv_cache_utils.py:1168] GPU KV cache size: {tokens} tokens, "
             "Maximum concurrency for 32,768 tokens per request: 9.44x\n")


def make_factory(fail=(), pool=None, log_extra=""):
    """A stand-in for VLLMServer: writes a log, 'starts' instantly, fails for the names in ``fail``."""
    class Fake:
        def __init__(self, model, revision, **kw):
            self.kw = kw
            self.pid = None

        def argv(self):
            return ["vllm", "serve", "m"] + list(self.kw["extra_args"])

        def start(self):
            extra = tuple(self.kw["extra_args"])
            if extra in fail:
                raise eng.ServerStartError("vllm exited with code 1 during startup: invalid argument")
            tokens = (pool or {}).get(extra, "309,248")
            pathlib.Path(self.kw["log_path"]).write_text(POOL_LINE.format(tokens=tokens) + log_extra)
            return self

        def stop(self):
            pass

    return Fake


class Counters:
    """Prefix-cache counters of a pretend engine: a repeated prompt hits only when prefix caching is on."""

    def __init__(self, prefix_on):
        self.prefix_on, self.q, self.h = prefix_on, 0.0, 0.0
        self.seen = False

    def __call__(self, host, port):
        return {"prefix_cache_queries_total": self.q, "prefix_cache_hits_total": self.h, "kv_cache_usage": 0.0}

    def note_request(self):
        self.q += 8192
        if self.prefix_on and self.seen:
            self.h += 8192
        self.seen = True


def probe(name, tmp_path, **factory_kw):
    prefix = pr.CONFIGS[name][0]
    counters = Counters(prefix)

    def scenario(payload):
        counters.note_request()
        return echoing_scenario(payload)

    with fake_server(scenario) as (host, port, _):
        return pr.run_probe(name, model=MODEL, revision="abc", prompt_text="hello", out_dir=tmp_path / name,
                            port=port, timeout_s=5.0, server_factory=make_factory(**factory_kw),
                            metrics_fn=counters, settle_s=0.0)


def test_baseline_is_a_control_and_prefix_cache_is_shown_active_only_by_a_hit(tmp_path):
    base = probe("baseline", tmp_path)
    pref = probe("prefix_cache", tmp_path)
    assert base["started"] and not base["second_request_hit_prefix_cache"] and base["same_output"]
    assert pref["second_request_hit_prefix_cache"]
    rows = {r["name"]: r for r in pr.classify([base, pref])}
    assert rows["baseline"]["verdict"].startswith("control")
    assert rows["prefix_cache"]["verdict"].startswith("active")


def test_a_flag_that_is_accepted_but_changes_nothing_is_not_called_active(tmp_path):
    base = probe("baseline", tmp_path)
    fp8 = probe("fp8_kv", tmp_path)                      # same pool as baseline: the flag did nothing visible
    row = {r["name"]: r for r in pr.classify([base, fp8])}["fp8_kv"]
    assert row["pool_vs_baseline"] == 1.0 and "not shown to be active" in row["verdict"]


def test_fp8_is_active_only_when_the_engine_reports_a_larger_pool(tmp_path):
    base = probe("baseline", tmp_path)
    fp8 = probe("fp8_kv", tmp_path, pool={("--kv-cache-dtype", "fp8"): "618,496"})
    row = {r["name"]: r for r in pr.classify([base, fp8])}["fp8_kv"]
    assert row["pool_vs_baseline"] == pytest.approx(2.0) and row["verdict"] == "active: pool grew"


def test_startup_failure_is_recorded_with_the_reason_and_the_probe_continues(tmp_path):
    bad = probe("kv_offload_native", tmp_path, fail={pr.CONFIGS["kv_offload_native"][1]})
    assert not bad["started"] and "invalid argument" in bad["failure"]
    assert bad["server_argv"][-4:] == ["--kv-offloading-backend", "native", "--kv-offloading-size", "8"]
    row = pr.classify([bad])[0]
    assert row["verdict"].startswith("did not start")


def test_offload_needs_a_log_mention_to_count_as_started_with_offloading(tmp_path):
    quiet = probe("kv_offload_native", tmp_path)
    loud = probe("kv_offload_native", tmp_path, log_extra="(EngineCore) INFO OffloadingConnector initialised\n")
    assert "not shown to be active" in pr.classify([quiet])[0]["verdict"]
    assert pr.classify([loud])[0]["verdict"].startswith("started and the log mentions offloading")


def test_evidence_lines_pick_relevant_lines_and_skip_progress_bars():
    text = "noise\nCapturing CUDA graphs (PIECEWISE): fp8\nINFO KV cache offloading enabled\nother\n"
    assert pr.evidence_lines(text) == ["INFO KV cache offloading enabled"]


def test_cgroup_v2_splits_anonymous_memory_from_page_cache(tmp_path):
    (tmp_path / "memory.stat").write_text("anon 1073741824\nfile 3221225472\nshmem 2147483648\nkernel 0\n")
    assert sp._cgroup_mem_stat_mib(str(tmp_path)) == {"container_anon_mib": 1024.0, "container_file_mib": 3072.0,
                                                      "container_shmem_mib": 2048.0}


def test_cgroup_v1_stat_and_missing_files(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.stat").write_text("total_rss 2097152\ntotal_cache 4194304\n")
    assert sp._cgroup_mem_stat_mib(str(tmp_path)) == {"container_anon_mib": 2.0, "container_file_mib": 4.0,
                                                      "container_shmem_mib": None}
    empty = tmp_path / "none"
    empty.mkdir()
    assert sp._cgroup_mem_stat_mib(str(empty)) == {"container_anon_mib": None, "container_file_mib": None,
                                                   "container_shmem_mib": None}


def test_cpu_block_reports_anon_and_file_peaks_and_old_records_still_work():
    m = sp.MemoryMonitor(cpu_reader=lambda: {}, interval_s=1.0)
    row = lambda t, a, f: {"t_s": t, "server_tree_rss_mib": 900.0, "host_mem_available_mib": 5000.0,  # noqa: E731
                           "container_mem_current_mib": 3000.0, "container_mem_limit_mib": 4096.0,
                           "container_anon_mib": a, "container_file_mib": f, "container_shmem_mib": f / 2,
                           "scope": "s", "source": "x"}
    m.samplers["cpu"].samples += [row(1.0, 800.0, 2000.0), row(2.0, 1200.0, 1800.0)]
    cpu = m.request_block(0.5, 2.5)["cpu"]
    assert cpu["container_anon_peak_mib"] == 1200.0 and cpu["container_file_peak_mib"] == 2000.0
    assert cpu["container_shmem_peak_mib"] == 1000.0
    m2 = sp.MemoryMonitor(cpu_reader=lambda: {}, interval_s=1.0)
    m2.samplers["cpu"].samples += [{"t_s": 1.0, "server_tree_rss_mib": 1.0, "host_mem_available_mib": 2.0,
                                    "scope": "s", "source": "x"}]
    assert m2.request_block(0.5, 1.5)["cpu"]["container_anon_peak_mib"] is None


def test_cpu_reader_includes_anon_and_file(monkeypatch):
    monkeypatch.setattr(sp, "_host_mem_mib", lambda: {"host_mem_available_mib": 1.0, "host_mem_total_mib": 2.0})
    monkeypatch.setattr(sp, "_cgroup_mem_mib", lambda: {"container_mem_limit_mib": None, "container_mem_current_mib": 5.0})
    monkeypatch.setattr(sp, "_cgroup_mem_stat_mib", lambda: {"container_anon_mib": 3.0, "container_file_mib": 2.0,
                                                                  "container_shmem_mib": 1.0})
    r = sp.make_cpu_reader(None)()
    assert (r["container_anon_mib"], r["container_file_mib"]) == (3.0, 2.0)


def test_probe_cell_compiles_and_uses_real_probe_flags():
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[1]
    cell = (root / "docs" / "kaggle_step9_probe.py").read_text(encoding="utf-8")
    compile(cell, "kaggle_step9_probe.py", "exec")
    helptext = subprocess.run([sys.executable, "-m", "benchmarks.probe_strategies", "--help"], cwd=root,
                              capture_output=True, text=True).stdout
    for flag in ("--workload-dir", "--out-dir", "--max-model-len"):
        assert flag in cell and flag in helptext
    for name in pr.CONFIGS:
        assert f'"{name}"' in cell                       # the cell reports every configuration the probe runs
