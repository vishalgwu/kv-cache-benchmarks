"""Step 9 working-set sweep: fake server with a simulated LRU prefix cache; nothing is written under results/."""

import collections
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import workingset as ws  # noqa: E402
from workloads import generate as g  # noqa: E402
from test_runner import MODEL, echoing_scenario, fake_counter, fake_server  # noqa: E402

LADDER = (4, 8)


@pytest.fixture()
def wl(tmp_path):
    manifest, prompts = g.generate_workload(
        base_seed=7, levels=[64], per_level=sum(LADDER) + 1, output_tokens=16, max_context=1024,
        count_tokens=fake_counter, model_id=MODEL, revision="abc123")
    return g.write_workload(tmp_path / "wl", manifest, prompts)


class SimCache:
    """A prefix cache that keeps the last ``capacity`` distinct contexts (LRU) and counts hits like vLLM."""

    def __init__(self, capacity):
        self.capacity, self.lru = capacity, collections.OrderedDict()
        self.queries = self.hits = 0.0

    def scenario(self, payload):
        text = payload["messages"][0]["content"]
        key = text.split("\n\nFollow-up")[0]
        tokens = fake_counter(text)
        ctx_tokens = fake_counter(key)
        self.queries += tokens
        if key in self.lru:
            self.hits += ctx_tokens
            self.lru.move_to_end(key)
        elif self.capacity:
            self.lru[key] = True
            while len(self.lru) > self.capacity:
                self.lru.popitem(last=False)
        return echoing_scenario(payload)

    def metrics(self, host, port):
        return {"prefix_cache_queries_total": self.queries, "prefix_cache_hits_total": self.hits}


def run(wl, tmp_path, sim, **over):
    with fake_server(sim.scenario) as (host, port, _):
        kw = dict(workload_dir=wl, host=host, port=port, model=MODEL, revision="abc123", strategy="prefix_cache",
                  sizes=[4, 8], rounds=1, revisit_order="same", seed=1, timeout_s=10.0, out_dir=tmp_path / "res",
                  server_settings=None, settle_s=0.0, ladder=LADDER, metrics_fn=sim.metrics)
        kw.update(over)
        return ws.run_workingset(**kw)


def test_session_blocks_and_visit_orders():
    assert [ws.session_offset(s) for s in ws.LADDER] == [0, 48, 144, 288]
    with pytest.raises(ValueError):
        ws.session_offset(50)
    assert ws.visit_order(5, 1, "same", 1) == [0, 1, 2, 3, 4]
    a, b = ws.visit_order(20, 1, "shuffled", 1), ws.visit_order(20, 1, "shuffled", 1)
    assert a == b and sorted(a) == list(range(20)) and a != list(range(20))
    assert ws.visit_order(20, 2, "shuffled", 1) != a            # a new order each round
    with pytest.raises(ValueError):
        ws.visit_order(3, 1, "random", 1)


def test_revisits_hit_when_the_working_set_fits_and_miss_when_it_does_not(wl, tmp_path):
    run_dir, records = run(wl, tmp_path, SimCache(capacity=6))
    assert mt.validate_run_dir(run_dir) == []
    cells = ws.summarize(records)
    small, big = cells["4"], cells["8"]
    assert small["fill"]["hit_token_share"] == 0 and small["fill"]["requests_mostly_cached"] == 0
    assert small["revisit"]["requests_mostly_cached"] == 4 and small["revisit"]["share_mostly_cached"] == 1.0
    assert big["fill"]["requests_mostly_cached"] == 0
    assert big["revisit"]["requests_mostly_cached"] == 0         # 8 sessions cycled through 6 slots: every revisit misses
    assert big["revisit"]["hit_token_share"] == 0


def test_more_capacity_keeps_the_larger_working_set_alive(wl, tmp_path):
    _, records = run(wl, tmp_path, SimCache(capacity=12))
    assert ws.summarize(records)["8"]["revisit"]["share_mostly_cached"] == 1.0


def test_no_caching_means_no_hits_anywhere(wl, tmp_path):
    _, records = run(wl, tmp_path, SimCache(capacity=0), strategy="baseline")
    cells = ws.summarize(records)
    assert all(c[p]["hit_token_share"] == 0 for c in cells.values() for p in ("fill", "revisit"))
    assert {r["strategy"] for r in records} == {"baseline"}


def test_each_size_uses_its_own_block_of_sessions_and_the_warmup_a_spare(wl, tmp_path):
    _, records = run(wl, tmp_path, SimCache(capacity=6))
    ids = {s: {r["context_id"] for r in records if r.get("working_set_sessions") == s} for s in (4, 8)}
    assert len(ids[4]) == 4 and len(ids[8]) == 8 and not ids[4] & ids[8]
    warm = [r for r in records if r["is_warmup"]]
    assert len(warm) == 1 and warm[0]["context_id"] not in ids[4] | ids[8] and warm[0]["phase"] == "warmup"
    assert len(records) == 1 + (4 * 2) + (8 * 2)                  # warmup, then fill + 1 revisit round per size


def test_every_request_has_a_distinct_question_and_records_carry_the_cell(wl, tmp_path):
    _, records = run(wl, tmp_path, SimCache(capacity=6), rounds=2)
    meas = [r for r in records if not r["is_warmup"]]
    assert len({r["prompt_sha256"] for r in meas}) == len(meas)     # same context, different turn text
    assert {(r["phase"], r["turn"]) for r in meas} == {("fill", 0), ("revisit", 1), ("revisit", 2)}
    assert all(r["schema"] == "kv-bench-request/4" and r["context_tokens"] == 64 for r in meas)


def test_a_failed_request_stops_the_sweep(wl, tmp_path):
    sim = SimCache(capacity=6)
    calls = []

    def flaky(payload):
        calls.append(1)
        if len(calls) == 7:
            from test_runner import DONE, finish_event, role_event, usage_event
            return [(0, role_event()), (0, finish_event()), (0, usage_event(5, 0)), (0, DONE)]
        return sim.scenario(payload)

    with fake_server(flaky) as (host, port, _):
        run_dir, records = ws.run_workingset(
            workload_dir=wl, host=host, port=port, model=MODEL, revision="abc123", strategy="prefix_cache",
            sizes=[4, 8], rounds=1, revisit_order="same", seed=1, timeout_s=10.0, out_dir=tmp_path / "res",
            server_settings=None, settle_s=0.0, ladder=LADDER, metrics_fn=sim.metrics)
    assert records[-1]["status"] != "ok" and len(records) == 7 and mt.validate_run_dir(run_dir) == []


def test_workload_with_too_few_contexts_is_refused(wl, tmp_path):
    with pytest.raises(ValueError, match="need"):
        run(wl, tmp_path, SimCache(capacity=6), ladder=(8, 16))


def test_table_names_the_strategy_and_shows_both_phases(wl, tmp_path):
    _, records = run(wl, tmp_path, SimCache(capacity=6))
    text = ws.format_table(records)
    assert text.splitlines()[0] == "prefix_cache" and "revisit" in text and "fill" in text
    assert len(text.splitlines()) == 1 + 1 + 4                      # strategy, header, two sizes x two phases


def test_cli_table_mode_prints_a_summary(wl, tmp_path, capsys, monkeypatch):
    run_dir, _ = run(wl, tmp_path, SimCache(capacity=6))
    monkeypatch.setattr(sys, "argv", ["workingset", "--table", str(run_dir)])
    ws.main()
    out = capsys.readouterr().out
    assert "validation OK" in out and "prefix_cache" in out


@pytest.mark.parametrize("name", ["kaggle_step9_ws_a.py", "kaggle_step9_ws_b.py"])
def test_kaggle_cells_use_real_flags_valid_strategies_and_ladder_sizes(name):
    import re
    import subprocess
    from benchmarks import probe_strategies as pr
    root = pathlib.Path(__file__).resolve().parents[1]
    cell = (root / "docs" / name).read_text(encoding="utf-8")
    compile(cell, name, "exec")
    helptext = subprocess.run([sys.executable, "-m", "benchmarks.workingset", "--help"], cwd=root,
                              capture_output=True, text=True).stdout
    cmd = re.search(r'python -m benchmarks\.workingset --workload-dir(.*?)--out-dir', cell, re.S).group(1)
    for flag in set(re.findall(r"--[a-z][a-z-]+", cmd)):
        assert flag in helptext, (name, flag)
    runs = eval(re.search(r"^RUNS = (\[.*\])$", cell, re.M).group(1))
    assert runs
    for strategy, sessions in runs:
        assert strategy in pr.CONFIGS and strategy != "fp8_kv"
        assert all(int(s) in ws.LADDER for s in sessions.split())
    # the workload holds every ladder block plus one spare context
    assert f"--per-level {sum(ws.LADDER) + 1}" in cell and "--levels 4096" in cell


def test_external_cache_hits_are_sampled_and_recorded_when_a_connector_exposes_them(wl, tmp_path):
    from benchmarks import sampling as sp
    samples, _ = sp.parse_prometheus('vllm:external_prefix_cache_hits_total{engine="0"} 128.0\n'
                                     'vllm:external_prefix_cache_queries_total{engine="0"} 4096.0\n')
    values = sp.extract_engine_values(samples)
    assert values["external_prefix_cache_hits_total"] == 128.0 and values["external_prefix_cache_queries_total"] == 4096.0
    assert sp.extract_engine_values([])["external_prefix_cache_hits_total"] is None
    sim = SimCache(capacity=6)
    ext = {"v": 0.0}

    def metrics(host, port):
        out = sim.metrics(host, port)
        out["external_prefix_cache_hits_total"] = ext["v"]
        return out

    def scenario(payload):
        ext["v"] += 10                                  # pretend a CPU tier supplied 10 tokens per request
        return sim.scenario(payload)

    with fake_server(scenario) as (host, port, _):
        _, records = ws.run_workingset(workload_dir=wl, host=host, port=port, model=MODEL, revision="abc123",
                                       strategy="kv_offload_native", sizes=[4], rounds=1, revisit_order="same",
                                       seed=1, timeout_s=10.0, out_dir=tmp_path / "res", server_settings=None,
                                       settle_s=0.0, ladder=LADDER, metrics_fn=metrics)
    assert {r["external_prefix_cache_hits_delta"] for r in records} == {10.0}
