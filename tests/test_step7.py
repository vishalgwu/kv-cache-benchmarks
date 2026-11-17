"""Step 7 additions: per-level warmup, rotated level order, per-level and determinism summaries.

Fake servers and synthetic workloads only; nothing here is written under results/.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import runner as rn  # noqa: E402
from workloads import generate as g  # noqa: E402
from test_runner import MODEL, echoing_scenario, fake_counter, fake_server, run  # noqa: E402


@pytest.fixture()
def spare_workload(tmp_path):
    manifest, prompts = g.generate_workload(
        base_seed=3, levels=[64, 128], per_level=3, output_tokens=16, max_context=1024,
        count_tokens=fake_counter, model_id=MODEL, revision="abc123")
    return g.write_workload(tmp_path / "wl3", manifest, prompts)


def test_rotate_order_shifts_levels_each_trial():
    assert rn.level_order([1, 2, 3], 4, "rotate") == [[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3]]
    assert rn.level_order([1, 2, 3], 2, "ascending") == [[1, 2, 3], [1, 2, 3]]
    with pytest.raises(ValueError):
        rn.level_order([1], 1, "random")


def test_per_level_warmup_uses_a_spare_unmeasured_prompt(spare_workload, tmp_path):
    with fake_server(echoing_scenario) as (host, port, _):
        run_dir, records = run(spare_workload, tmp_path, host, port, warmup=0,
                               warmup_per_level=True, trials=2, order="rotate")
    warm = [r for r in records if r["is_warmup"]]
    meas = [r for r in records if not r["is_warmup"]]
    assert [r["prompt_id"] for r in warm] == ["L64-001", "L128-001"]
    assert not {r["prompt_id"] for r in warm} & {r["prompt_id"] for r in meas}
    assert [r["prompt_id"] for r in meas] == ["L64-000", "L128-000", "L128-000", "L64-000"]
    assert [r["output_target_tokens"] for r in warm] == [16, 16]  # warms decode too
    assert (run_dir / "workload_manifest.json").exists()
    cfg = json.loads((run_dir / "run_manifest.json").read_text())["config"]
    assert cfg["warmup_prompt_ids"] == ["L64-001", "L128-001"] and cfg["level_order"] == "rotate"
    assert mt.validate_run_dir(run_dir) == []


def test_per_level_warmup_without_a_spare_prompt_is_refused(spare_workload, tmp_path):
    with fake_server(echoing_scenario) as (host, port, _):
        with pytest.raises(ValueError, match="spare prompt"):
            run(spare_workload, tmp_path, host, port, warmup_per_level=True, limit_per_level=3)


def test_by_level_and_determinism_summaries(spare_workload, tmp_path):
    with fake_server(echoing_scenario) as (host, port, _):
        run_dir, records = run(spare_workload, tmp_path, host, port, warmup=0, trials=3)
    s = mt.summarize_run(records)
    assert s["by_level"]["64"]["n"] == 3 and s["by_level"]["64"]["prompts"] == 1
    assert s["by_level"]["64"]["ttft_s"]["p95"] is None  # n < 20
    assert s["determinism"]["prompts_with_repeats"] == 2 and s["determinism"]["all_identical"]
    records[0]["output_sha256"] = "x"
    d = mt.determinism(records)
    assert not d["all_identical"] and d["differing_prompt_ids"] == [records[0]["prompt_id"]]
