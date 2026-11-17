import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import envcheck as ec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_parse_lock_ignores_comments_and_normalises_names():
    lock = ec.parse_lock("# note\nvllm==0.34.0  # engine\nFlashInfer_Python==0.7.3\n\nnot a pin\n")
    assert lock == {"vllm": "0.34.0", "flashinfer-python": "0.7.3"}


def test_compare_flags_a_different_engine_as_blocking():
    lock = {"vllm": "0.34.0", "torch": "2.14.0"}
    assert ec.compare({"vllm": "0.34.0", "torch": "2.14.0"}, lock) == []
    probs = ec.compare({"vllm": "0.33.0", "torch": "2.14.0"}, lock)
    assert probs == [{"package": "vllm", "locked": "0.34.0", "installed": "0.33.0", "blocking": True}]
    missing = ec.compare({"vllm": "0.34.0"}, lock)
    assert missing[0]["package"] == "torch" and missing[0]["installed"] is None and not missing[0]["blocking"]


def test_check_returns_none_without_a_lock(tmp_path):
    assert ec.check(tmp_path / "missing.txt") is None


def test_committed_lock_agrees_with_the_committed_freeze():
    lock = ec.parse_lock((ROOT / "configs" / "requirements.lock.txt").read_text(encoding="utf-8"))
    freeze = ec.parse_lock((ROOT / "configs" / "kaggle_pip_freeze_2026-11-17.txt").read_text(encoding="utf-8"))
    assert lock["vllm"] == "0.34.0"
    assert ec.compare(freeze, lock) == []  # every locked version is the one in the Kaggle freeze
