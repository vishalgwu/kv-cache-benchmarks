import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from workloads import generate as g  # noqa: E402


def fake_counter(text: str) -> int:
    """Test-only stand-in for a tokenizer: 10 template tokens, long words cost 2.

    This exercises the length-fitting logic. It is NOT a real tokenizer and its
    output must never appear in measured data.
    """
    return 10 + sum(1 if len(w) <= 5 else 2 for w in text.split())


KW = dict(
    base_seed=7, levels=[64, 256], per_level=3, output_tokens=16, max_context=1024,
    count_tokens=fake_counter, model_id="test/model", revision="abc123",
)


def test_same_seed_reproduces_workload():
    m1, p1 = g.generate_workload(**KW)
    m2, p2 = g.generate_workload(**KW)
    assert m1 == m2 and p1 == p2


def test_different_seed_changes_workload():
    m1, _ = g.generate_workload(**KW)
    m2, _ = g.generate_workload(**{**KW, "base_seed": 8})
    assert m1["workload_id"] != m2["workload_id"]


def test_actual_length_within_tolerance_and_never_over_target():
    _, prompts = g.generate_workload(**KW)
    for p in prompts:
        assert p["actual_input_tokens"] <= p["target_input_tokens"]
        assert p["target_input_tokens"] - p["actual_input_tokens"] <= g.DEFAULT_TOLERANCE
        assert fake_counter(p["text"]) == p["actual_input_tokens"]


def test_input_plus_output_must_fit_context():
    with pytest.raises(ValueError, match="max context"):
        g.generate_workload(**{**KW, "levels": [1020], "output_tokens": 16})


def test_prompts_are_distinct_with_no_shared_prefix():
    _, prompts = g.generate_workload(**{**KW, "per_level": 5})
    texts = [p["text"] for p in prompts]
    assert len(set(texts)) == len(texts)
    prefixes = {" ".join(t.split()[:8]) for t in texts}
    assert len(prefixes) == len(texts)


def test_hash_and_manifest_fields_present():
    manifest, prompts = g.generate_workload(**KW)
    assert len(manifest["requests"]) == 6
    for p, r in zip(prompts, manifest["requests"]):
        assert hashlib.sha256(p["text"].encode()).hexdigest() == r["prompt_sha256"]
        assert "text" not in r
        for key in ("prompt_id", "seed", "target_input_tokens", "actual_input_tokens",
                    "output_target_tokens", "prompt_sha256"):
            assert key in r
    assert manifest["truncation_policy"].startswith("none")
    assert manifest["revision"] == "abc123"


def test_target_smaller_than_template_overhead_fails():
    with pytest.raises(ValueError, match="overhead"):
        g.build_prompt(1, 5, fake_counter)


def test_counter_that_never_grows_fails():
    with pytest.raises(ValueError, match="did not grow"):
        g.build_prompt(1, 50, lambda text: 10)


def test_write_workload_roundtrip(tmp_path):
    manifest, prompts = g.generate_workload(**KW)
    out = g.write_workload(tmp_path, manifest, prompts)
    assert json.loads((out / "manifest.json").read_text()) == manifest
    lines = (out / "prompts.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == prompts


def test_derived_seed_is_stable_and_distinct():
    assert g.derive_seed(1, 64, 0) == g.derive_seed(1, 64, 0)
    assert g.derive_seed(1, 64, 0) != g.derive_seed(1, 64, 1)
    assert g.derive_seed(1, 64, 0) != g.derive_seed(1, 256, 0)
