import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import cache_estimator as ce  # noqa: E402

# Synthetic config used only to test the formula. These are NOT real model facts.
CFG = ce.CacheConfig(num_layers=10, num_kv_heads=4, head_dim=64, bytes_per_element=2)


def test_formula_matches_hand_calculation():
    # 2 * 10 * 4 * 64 * 2 = 10240 bytes per token
    assert CFG.bytes_per_token == 10240
    assert ce.estimate_cache_bytes(CFG, [100]) == 1_024_000


def test_doubling_retained_tokens_doubles_estimate():
    one = ce.estimate_cache_bytes(CFG, [500])
    two = ce.estimate_cache_bytes(CFG, [1000])
    assert two == 2 * one


def test_sums_over_sequences():
    assert ce.estimate_cache_bytes(CFG, [100, 200, 300]) == ce.estimate_cache_bytes(CFG, [600])


def test_input_and_generated_tokens_both_count():
    assert ce.retained_tokens(1000, 128) == 1128
    assert ce.estimate_cache_bytes(CFG, [ce.retained_tokens(1000, 128)]) > ce.estimate_cache_bytes(
        CFG, [1000]
    )


def test_uses_kv_heads_not_query_heads():
    hf = {
        "num_hidden_layers": 10,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
    }
    cfg = ce.config_from_hf(hf, "fp16")
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 64  # 2048 / 32
    assert cfg.bytes_per_token == 10240


def test_missing_kv_heads_falls_back_to_query_heads():
    hf = {"num_hidden_layers": 2, "num_attention_heads": 8, "hidden_size": 512}
    assert ce.config_from_hf(hf).num_kv_heads == 8


def test_explicit_head_dim_is_used():
    hf = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 512,
        "head_dim": 128,
    }
    assert ce.config_from_hf(hf).head_dim == 128


def test_sliding_window_rejected():
    hf = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "hidden_size": 512,
        "use_sliding_window": True,
    }
    with pytest.raises(ce.UnsupportedArchitecture):
        ce.config_from_hf(hf)


def test_hybrid_layer_types_rejected():
    hf = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "hidden_size": 512,
        "layer_types": ["full_attention", "sliding_attention"],
    }
    with pytest.raises(ce.UnsupportedArchitecture):
        ce.config_from_hf(hf)


def test_unknown_dtype_rejected():
    with pytest.raises(ValueError):
        ce.config_from_hf({"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 8}, "fp7")


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        ce.CacheConfig(0, 1, 1, 2)
    with pytest.raises(ValueError):
        ce.estimate_cache_bytes(CFG, [-1])
    with pytest.raises(ValueError):
        ce.retained_tokens(-1, 0)


def test_units_are_binary_and_labeled():
    assert ce.MIB == 2**20 and ce.GIB == 2**30
    assert "MiB" in ce.format_bytes(5 * ce.MIB)
    assert "GiB" in ce.format_bytes(3 * ce.GIB)
    assert "estimated" in ce.format_bytes(10)


def test_real_qwen_config_matches_vllm_reported_capacity():
    # Real values from Qwen/Qwen2.5-1.5B-Instruct config.json (Kaggle run, 2026-11-16).
    hf = {
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "hidden_size": 1536,
        "layer_types": ["full_attention"] * 28,
        "use_sliding_window": False,
        "sliding_window": None,
    }
    cfg = ce.config_from_hf(hf, "fp16")
    assert cfg.bytes_per_token == 28672  # 28 KiB per token
    # vLLM 0.33.0 reported 324,928 cache tokens in "8.68 GiB available".
    gib = ce.estimate_cache_bytes(cfg, [324_928]) / ce.GIB
    assert gib == pytest.approx(8.68, abs=0.01)


def test_summary_is_labeled_estimate():
    s = ce.summarize(CFG, 1000, 128, 4)
    assert s["label"] == "estimated"
    assert s["estimated_bytes"] == CFG.bytes_per_token * 1128 * 4
