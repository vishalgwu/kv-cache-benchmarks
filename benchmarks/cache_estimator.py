"""Analytical KV-cache size estimator (Step 3).

All values returned by this module are **estimates**, not hardware measurements.

Formula (full attention, uniform KV shapes):

    bytes = 2 * layers * kv_heads * head_dim * bytes_per_element * retained_tokens

where the leading 2 covers keys and values, and ``retained_tokens`` is the sum of
(input + generated) tokens over all active sequences.

Assumptions (see docs/architecture_note.md):
* uniform KV shape in every layer, full attention in every layer
* no prefix sharing between sequences
* no cache compression or quantization beyond ``bytes_per_element``
* no tensor-parallel sharding
* allocator overhead, block padding, and activation buffers are excluded
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

KIB = 2**10
MIB = 2**20  # mebibyte, 2^20 bytes
GIB = 2**30  # gibibyte, 2^30 bytes

BYTES_PER_ELEMENT = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
}

# Marks every output of this module so it is never mistaken for a measurement.
LABEL = "estimated"


class UnsupportedArchitecture(ValueError):
    """Raised when the simple full-attention formula does not apply."""


@dataclass(frozen=True)
class CacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_dim", "bytes_per_element"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def bytes_per_token(self) -> int:
        """Estimated KV bytes for one retained token in one sequence."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element


def config_from_hf(hf_config: Mapping, dtype: str = "fp16") -> CacheConfig:
    """Build a CacheConfig from a Hugging Face ``config.json`` dictionary.

    Uses ``num_key_value_heads`` (KV heads), NOT ``num_attention_heads`` (query
    heads). Falls back to query heads only when the config has no KV-head field,
    which is the multi-head-attention case.

    Raises UnsupportedArchitecture for sliding-window attention, because the
    simple formula would overestimate the cache.
    """
    if hf_config.get("use_sliding_window"):
        raise UnsupportedArchitecture(
            "use_sliding_window is true: windowed layers retain fewer tokens; "
            "treat this architecture separately."
        )
    if hf_config.get("layer_types") and any(
        t != "full_attention" for t in hf_config["layer_types"]
    ):
        raise UnsupportedArchitecture("hybrid layer_types found; treat separately.")

    layers = hf_config["num_hidden_layers"]
    query_heads = hf_config["num_attention_heads"]
    kv_heads = hf_config.get("num_key_value_heads", query_heads)
    head_dim = hf_config.get("head_dim") or hf_config["hidden_size"] // query_heads

    key = dtype.lower()
    if key not in BYTES_PER_ELEMENT:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(BYTES_PER_ELEMENT)}")
    return CacheConfig(layers, kv_heads, head_dim, BYTES_PER_ELEMENT[key])


def retained_tokens(input_tokens: int, generated_tokens: int) -> int:
    """Tokens a sequence holds in cache: prompt tokens plus generated tokens."""
    if input_tokens < 0 or generated_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return input_tokens + generated_tokens


def estimate_cache_bytes(config: CacheConfig, tokens_per_sequence: Iterable[int]) -> int:
    """Estimated KV-cache bytes for the given active sequences."""
    total_tokens = 0
    for tokens in tokens_per_sequence:
        if tokens < 0:
            raise ValueError("token counts must be non-negative")
        total_tokens += tokens
    return config.bytes_per_token * total_tokens


def format_bytes(num_bytes: int) -> str:
    """Human-readable size with explicit binary units and the 'estimated' label."""
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.2f} GiB ({LABEL})"
    if num_bytes >= MIB:
        return f"{num_bytes / MIB:.2f} MiB ({LABEL})"
    return f"{num_bytes} bytes ({LABEL})"


def summarize(config: CacheConfig, input_tokens: int, generated_tokens: int, concurrency: int) -> dict:
    """Estimate for ``concurrency`` identical sequences; every field is labeled."""
    per_seq = retained_tokens(input_tokens, generated_tokens)
    total = estimate_cache_bytes(config, [per_seq] * concurrency)
    return {
        "label": LABEL,
        "bytes_per_token": config.bytes_per_token,
        "retained_tokens_per_sequence": per_seq,
        "concurrency": concurrency,
        "estimated_bytes": total,
        "estimated_mib": total / MIB,
        "estimated_gib": total / GIB,
    }
