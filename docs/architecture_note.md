# KV-cache estimator: architecture note (Step 3)

`benchmarks/cache_estimator.py` computes an **estimate**. It is not a hardware measurement and is always labeled `estimated`.

```
bytes = 2 x layers x kv_heads x head_dim x bytes_per_element x sum(retained_tokens per active sequence)
```

- The 2 is one key tensor plus one value tensor.
- `kv_heads` is `num_key_value_heads` from the model's `config.json`, not the number of query heads. With grouped-query attention it is smaller, which shrinks the cache.
- `retained_tokens` = input tokens + generated tokens, per sequence.
- Units: bytes, MiB (2^20), GiB (2^30).

## Assumptions

1. Every layer has the same KV shape and uses full attention.
2. No prefix sharing between sequences.
3. No cache compression or quantization beyond `bytes_per_element` (fp16 = 2).
4. No tensor-parallel sharding.
5. Allocator overhead, block padding, and activation buffers are excluded.

Configs with `use_sliding_window: true` or non-full-attention `layer_types` are rejected rather than silently overestimated.

## Why the estimate can differ from what the GPU shows

- Engines such as vLLM reserve a cache pool up front, so total GPU memory can stay flat while active cache grows.
- Block-based allocation rounds usage up.
- Total reserved memory also holds weights, activations, and framework buffers.

For these reasons the estimate is compared against engine-reported cache metrics in Step 6, not against `nvidia-smi` totals. Layer, KV-head, and head-dim values are read from the pinned model's `config.json` on the first Kaggle run, not assumed.
