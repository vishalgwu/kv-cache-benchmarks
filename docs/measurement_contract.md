# Measurement contract (Step 6)

Every number this project reports is one of three kinds, and says which:

- **measured**: read from hardware, the engine, or a clock during a run.
- **estimated**: computed from a formula or model parameters (for example the KV-cache size estimate). Never mixed into measured columns.
- **unavailable**: could not be obtained. It is stored as `null` with a reason, never as 0 and never invented.

Each value below has a unit, a scope (what it covers), and a source. A value without all of these is not reportable. Reproduce any figure with `python -m benchmarks.metrics results/<run_id>`, which recomputes it from `requests.jsonl`.

## 1. Request timing (client side)

All timestamps come from `time.monotonic()` in the runner process. Wall-clock time is stored only to order runs, never to compute durations.

| Metric | Unit | Definition | Source |
|---|---|---|---|
| `t_start_s` | s (monotonic) | taken immediately before the HTTP request is written; TCP connect happens inside the request | measured |
| `t_first_content_s` | s | arrival of the first stream event with non-empty text. Role-only, empty, and usage-only events are ignored | measured |
| `t_last_content_s` | s | arrival of the last such event | measured |
| `t_complete_s` | s | `[DONE]` received | measured |
| `ttft_s` | s | `t_first_content_s - t_start_s`; includes connect, queueing, prefill, and transport | measured |
| `e2e_latency_s` | s | `t_complete_s - t_start_s` | measured |
| `decode_time_s` | s | `t_last_content_s - t_first_content_s` | measured |
| `decode_tokens_per_s` | tokens/s | `(output_tokens - 1) / decode_time_s`; undefined (null) for fewer than 2 tokens | measured tokens over chunk timing (a proxy: one event may carry several tokens) |

"Arrival" means the client finished reading that stream line. Client TTFT is not engine prefill time; it also contains queueing and transport.

## 2. Token counts

| Metric | Unit | Source |
|---|---|---|
| `prompt_tokens`, `output_tokens` | tokens | the engine's own `usage` block |
| `input_tokens_manifest` | tokens | the workload generator's count (chat template applied) |

Token counts are never derived from stream-event counts (`content_chunks` is an event count and is labeled so). A request whose `usage` is missing is not usable. `prompt_tokens` must equal `input_tokens_manifest`; a mismatch invalidates the record.

## 3. Aggregates

Only records with `usable_for_analysis: true` enter a statistic: status ok, not a warmup, no validation errors.

| Metric | Definition |
|---|---|
| `output_tokens_per_s_over_span` | total output tokens / (last `t_complete_s` - first `t_start_s`) over the included records. Use for overlapping requests (Step 8). Includes idle gaps between sequential requests |
| `output_tokens_per_s_over_busy_time` | total output tokens / sum of `e2e_latency_s` |
| percentiles | linear interpolation, always reported with the sample count `n`. **p95 needs n >= 20, p99 needs n >= 100**; below that the value is null with a note, because a small trial set cannot establish tail latency |

Failures (timeouts, HTTP errors, malformed or incomplete streams, empty outputs, out-of-memory) are counted by category and reported separately. They never enter timing statistics.

## 4. Memory and engine metrics

Sampled on background threads in the runner process, on the same monotonic clock as the request timestamps. The default interval is 0.5 s. Each sample's time is the midpoint of the read.

| Metric | Unit | Scope | Source | Label |
|---|---|---|---|---|
| `gpu.memory_used_mib` | MiB | **whole device** (all processes on the GPU), not per process | NVML if available, else `nvidia-smi` | measured |
| `gpu.baseline_used_mib` | MiB | same | median of samples in an idle window before the first request | measured |
| `gpu.peak_used_mib_in_window` | MiB | same | maximum sample inside the request window | measured, **sampled peak** |
| `gpu.used_mib_at_window_start`, `gpu.peak_minus_window_start_mib` | MiB | same | last sample at or before the request start; the window peak minus that value (growth during this request) | measured |
| `engine.kv_cache_usage_peak` | fraction of the cache pool (0 to 1; "1 means 100 percent" in the engine's HELP text, confirmed on vLLM 0.33.0) | the engine's KV-cache pool | vLLM `/metrics` `vllm:kv_cache_usage_perc` | measured; sampled peak |
| `engine.requests_running_peak`, `requests_waiting_peak` | requests | the engine scheduler | vLLM `/metrics` | measured |
| `engine.preemptions_in_window` | count | the engine | vLLM `/metrics` counter delta | measured |
| `engine_reported_startup.*` | GiB, tokens | the engine at startup | vLLM server log | engine-reported (parsed text) |
| `cpu.server_tree_rss_peak_mib` | MiB | RSS summed over the server process tree | psutil or `/proc` | measured; may double count shared pages |
| `cpu.host_mem_available_min_mib` | MiB | whole machine (inside a container this is the machine, not the container's limit) | `/proc/meminfo` MemAvailable | measured |
| `cpu.container_mem_limit_mib`, `cpu.container_mem_current_peak_mib` | MiB | the container's cgroup | `/sys/fs/cgroup` (v2, then v1) | measured; null when unreadable or unlimited |
| `engine.prefix_cache_hits_in_window`, `engine.prefix_cache_queries_in_window` | tokens | the engine | vLLM counters, delta over the window | measured; the run-level delta in the manifest is the reliable check that nothing was reused |
| KV-cache size from the formula | bytes | one model configuration | `benchmarks/cache_estimator.py` | **estimated** |

### Limits that every reader must respect

1. **Device memory is not KV-cache usage.** vLLM preallocates its cache pool at startup (`gpu_memory_utilization`), so device memory is nearly flat while the active cache grows, and the amount used does not equal the amount of cache in use. Cache occupancy comes from the engine's own metrics or its startup log, not from device memory minus model weights.
2. **A sampled peak is a lower bound** on the true peak. Anything between two samples is invisible.
3. **A request shorter than the sampling interval may contain no sample.** It is then `unavailable` with the reason `no GPU sample fell inside the request window`. That is a valid, explicit gap, not an error and not a zero.
4. **Device scope assumes one process uses the GPU.** Memory is attributed to the vLLM server because it is the only workload on GPU 0. In a container, per-process GPU memory is usually unreadable, so none is reported.
5. **GPU indices are physical** (NVML and `nvidia-smi` ignore `CUDA_VISIBLE_DEVICES`). The study uses physical GPU 0 on a machine with identical GPUs.
6. **Sampling has a cost.** Reading `/metrics` and NVML adds small load to the server. The sampler's own read errors are counted in the run manifest.
7. **Idle baseline is taken after the server has started and before the first request.** It includes the preallocated cache pool, so a request's peak minus the baseline is the extra memory used above the pool, mainly activations.

8. **Device memory is a retained high-water mark, not a per-request value.** The allocator keeps memory after a request finishes. In the first real run (2026-11-16), device memory rose 54 MiB at the first 1024-token request and another 106 MiB at the first 4096-token request, then stayed there; repeat requests of the same size showed zero growth and the same "peak". A request's `peak_used_mib_in_window` therefore includes growth from earlier requests. Use `peak_minus_window_start_mib` for growth during one request, and treat peaks as order-dependent. For a clean per-cell figure, restart the server for each cell.
9. **`/proc/meminfo` inside a container reports the machine, not the container.** On Kaggle it showed about 27 GiB available while vLLM's own startup log said 3.85 GiB (and 6.18 GiB in an earlier run). The cgroup fields above are the container's view; CPU-offload headroom must be judged from those, not from `host_mem_available_min_mib`.
10. **The engine's occupancy reproduces the size formula, with these inferred details.** Reported usage equals `ceil((prompt + output - 1) / 16) / (total_blocks - 1)`: 16-token blocks, one block reserved by the engine, and the last generated token not yet cached. This held to six digits for all four measured requests in the first real run. It is inferred from the data, not read from vLLM's documentation.
11. **Counters can lag.** Engine counters (preemptions, prefix-cache hits) may update just after a request ends, so one window can under-report by a request. Use the run-level deltas in `run_manifest.json` (`memory_monitoring.engine_counter_deltas_over_run`).

### Engine metric names

vLLM metric names change between versions. The runner records the engine's actual catalog (name, type, HELP) in `engine_metrics_catalog.tsv` for each run and looks for the canonical metrics under known candidate names. A metric the engine does not expose is `unavailable`, not zero.

Confirmed on vLLM 0.33.0 (2026-11-16): `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_preemptions_total`, `vllm:prefix_cache_hits_total`, and `vllm:prefix_cache_queries_total` all exist. The catalog also exposes server-side phase timings (`request_queue_time`, `request_prefill_time`, `request_decode_time`, `time_to_first_token` histograms), which could split client TTFT into queueing and prefill, and `prompt_tokens_cached_by_source`, whose help text names a host (DRAM) tier. Neither is used yet.

## 5. Record and schema rules

- A record is written for every request, including warmups and failures, with `status`, `error_category`, and `error_message`.
- `usable_for_analysis` is true only for status ok, non-warmup records with no validation errors.
- `kv-bench-request/2` adds the optional `memory` block (sections `gpu`, `engine`, `cpu`, each with `label`, and `reason` when unavailable). `kv-bench-request/1` records (no memory block) stay valid.
- `validate_run_dir` checks that the config hash matches the config, manifest counts match the records, request IDs are unique, stored validation errors match a recomputation, timestamps are consistent, and failed or warmup records are never usable.
- Test fixtures (fake servers, synthetic records, scripted samples) are never written under `results/`.
