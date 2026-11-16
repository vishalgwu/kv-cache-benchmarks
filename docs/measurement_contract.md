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
| `engine.kv_cache_usage_peak` | raw engine value | the engine's KV-cache pool | vLLM `/metrics` | measured; scale (fraction or percent) is confirmed from the HELP text saved in `engine_metrics_catalog.tsv` |
| `engine.requests_running_peak`, `requests_waiting_peak` | requests | the engine scheduler | vLLM `/metrics` | measured |
| `engine.preemptions_in_window` | count | the engine | vLLM `/metrics` counter delta | measured |
| `engine_reported_startup.*` | GiB, tokens | the engine at startup | vLLM server log | engine-reported (parsed text) |
| `cpu.server_tree_rss_peak_mib` | MiB | RSS summed over the server process tree | psutil or `/proc` | measured; may double count shared pages |
| `cpu.host_mem_available_min_mib` | MiB | whole machine | `/proc/meminfo` MemAvailable | measured |
| KV-cache size from the formula | bytes | one model configuration | `benchmarks/cache_estimator.py` | **estimated** |

### Limits that every reader must respect

1. **Device memory is not KV-cache usage.** vLLM preallocates its cache pool at startup (`gpu_memory_utilization`), so device memory is nearly flat while the active cache grows, and the amount used does not equal the amount of cache in use. Cache occupancy comes from the engine's own metrics or its startup log, not from device memory minus model weights.
2. **A sampled peak is a lower bound** on the true peak. Anything between two samples is invisible.
3. **A request shorter than the sampling interval may contain no sample.** It is then `unavailable` with the reason `no GPU sample fell inside the request window`. That is a valid, explicit gap, not an error and not a zero.
4. **Device scope assumes one process uses the GPU.** Memory is attributed to the vLLM server because it is the only workload on GPU 0. In a container, per-process GPU memory is usually unreadable, so none is reported.
5. **GPU indices are physical** (NVML and `nvidia-smi` ignore `CUDA_VISIBLE_DEVICES`). The study uses physical GPU 0 on a machine with identical GPUs.
6. **Sampling has a cost.** Reading `/metrics` and NVML adds small load to the server. The sampler's own read errors are counted in the run manifest.
7. **Idle baseline is taken after the server has started and before the first request.** It includes the preallocated cache pool, so a request's peak minus the baseline is the extra memory used above the pool, mainly activations.

### Engine metric names

vLLM metric names change between versions. The runner records the engine's actual catalog (name, type, HELP) in `engine_metrics_catalog.tsv` for each run and looks for the canonical metrics under known candidate names. A metric the engine does not expose is `unavailable`, not zero.

## 5. Record and schema rules

- A record is written for every request, including warmups and failures, with `status`, `error_category`, and `error_message`.
- `usable_for_analysis` is true only for status ok, non-warmup records with no validation errors.
- `kv-bench-request/2` adds the optional `memory` block (sections `gpu`, `engine`, `cpu`, each with `label`, and `reason` when unavailable). `kv-bench-request/1` records (no memory block) stay valid.
- `validate_run_dir` checks that the config hash matches the config, manifest counts match the records, request IDs are unique, stored validation errors match a recomputation, timestamps are consistent, and failed or warmup records are never usable.
- Test fixtures (fake servers, synthetic records, scripted samples) are never written under `results/`.
