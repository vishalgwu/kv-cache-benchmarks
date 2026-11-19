# KV-Cache Pressure Benchmarks

A reproducible benchmark study of how **context length**, **concurrent requests**, and **supported KV-cache memory strategies** affect GPU/CPU memory use, time to first token (TTFT), and token throughput during LLM inference.

> **Status: foundation and first benchmark run done; no measured findings yet.** The environment, model, cache estimator, workload generator, and streaming runner exist and have run once on a Kaggle T4 (a pipeline check, not a result). Nothing in this repository should be read as a performance finding until the sweep steps (7-9) are marked accepted. See [docs/progress_tracker.md](docs/progress_tracker.md).

## Central question

How do context length, concurrent requests, and supported KV-cache memory/offload strategies change inference memory usage, time to first token, and throughput?

## What this project will produce

- A reproducible benchmark runner that sends real requests to a real model.
- Structured measurement records (JSONL request records plus a run manifest).
- Charts: TTFT vs. context length, throughput vs. concurrency, peak memory vs. context and concurrency.
- A technical report explaining where KV-cache pressure appears and which strategies improve capacity or performance, under explicitly recorded conditions.
- A tagged release with commands to reproduce the study.

## Scope

**In scope**
- Single-device study: one model, one inference engine, one hardware configuration.
- Context-length sweep and concurrency sweep on a small grid (candidate levels: 1K/4K/8K input tokens, concurrency 1/2/4, adjusted to hardware).
- Comparison of KV-cache strategies **only where the engine genuinely supports them** (for example, KV offload to CPU memory).
- Analytical KV-cache size estimator, compared against measured engine metrics.

**Out of scope**
- FabricOps (a separate infrastructure investigation copilot).
- The legacy diagnostics demo and its synthetic values (for example, "92% cache occupancy"). These are not benchmark evidence.
- Rack-scale or fabric claims. A single-device study cannot support them; rack-scale claims require rack-scale measurements.
- Model training. This project measures serving behavior only.

## Measurement rules

- Use a monotonic clock for all timings.
- **TTFT:** first generated-token arrival minus request submission, including observed queueing and transport. Empty or metadata-only stream events are ignored.
- **Decode throughput per request:** tokens after the first token divided by time from first-token arrival to last-token arrival (undefined for fewer than two tokens).
- **Aggregate output throughput:** total generated tokens divided by the declared batch wall-clock interval. Failures are reported separately.
- **Memory:** baseline and sampled peak GPU memory with scope and sampling interval. Total reserved GPU memory is not KV-cache usage; cache occupancy is reported only if the engine exposes it reliably. CPU memory is recorded for offload experiments where measurable.
- Every number is labeled **measured**, **estimated**, or **unavailable**.
- Warmups are excluded and identified. At least three repeated trials per matrix cell initially.
- Failures (out-of-memory, timeouts, rejected contexts) are retained with reasons; unsupported cells are marked skipped.
- Prefix-cache policy is recorded and unintended shared-prefix reuse is prevented.
- Runs with silently different models or precision are never compared.
- Percentiles are reported with sample counts; small trial sets do not establish tail latency.

## Roadmap (12 steps)

Progress is evidence-based. A step is accepted only when its completion gate is met.

| # | Step | Status |
|---|------|--------|
| 1 | Freeze the experiment and baseline scope | In progress (plan drafted; owner's two self-check answers pending) |
| 2 | Choose hardware, model, and engine | Accepted (Kaggle T4, Qwen2.5-1.5B-Instruct fp16, vLLM 0.33.0) |
| 3 | Understand and estimate KV-cache growth | Accepted (28 KiB/token estimate matches vLLM's reported cache capacity) |
| 4 | Create reproducible token workloads | In progress (generator verified with the real tokenizer; final sweep manifest pending) |
| 5 | Build the first streamed benchmark run | Accepted (runner works end to end; first run is a pipeline check) |
| 6 | Validate memory and timing measurements | Accepted (samplers verified on real hardware; engine cache occupancy matches the size formula) |
| 7 | Measure the context-length sweep | In progress (full 1K-32K sweep run, 0 failures, committed; next: review the sweep, then Step 8) |
| 8 | Add true concurrent-request experiments | In progress (1K-32K concurrency data collected and committed, 0 failures; written findings and acceptance next) |
| 9 | Compare real cache and offload strategies | In progress (probe run: prefix caching active, fp8 KV unsupported on the T4, CPU offload starts; working-set sweep part A run: prefix cache alone loses all revisits at 96+ sessions; part B (offload) next) |
| 10 | Analyze results and create evidence-backed charts | Pending |
| 11 | Prove reproducibility and package the benchmark | Pending |
| 12 | Complete the report and final release | Pending |

**Milestone gates:** A Foundation (1-4) · B First valid benchmark (5-6) · C Experimental evidence (7-9) · D Finished study (10-12).

### Next: Step 7, measure the context-length sweep
Decisions the first real runs call for, to settle before the sweep:
- **Server settings:** launch once with `max_model_len` 32768 and keep it fixed (the cache pool size depends on it). Compare `enforce_eager` on and off first, because decode used only about half the GPU in eager mode.
- **Output length:** decide whether to force 128 output tokens (`--ignore-eos`); 1 of 4 early requests stopped at 113.
- **Warmup:** one warmup request at each context length, not a short prompt.
- **Order and repeats:** at least three trials per level, order rotated or randomized, plus a same-prompt-twice determinism check.
- **Device memory:** treat per-request peaks as order-dependent (the allocator retains memory); restart the server per cell if a clean per-cell value is needed.
- Done when each planned context level has repeated measured results or an explicit failure or skip.

## Concepts covered

Inference vs. training · the KV cache and its size formula (`2 x layers x KV_heads x head_dim x bytes_per_element x retained_tokens`) · causal attention and grouped-query attention · tokenization and chat templates · prefill vs. decode · allocated vs. reserved GPU memory and cache pools · concurrency vs. continuous batching · prefix caching and KV offload · benchmark methodology (warmups, repeated trials, percentiles, failure handling) · reproducibility.

## Planned layout

```
benchmarks/   runner and engine adapters
configs/      experiment definitions and environment manifest
workloads/    prompt generation rules and seeds
results/      run manifests and structured request records
analysis/     aggregation and chart scripts
tests/        measurement and runner checks
docs/         methods, findings, legacy notes
```

## Hardware

Kaggle free-tier notebook with one NVIDIA Tesla T4 (15,360 MiB) used for every run; the machine exposes two T4s but the study sets `CUDA_VISIBLE_DEVICES=0`. Model: `Qwen/Qwen2.5-1.5B-Instruct` at a pinned revision, fp16. Engine: vLLM 0.33.0. Free shared GPUs are noisier than dedicated hardware, so the exact GPU, driver, and date are recorded with every run (`configs/environment.json`). Results are only valid for the hardware manifest they ship with.

## Documents

- [Project overview and completion plan](KV_Cache_Project_Overview_and_Completion_Plan.pdf)
- [Complete step-by-step roadmap](KV_Cache_Complete_Step_by_Step_GenAI_Roadmap.pdf)

## Reproducibility and hygiene

- Dependencies and model/engine revisions will be pinned.
- Secrets and model weights are kept out of Git.
- Small reference results are committed; large artifacts are documented with retrieval instructions.
