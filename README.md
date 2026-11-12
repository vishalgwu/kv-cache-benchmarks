# KV-Cache Pressure Benchmarks

A reproducible benchmark study of how **context length**, **concurrent requests**, and **supported KV-cache memory strategies** affect GPU/CPU memory use, time to first token (TTFT), and token throughput during LLM inference.

> **Status: planning and setup.** No benchmark runner, hardware manifest, or measured results exist yet. Nothing in this repository should be read as a performance finding until the corresponding step below is marked accepted.

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
| 1 | Freeze the experiment and baseline scope | Pending |
| 2 | Choose hardware, model, and engine | Pending |
| 3 | Understand and estimate KV-cache growth | Pending |
| 4 | Create reproducible token workloads | Pending |
| 5 | Build the first streamed benchmark run | Pending |
| 6 | Validate memory and timing measurements | Pending |
| 7 | Measure the context-length sweep | Pending |
| 8 | Add true concurrent-request experiments | Pending |
| 9 | Compare real cache and offload strategies | Pending |
| 10 | Analyze results and create evidence-backed charts | Pending |
| 11 | Prove reproducibility and package the benchmark | Pending |
| 12 | Complete the report and final release | Pending |

**Milestone gates:** A Foundation (1-4) · B First valid benchmark (5-6) · C Experimental evidence (7-9) · D Finished study (10-12).

### Step 1 (next): freeze the experiment and baseline scope
- Write `docs/experiment_plan.md`: question, hypotheses, variables, stopping rules.
- Define the minimum study: one model, one engine, one hardware configuration, a context/concurrency grid, and a cache-strategy comparison if supported.
- Create `benchmarks/`, `configs/`, `workloads/`, `results/`, `analysis/`, `tests/`.
- Done when you can explain exactly what the study will and will not prove.

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

Not yet selected. The first experiment is planned for a single NVIDIA GPU (a free-tier T4 with 16 GB VRAM is a candidate). Free shared GPUs are noisier than dedicated hardware, so the exact GPU, driver, and date are recorded with every run. Results are only valid for the hardware manifest they ship with.

## Documents

- [Project overview and completion plan](KV_Cache_Project_Overview_and_Completion_Plan.pdf)
- [Complete step-by-step roadmap](KV_Cache_Complete_Step_by_Step_GenAI_Roadmap.pdf)

## Reproducibility and hygiene

- Dependencies and model/engine revisions will be pinned.
- Secrets and model weights are kept out of Git.
- Small reference results are committed; large artifacts are documented with retrieval instructions.
