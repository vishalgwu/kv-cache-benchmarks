# Experiment Plan (Step 1: frozen scope)

Status: **DRAFT.** Fields marked `TBD` are decided in Step 2 (hardware, model, engine). Once the TBD fields are filled and this file is committed, the plan is frozen: later changes must be recorded in the "Deviations" section at the bottom, not silently edited.

## 1. Question

How do **context length**, **concurrent requests**, and **supported KV-cache memory strategies** change GPU/CPU memory use, time to first token (TTFT), and token throughput during LLM inference on one device?

## 2. What this study will and will not prove

**Can support (if measured and verified):**
- How TTFT, throughput, and peak memory change with input length and concurrency for one model/engine/GPU.
- Where this setup hits a capacity boundary (out-of-memory, timeout, rejected context).
- Whether a supported KV-cache strategy (for example, CPU offload) changes those results, under matched workloads.
- How close an analytical KV-cache size estimate is to engine-reported cache metrics.

**Cannot support:**
- Anything about other models, GPUs, engines, or precisions.
- Rack-scale, multi-GPU, or fabric behavior.
- That KV-cache pressure *caused* a slowdown, unless an experiment isolates it (longer prefill work alone also raises TTFT).
- Tail latency (p99) from a small number of trials.
- Production performance. Free shared GPUs are noisy.

The legacy diagnostics demo and its synthetic values are **excluded** from all evidence.

## 3. Hypotheses

Stated before measuring. They may be wrong; that is a valid result.

| ID | Hypothesis | Evidence that would refute it |
|----|-----------|-------------------------------|
| H1 | TTFT increases with input length (more prefill work). | TTFT flat or decreasing across context levels. |
| H2 | Aggregate throughput rises with concurrency until a limit, while per-request latency worsens. | Throughput does not rise, or latency does not worsen. |
| H3 | Peak GPU memory use rises with context length and concurrency as KV cache grows, but total reserved memory may stay flat if the engine preallocates a cache pool. | Memory moves in a way the cache estimate cannot explain, or reserved memory tracks demand exactly. |
| H4 | The analytical estimate (2 x layers x KV_heads x head_dim x bytes x retained tokens) scales linearly with retained tokens and matches engine cache metrics within a documented margin. | Large unexplained gap between estimate and engine-reported cache bytes. |
| H5 | If KV offload is supported, offload extends the context/concurrency that fits but increases TTFT or lowers throughput versus GPU-resident cache. | Offload shows no capacity gain or no cost. |

## 4. Variables

**Independent (changed on purpose):**

| Variable | Planned levels (candidates, adjust to hardware) |
|----------|-------------------------------------------------|
| Input length | 1K, 4K, 8K, 16K, 32K tokens (candidates; the model advertises a 32K window, so keep input + output within the limit and stop at the first capacity failure) |
| Concurrency | 1, 2, 4, 8, 16 (add 32 only if 16 still shows no pressure) |
| Cache strategy | Default GPU-resident baseline; CPU offload only if the engine supports it |

**Controlled (held fixed across compared cells):**
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, not gated). Revision: commit hash `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (stored in `configs/environment.json`). Never run from an unpinned `main`.
- Precision/dtype: fp16 (a T4 has no bf16)
- Inference engine and version: vLLM 0.33.0, run as an OpenAI-compatible server and measured with a streaming client (attention backend TRITON_ATTN on the T4). Chosen after it generated correctly on the Kaggle T4 in fp16.
- Hardware: Kaggle free-tier NVIDIA T4 (16 GB VRAM), shared and variable. Exact GPU, driver, CUDA, Python, library versions, and date are recorded per run in `configs/environment.json`. Driver 580.178.04, CUDA 13.0, Python 3.13.16, torch 2.14.0 (full list in `configs/environment.json` and `configs/requirements.lock.txt`).
- Execution: code is developed locally, pushed to GitHub, cloned into a Kaggle notebook, run there, and result files are downloaded and committed.
- Architecture facts (read from the pinned model's `config.json`): 28 layers, 12 query heads, 2 KV heads, head dim 128, all full attention, 32,768 max positions. Estimated KV cache: 28,672 bytes (28 KiB) per token in fp16.
- Output length target: 128 tokens (record actual length and stop reason)
- Sampling: greedy (temperature 0), with the server launched using `--generation-config vllm` so the model's `generation_config.json` defaults (repetition_penalty 1.1, top_k 20, top_p 0.8) do not silently apply
- Server launch: one launch with a fixed `max_model_len` for the whole sweep, so the KV-cache pool size is identical across cells (it varies with `max_model_len`)
- Prompt policy: seeded synthetic text, distinct prompts, no unintended shared-prefix reuse
- Prefix-caching policy: recorded explicitly per run
- Load policy: fixed-concurrency batches (request-rate sweeps are a later extension)

**Dependent (measured outcomes):**
- TTFT: first non-empty generated token arrival minus request submission
- Per-request decode throughput and aggregate output throughput
- Peak and baseline GPU memory (with scope and sampling interval); CPU memory for offload runs
- Engine cache occupancy, only if reliably exposed
- Success, timeout, out-of-memory, and rejection counts

## 5. Minimum study

One model, one engine, one hardware configuration. Start with a small pilot grid (1K/4K/8K x concurrency 1/2/4 = 9 cells, at least 3 trials each after warmup) to validate the runner, then extend toward 16K/32K and concurrency 8/16 to look for the capacity boundary.

Rationale: if the model uses grouped-query attention with few KV heads, its cache cost per token is small (an estimate to be confirmed in Step 3), so the small pilot grid may show no memory pressure. The extended grid is what can reveal it. If no boundary appears even at the extended grid, report that as a finding rather than adding cells until something breaks.

Strategy comparison repeats the same grid for each genuinely supported strategy.

## 6. Planned charts

Every chart has a named input variable and measured outcome.

1. TTFT vs. input length (concurrency 1).
2. Aggregate throughput vs. concurrency (fixed context).
3. Per-request latency vs. concurrency.
4. Peak GPU memory vs. input length and concurrency.
5. Estimated vs. engine-reported KV-cache size (if available).
6. Capacity boundary map: success/failure per (context, concurrency) cell.
7. Strategy comparison (baseline vs. offload) for TTFT and throughput, if supported.

## 7. Stopping rules

- Stop increasing context or concurrency after the first out-of-memory, timeout, or rejection until the next level is reviewed by hand.
- Retry cap: maximum 1 retry per failed request; retries are logged and never counted as clean samples.
- Total budget: Kaggle free-tier weekly GPU quota (current limit `TBD`, read from the Kaggle account page). No paid resources. Stop when the quota is reached and report what was completed.
- Kaggle sessions can end without warning: write every request record to disk as it completes and download results before the session closes.
- If more than 10% of requests in a cell fail for reasons other than a capacity boundary, pause and investigate before continuing.

## 8. Fallbacks

- **Offload unsupported:** finish the context and concurrency study, document the limitation, and make no offload claims.
- **No suitable GPU:** use a free-tier cloud GPU or rent one with a recorded spending limit and shutdown procedure.
- **Target context too large:** lower the top level and record the reduced grid.

## 9. Validity rules

- Every reported number traces to a raw record or is labeled as an analytical estimate.
- Warmups are excluded and identified; cold-start behavior is recorded separately.
- Run order is rotated or randomized to reduce drift.
- Runs with different models or precision are never compared.
- Failures are retained with reasons; unsupported cells are marked skipped.
- Percentiles are reported only with sample counts.

## 10. Acceptance checklist for Step 1

- [ ] Every proposed chart names an input variable and a measured outcome (Section 6).
- [ ] Synthetic values from the legacy demo are explicitly excluded from benchmark evidence (Sections 2, 9).
- [ ] Unsupported offload has a documented fallback (Section 8).
- [ ] I can explain exactly what the study will and will not prove (Section 2).
- [ ] I can explain why a working diagnostics dashboard does not prove KV-cache performance.

## Deviations (append only)

- **2026-11-16, clarification (sampling):** the first real run's server log showed vLLM applying the model's `generation_config.json` defaults on top of the request (repetition_penalty 1.1, top_k 20, top_p 0.8). The plan did not control this. The launcher now passes `--generation-config vllm`, and run `20261116T210839Z-d6a00143` is treated as a pipeline check, not data.
- **2026-11-16, clarification (cache pool):** the KV-cache pool size depends on the `max_model_len` the server starts with (8.68 GiB at 4096, 8.92 GiB at 1216, because startup profiling measured different peak activation). The sweep uses one fixed launch setting.
- **2026-11-16, clarification (warmup):** a short warmup prompt does not warm the shapes of a long prefill, so Step 7 adds a warmup at each context length.
