# Progress Tracker

A step is **Accepted** only when its completion gate is met and the evidence is committed. Status values: Pending / In progress / Accepted.

| # | Step | Status | Evidence | Completion gate |
|---|------|--------|----------|-----------------|
| 1 | Freeze the experiment and baseline scope | In progress | `docs/experiment_plan.md` | Can explain what the study will and will not prove |
| 2 | Choose hardware, model, and engine | Accepted 2026-11-16: T4 fp16, pinned revision, vLLM 0.33.0 generation passed on one GPU, versions and key-package lock committed | `configs/environment.json`, dependency lock | One real generation works in a repeatable environment |
| 3 | Understand and estimate KV-cache growth | Accepted 2026-11-16: real config passes the guards, 28 KiB/token estimate matches vLLM's reported 324,928-token / 8.68 GiB capacity | `benchmarks/cache_estimator.py`, tests | Can calculate an estimate without presenting it as a measurement |
| 4 | Create reproducible token workloads | In progress (generator verified with the real tokenizer: 1K/4K/8K targets hit exactly; manifest for the final sweep to be generated and committed once the level list is fixed) | `workloads/generate.py`, manifest | Every request has an auditable token budget and identity |
| 5 | Build the first streamed benchmark run | Accepted 2026-11-16: runner and vLLM adapter ran on Kaggle; run 20261116T210839Z-d6a00143 (committed under results/) gave one usable record, engine prompt tokens matched the manifest, timestamps ordered. It is a pipeline check, not a baseline (see results/README.md) | `benchmarks/runner.py`, one real record | One command produces a measured request and its raw record |
| 6 | Validate memory and timing measurements | Accepted 2026-11-16: contract, samplers, metrics, and run validation verified on real hardware (NVML and vLLM /metrics, run 20261116T223433Z-53df2b80 committed); engine cache occupancy reproduces the size formula. Container-memory and prefix-counter fields were added after that run and get real-hardware validation in the next run | Measurement contract, sampler, tests | Metrics recomputable or limits explicit |
| 7 | Measure the context-length sweep | In progress: runner now supports per-level warmup, rotated order, forced output length, per-level and determinism summaries (130 tests); pilot run 20261117T032543Z-acb4745b committed (eager off, 1K/4K/8K, n=6 per level, vLLM 0.34.0 not the pinned 0.33.0); engine pinned to 0.34.0 with an envcheck guard, eager on/off pair and an eager-off repeat committed (20261117T040952Z-6f8b82fb, 20261117T041433Z-acb4745b); full 1K-32K sweep run and committed (20261117T053513Z-615f165f: 5 levels, n=6 each, 0 failures, 175 tests); acceptance (Step 1 answers, written findings) still open | Raw trial records | Each cell has repeated results or an explicit failure/skip |
| 8 | Add true concurrent-request experiments | In progress: wave runner (`benchmarks/concurrent.py`), wave/cell statistics (`benchmarks/waves.py`), `kv-bench-request/3` records, 12 tests (187 total); first grid run and committed (20261117T062402Z-a80c526b: 1K/4K/8K x N 1-16, 3 waves per cell, 0 failures, 198 tests); 16K/32K concurrency: cells `docs/kaggle_step8_16k.py` and `docs/kaggle_step8_32k.py` written, not yet run | Concurrent runner, matrix | Concurrency measured without confusing overlap with batching |
| 9 | Compare real cache and offload strategies | Pending | Support matrix, paired comparisons | Each claimed strategy actually active, workloads matched |
| 10 | Analyze results and create evidence-backed charts | Pending | Analysis script, charts, findings table | Every number traces to raw records or a labeled estimate |
| 11 | Prove reproducibility and package the benchmark | Pending | Pinned deps, data dictionary, tests | Clean environment reproduces charts from stored results |
| 12 | Complete the report and final release | Pending | Report, README quickstart, tagged release | Reviewer can reproduce method and audit key results |

## Milestone gates

- **A Foundation:** Steps 1-4
- **B First valid benchmark:** Steps 5-6
- **C Experimental evidence:** Steps 7-9
- **D Finished study:** Steps 10-12
