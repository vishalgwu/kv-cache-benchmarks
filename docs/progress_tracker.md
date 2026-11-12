# Progress Tracker

A step is **Accepted** only when its completion gate is met and the evidence is committed. Status values: Pending / In progress / Accepted.

| # | Step | Status | Evidence | Completion gate |
|---|------|--------|----------|-----------------|
| 1 | Freeze the experiment and baseline scope | In progress | `docs/experiment_plan.md` | Can explain what the study will and will not prove |
| 2 | Choose hardware, model, and engine | In progress (model, dtype, and hardware chosen; engine, revision hash, and environment manifest pending) | `configs/environment.json`, dependency lock | One real generation works in a repeatable environment |
| 3 | Understand and estimate KV-cache growth | Pending | `benchmarks/cache_estimator.py`, tests | Can calculate an estimate without presenting it as a measurement |
| 4 | Create reproducible token workloads | Pending | `workloads/generate.py`, manifest | Every request has an auditable token budget and identity |
| 5 | Build the first streamed benchmark run | Pending | `benchmarks/runner.py`, one real record | One command produces a measured request and its raw record |
| 6 | Validate memory and timing measurements | Pending | Measurement contract, sampler, tests | Metrics recomputable or limits explicit |
| 7 | Measure the context-length sweep | Pending | Raw trial records | Each cell has repeated results or an explicit failure/skip |
| 8 | Add true concurrent-request experiments | Pending | Concurrent runner, matrix | Concurrency measured without confusing overlap with batching |
| 9 | Compare real cache and offload strategies | Pending | Support matrix, paired comparisons | Each claimed strategy actually active, workloads matched |
| 10 | Analyze results and create evidence-backed charts | Pending | Analysis script, charts, findings table | Every number traces to raw records or a labeled estimate |
| 11 | Prove reproducibility and package the benchmark | Pending | Pinned deps, data dictionary, tests | Clean environment reproduces charts from stored results |
| 12 | Complete the report and final release | Pending | Report, README quickstart, tagged release | Reviewer can reproduce method and audit key results |

## Milestone gates

- **A Foundation:** Steps 1-4
- **B First valid benchmark:** Steps 5-6
- **C Experimental evidence:** Steps 7-9
- **D Finished study:** Steps 10-12
