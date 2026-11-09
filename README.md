# KV-cache Investigation and Benchmarking

A performance engineering project measuring how context length, concurrent requests, and KV-cache placement affect inference memory usage, time to first token (TTFT), and throughput.

## Project question

Under a fixed model, inference engine, and hardware configuration, when does KV-cache pressure become a capacity or latency bottleneck, and what tradeoffs do supported memory/offload strategies introduce?

The intended output is a reproducible benchmark suite and an evidence-backed performance report. FabricOps remains the separate infrastructure investigation copilot project.

## Current status

**Scope defined; benchmark implementation has not started.** The existing Python workflow, synthetic incidents, tests, and Next.js dashboard belong to an earlier diagnostics demo. They do not run inference, measure KV-cache consumption, or demonstrate offload performance. Existing synthetic incident values and simulated token counts are not benchmark results.

Earlier documentation is preserved in [docs/LEGACY_MISSION_CONTROL.md](docs/LEGACY_MISSION_CONTROL.md). Existing setup and hiring-pitch documents describe that legacy demo.

## Implementation phases

1. **Define the experiment:** inventory available hardware; choose a model and inference engine; specify supported context lengths, concurrency levels, memory strategies, and resource limits.
2. **Build a real baseline:** run streamed inference with a reproducible workload; capture actual input/output token counts, client-observed TTFT, throughput, errors, and device memory measurements.
3. **Sweep context and concurrency:** vary input length and simultaneous requests while holding model, output length, and engine settings fixed. Include warmups, repeated runs, and explicit failure records.
4. **Compare memory strategies:** benchmark strategies supported by the selected engine and hardware. Distinguish KV-cache offload from model-weight offload, prefix caching, and KV-cache quantization. Mark unsupported strategies unavailable.
5. **Analyze tradeoffs:** plot memory, latency, and throughput; identify capacity boundaries and regressions; report variability and measurement limitations.
6. **Package the evidence:** publish configuration files, raw run records, reproducible commands, charts, and a report explaining which strategy helps under which conditions.

## Measurement rules

- Record hardware, model revision, engine/version, precision, cache settings, workload seed, and run configuration.
- Use actual tokenized lengths; input plus generated tokens must fit the supported context.
- Measure TTFT from client request submission to the first generated token, documenting transport and streaming behavior.
- Report per-request output tokens/second and aggregate output tokens/second separately, with explicit timing boundaries.
- Distinguish total GPU memory from KV-cache memory. Mark cache-specific values estimated unless the engine exposes reliable measurements.
- Record CPU memory and transfer metrics when available for offload experiments.
- Separate warmup runs, repeat measurements, and preserve timeouts and out-of-memory failures.
- Label synthetic workloads and analytical estimates; never present them as measured hardware performance.

The first milestone is one reproducible baseline inference run with real measurements. Hardware and engine selection must precede performance claims.


[project on process]
