# Results

Each run lives in `results/<run_id>/` and holds:

- `run_manifest.json`: resolved configuration, config hash, git commit, environment file hash, counts.
- `requests.jsonl`: one record per request (warmups and failures included). Only records with `usable_for_analysis: true` count.
- `server.log`: the vLLM server log for that run.

`results/` is stored byte-exact (`.gitattributes`: `-text`), so file hashes survive Windows checkouts. Never edit these files by hand.

## Runs

| Run | Status | What it is |
|-----|--------|-----------|
| `20261116T223433Z-53df2b80` | Sampler validation only | Second real run: 1 warmup + 4 measured requests (2 each at 1024 and 4096 input tokens, 128 output target), with the GPU, engine `/metrics`, and CPU samplers running at 0.25 s. It validated the measurement pipeline on real hardware: NVML and `/metrics` worked, the run validates, and the engine's cache occupancy reproduces the size formula. **Not a baseline or a finding**: one trial, two prompts per level, `enforce_eager` on, a 37-token warmup that does not match the measured shapes, and sample files kept as evidence. Sample files: `samples_gpu.jsonl`, `samples_engine.jsonl`, `samples_cpu.jsonl`; engine metric names: `engine_metrics_catalog.tsv`. |
| `20261116T210839Z-d6a00143` | Pipeline verification only | First real run: 1 warmup + 1 measured request (1024 input tokens, 128 output tokens). It proved the runner works end to end and that the engine's prompt token count matches the workload manifest. **Do not use it as a baseline or a finding.** It is one sample, the server applied the model's `generation_config.json` defaults (repetition_penalty 1.1, top_k 20, top_p 0.8; see the warning in `server.log`), and the warmup did not match the measured shape. The launcher now passes `--generation-config vllm`. |
