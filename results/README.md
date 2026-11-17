# Results

Each run lives in `results/<run_id>/` and holds:

- `run_manifest.json`: resolved configuration, config hash, git commit, environment file hash, counts.
- `requests.jsonl`: one record per request (warmups and failures included). Only records with `usable_for_analysis: true` count.
- `server.log`: the vLLM server log for that run.

`results/` is stored byte-exact (`.gitattributes`: `-text`), so file hashes survive Windows checkouts. Never edit these files by hand.

## Runs

| Run | Status | What it is |
|-----|--------|-----------|
| `20261117T041433Z-acb4745b` | Pilot only | Exact repeat of the eager-off pilot below (same config hash) on the pinned stack (manifest records vllm 0.34.0, no lock mismatch). Medians agree with the first run within 1 percent. |
| `20261117T040952Z-6f8b82fb` | Pilot only | The same pilot with `enforce_eager` ON on the pinned stack (vLLM 0.34.0). Decode about 26 tok/s at every level with the GPU about 42-58 percent busy in decode. Pairs with the eager-off runs; n = 6 per level, no findings. |
| `20261117T032543Z-acb4745b` | Pilot only | Step 7 pilot, `enforce_eager` OFF (CUDA graphs): 4 warmups (short + one full-length per level) and 18 measured requests (1K/4K/8K input, 2 prompts per level, 3 trials, level order rotated), 128 output tokens forced. Validates; all repeats bit-identical. **Ran on vLLM 0.34.0, not the pinned 0.33.0** (see `configs/environment.json`). n = 6 per level: no tail percentiles, no findings. |
| `20261116T223433Z-53df2b80` | Sampler validation only | Second real run: 1 warmup + 4 measured requests (2 each at 1024 and 4096 input tokens, 128 output target), with the GPU, engine `/metrics`, and CPU samplers running at 0.25 s. It validated the measurement pipeline on real hardware: NVML and `/metrics` worked, the run validates, and the engine's cache occupancy reproduces the size formula. **Not a baseline or a finding**: one trial, two prompts per level, `enforce_eager` on, a 37-token warmup that does not match the measured shapes, and sample files kept as evidence. Sample files: `samples_gpu.jsonl`, `samples_engine.jsonl`, `samples_cpu.jsonl`; engine metric names: `engine_metrics_catalog.tsv`. |
| `20261116T210839Z-d6a00143` | Pipeline verification only | First real run: 1 warmup + 1 measured request (1024 input tokens, 128 output tokens). It proved the runner works end to end and that the engine's prompt token count matches the workload manifest. **Do not use it as a baseline or a finding.** It is one sample, the server applied the model's `generation_config.json` defaults (repetition_penalty 1.1, top_k 20, top_p 0.8; see the warning in `server.log`), and the warmup did not match the measured shape. The launcher now passes `--generation-config vllm`. |
