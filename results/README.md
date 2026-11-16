# Results

Each run lives in `results/<run_id>/` and holds:

- `run_manifest.json`: resolved configuration, config hash, git commit, environment file hash, counts.
- `requests.jsonl`: one record per request (warmups and failures included). Only records with `usable_for_analysis: true` count.
- `server.log`: the vLLM server log for that run.

`results/` is stored byte-exact (`.gitattributes`: `-text`), so file hashes survive Windows checkouts. Never edit these files by hand.

## Runs

| Run | Status | What it is |
|-----|--------|-----------|
| `20261116T210839Z-d6a00143` | Pipeline verification only | First real run: 1 warmup + 1 measured request (1024 input tokens, 128 output tokens). It proved the runner works end to end and that the engine's prompt token count matches the workload manifest. **Do not use it as a baseline or a finding.** It is one sample, the server applied the model's `generation_config.json` defaults (repetition_penalty 1.1, top_k 20, top_p 0.8; see the warning in `server.log`), and the warmup did not match the measured shape. The launcher now passes `--generation-config vllm`. |
