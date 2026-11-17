# Step 7 context-length sweep: input 1K / 4K / 8K / 16K / 32K tokens, enforce_eager OFF (CUDA graphs).
# Paste this whole file into ONE Kaggle cell (GPU T4, internet on) and send the full output plus the zip.
#
# One server launch (max_model_len 32768, prefix caching off, greedy, 128 output tokens forced), a warmup
# at every level, 3 trials with the level order rotated each trial, 2 prompts per level (n = 6 per level).
# The first trial runs the levels in ascending order, so a failure at 16K or 32K stops the run (as the plan
# requires) after the earlier levels already have one trial each.
# The top level is 32,000 input tokens: 32,768 would leave no room for 128 output tokens in a 32,768 window.
# Expect roughly 30-40 minutes (32K prefill is estimated at about 3 minutes per request: an extrapolation).
import subprocess, glob, json, os, shutil, time

WORK = "/kaggle/working"
REPO = f"{WORK}/kv-cache-benchmarks"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LEVELS = "1024 4096 8192 16384 32000"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
T0 = time.time()


def sh(cmd, cwd=REPO, timeout=7200):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


os.chdir(WORK)
if os.path.isdir(REPO):
    print(sh("git pull --ff-only")[1][-300:])
else:
    print(sh(f"git clone https://github.com/vishalgwu/kv-cache-benchmarks.git {REPO}", cwd=WORK)[1][-300:])
print("commit:", sh("git log --oneline -1")[1])

# exactly the locked vLLM, never "latest"; stop if the engine does not match
LOCKED_VLLM = [l.split("==")[1].split("#")[0].strip() for l in open(f"{REPO}/configs/requirements.lock.txt")
               if l.startswith("vllm==")][0]
if sh("python -m benchmarks.envcheck")[0] != 0:
    print(f"installing vllm=={LOCKED_VLLM} ..."); print(sh(f"pip install -q vllm=={LOCKED_VLLM}", timeout=1800)[1][-300:])
rc, out = sh("python -m benchmarks.envcheck")
print("== environment vs lock =="); print(out)
if rc != 0:
    raise SystemExit("vllm does not match the lock; stopping before any benchmark")

print("== tests =="); print(sh("python -m pytest tests -q")[1][-200:])

shutil.rmtree(f"{WORK}/workload_sweep", ignore_errors=True)
print("== workload (3 prompts per level: 2 measured + 1 spare for the per-level warmup) ==")
print(sh(f"python -m workloads.generate --model-id Qwen/Qwen2.5-1.5B-Instruct --revision {REV} "
         f"--levels {LEVELS} --per-level 3 --seed 1234 --output-tokens 128 --max-context 32768 "
         f"--out-dir {WORK}/workload_sweep")[1][-700:])
wl = max(glob.glob(f"{WORK}/workload_sweep/*"), key=os.path.getmtime)

out_dir = f"{WORK}/results_step7_sweep"
shutil.rmtree(out_dir, ignore_errors=True)
print(f"\n===== sweep: launches vllm serve (max_model_len 32768, CUDA graphs ON) =====  [{time.time()-T0:.0f}s elapsed]")
rc, out = sh(f"python -m benchmarks.runner --workload-dir {wl} --launch-server --no-enforce-eager "
             f"--max-model-len 32768 --levels {LEVELS} --limit-per-level 2 --warmup 1 "
             f"--warmup-per-level --order rotate --trials 3 --ignore-eos --timeout 900 "
             f"--sample-interval 0.5 --baseline-seconds 3 --out-dir {out_dir}")
print("runner exit code:", rc, f"[{time.time()-T0:.0f}s elapsed]"); print(out[-4500:])

cands = glob.glob(f"{out_dir}/2*")
if not cands:
    raise SystemExit("no run directory was written; see the runner output above")
run = max(cands, key=os.path.getmtime)

print("\n===== validation + per-level table =====")
print(sh(f"python -m benchmarks.metrics --table {run}")[1])
print("===== run-level checks =====")
print(sh(f"python -m benchmarks.metrics {run} | python -c \"import json,sys; t=sys.stdin.read(); "
         f"i=t.index('{{'); d=json.loads(t[i:]); print('counts', d['counts']); print('failures', d['failures_by_category']); "
         f"print('determinism', d['determinism'])\"")[1])
man = json.load(open(f"{run}/run_manifest.json"))
print("package_versions:", man.get("package_versions"), "| lock_mismatches:", man.get("lock_mismatches"))
print("-- engine_reported_startup --"); print(json.dumps(man.get("engine_reported_startup"), indent=1))
print("-- engine counter deltas over run --")
print(json.dumps((man.get("memory_monitoring") or {}).get("engine_counter_deltas_over_run"), indent=1))
print("-- sampler health --")
print({k: (v.get("n_samples"), v.get("max_gap_s"), v.get("read_errors"))
       for k, v in (man.get("memory_monitoring") or {}).items() if isinstance(v, dict) and "n_samples" in v})

print("\n-- per-request: W=warmup trial id status out_tokens ttft_s decode_tok/s | gpu +growth_MiB sm_clock_min temp_max throttle | kv_peak --")
for line in open(f"{run}/requests.jsonl"):
    r = json.loads(line); g = (r.get("memory") or {}).get("gpu", {}); e = (r.get("memory") or {}).get("engine", {})
    f = lambda v, p=3: None if v is None else round(v, p)
    print("W" if r["is_warmup"] else " ", r["trial_index"], r["prompt_id"], r["status"], r["output_tokens"],
          f(r["ttft_s"]), f(r["decode_tokens_per_s"], 1), "| gpu +", g.get("peak_minus_window_start_mib"),
          g.get("sm_clock_mhz_min_in_window"), g.get("temperature_c_max_in_window"),
          g.get("throttle_reasons_seen_in_window"), "| kv", f(e.get("kv_cache_usage_peak"), 4),
          "" if r["status"] == "ok" else f"| ERROR {r['error_category']}: {str(r['error_message'])[:120]}")

print("\n-- server.log: errors / OOM / warnings that mention memory or length --")
print(sh(f"grep -iE 'out of memory|OOM|error|exceed|too long|preempt' {run}/server.log | grep -v 'FA2\\|fa_utils' | cut -c1-220 | head -12")[1])
shutil.make_archive(f"{WORK}/step7_sweep", "zip", run)
print(f"\ntotal elapsed {time.time()-T0:.0f}s")
print("zip for download:", f"{WORK}/step7_sweep.zip")
