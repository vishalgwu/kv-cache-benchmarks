# Step 7 pilot: context-length sweep at 1K / 4K / 8K input tokens, run twice on one T4:
#   A) enforce_eager ON   B) enforce_eager OFF (CUDA graphs)
# Paste this whole file into ONE Kaggle cell (GPU T4 x1 or x2, internet on) and send the full output.
#
# Fixed for both runs: max_model_len 32768, prefix caching off, greedy, 128 output tokens forced
# (--ignore-eos), a warmup at every level, 3 trials with the level order rotated each trial,
# 2 prompts per level (so n = 6 requests per level per mode). This is a pilot: n is small, so
# no tail percentiles and no conclusions beyond "what to run next".
import subprocess, glob, json, os, shutil

WORK = "/kaggle/working"
REPO = f"{WORK}/kv-cache-benchmarks"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sh(cmd, cwd=REPO, timeout=3600):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


os.chdir(WORK)
if os.path.isdir(REPO):
    print(sh("git pull --ff-only")[1][-300:])
else:
    print(sh(f"git clone https://github.com/vishalgwu/kv-cache-benchmarks.git {REPO}", cwd=WORK)[1][-300:])
print("commit:", sh("git log --oneline -1")[1])
# Install exactly the vLLM version in the lock (never "latest"), then verify the whole lock.
LOCKED_VLLM = [l.split("==")[1].split("#")[0].strip() for l in open(f"{REPO}/configs/requirements.lock.txt")
               if l.startswith("vllm==")][0]
if sh(f"python -m benchmarks.envcheck")[0] != 0:
    print(f"installing vllm=={LOCKED_VLLM} ..."); print(sh(f"pip install -q vllm=={LOCKED_VLLM}", timeout=1800)[1][-300:])
rc, out = sh("python -m benchmarks.envcheck")
print("== environment vs lock =="); print(out)
if rc != 0:
    raise SystemExit("vllm does not match the lock; stopping before any benchmark")

print("== tests =="); print(sh("python -m pytest tests -q")[1][-200:])

LEVELS = "1024 4096 8192"
shutil.rmtree(f"{WORK}/workload_step7", ignore_errors=True)
print("== workload (3 prompts per level: 2 measured + 1 spare for the per-level warmup) ==")
print(sh(f"python -m workloads.generate --model-id Qwen/Qwen2.5-1.5B-Instruct --revision {REV} "
         f"--levels {LEVELS} --per-level 3 --seed 1234 --output-tokens 128 --max-context 32768 "
         f"--out-dir {WORK}/workload_step7")[1][-400:])
wl = max(glob.glob(f"{WORK}/workload_step7/*"), key=os.path.getmtime)

runs = {}
for label, eager in (("eager_on", "--enforce-eager"), ("eager_off", "--no-enforce-eager")):
    out_dir = f"{WORK}/results_step7_{label}"
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"\n===== run {label}: launches vllm serve (max_model_len 32768) =====")
    rc, out = sh(f"python -m benchmarks.runner --workload-dir {wl} --launch-server {eager} "
                 f"--max-model-len 32768 --levels {LEVELS} --limit-per-level 2 --warmup 1 "
                 f"--warmup-per-level --order rotate --trials 3 --ignore-eos --timeout 900 "
                 f"--sample-interval 0.25 --baseline-seconds 3 --out-dir {out_dir}")
    print("exit code:", rc); print(out[-3500:])
    cands = glob.glob(f"{out_dir}/2*")
    if cands:
        runs[label] = max(cands, key=os.path.getmtime)

for label, run in runs.items():
    print(f"\n===== {label}: validation + summary =====")
    print(sh(f"python -m benchmarks.metrics {run}")[1][:6000])
    man = json.load(open(f"{run}/run_manifest.json"))
    print("-- engine_reported_startup --"); print(json.dumps(man.get("engine_reported_startup"), indent=1))
    print("-- engine counter deltas over run --")
    print(json.dumps((man.get("memory_monitoring") or {}).get("engine_counter_deltas_over_run"), indent=1))
    print("-- cpu block of the last request (container limit / peak) --")
    last = [json.loads(l) for l in open(f"{run}/requests.jsonl")][-1]
    print(json.dumps((last.get("memory") or {}).get("cpu"), indent=1)[:1500])
    print("-- per-request (trial, prompt, ttft, decode tok/s, gpu growth in window, engine kv peak) --")
    for line in open(f"{run}/requests.jsonl"):
        r = json.loads(line); g = (r.get("memory") or {}).get("gpu", {}); e = (r.get("memory") or {}).get("engine", {})
        print("W" if r["is_warmup"] else " ", r["trial_index"], r["prompt_id"], r["status"],
              r["finish_reason"], r["output_tokens"],
              None if r["ttft_s"] is None else round(r["ttft_s"], 3),
              None if r["decode_tokens_per_s"] is None else round(r["decode_tokens_per_s"], 1),
              "| gpu peak", g.get("peak_used_mib_in_window"), "+", g.get("peak_minus_window_start_mib"),
              "| kv", e.get("kv_cache_usage_peak"))
    print("-- generation-config / attention / graph lines in server.log --")
    print(sh(f"grep -iE 'generation|attention backend|cuda graph|enforce' {run}/server.log | cut -c1-240 | head -12")[1])
    shutil.make_archive(f"{WORK}/step7_pilot_{label}", "zip", run)

print("\nzips for download:", [f"{WORK}/step7_pilot_{k}.zip" for k in runs])
