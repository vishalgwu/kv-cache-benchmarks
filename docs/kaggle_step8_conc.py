# Step 8 concurrency grid: input 1K / 4K / 8K tokens x concurrency 1 / 2 / 4 / 8 / 16, CUDA graphs ON.
# Paste this whole file into ONE Kaggle cell (GPU T4, internet on) and send the full output plus the zip.
#
# Same server settings as the Step 7 sweep (max_model_len 32768, prefix caching off, greedy, 128 output
# tokens forced), so the KV-cache pool is identical (309,248 tokens). Load policy: a wave of N requests
# with N DISTINCT prompts of one length is submitted together; the next wave starts when all have finished.
# Every cell gets one unmeasured warmup wave and then 3 measured waves; cell order is rotated per wave.
# The run stops after the first wave that contains a failure (so a capacity boundary is not run past).
# Expect roughly 40-50 minutes: prefill is serialised by the GPU, so an 8K wave of 16 takes about 3+ minutes.
import subprocess, glob, json, os, shutil, time

WORK = "/kaggle/working"
REPO = f"{WORK}/kv-cache-benchmarks"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LEVELS = "1024 4096 8192"
CONC = "1 2 4 8 16"
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

LOCKED_VLLM = [l.split("==")[1].split("#")[0].strip() for l in open(f"{REPO}/configs/requirements.lock.txt")
               if l.startswith("vllm==")][0]
if sh("python -m benchmarks.envcheck")[0] != 0:
    print(f"installing vllm=={LOCKED_VLLM} ..."); print(sh(f"pip install -q vllm=={LOCKED_VLLM}", timeout=1800)[1][-300:])
rc, out = sh("python -m benchmarks.envcheck")
print("== environment vs lock =="); print(out)
if rc != 0:
    raise SystemExit("vllm does not match the lock; stopping before any benchmark")

print("== tests =="); print(sh("python -m pytest tests -q")[1][-200:])

shutil.rmtree(f"{WORK}/workload_conc", ignore_errors=True)
print("== workload (32 prompts per level: 16 measured + 16 spare for the warmup waves) ==")
print(sh(f"python -m workloads.generate --model-id Qwen/Qwen2.5-1.5B-Instruct --revision {REV} "
         f"--levels {LEVELS} --per-level 32 --seed 1234 --output-tokens 128 --max-context 32768 "
         f"--out-dir {WORK}/workload_conc")[1][-500:])
wl = max(glob.glob(f"{WORK}/workload_conc/*"), key=os.path.getmtime)

out_dir = f"{WORK}/results_step8_conc"
shutil.rmtree(out_dir, ignore_errors=True)
print(f"\n===== concurrency grid: launches vllm serve (max_model_len 32768, CUDA graphs ON) =====  [{time.time()-T0:.0f}s elapsed]")
rc, out = sh(f"python -m benchmarks.runner --workload-dir {wl} --launch-server --no-enforce-eager "
             f"--max-model-len 32768 --levels {LEVELS} --concurrency-levels {CONC} --waves 3 --warmup 1 "
             f"--order rotate --ignore-eos --timeout 900 --sample-interval 0.5 --baseline-seconds 3 "
             f"--out-dir {out_dir}")
print("runner exit code:", rc, f"[{time.time()-T0:.0f}s elapsed]"); print(out[-3500:])

cands = glob.glob(f"{out_dir}/2*")
if not cands:
    raise SystemExit("no run directory was written; see the runner output above")
run = max(cands, key=os.path.getmtime)

print("\n===== validation + per-cell table =====")
print(sh(f"python -m benchmarks.metrics --table {run}")[1])
man = json.load(open(f"{run}/run_manifest.json"))
print("counts:", man.get("counts"))
print("package_versions:", man.get("package_versions"), "| lock_mismatches:", man.get("lock_mismatches"))
print("-- engine_reported_startup --"); print(json.dumps(man.get("engine_reported_startup"), indent=1))
print("-- engine counter deltas over run --")
print(json.dumps((man.get("memory_monitoring") or {}).get("engine_counter_deltas_over_run"), indent=1))
print("-- sampler health --")
print({k: (v.get("n_samples"), v.get("max_gap_s"), v.get("read_errors"))
       for k, v in (man.get("memory_monitoring") or {}).items() if isinstance(v, dict) and "n_samples" in v})

print("\n-- per-wave: wave_id, status counts, span_s, agg tok/s, ttft first/last, in-flight, co-decode, engine run/wait peak, kv peak, gpu MiB --")
code = r'''
import json, sys, collections
sys.path.insert(0, ".")
from benchmarks import waves
recs = [json.loads(l) for l in open(sys.argv[1] + "/requests.jsonl")]
groups = collections.OrderedDict()
for r in recs:
    if r.get("wave_id") and r["wave_id"] != "warmup":
        groups.setdefault(r["wave_id"], []).append(r)
for wid, rs in groups.items():
    ok = [r for r in rs if r["status"] == "ok"]
    tag = "W" if rs[0]["is_warmup"] else " "
    if len(ok) == len(rs):
        s = waves.wave_stats(rs)
        print(tag, wid, f"ok {len(ok)}/{len(rs)}", f"span {s['span_s']:.1f}", f"agg {s['agg_output_tok_s']:.1f}",
              f"ttft {s['ttft_first_s']:.2f}/{s['ttft_last_s']:.2f}", f"inflt {s['max_in_flight']}",
              f"codec {s['co_decode_s']:.2f}", f"run/wait {s['engine_running_peak']}/{s['engine_waiting_peak']}",
              f"kv {s['kv_cache_usage_peak']}", f"gpu {s['gpu_peak_used_mib']}", f"skew {s['start_skew_s']:.3f}")
    else:
        print(tag, wid, f"ok {len(ok)}/{len(rs)}", [(r["slot"], r["error_category"], str(r["error_message"])[:100]) for r in rs if r["status"] != "ok"])
'''
open(f"{WORK}/wave_print.py", "w").write(code)
print(sh(f"python {WORK}/wave_print.py {run}")[1])

print("\n-- server.log: errors / OOM / preemption warnings --")
print(sh(f"grep -iE 'out of memory|OOM|error|exceed|too long|preempt' {run}/server.log | grep -v 'FA2\\|fa_utils' | cut -c1-220 | head -12")[1])
shutil.make_archive(f"{WORK}/step8_conc", "zip", run)
print(f"\ntotal elapsed {time.time()-T0:.0f}s")
print("zip for download:", f"{WORK}/step8_conc.zip")
