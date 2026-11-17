# Step 8 extension, part B: input 16K tokens (16,384) x concurrency 1 / 2 / 4 / 8 / 16, CUDA graphs ON.
# Paste this whole file into ONE Kaggle cell (GPU T4, internet on) and send the full output plus the zip.
#
# Same server as the Step 7 sweep and the Step 8 grid (max_model_len 32768, prefix caching off, greedy, 128 output
# tokens forced; KV pool 309,248 tokens). 16K x 16 requests need about 85 percent of that pool.
# Waves: ONE unmeasured warmup wave at N=1, then 2 measured waves per cell (the 1K-8K grid showed wave-to-wave spread
# under 1 percent and warmup waves equal to measured ones, so per-cell warmups are skipped to save time).
# The per-request timeout is 3600 s because the last request of a 16-wave waits behind 15 serial prefills (~12 minutes).
# Expect roughly 50-60 minutes: one 16K prefill is ~45 s and prefill is serial, so a wave of 16 is ~12.5 minutes.
import subprocess, glob, json, os, shutil, time

WORK = "/kaggle/working"
REPO = f"{WORK}/kv-cache-benchmarks"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LEVELS = "16384"
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

shutil.rmtree(f"{WORK}/workload_conc16k", ignore_errors=True)
print("== workload (17 prompts: measured plus 1 spare for the warmup wave) ==")
print(sh(f"python -m workloads.generate --model-id Qwen/Qwen2.5-1.5B-Instruct --revision {REV} "
         f"--levels {LEVELS} --per-level 17 --seed 1234 --output-tokens 128 --max-context 32768 "
         f"--out-dir {WORK}/workload_conc16k")[1][-500:])
wl = max(glob.glob(f"{WORK}/workload_conc16k/*"), key=os.path.getmtime)

out_dir = f"{WORK}/results_step8_16k"
shutil.rmtree(out_dir, ignore_errors=True)
print(f"\n===== concurrency grid: launches vllm serve (max_model_len 32768, CUDA graphs ON) =====  [{time.time()-T0:.0f}s elapsed]")
rc, out = sh(f"python -m benchmarks.runner --workload-dir {wl} --launch-server --no-enforce-eager "
             f"--max-model-len 32768 --levels {LEVELS} --concurrency-levels {CONC} --waves 2 --warmup 1 --warmup-cells smallest "
             f"--order rotate --ignore-eos --timeout 3600 --sample-interval 0.5 --baseline-seconds 3 "
             f"--out-dir {out_dir}", timeout=21600)
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
open(f"{WORK}/wave_print_step8_16k.py", "w").write(code)
print(sh(f"python {WORK}/wave_print_step8_16k.py {run}")[1])

print("\n-- server.log: errors / OOM / preemption warnings --")
print(sh(f"grep -iE 'out of memory|OOM|error|exceed|too long|preempt' {run}/server.log | grep -v 'FA2\\|fa_utils' | cut -c1-220 | head -12")[1])
shutil.make_archive(f"{WORK}/step8_16k", "zip", run)
print(f"\ntotal elapsed {time.time()-T0:.0f}s")
print("zip for download:", f"{WORK}/step8_16k.zip")
