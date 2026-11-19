# Step 9 working-set sweep, part A: baseline (no prefix cache) and GPU prefix caching only.
# Paste this whole file into ONE Kaggle cell (GPU T4, internet on) and send the full output plus the zip.
#
# Question: how many long conversations can each strategy keep warm? S sessions, each with its own distinct 4,096-token
# context, are filled once and then revisited once, in the same order, with a short distinct follow-up question; one request
# at a time. The GPU cache pool holds 309,248 tokens, about 74 such sessions. Working-set sizes S = 48 / 96 / 144 / 192
# (about 64 / 129 / 193 / 257 percent of the pool). Revisiting in the same order is the worst case for an LRU cache, so
# once S exceeds the pool the prefix cache is expected to lose its hits: that is an expectation, the run decides.
# Baseline (no prefix cache) runs only S = 48: it has nothing to keep, so it is the no-reuse reference.
# Expect roughly 70-80 minutes (a revisit that misses costs a full 3.2 s prefill; counters need 0.5 s per request).
# A failure stops that strategy's run and the next strategy still runs: send everything anyway.
import subprocess, glob, json, os, shutil, time

WORK = "/kaggle/working"
REPO = f"{WORK}/kv-cache-benchmarks"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
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
print("== GPU =="); print(sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")[1])

LOCKED_VLLM = [l.split("==")[1].split("#")[0].strip() for l in open(f"{REPO}/configs/requirements.lock.txt")
               if l.startswith("vllm==")][0]
if sh("python -m benchmarks.envcheck")[0] != 0:
    print(f"installing vllm=={LOCKED_VLLM} ..."); print(sh(f"pip install -q vllm=={LOCKED_VLLM}", timeout=1800)[1][-300:])
rc, out = sh("python -m benchmarks.envcheck")
print("== environment vs lock =="); print(out)
if rc != 0:
    raise SystemExit("vllm does not match the lock; stopping before any run")
print("== tests =="); print(sh("python -m pytest tests -q")[1][-200:])

print("== container memory and /dev/shm before any server ==")
print(sh("cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current 2>/dev/null; "
         "grep -E '^(anon|file|shmem) ' /sys/fs/cgroup/memory.stat 2>/dev/null; df -h /dev/shm; nproc")[1])

# 481 contexts of 4,096 tokens: 48+96+144+192 sessions (one block per working-set size) plus 1 spare for the warmup.
# The same seed gives the same texts in every strategy's cell, so the strategies see identical sessions.
shutil.rmtree(f"{WORK}/workload_ws", ignore_errors=True)
print(sh(f"python -m workloads.generate --model-id Qwen/Qwen2.5-1.5B-Instruct --revision {REV} "
         f"--levels 4096 --per-level 481 --seed 1234 --output-tokens 16 --max-context 32768 "
         f"--out-dir {WORK}/workload_ws", timeout=3600)[1][-400:])
wl = max(glob.glob(f"{WORK}/workload_ws/*"), key=os.path.getmtime)

out_dir = f"{WORK}/results_step9_ws_a"
shutil.rmtree(out_dir, ignore_errors=True)
RUNS = [("baseline", "48"), ("prefix_cache", "48 96 144 192")]
for strategy, sessions in RUNS:
    print(f"\n===== {strategy}: sessions {sessions} =====  [{time.time()-T0:.0f}s elapsed]")
    rc, out = sh(f"python -m benchmarks.workingset --workload-dir {wl} --strategy {strategy} "
                 f"--sessions {sessions} --rounds 1 --revisit-order same --timeout 900 --settle 0.5 "
                 f"--sample-interval 0.5 --baseline-seconds 3 --out-dir {out_dir}/{strategy}", timeout=21600)
    print("exit code:", rc, f"[{time.time()-T0:.0f}s elapsed]"); print(out[-3500:])

print("\n===== validation + tables =====")
for strategy, _ in RUNS:
    for d in sorted(glob.glob(f"{out_dir}/{strategy}/2*")):
        print(sh(f"python -m benchmarks.workingset --table {d}")[1])
        man = json.load(open(f"{d}/run_manifest.json"))
        print("counts:", man.get("counts"), "| package_versions:", man.get("package_versions"),
              "| lock_mismatches:", man.get("lock_mismatches"))
        print("kv pool:", (man.get("engine_reported_startup") or {}).get("kv_cache_tokens"), "tokens |",
              "counter deltas:", (man.get("memory_monitoring") or {}).get("engine_counter_deltas_over_run"))
        print("sampler health:", {k: (v.get("n_samples"), v.get("read_errors")) for k, v in
                                  (man.get("memory_monitoring") or {}).items() if isinstance(v, dict) and "n_samples" in v})
print("\n-- /dev/shm and container memory now --")
print(sh("df -h /dev/shm; grep -E '^(anon|file|shmem) ' /sys/fs/cgroup/memory.stat 2>/dev/null; cat /sys/fs/cgroup/memory.current")[1])
print("\n-- server logs: errors / OOM / preemption (first 6 per run) --")
for log in sorted(glob.glob(f"{out_dir}/*/2*/server.log")):
    print(f"[{log.split('/')[-3]}]")
    print(sh(f"grep -iE 'out of memory|OOM|Traceback|ValueError|RuntimeError|preempt|offload.*(fail|error|drop)' {log} "
             f"| grep -v 'FA2\\|fa_utils\\|FA version' | cut -c1-240 | head -6")[1])
shutil.make_archive(f"{WORK}/step9_ws_a", "zip", out_dir)
print(f"\ntotal elapsed {time.time()-T0:.0f}s")
print("zip for download:", f"{WORK}/step9_ws_a.zip")
