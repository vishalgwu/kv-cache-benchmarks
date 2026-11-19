# Step 9 capability probe: which KV-cache strategies actually work on this T4 with vLLM 0.34.0?
# Paste this whole file into ONE Kaggle cell (GPU T4, internet on) and send the full output plus the zip.
#
# It launches vLLM four times (each launch ~2-5 minutes): baseline (prefix caching off), prefix caching on,
# fp8 KV cache, and native CPU KV offloading (8 GiB). For each it sends the SAME 8K prompt twice and records:
# whether the server started (and why not, if not), the engine-reported KV pool size, prefix-cache hit counters,
# TTFT of both requests, whether the outputs match, log lines that mention offloading/fp8, and container
# anonymous memory vs page cache. A strategy is called ACTIVE only when the evidence shows it, never because the
# flag was accepted. This is a probe, not a benchmark: two requests per configuration prove nothing about speed.
# A configuration that fails to start is a RESULT; the probe carries on with the next one.
# Expect roughly 15-25 minutes.
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
    raise SystemExit("vllm does not match the lock; stopping before any probe")
print("== tests =="); print(sh("python -m pytest tests -q")[1][-200:])

# memory facts BEFORE any server: what the container offers for CPU offloading
print("== container memory before any server ==")
print(sh("cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current 2>/dev/null; "
         "grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat 2>/dev/null; "
         "grep -E 'MemTotal|MemAvailable' /proc/meminfo; nproc")[1])

shutil.rmtree(f"{WORK}/workload_probe", ignore_errors=True)
print(sh(f"python -m workloads.generate --model-id Qwen/Qwen2.5-1.5B-Instruct --revision {REV} "
         f"--levels 8192 --per-level 2 --seed 1234 --output-tokens 128 --max-context 32768 "
         f"--out-dir {WORK}/workload_probe")[1][-300:])
wl = max(glob.glob(f"{WORK}/workload_probe/*"), key=os.path.getmtime)

out_dir = f"{WORK}/results_step9_probe"
shutil.rmtree(out_dir, ignore_errors=True)
print(f"\n===== probing 4 configurations =====  [{time.time()-T0:.0f}s elapsed]")
rc, out = sh(f"python -m benchmarks.probe_strategies --workload-dir {wl} --out-dir {out_dir} --max-model-len 32768",
             timeout=7200)
print("probe exit code:", rc, f"[{time.time()-T0:.0f}s elapsed]"); print(out[-6000:])

print("\n===== per configuration =====")
for name in ("baseline", "prefix_cache", "fp8_kv", "kv_offload_native"):
    f = f"{out_dir}/probe_{name}.json"
    if not os.path.exists(f):
        print(name, ": no result file"); continue
    d = json.load(open(f))
    print(f"\n--- {name}: started={d['started']} startup_s={d['startup_s'] and round(d['startup_s'])}")
    print("argv:", " ".join(d.get("server_argv") or []))
    if d["failure"]:
        print("FAILURE:", d["failure"][-1500:])
    for k in ("request_1", "request_2"):
        r = d.get(k)
        if r:
            print(k, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in r.items() if x != "output_text"})
    for k in ("same_output", "second_request_hit_prefix_cache", "ttft_ratio_second_over_first"):
        print(k, d.get(k))
    print("kv pool:", (d.get("engine_reported_startup") or {}).get("kv_cache_tokens"),
          "tokens;", (d.get("engine_reported_startup") or {}).get("kv_cache_final_gib"), "GiB")
    print("memory before:", d.get("memory_before")); print("memory after: ", d.get("memory_after"))
    print("log evidence:"); [print("   ", l) for l in (d.get("log_evidence") or [])[:20]]

print("\n-- server logs: errors / tracebacks (first 6 per config) --")
for name in ("baseline", "prefix_cache", "fp8_kv", "kv_offload_native"):
    print(f"[{name}]")
    print(sh(f"grep -E 'Error|error:|Traceback|not supported|Unsupported|unsupported|NotImplemented' "
             f"{out_dir}/probe_{name}.server.log | grep -v 'FA2\\|fa_utils\\|FA version' | cut -c1-240 | head -6")[1])

shutil.make_archive(f"{WORK}/step9_probe", "zip", out_dir)
print(f"\ntotal elapsed {time.time()-T0:.0f}s")
print("zip for download:", f"{WORK}/step9_probe.zip")
