"""Step 9 capability probe: which KV-cache strategies actually work on this device and engine version?

For each candidate configuration the probe launches vLLM, sends the same long prompt twice, and records
evidence rather than assumptions:

* did the server start (and if not, the last log lines),
* the engine-reported KV pool size (does fp8 really enlarge it? does offload change it?),
* prefix-cache counters before and after each request (a repeat request should hit the cache only when
  prefix caching is really on),
* TTFT of the first and second request and whether their outputs are identical,
* log lines that mention offloading, fp8 or connectors (so a flag that was silently ignored shows up),
* container anonymous memory and page cache (cgroup ``memory.stat``) before and after.

A strategy counts as ACTIVE only when evidence shows it, never because the flag was accepted. The probe
makes no performance claim: two requests per configuration cannot support one.

Command line:  python -m benchmarks.probe_strategies --workload-dir <dir> --out-dir <dir> [--only name ...]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from typing import Callable, Optional

from benchmarks import runner, sampling
from benchmarks import vllm_engine as engine

PROBE_MAX_TOKENS = 16
EVIDENCE_PATTERN = re.compile(r"offload|kv_transfer|KVTransfer|connector|fp8|kv_cache_dtype|prefix caching|"
                              r"KV cache|kv cache", re.IGNORECASE)

# name -> (enable_prefix_caching, extra vllm serve arguments, what it tests)
CONFIGS = {
    "baseline": (False, (), "GPU-resident KV cache, prefix caching off (the configuration used in Steps 7 and 8)"),
    "prefix_cache": (True, (), "GPU-resident KV cache with automatic prefix caching on"),
    "fp8_kv": (False, ("--kv-cache-dtype", "fp8"), "KV cache stored as fp8 (half the bytes per token); T4 support unverified"),
    "kv_offload_native": (True, ("--kv-offloading-backend", "native", "--kv-offloading-size", "8"),
                          "native CPU KV offloading, 8 GiB buffer, with prefix caching; support in this version unverified"),
}


def evidence_lines(log_text: str, limit: int = 30) -> list:
    out = []
    for line in log_text.splitlines():
        if EVIDENCE_PATTERN.search(line) and "Capturing CUDA graphs" not in line:
            out.append(line.strip()[:260])
            if len(out) >= limit:
                break
    return out


def container_memory(pid: Optional[int]) -> dict:
    keys = ("server_tree_rss_mib", "container_mem_current_mib", "container_anon_mib", "container_file_mib",
            "container_mem_limit_mib")
    try:
        r = sampling.make_cpu_reader(pid)()
    except Exception as exc:  # no /proc on this machine: record the gap instead of failing the probe
        return {**{k: None for k in keys}, "error": f"{type(exc).__name__}: {exc}"}
    return {k: r.get(k) for k in keys}


def counters(host: str, port: int, metrics_fn: Callable = sampling.read_engine_metrics) -> dict:
    try:
        m = metrics_fn(host, port)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {k: m.get(k) for k in ("prefix_cache_queries_total", "prefix_cache_hits_total", "kv_cache_usage")}


def delta(before: dict, after: dict, key: str) -> Optional[float]:
    a, b = before.get(key), after.get(key)
    return None if a is None or b is None else b - a


def one_request(host: str, port: int, model: str, text: str, timeout_s: float, metrics_fn: Callable) -> dict:
    before = counters(host, port, metrics_fn)
    res = engine.stream_chat(host, port, engine.build_payload(model, text, PROBE_MAX_TOKENS, ignore_eos=True),
                             timeout_s=timeout_s)
    time.sleep(0.6)  # engine counters can lag a request (measurement contract limit 11)
    after = counters(host, port, metrics_fn)
    return {"status": res.status, "error": res.error_message,
            "ttft_s": (res.t_first_content - res.t_start) if res.t_first_content is not None else None,
            "prompt_tokens": res.prompt_tokens, "output_tokens": res.completion_tokens,
            "output_text": res.text,
            "prefix_cache_queries_delta": delta(before, after, "prefix_cache_queries_total"),
            "prefix_cache_hits_delta": delta(before, after, "prefix_cache_hits_total")}


def run_probe(name: str, *, model: str, revision: str, prompt_text: str, out_dir: pathlib.Path,
              port: int = 8000, max_model_len: int = 32768, timeout_s: float = 900.0,
              startup_timeout_s: float = 900.0, server_factory: Callable = engine.VLLMServer,
              metrics_fn: Callable = sampling.read_engine_metrics, settle_s: float = 10.0) -> dict:
    prefix, extra, purpose = CONFIGS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / f"probe_{name}.server.log")
    result = {"name": name, "purpose": purpose, "enable_prefix_caching": prefix, "extra_args": list(extra),
              "started": False, "startup_s": None, "failure": None}
    server = server_factory(model, revision, port=port, max_model_len=max_model_len,
                            enable_prefix_caching=prefix, enforce_eager=False, log_path=log_path,
                            startup_timeout_s=startup_timeout_s, extra_args=extra)
    result["server_argv"] = None
    try:
        result["server_argv"] = server.argv()
    except Exception as exc:  # e.g. the vllm command is missing
        result["failure"] = f"{type(exc).__name__}: {exc}"
        return result
    t0 = time.monotonic()
    try:
        server.start()
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {str(exc)[-1800:]}"
        result["startup_s"] = time.monotonic() - t0
        result["log_evidence"] = _log_evidence(log_path)
        server.stop()
        return result
    result["started"], result["startup_s"] = True, time.monotonic() - t0
    try:
        result["memory_before"] = container_memory(server.pid)
        result["request_1"] = one_request("127.0.0.1", port, model, prompt_text, timeout_s, metrics_fn)
        result["request_2"] = one_request("127.0.0.1", port, model, prompt_text, timeout_s, metrics_fn)
        result["memory_after"] = container_memory(server.pid)
        r1, r2 = result["request_1"], result["request_2"]
        result["same_output"] = (r1["status"] == "ok" and r2["status"] == "ok"
                                 and r1["output_text"] == r2["output_text"])
        result["second_request_hit_prefix_cache"] = bool((r2["prefix_cache_hits_delta"] or 0) > 0)
        result["ttft_ratio_second_over_first"] = (r2["ttft_s"] / r1["ttft_s"]
                                                  if r1["ttft_s"] and r2["ttft_s"] else None)
    finally:
        server.stop()
    result["log_evidence"] = _log_evidence(log_path)
    try:
        text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
        result["engine_reported_startup"] = sampling.parse_vllm_startup_log(text)
    except OSError:
        result["engine_reported_startup"] = None
    time.sleep(settle_s)  # let the GPU memory be released before the next launch
    return result


def _log_evidence(log_path: str) -> list:
    try:
        return evidence_lines(pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def classify(results: list) -> list:
    """Plain-language support verdict per configuration, from evidence only."""
    base = next((r for r in results if r["name"] == "baseline" and r["started"]), None)
    base_tokens = ((base or {}).get("engine_reported_startup") or {}).get("kv_cache_tokens")
    rows = []
    for r in results:
        pool = (r.get("engine_reported_startup") or {}).get("kv_cache_tokens")
        row = {"name": r["name"], "started": r["started"], "kv_cache_tokens": pool,
               "pool_vs_baseline": (pool / base_tokens) if pool and base_tokens else None,
               "second_request_hit_prefix_cache": r.get("second_request_hit_prefix_cache"),
               "ttft_ratio_second_over_first": r.get("ttft_ratio_second_over_first"),
               "same_output": r.get("same_output"), "evidence_lines": len(r.get("log_evidence") or [])}
        if not r["started"]:
            row["verdict"] = "did not start (unsupported or invalid flags on this device/version)"
        elif r["name"] == "baseline":
            row["verdict"] = ("control: no prefix hit expected" if not row["second_request_hit_prefix_cache"]
                              else "UNEXPECTED: baseline hit the prefix cache")
        elif r["name"] == "prefix_cache":
            row["verdict"] = ("active: repeat request hit the prefix cache" if row["second_request_hit_prefix_cache"]
                              else "flag accepted but no hit seen: not shown to be active")
        elif r["name"] == "fp8_kv":
            row["verdict"] = ("active: pool grew" if (row["pool_vs_baseline"] or 0) > 1.5
                              else "flag accepted but the pool did not grow: not shown to be active")
        elif r["name"] == "kv_offload_native":
            mentioned = any(re.search(r"offload", ln, re.I) for ln in (r.get("log_evidence") or []))
            row["verdict"] = ("started and the log mentions offloading; activity under pressure still unproven"
                              if mentioned else "started but the log never mentions offloading: not shown to be active")
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 9 KV-cache strategy capability probe")
    ap.add_argument("--workload-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--only", nargs="+", choices=list(CONFIGS), help="probe only these configurations")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--level", type=int, default=8192, help="input length of the probe prompt")
    args = ap.parse_args()

    wl_manifest, prompts = runner.load_workload(args.workload_dir)
    prompt = next(p for p in prompts if p["level"] == args.level)
    results = []
    for name in (args.only or list(CONFIGS)):
        print(f"\n=== probing {name}: {CONFIGS[name][2]}")
        res = run_probe(name, model=wl_manifest["model_id"], revision=wl_manifest["revision"],
                        prompt_text=prompt["text"], out_dir=args.out_dir, port=args.port,
                        max_model_len=args.max_model_len)
        res["probe_prompt_id"] = prompt["prompt_id"]
        results.append(res)
        (args.out_dir / f"probe_{name}.json").write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"started={res['started']} startup_s={res['startup_s']} failure={str(res['failure'])[:300]}")
    summary = classify(results)
    (args.out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\n=== summary")
    for row in summary:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
