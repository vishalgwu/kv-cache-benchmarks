"""Memory and engine-metric samplers (Step 6).

Three background samplers run in the runner process, so their timestamps share the
runner's monotonic clock and line up with request timestamps:

* GPU      device memory used/total (MiB) and utilization, via NVML or nvidia-smi
* engine   vLLM ``/metrics``: KV-cache usage, running/waiting requests, preemptions
* CPU      resident memory of the server process tree and host available memory

Limits that every consumer must respect (see docs/measurement_contract.md):

* GPU memory is DEVICE scope (all processes on that GPU), not per process.
* vLLM preallocates its KV-cache pool at startup, so device memory is nearly flat
  while the active cache grows. Device memory therefore does NOT measure KV-cache
  occupancy; engine cache usage does, where the engine exposes it.
* A sampled peak is a LOWER BOUND on the true peak: anything between two samples
  is invisible, and a request shorter than the interval may contain no sample.
  Such a request gets ``unavailable`` rather than an invented number.
* Summed process RSS can double count shared pages.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from typing import Callable, Optional

MIB = 2**20
LABELS = {"measured", "estimated", "unavailable"}

# Engine metric names differ between vLLM versions, so each canonical metric lists
# candidates. The run saves the engine's real catalog so the names can be verified.
ENGINE_METRICS = {
    "kv_cache_usage": (("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"), "max"),
    "requests_running": (("vllm:num_requests_running",), "sum"),
    "requests_waiting": (("vllm:num_requests_waiting",), "sum"),
    "preemptions_total": (("vllm:num_preemptions_total", "vllm:num_preemptions"), "sum"),
    # Counters that prove prefix reuse did (or did not) happen; with prefix caching off
    # they should not grow.
    "prefix_cache_hits_total": (("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"), "sum"),
    "prefix_cache_queries_total": (("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"), "sum"),
}


# ----------------------------------------------------------------------------- sampler
class PeriodicSampler:
    """Call ``read_fn`` every ``interval_s`` on a thread; keep timestamped samples.

    A sample's time ``t_s`` is the midpoint of the read (start and end are kept too),
    so a slow read is not attributed entirely to its start.
    """

    def __init__(self, name: str, read_fn: Callable[[], Optional[dict]], interval_s: float = 0.5,
                 clock: Callable[[], float] = time.monotonic):
        self.name, self.read_fn, self.interval_s, self.clock = name, read_fn, interval_s, clock
        self.samples: list = []
        self.read_errors = 0
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        next_due = self.clock()
        while not self._stop.is_set():
            t0 = self.clock()
            try:
                values = self.read_fn()
            except Exception as exc:  # keep sampling; count and remember the failure
                self.read_errors += 1
                self.last_error = repr(exc)[:200]
                values = None
            t1 = self.clock()
            if values is not None:
                self.samples.append({"t_s": (t0 + t1) / 2, "t_read_start_s": t0, "t_read_end_s": t1,
                                     "wall_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **values})
            next_due += self.interval_s
            self._stop.wait(max(0.0, next_due - self.clock()))

    def start(self) -> "PeriodicSampler":
        self._thread = threading.Thread(target=self._run, name=f"sampler-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


# ----------------------------------------------------------------------------- windows
def in_window(samples: list, t0: float, t1: float) -> list:
    return [s for s in samples if t0 <= s["t_s"] <= t1]


def window_stat(samples: list, t0: float, t1: float, key: str, fn: Callable) -> tuple:
    """(value, n) over non-None ``key`` values inside [t0, t1]; value is None when n == 0."""
    vals = [s[key] for s in in_window(samples, t0, t1) if s.get(key) is not None]
    return (fn(vals) if vals else None), len(vals)


def counter_delta(samples: list, key: str, t0: float, t1: float) -> Optional[float]:
    """Growth of a cumulative counter over [t0, t1]: last in-window value minus the last
    value before the window (or the first in-window value if none). None without data.
    Counters can update after a request ends, so a window may under-report by one request."""
    in_win = [s for s in samples if t0 <= s["t_s"] <= t1 and s.get(key) is not None]
    if not in_win:
        return None
    before = [s for s in samples if s["t_s"] < t0 and s.get(key) is not None]
    ref = before[-1][key] if before else in_win[0][key]
    return in_win[-1][key] - ref


def interval_stats(samples: list, nominal_s: float) -> dict:
    gaps = [b["t_s"] - a["t_s"] for a, b in zip(samples, samples[1:])]
    return {"nominal_interval_s": nominal_s, "n_samples": len(samples),
            "mean_gap_s": (sum(gaps) / len(gaps)) if gaps else None,
            "max_gap_s": max(gaps) if gaps else None}


# ----------------------------------------------------------------------------- GPU readers
def read_gpu_nvidia_smi(gpu_index: int = 0, timeout_s: float = 5.0) -> dict:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout_s)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {(out.stderr or out.stdout).strip()[:200]}")
    used, total, util = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
    return {"memory_used_mib": float(used), "memory_total_mib": float(total),
            "gpu_util_pct": None if "N/A" in util else float(util),
            "source": "nvidia-smi", "scope": "device"}


def _throttle_mask(pynvml, handle) -> int:
    """Current clock-throttle reason bitmask (NVML renamed the call across versions)."""
    for name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
        fn = getattr(pynvml, name, None)
        if fn is not None:
            return fn(handle)
    raise RuntimeError("no throttle-reason call in this NVML binding")


def make_gpu_reader(gpu_index: int = 0) -> Callable[[], dict]:
    """Prefer NVML (lower overhead); fall back to calling nvidia-smi.

    Indices are physical (NVML/nvidia-smi ignore CUDA_VISIBLE_DEVICES). On a machine
    with identical GPUs, physical index 0 is the device the study uses.
    """
    try:
        import pynvml  # provided by the nvidia-ml-py package

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        def read() -> dict:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            out = {"memory_used_mib": mem.used / MIB, "memory_total_mib": mem.total / MIB,
                   "gpu_util_pct": float(util.gpu), "source": "nvml", "scope": "device"}
            # Best-effort extras for long runs (thermal or power throttling would bias later trials).
            # Each is None when the driver does not offer it; they never break the memory reading.
            extras = {
                "temperature_c": lambda: float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                "sm_clock_mhz": lambda: float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                "power_w": lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                "throttle_reasons_mask": lambda: int(_throttle_mask(pynvml, handle)),
            }
            for key, fn in extras.items():
                try:
                    out[key] = fn()
                except Exception:
                    out[key] = None
            return out

        read()  # fail now, not on the sampler thread
        return read
    except Exception:
        return lambda: read_gpu_nvidia_smi(gpu_index)


# ----------------------------------------------------------------------------- engine metrics
_PROM = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text: str) -> tuple:
    """Return (samples, meta). samples: [(name, labels, value)]; meta: name -> {type, help}."""
    samples, meta = [], {}
    for line in text.splitlines():
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                entry = meta.setdefault(parts[2], {})
                entry["help" if parts[1] == "HELP" else "type"] = parts[3] if len(parts) > 3 else ""
            continue
        m = _PROM.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        if value != value:  # NaN
            continue
        samples.append((m.group(1), m.group(2) or "", value))
    return samples, meta


def extract_engine_values(samples: list) -> dict:
    """Pick the canonical engine metrics out of parsed samples; missing ones are None."""
    out = {"metric_names": {}}
    for canonical, (candidates, agg) in ENGINE_METRICS.items():
        out[canonical], out["metric_names"][canonical] = None, None
        for name in candidates:
            vals = [v for n, _, v in samples if n == name]
            if vals:
                out[canonical] = max(vals) if agg == "max" else sum(vals)
                out["metric_names"][canonical] = name
                break
    return out


def _http_get(host: str, port: int, path: str, timeout_s: float) -> str:
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", "replace")


def read_engine_metrics(host: str, port: int, timeout_s: float = 2.0) -> dict:
    samples, _ = parse_prometheus(_http_get(host, port, "/metrics", timeout_s))
    return {**extract_engine_values(samples), "source": "vllm /metrics"}


def fetch_engine_catalog(host: str, port: int, timeout_s: float = 5.0) -> list:
    """All vllm: metric names with TYPE and HELP, so names and scales can be verified."""
    samples, meta = parse_prometheus(_http_get(host, port, "/metrics", timeout_s))
    names = sorted({n for n, _, _ in samples if n.startswith("vllm:")})
    return [(n, meta.get(n, {}).get("type", ""), meta.get(n, {}).get("help", "")) for n in names]


# ----------------------------------------------------------------------------- CPU readers
def _proc_tree_rss_mib(pid: int, proc_root: str = "/proc") -> Optional[float]:
    children, rss = {}, {}
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            status = pathlib.Path(proc_root, entry, "status").read_text()
        except OSError:
            continue
        ppid = re.search(r"PPid:\s+(\d+)", status)
        vm = re.search(r"VmRSS:\s+(\d+)\s*kB", status)
        children.setdefault(int(ppid.group(1)) if ppid else 0, []).append(int(entry))
        rss[int(entry)] = int(vm.group(1)) if vm else 0
    if pid not in rss:
        return None
    total, stack, seen = 0, [pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        total += rss.get(p, 0)
        stack.extend(children.get(p, []))
    return total / 1024  # kB -> MiB


def _read_int(path: str) -> Optional[int]:
    try:
        text = pathlib.Path(path).read_text().strip()
    except OSError:
        return None
    if text in ("", "max"):
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value >= 2**60 else value  # cgroup v1 reports ~9.2e18 for "unlimited"


def _cgroup_mem_mib(root: str = "/sys/fs/cgroup") -> dict:
    """Container memory limit and current use (cgroup v2, then v1). None when unreadable.

    /proc/meminfo inside a container describes the whole machine, not the container's
    limit, so it can overstate how much memory the notebook may actually use.
    """
    for limit_rel, current_rel in (("memory.max", "memory.current"),
                                   ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")):
        limit, current = _read_int(f"{root}/{limit_rel}"), _read_int(f"{root}/{current_rel}")
        if limit is not None or current is not None:
            return {"container_mem_limit_mib": None if limit is None else limit / MIB,
                    "container_mem_current_mib": None if current is None else current / MIB}
    return {"container_mem_limit_mib": None, "container_mem_current_mib": None}


def _cgroup_mem_stat_mib(root: str = "/sys/fs/cgroup") -> dict:
    """Container anonymous memory, page cache and shared memory, in MiB (cgroup v2 ``memory.stat``, then v1).

    ``memory.current`` includes page cache (measurement contract limit 13), so it cannot show how much
    memory the server really holds. ``anon`` is ordinary process memory; ``file`` is page cache; ``shmem``
    is shared memory (tmpfs such as /dev/shm), which cgroup v2 counts inside ``file`` as well.
    The Step 9 probe showed that vLLM's native CPU KV offload buffer (an mmap'd /dev/shm file) rose in
    ``file`` and in the server's RSS but NOT in ``anon``. None when unreadable.
    """
    for rel, keys in (("memory.stat", ("anon", "file", "shmem")),
                      ("memory/memory.stat", ("total_rss", "total_cache", "total_shmem"))):
        try:
            text = pathlib.Path(f"{root}/{rel}").read_text()
        except OSError:
            continue
        fields = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                fields[parts[0]] = int(parts[1])
        if any(k in fields for k in keys):
            get = lambda k: fields[k] / MIB if k in fields else None  # noqa: E731
            return {"container_anon_mib": get(keys[0]), "container_file_mib": get(keys[1]),
                    "container_shmem_mib": get(keys[2])}
    return {"container_anon_mib": None, "container_file_mib": None, "container_shmem_mib": None}


def _host_mem_mib(meminfo_path: str = "/proc/meminfo") -> dict:
    text = pathlib.Path(meminfo_path).read_text()
    get = lambda key: int(re.search(rf"{key}:\s+(\d+)\s*kB", text).group(1)) / 1024  # noqa: E731
    return {"host_mem_available_mib": get("MemAvailable"), "host_mem_total_mib": get("MemTotal")}


def make_cpu_reader(pid: Optional[int]) -> Callable[[], dict]:
    def read() -> dict:
        values = {"server_tree_rss_mib": None, "source": "/proc", "scope": (
            "sum of RSS over the server process tree (may double count shared pages); "
            "host MemAvailable for the whole machine")}
        if pid is not None:
            try:
                import psutil

                proc = psutil.Process(pid)
                tree = [proc] + proc.children(recursive=True)
                values["server_tree_rss_mib"] = sum(p.memory_info().rss for p in tree) / MIB
                values["source"] = "psutil + /proc/meminfo"
            except ImportError:
                values["server_tree_rss_mib"] = _proc_tree_rss_mib(pid)
            except Exception:
                values["server_tree_rss_mib"] = None
        values.update(_host_mem_mib())
        values.update(_cgroup_mem_mib())
        values.update(_cgroup_mem_stat_mib())
        return values

    return read


# ----------------------------------------------------------------------------- startup log
def parse_vllm_startup_log(text: str) -> dict:
    """Engine-reported startup figures from a vLLM server log. Missing values are None."""
    def first(pattern, cast=float, group=1):
        m = re.search(pattern, text)
        return cast(m.group(group).replace(",", "")) if m else None

    usage = re.search(r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), "
                      r"([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph", text)
    free = re.search(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)", text)
    backend = re.search(r"Using (\w+) attention backend", text)
    version = re.search(r"Initializing a V1 LLM engine \(v([\w.+-]+)\)", text)
    blocks = re.search(r"([\d,]+) KV cache blocks|Extended KV cache to ([\d,]+) blocks", text)
    n_blocks = int((blocks.group(1) or blocks.group(2)).replace(",", "")) if blocks else None
    tokens = first(r"GPU KV cache size: ([\d,]+) tokens", int)
    block_size = tokens // n_blocks if (n_blocks and tokens and tokens % n_blocks == 0) else None
    return {
        "source": "vLLM server log (engine-reported)",
        "vllm_version_in_log": version.group(1) if version else None,
        # vLLM 0.34 prints a profiling estimate ("Available KV cache memory") that is NOT the pool it
        # finally builds; the extensible cache reports the real size here (None on older logs).
        "kv_cache_final_gib": first(r"Extended KV cache to [\d,]+ blocks \(([\d.]+) GiB\)"),
        "kv_cache_blocks": n_blocks,
        "kv_block_size_tokens": block_size,
        "model_loading_gib": first(r"Model loading took ([\d.]+) GiB"),
        "available_kv_cache_gib": first(r"Available KV cache memory: ([\d.]+) GiB"),
        "kv_cache_tokens": first(r"GPU KV cache size: ([\d,]+) tokens", int),
        "max_concurrency_reported": first(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
        "max_concurrency_tokens_per_request": first(
            r"Maximum concurrency for ([\d,]+) tokens per request", int),
        "free_memory_gib_at_startup": float(free.group(1)) if free else None,
        "device_total_gib": float(free.group(2)) if free else None,
        "consumed_weights_plus_non_torch_gib": float(usage.group(1)) if usage else None,
        "peak_activation_gib": float(usage.group(2)) if usage else None,
        "cuda_graph_gib": float(usage.group(3)) if usage else None,
        "attention_backend": backend.group(1) if backend else None,
    }


# ----------------------------------------------------------------------------- monitor
class MemoryMonitor:
    """Bundle of GPU, engine, and CPU samplers plus per-request window summaries."""

    def __init__(self, *, gpu_reader=None, engine_reader=None, cpu_reader=None,
                 interval_s: float = 0.5, clock: Callable[[], float] = time.monotonic):
        self.interval_s, self.clock = interval_s, clock
        readers = {"gpu": gpu_reader, "engine": engine_reader, "cpu": cpu_reader}
        self.samplers = {name: PeriodicSampler(name, fn, interval_s, clock)
                         for name, fn in readers.items() if fn is not None}
        self.baseline_window: Optional[tuple] = None
        self.engine_catalog: list = []

    def start(self) -> None:
        for s in self.samplers.values():
            s.start()

    def stop(self) -> None:
        for s in self.samplers.values():
            s.stop()

    def record_baseline(self, duration_s: float) -> None:
        """Idle baseline: sample with no requests in flight for ``duration_s`` seconds."""
        t0 = self.clock()
        time.sleep(duration_s)
        self.baseline_window = (t0, self.clock())

    def _samples(self, name: str) -> list:
        return list(self.samplers[name].samples) if name in self.samplers else []

    def baseline_gpu_mib(self) -> tuple:
        if self.baseline_window is None:
            return None, 0
        return window_stat(self._samples("gpu"), *self.baseline_window, "memory_used_mib",
                           statistics.median)

    def request_block(self, t0: float, t1: float) -> dict:
        """Memory/engine summary for one request window [t0, t1] (monotonic seconds)."""
        block = {"window": "request submission to completion, monotonic clock",
                 "sampling_interval_s": self.interval_s}
        block["gpu"] = self._gpu_section(t0, t1) if "gpu" in self.samplers else _unavailable(
            "gpu sampler not configured")
        block["engine"] = self._engine_section(t0, t1) if "engine" in self.samplers else _unavailable(
            "engine metrics sampler not configured")
        block["cpu"] = self._cpu_section(t0, t1) if "cpu" in self.samplers else _unavailable(
            "cpu sampler not configured")
        return block

    def _gpu_section(self, t0: float, t1: float) -> dict:
        samples = self._samples("gpu")
        peak, n = window_stat(samples, t0, t1, "memory_used_mib", max)
        base, n_base = self.baseline_gpu_mib()
        before = [s for s in samples if s["t_s"] <= t0 and s.get("memory_used_mib") is not None]
        start = before[-1]["memory_used_mib"] if before else None
        common = {"unit": "MiB", "scope": "device (all processes on the GPU; not per process)",
                  "source": (samples[-1]["source"] if samples else None),
                  "baseline_used_mib": base, "baseline_n_samples": n_base,
                  "used_mib_at_window_start": start,
                  "peak_minus_window_start_mib": (peak - start) if (peak is not None and start is not None)
                  else None,
                  "sm_clock_mhz_min_in_window": window_stat(samples, t0, t1, "sm_clock_mhz", min)[0],
                  "temperature_c_max_in_window": window_stat(samples, t0, t1, "temperature_c", max)[0],
                  "power_w_max_in_window": window_stat(samples, t0, t1, "power_w", max)[0],
                  "throttle_reasons_seen_in_window": sorted({
                      s["throttle_reasons_mask"] for s in in_window(samples, t0, t1)
                      if s.get("throttle_reasons_mask") is not None}) or None,
                  "n_samples_in_window": n, "peak_note": "sampled peak; a lower bound on the true peak",
                  "retention_note": "the allocator keeps memory after a request ends, so a window's peak "
                                    "includes growth from earlier requests; peak_minus_window_start_mib is "
                                    "the growth during this request"}
        if peak is None:
            return {**common, "label": "unavailable", "peak_used_mib_in_window": None,
                    "reason": "no GPU sample fell inside the request window"}
        return {**common, "label": "measured", "peak_used_mib_in_window": peak, "reason": None}

    def _engine_section(self, t0: float, t1: float) -> dict:
        samples = self._samples("engine")
        in_win = in_window(samples, t0, t1)
        names = next((s["metric_names"] for s in reversed(samples) if s.get("metric_names")), {})
        sec = {"source": "vllm /metrics", "metric_names": names,
               "n_samples_in_window": len(in_win),
               "kv_cache_usage_scale_note": "raw value as reported by the engine; confirm the scale "
                                            "(fraction vs percent) from the metric HELP text in "
                                            "engine_metrics_catalog.tsv"}
        if not in_win:
            return {**sec, "label": "unavailable", "kv_cache_usage_peak": None,
                    "requests_running_peak": None, "requests_waiting_peak": None,
                    "preemptions_in_window": None, "prefix_cache_hits_in_window": None,
                    "prefix_cache_queries_in_window": None,
                    "reason": "no engine sample fell inside the request window"}
        peak = lambda key: window_stat(samples, t0, t1, key, max)[0]  # noqa: E731
        have_any = any(peak(k) is not None for k in ("kv_cache_usage", "requests_running", "requests_waiting"))
        return {**sec, "label": "measured" if have_any else "unavailable",
                "kv_cache_usage_peak": peak("kv_cache_usage"),
                "requests_running_peak": peak("requests_running"),
                "requests_waiting_peak": peak("requests_waiting"),
                "preemptions_in_window": counter_delta(samples, "preemptions_total", t0, t1),
                "prefix_cache_hits_in_window": counter_delta(samples, "prefix_cache_hits_total", t0, t1),
                "prefix_cache_queries_in_window": counter_delta(samples, "prefix_cache_queries_total", t0, t1),
                "counter_note": "counters may update just after a request ends, so a window can under-report "
                                "by one request; the run-level deltas in the manifest are the reliable check",
                "reason": None if have_any else "engine exposes none of the expected metric names"}

    def _cpu_section(self, t0: float, t1: float) -> dict:
        samples = self._samples("cpu")
        rss, n = window_stat(samples, t0, t1, "server_tree_rss_mib", max)
        avail, n_avail = window_stat(samples, t0, t1, "host_mem_available_mib", min)
        cont_peak, _ = window_stat(samples, t0, t1, "container_mem_current_mib", max)
        anon_peak, _ = window_stat(samples, t0, t1, "container_anon_mib", max)
        file_peak, _ = window_stat(samples, t0, t1, "container_file_mib", max)
        shmem_peak, _ = window_stat(samples, t0, t1, "container_shmem_mib", max)
        cont_limit = next((s["container_mem_limit_mib"] for s in reversed(samples)
                           if s.get("container_mem_limit_mib") is not None), None)
        sec = {"unit": "MiB", "scope": (samples[-1]["scope"] if samples else None),
               "source": (samples[-1]["source"] if samples else None),
               "n_samples_in_window": max(n, n_avail),
               "container_mem_current_peak_mib": cont_peak, "container_mem_limit_mib": cont_limit,
               "container_anon_peak_mib": anon_peak, "container_file_peak_mib": file_peak,
               "container_shmem_peak_mib": shmem_peak}
        if rss is None and avail is None:
            return {**sec, "label": "unavailable", "server_tree_rss_peak_mib": None,
                    "host_mem_available_min_mib": None,
                    "reason": "no CPU sample fell inside the request window"}
        return {**sec, "label": "measured", "server_tree_rss_peak_mib": rss,
                "host_mem_available_min_mib": avail, "reason": None}

    def describe(self) -> dict:
        out = {"interval_nominal_s": self.interval_s, "baseline_window_s": (
            None if self.baseline_window is None else self.baseline_window[1] - self.baseline_window[0])}
        for name, s in self.samplers.items():
            out[name] = {**interval_stats(s.samples, self.interval_s), "read_errors": s.read_errors,
                         "last_error": s.last_error}
        base, n_base = self.baseline_gpu_mib()
        out["gpu_idle_baseline_used_mib"] = {"value": base, "n_samples": n_base,
                                             "label": "measured" if base is not None else "unavailable"}
        eng = self._samples("engine")
        deltas = {}
        for key in ("preemptions_total", "prefix_cache_hits_total", "prefix_cache_queries_total"):
            vals = [s[key] for s in eng if s.get(key) is not None]
            deltas[key] = (vals[-1] - vals[0]) if len(vals) >= 2 else None
        out["engine_counter_deltas_over_run"] = deltas
        return out

    def write(self, run_dir: pathlib.Path) -> None:
        for name, s in self.samplers.items():
            with open(run_dir / f"samples_{name}.jsonl", "w", encoding="utf-8") as fh:
                for sample in s.samples:
                    fh.write(json.dumps(sample, sort_keys=True) + "\n")
        if self.engine_catalog:
            with open(run_dir / "engine_metrics_catalog.tsv", "w", encoding="utf-8") as fh:
                fh.write("name\ttype\thelp\n")
                for name, kind, help_text in self.engine_catalog:
                    fh.write(f"{name}\t{kind}\t{help_text}\n")


def _unavailable(reason: str) -> dict:
    return {"label": "unavailable", "reason": reason}


# ----------------------------------------------------------------------------- validation
def validate_memory_block(block: dict) -> list:
    """Schema check for a per-request ``memory`` block. Empty list means valid."""
    errs = []
    for section in ("gpu", "engine", "cpu"):
        sec = block.get(section)
        if not isinstance(sec, dict):
            errs.append(f"memory.{section}: missing")
            continue
        label = sec.get("label")
        if label not in LABELS:
            errs.append(f"memory.{section}: bad label {label!r}")
            continue
        if label == "unavailable" and not sec.get("reason"):
            errs.append(f"memory.{section}: unavailable without a reason")
        if label == "measured":
            if sec.get("n_samples_in_window", 0) < 1:
                errs.append(f"memory.{section}: measured with no samples in window")
            if section in ("gpu", "cpu") and not sec.get("unit"):
                errs.append(f"memory.{section}: measured value without a unit")
            if section == "gpu" and sec.get("peak_used_mib_in_window") is None:
                errs.append("memory.gpu: measured but peak is None")
            if section == "gpu" and not sec.get("scope"):
                errs.append("memory.gpu: measured value without a scope")
    return errs
