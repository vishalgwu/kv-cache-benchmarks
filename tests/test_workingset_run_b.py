"""Step 9 working-set sweep, part B: prefix cache plus native CPU KV offload (8 GiB), real Kaggle T4 data.

offload 20261119T034805Z-1fee2266 (2026-11-19, vLLM 0.34.0) is paired with the prefix-only run
20261119T022723Z-402f880f from part A: same 480 session contexts, same order, same server settings, except for
`--kv-offloading-backend native --kv-offloading-size 8`. These tests pin what was recorded.
"""

import json
import pathlib
import statistics
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import workingset as ws  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
OFF, PREFIX = RES / "20261119T034805Z-1fee2266", RES / "20261119T022723Z-402f880f"
pytestmark = pytest.mark.skipif(not (OFF.exists() and PREFIX.exists()), reason="part A/B runs not present")


def recs(d):
    return [r for r in mt.load_records(d / "requests.jsonl") if r["phase"] in ("fill", "revisit")]


def key(r):
    return (r["working_set_sessions"], r["phase"], r["session"])


def test_run_validates_and_really_used_the_offload_configuration():
    assert mt.validate_run_dir(OFF) == []
    m = json.loads((OFF / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["config"]["strategy"] == "kv_offload_native"
    assert m["package_versions"]["vllm"] == "0.34.0" and m["lock_mismatches"] == []
    assert m["counts"] == {"requests": 961, "ok": 961, "errors": 0, "warmup": 1, "usable": 960}
    text = (OFF / "server.log").read_text(encoding="utf-8", errors="replace")
    assert "--kv-offloading-backend native" in text or "OffloadingConnector" in text
    assert "Creating offloading spec with name: CPUOffloadingSpec" in text
    assert "Created mmap file /dev/shm/vllm_offload_" in text and "(8.59 GB)" in text


def test_offload_gave_exactly_the_same_cache_hits_and_first_tokens_as_prefix_caching_alone():
    a, b = {key(r): r for r in recs(PREFIX)}, {key(r): r for r in recs(OFF)}
    assert a.keys() == b.keys() and len(a) == 960
    assert sum(r["prefix_cache_hits_delta"] or 0 for r in a.values()) == 193536
    assert sum(r["prefix_cache_hits_delta"] or 0 for r in b.values()) == 193536
    assert all(a[k]["prefix_cache_hits_delta"] == b[k]["prefix_cache_hits_delta"] for k in a)   # request by request
    assert [k for k in b if (b[k]["prefix_cache_hits_delta"] or 0) > 0 and k[0] != 48] == []
    diffs = [b[k]["ttft_s"] - a[k]["ttft_s"] for k in a]
    assert abs(statistics.median(diffs)) < 0.01                                  # median difference under 10 ms
    cells = ws.summarize(mt.load_records(OFF / "requests.jsonl"))
    for size in ("96", "144", "192"):
        assert cells[size]["revisit"]["requests_mostly_cached"] == 0
        assert cells[size]["revisit"]["ttft_s"]["median"] == pytest.approx(3.25, abs=0.1)
    assert cells["48"]["revisit"]["requests_mostly_cached"] == 48


def test_the_engine_log_shows_blocks_written_to_the_cpu_but_never_a_cpu_hit():
    ev = ws.offload_evidence((OFF / "server.log").read_text(encoding="utf-8", errors="replace"))
    assert ev["intervals"] == 375 and ev["block_bytes"] == 1835008
    assert ev["store_blocks_total"] == pytest.approx(58288)                      # about 107 GB written in all
    assert ev["load_bytes_total"] == 0 and ev["cpu_cache_read_usage_max"] == 0
    assert ev["external_hit_rate_max_pct"] == 0.0                                # not one prefix hit from another tier
    assert ev["cpu_cache_usage_max"] == pytest.approx(0.01367, abs=1e-4)         # about one request's blocks at a time
    base = ws.offload_evidence((PREFIX / "server.log").read_text(encoding="utf-8", errors="replace"))
    assert base["intervals"] == 0 and base["store_bytes_total"] == 0


def test_the_buffer_is_8gib_of_shared_memory_and_container_memory_was_near_the_limit():
    off, base = recs(OFF), recs(PREFIX)
    pick = lambda rs, k: {r["memory"]["cpu"][k] for r in rs if r["memory"]["cpu"].get(k) is not None}  # noqa: E731
    (shm,) = pick(off, "container_shmem_peak_mib")                               # the first direct shmem reading
    assert shm == pytest.approx(8237.75, abs=0.5)                                # 8.59 GB buffer + 46 MiB baseline
    (shm_base,) = pick(base, "container_shmem_peak_mib")
    assert shm_base == pytest.approx(46.0, abs=1.0) and shm - shm_base == pytest.approx(8192, abs=2)
    assert 7.5 * 1024 < min(pick(off, "container_file_peak_mib")) - max(pick(base, "container_file_peak_mib")) < 8.2 * 1024  # about +7.7 GiB: sessions differ in page cache by a few hundred MiB
    assert max(pick(off, "container_anon_peak_mib")) - min(pick(base, "container_anon_peak_mib")) < 100
    cur = pick(off, "container_mem_current_peak_mib")
    assert max(cur) < 30720 and max(cur) > 27 * 1024                             # 27.7 of 30 GiB, flat all run
    assert max(cur) - min(cur) < 200


def test_offload_did_not_slow_the_cache_misses_noticeably():
    a = statistics.median(r["ttft_s"] for r in recs(PREFIX) if r["phase"] == "fill")
    b = statistics.median(r["ttft_s"] for r in recs(OFF) if r["phase"] == "fill")
    assert abs(b - a) < 0.02                                                     # writing every request to CPU cost under 20 ms


SYNTH = (
    "(Core) INFO [gpu_worker.py:244] Registering full CPU offload region rank=0 (row=1000, slot=1000 bytes)\n"
    "(API) INFO [loggers.py:335] Engine 000: Avg prompt throughput: 1.0 tokens/s, Prefix cache hit rate: 25.5%, "
    "External prefix cache hit rate: 12.5%\n"
    "(API) INFO [metrics.py:108] KV Transfer metrics: vllm:kv_offload_store_bytes=4000, "
    "vllm:kv_offload_load_bytes=2000, vllm:kv_offload_cpu_cache_usage_perc=0.5, vllm:kv_offload_cpu_cache_read_usage_perc=0.25\n"
    "(API) INFO [metrics.py:108] KV Transfer metrics: vllm:kv_offload_store_bytes=1000, "
    "vllm:kv_offload_cpu_cache_usage_perc=0.75\n")


def test_offload_evidence_parser_on_synthetic_lines():
    ev = ws.offload_evidence(SYNTH)
    assert ev["intervals"] == 2 and ev["store_bytes_total"] == 5000 and ev["load_bytes_total"] == 2000
    assert (ev["store_blocks_total"], ev["load_blocks_total"]) == (5.0, 2.0)
    assert ev["cpu_cache_usage_max"] == 0.75 and ev["cpu_cache_read_usage_max"] == 0.25
    assert ev["external_hit_rate_max_pct"] == 12.5 and ev["prefix_hit_rate_max_pct"] == 25.5
    empty = ws.offload_evidence("")
    assert empty["intervals"] == 0 and empty["cpu_cache_usage_max"] is None and "store_blocks_total" not in empty
