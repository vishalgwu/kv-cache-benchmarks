"""Runner and streaming-client tests.

These run against a scripted local HTTP server (a TEST FIXTURE). Nothing here is
real inference and nothing it produces may be written to ``results/``.
"""

import json
import pathlib
import socket
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import runner as rn  # noqa: E402
from benchmarks import vllm_engine as eng  # noqa: E402
from workloads import generate as g  # noqa: E402

MODEL = "fake/model"


def sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


def role_event():
    return sse({"model": MODEL, "choices": [{"delta": {"role": "assistant", "content": ""}}]})


def content_event(text):
    return sse({"model": MODEL, "choices": [{"delta": {"content": text}}]})


def finish_event(reason="stop"):
    return sse({"model": MODEL, "choices": [{"delta": {}, "finish_reason": reason}]})


def usage_event(prompt_tokens, completion_tokens):
    return sse({"model": MODEL, "choices": [],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})


DONE = "data: [DONE]\n\n"


def good_script(first_delay=0.0, prompt_tokens=20, completion_tokens=3):
    return [(0, role_event()), (first_delay, content_event("Hel")), (0.02, content_event("lo ")),
            (0.02, content_event("world")), (0, finish_event()),
            (0, usage_event(prompt_tokens, completion_tokens)), (0, DONE)]


@contextmanager
def fake_server(scenario, status=200):
    """scenario: callable(payload) -> list of (delay_seconds, text) to stream."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            Handler.last_payload = payload
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if status != 200:
                self.wfile.write(b"boom")
                return
            try:
                for delay, text in scenario(payload):
                    time.sleep(delay)
                    self.wfile.write(text.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", server.server_address[1], Handler
    finally:
        server.shutdown()
        server.server_close()


def call(scenario_script, timeout_s=5.0, status=200):
    with fake_server(lambda p: scenario_script, status) as (host, port, _):
        payload = eng.build_payload(MODEL, "hi", 16)
        return eng.stream_chat(host, port, payload, timeout_s=timeout_s)


# ----------------------------------------------------------------------------- client
def test_normal_stream_timing_and_usage():
    res = call(good_script(first_delay=0.15))
    assert res.status == "ok" and res.error_category is None
    assert res.text == "Hello world"
    assert res.content_chunks == 3  # events, not tokens
    assert res.completion_tokens == 3 and res.prompt_tokens == 20  # from engine usage
    assert res.finish_reason == "stop"
    assert res.t_start <= res.t_first_content <= res.t_last_content <= res.t_complete
    assert res.t_first_content - res.t_start >= 0.15  # the injected delay is inside TTFT
    assert res.response_model == MODEL


def test_empty_and_metadata_events_do_not_count_as_first_token():
    # Role event arrives immediately; real content arrives after 0.2 s.
    res = call(good_script(first_delay=0.2))
    assert res.t_first_content - res.t_start >= 0.2


def test_stream_with_no_content_is_an_error_not_a_success():
    res = call([(0, role_event()), (0, finish_event()), (0, usage_event(5, 0)), (0, DONE)])
    assert res.status == "error" and res.error_category == "empty_output"


def test_missing_usage_leaves_token_counts_unavailable():
    script = [(0, role_event()), (0, content_event("a")), (0, content_event("b")),
              (0, finish_event()), (0, DONE)]
    res = call(script)
    assert res.status == "ok"
    assert res.completion_tokens is None and res.prompt_tokens is None
    assert res.content_chunks == 2  # never substituted for a token count


def test_http_error_is_recorded():
    res = call([], status=500)
    assert res.status == "error" and res.error_category == "http_error"
    assert res.http_status == 500


def test_timeout_is_recorded():
    res = call([(2.0, content_event("late")), (0, DONE)], timeout_s=0.4)
    assert res.status == "error" and res.error_category == "timeout"


def test_connection_refused_is_recorded():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    res = eng.stream_chat("127.0.0.1", port, eng.build_payload(MODEL, "hi", 4), timeout_s=1.0)
    # Linux refuses immediately (connection_error); Windows hangs until the connect
    # times out (timeout). Either way the failure is recorded, never reported as ok.
    assert res.status == "error" and res.error_category in {"connection_error", "timeout"}


def test_malformed_json_is_recorded():
    res = call([(0, "data: {not json\n\n"), (0, DONE)])
    assert res.status == "error" and res.error_category == "malformed_stream"


def test_stream_without_done_is_incomplete():
    res = call([(0, content_event("a")), (0, finish_event())])
    assert res.status == "error" and res.error_category == "incomplete_stream"


def test_server_error_event_is_recorded():
    res = call([(0, sse({"error": {"message": "out of memory"}}))])
    assert res.status == "error" and res.error_category == "server_error"


def test_payload_asks_for_engine_usage_and_greedy_decoding():
    p = eng.build_payload("m", "hello", 128)
    assert p["stream"] is True and p["stream_options"] == {"include_usage": True}
    assert p["temperature"] == 0 and p["max_tokens"] == 128 and "ignore_eos" not in p
    assert eng.build_payload("m", "hello", 128, ignore_eos=True)["ignore_eos"] is True


# ----------------------------------------------------------------------------- records
def make_prompt(actual=20):
    return {"prompt_id": "L1-000", "prompt_sha256": "x", "target_input_tokens": 20,
            "actual_input_tokens": actual, "text": "hi"}


def record_for(res, actual=20, model=MODEL, max_tokens=16):
    return rn.build_record(run_id="r", cfg_hash="h", request_id="r-0", prompt=make_prompt(actual),
                           is_warmup=False, trial_index=0, result=res, expected_model=model,
                           max_tokens=max_tokens, ignore_eos=False)


def test_good_record_is_usable_and_metrics_are_derived():
    rec = record_for(call(good_script(first_delay=0.1)))
    assert rec["validation_errors"] == [] and rec["usable_for_analysis"] is True
    assert rec["ttft_s"] >= 0.1
    assert rec["e2e_latency_s"] >= rec["ttft_s"]
    assert rec["decode_tokens_per_s"] == pytest.approx(2 / rec["decode_time_s"])  # (3 - 1) tokens
    assert rec["prompt_tokens_source"] == "engine_usage"
    assert rec["clock"] == "time.monotonic" and "first non-empty content" in rec["ttft_basis"]


def test_one_token_output_has_no_decode_throughput():
    script = [(0, content_event("a")), (0, finish_event("length")), (0, usage_event(20, 1)), (0, DONE)]
    rec = record_for(call(script))
    assert rec["decode_tokens_per_s"] is None


def test_record_without_usage_is_not_usable():
    script = [(0, content_event("a")), (0, content_event("b")), (0, finish_event()), (0, DONE)]
    rec = record_for(call(script))
    assert "missing_engine_usage" in rec["validation_errors"]
    assert rec["usable_for_analysis"] is False and rec["output_tokens"] is None


def test_failed_record_is_kept_but_not_usable():
    rec = record_for(call([], status=500))
    assert rec["status"] == "error" and rec["error_category"] == "http_error"
    assert rec["usable_for_analysis"] is False and rec["validation_errors"] == []


def test_input_token_mismatch_is_flagged():
    rec = record_for(call(good_script(prompt_tokens=25)), actual=20)
    assert "input_token_mismatch" in rec["validation_errors"]
    assert rec["usable_for_analysis"] is False


def test_model_mismatch_is_flagged():
    rec = record_for(call(good_script()), model="some/other-model")
    assert "model_mismatch" in rec["validation_errors"]


def test_output_over_max_tokens_is_flagged():
    rec = record_for(call(good_script(completion_tokens=99)), max_tokens=16)
    assert "output_exceeds_max_tokens" in rec["validation_errors"]


def test_out_of_order_timestamps_are_flagged():
    rec = record_for(call(good_script()))
    rec["t_first_content_s"] = rec["t_start_s"] - 1
    assert "timestamps_out_of_order" in rn.validate_record(rec)


# ----------------------------------------------------------------------------- run
def fake_counter(text):
    return 10 + sum(1 if len(w) <= 5 else 2 for w in text.split())


@pytest.fixture()
def workload_dir(tmp_path):
    manifest, prompts = g.generate_workload(
        base_seed=3, levels=[64, 128], per_level=2, output_tokens=16, max_context=1024,
        count_tokens=fake_counter, model_id=MODEL, revision="abc123")
    return g.write_workload(tmp_path / "wl", manifest, prompts)


def echoing_scenario(payload):
    prompt_tokens = fake_counter(payload["messages"][0]["content"])
    return good_script(prompt_tokens=prompt_tokens)


def run(workload_dir, tmp_path, host, port, **over):
    kwargs = dict(workload_dir=workload_dir, host=host, port=port, model=MODEL, revision="abc123",
                  levels=[64, 128], limit_per_level=1, trials=1, warmup=1, timeout_s=5.0,
                  ignore_eos=False, out_dir=tmp_path / "results", server_settings=None)
    kwargs.update(over)
    return rn.run_benchmark(**kwargs)


def test_end_to_end_writes_manifest_and_one_record_per_request(workload_dir, tmp_path):
    with fake_server(echoing_scenario) as (host, port, _):
        run_dir, records = run(workload_dir, tmp_path, host, port)
    lines = [json.loads(x) for x in (run_dir / "requests.jsonl").read_text().splitlines()]
    assert len(lines) == len(records) == 3  # 1 warmup + 2 measured prompts
    assert [r["is_warmup"] for r in lines] == [True, False, False]
    assert all(r["usable_for_analysis"] for r in lines if not r["is_warmup"])
    assert not any(r["usable_for_analysis"] for r in lines if r["is_warmup"])
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["config_hash"] == lines[0]["config_hash"]
    assert manifest["counts"]["requests"] == 3 and manifest["counts"]["usable"] == 2
    assert manifest["config"]["sampling"]["max_tokens"] == 16
    assert manifest["finished_utc"] is not None


def test_manifest_is_written_before_the_first_request(workload_dir, tmp_path):
    seen = {}

    def scenario(payload):
        runs = list((tmp_path / "results").glob("*/run_manifest.json"))
        seen["manifest_existed"] = bool(runs)
        seen["finished"] = json.loads(runs[0].read_text())["finished_utc"] if runs else "n/a"
        return echoing_scenario(payload)

    with fake_server(scenario) as (host, port, _):
        run(workload_dir, tmp_path, host, port, warmup=0, levels=[64])
    assert seen["manifest_existed"] is True and seen["finished"] is None


def test_request_failure_still_produces_a_record_and_stops(workload_dir, tmp_path):
    with fake_server(lambda p: [], status=500) as (host, port, _):
        run_dir, records = run(workload_dir, tmp_path, host, port)
    assert len(records) == 1  # stop_on_error ends the run after the first failure
    assert records[0]["status"] == "error" and records[0]["is_warmup"] is True
    assert (run_dir / "requests.jsonl").read_text().strip() != ""


def test_no_stop_on_error_keeps_going(workload_dir, tmp_path):
    with fake_server(lambda p: [], status=500) as (host, port, _):
        _, records = run(workload_dir, tmp_path, host, port, stop_on_error=False)
    assert len(records) == 3 and all(r["status"] == "error" for r in records)


def test_concurrency_above_one_is_rejected(workload_dir, tmp_path):
    with pytest.raises(ValueError, match="Step 8"):
        run(workload_dir, tmp_path, "127.0.0.1", 1, concurrency=2)


def test_unknown_level_selects_nothing(workload_dir, tmp_path):
    with pytest.raises(ValueError, match="no prompts selected"):
        run(workload_dir, tmp_path, "127.0.0.1", 1, levels=[999])


def test_config_hash_is_stable_and_sensitive():
    base = {"a": 1, "b": {"c": 2}}
    assert rn.config_hash(base) == rn.config_hash({"b": {"c": 2}, "a": 1})
    assert rn.config_hash(base) != rn.config_hash({"a": 1, "b": {"c": 3}})


def test_ignore_eos_is_sent_and_recorded(workload_dir, tmp_path):
    with fake_server(echoing_scenario) as (host, port, handler):
        run_dir, records = run(workload_dir, tmp_path, host, port, ignore_eos=True, warmup=0,
                               levels=[64])
        assert handler.last_payload["ignore_eos"] is True
    assert records[0]["ignore_eos"] is True
    assert json.loads((run_dir / "run_manifest.json").read_text())["config"]["sampling"]["ignore_eos"]
