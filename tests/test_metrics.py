"""Metric and validation tests.

Request records below are built from SYNTHETIC TEST DATA with known timings so the
formulas can be checked by hand. They must never be written to results/.
"""

import json
import pathlib
import shutil
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import metrics as mt  # noqa: E402
from benchmarks import runner as rn  # noqa: E402
from benchmarks import vllm_engine as eng  # noqa: E402

REAL_RUN = pathlib.Path(__file__).resolve().parents[1] / "results" / "20261116T210839Z-d6a00143"
MODEL = "fake/model"


def synth(request_id="r-0", t_start=100.0, t_first=100.5, t_last=102.5, t_complete=102.6,
          prompt_tokens=20, completion_tokens=101, status="ok", error_category=None,
          is_warmup=False, finish_reason="length", max_tokens=128, model=MODEL):
    res = eng.StreamResult(
        status=status, error_category=error_category, t_start=t_start,
        t_first_content=t_first if status == "ok" else None,
        t_last_content=t_last if status == "ok" else None,
        t_complete=t_complete if status == "ok" else None, content_chunks=3, text="hi",
        finish_reason=finish_reason if status == "ok" else None, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, response_model=model)
    prompt = {"prompt_id": "p", "prompt_sha256": "x", "target_input_tokens": 20,
              "actual_input_tokens": 20}
    return rn.build_record(run_id="r", cfg_hash="h", request_id=request_id, prompt=prompt,
                           is_warmup=is_warmup, trial_index=0, result=res, expected_model=MODEL,
                           max_tokens=max_tokens, ignore_eos=False)


# ----------------------------------------------------------------------------- percentiles
def test_percentile_known_values():
    xs = [1, 2, 3, 4, 5]
    assert mt.percentile(xs, 0) == 1 and mt.percentile(xs, 100) == 5
    assert mt.percentile(xs, 50) == 3
    assert mt.percentile(xs, 95) == pytest.approx(4.8)
    assert mt.percentile([10, 20], 25) == pytest.approx(12.5)
    assert mt.percentile([7], 99) == 7
    assert mt.percentile([], 50) is None
    assert mt.percentile([5, 1, 3], 50) == 3  # input order does not matter


def test_describe_gates_tail_percentiles_on_sample_size():
    small = mt.describe(range(19))
    assert small["n"] == 19 and small["p95"] is None and "p95 needs n>=20" in small["tail_note"]
    assert mt.describe(range(20))["p95"] is not None
    assert mt.describe(range(99))["p99"] is None
    assert mt.describe(range(100))["p99"] is not None
    assert mt.describe(range(100))["tail_note"] is None


def test_describe_ignores_none_and_handles_empty():
    d = mt.describe([1.0, None, 3.0])
    assert d["n"] == 2 and d["mean"] == 2.0 and d["median"] == 2.0
    empty = mt.describe([])
    assert empty["n"] == 0 and empty["mean"] is None and empty["p95"] is None


# ----------------------------------------------------------------------------- known events
def test_known_events_give_hand_computed_metrics():
    # start 100.0, first token 100.5, last token 102.5, complete 102.6, 101 output tokens.
    rec = synth()
    assert rec["ttft_s"] == pytest.approx(0.5)
    assert rec["e2e_latency_s"] == pytest.approx(2.6)
    assert rec["decode_time_s"] == pytest.approx(2.0)
    assert rec["decode_tokens_per_s"] == pytest.approx(50.0)  # (101 - 1) / 2.0
    assert rec["usable_for_analysis"] is True and rec["validation_errors"] == []


def test_one_token_output_is_usable_but_has_no_decode_rate():
    rec = synth(completion_tokens=1, t_first=100.5, t_last=100.5, t_complete=100.6)
    assert rec["decode_tokens_per_s"] is None and rec["usable_for_analysis"] is True


def test_zero_token_output_is_not_usable():
    rec = synth(completion_tokens=0)
    assert "zero_output_tokens" in rec["validation_errors"] and rec["usable_for_analysis"] is False


def test_missing_usage_is_not_usable():
    rec = synth(prompt_tokens=None, completion_tokens=None)
    assert "missing_engine_usage" in rec["validation_errors"] and rec["usable_for_analysis"] is False


def test_failed_record_is_never_a_success_sample():
    rec = synth(status="error", error_category="timeout")
    assert rec["usable_for_analysis"] is False and rec["ttft_s"] is None and rec["decode_tokens_per_s"] is None


# ----------------------------------------------------------------------------- aggregation
def three_requests():
    # (start, finish, output tokens): 1.0 s + 1.5 s + 2.0 s busy, tokens 100 + 50 + 100.
    return [
        synth("a", 0.0, 0.2, 0.9, 1.0, completion_tokens=100),
        synth("b", 2.0, 2.3, 3.4, 3.5, completion_tokens=50),
        synth("c", 4.0, 4.4, 5.9, 6.0, completion_tokens=100),
    ]


def test_aggregate_throughput_matches_hand_calculation():
    agg = mt.aggregate_throughput(three_requests())
    assert agg["label"] == "measured" and agg["n_requests"] == 3 and agg["output_tokens_total"] == 250
    assert agg["span_s"] == pytest.approx(6.0)
    assert agg["output_tokens_per_s_over_span"] == pytest.approx(250 / 6.0)
    assert agg["busy_s"] == pytest.approx(4.5)
    assert agg["output_tokens_per_s_over_busy_time"] == pytest.approx(250 / 4.5)


def test_failures_and_warmups_do_not_change_throughput_but_are_counted():
    base = three_requests()
    extra = [synth("w", 50.0, 50.1, 50.5, 50.6, completion_tokens=8, is_warmup=True),
             synth("t", 60.0, error_category="timeout", status="error", completion_tokens=None),
             synth("h", 70.0, error_category="http_error", status="error", completion_tokens=None),
             synth("m", 80.0, 80.1, 80.5, 80.6, prompt_tokens=None, completion_tokens=None)]
    assert mt.aggregate_throughput(base + extra) == mt.aggregate_throughput(base)
    s = mt.summarize_run(base + extra)
    assert s["counts"] == {"total": 7, "warmup": 1, "measured": 6, "usable": 3, "failed": 2,
                           "ok_but_invalid": 1}
    assert s["failures_by_category"] == {"timeout": 1, "http_error": 1}
    assert s["ok_but_invalid_request_ids"] == ["m"]
    assert s["ttft_s"]["n"] == 3


def test_summary_of_a_run_with_no_usable_records_is_explicit():
    s = mt.summarize_run([synth(status="error", error_category="timeout")])
    assert s["aggregate"]["label"] == "unavailable" and s["ttft_s"]["n"] == 0


def test_summary_statistics_use_only_usable_records():
    s = mt.summarize_run(three_requests())
    assert s["ttft_s"]["n"] == 3 and s["ttft_s"]["min"] == pytest.approx(0.2)
    assert s["ttft_s"]["p95"] is None  # n=3 cannot establish a tail
    assert s["decode_tokens_per_s"]["n"] == 3


# ----------------------------------------------------------------------------- run validation
def copy_real_run(tmp_path):
    dst = tmp_path / "run"
    shutil.copytree(REAL_RUN, dst)
    return dst


def rewrite(run, mutate):
    recs = mt.load_records(run / "requests.jsonl")
    mutate(recs)
    (run / "requests.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))


@pytest.mark.skipif(not REAL_RUN.exists(), reason="committed real run not present")
def test_the_committed_real_run_validates_cleanly():
    assert mt.validate_run_dir(REAL_RUN) == []
    s = mt.summarize_run(mt.load_records(REAL_RUN / "requests.jsonl"))
    assert s["counts"]["usable"] == 1 and s["aggregate"]["output_tokens_total"] == 128


@pytest.mark.skipif(not REAL_RUN.exists(), reason="committed real run not present")
def test_validation_catches_tampering(tmp_path):
    run = copy_real_run(tmp_path)
    rewrite(run, lambda recs: recs[1].update(usable_for_analysis=False))
    assert any("should be True" in p for p in mt.validate_run_dir(run))

    run2 = tmp_path / "run2"
    shutil.copytree(REAL_RUN, run2)
    rewrite(run2, lambda recs: recs[1].update(ttft_s=9.99))
    assert any("ttft_s cannot be recomputed" in p for p in mt.validate_run_dir(run2))

    run3 = tmp_path / "run3"
    shutil.copytree(REAL_RUN, run3)
    rewrite(run3, lambda recs: recs[1].update(config_hash="deadbeef"))
    assert any("config_hash differs" in p for p in mt.validate_run_dir(run3))

    run4 = tmp_path / "run4"
    shutil.copytree(REAL_RUN, run4)
    rewrite(run4, lambda recs: recs.append(dict(recs[1])))
    problems = mt.validate_run_dir(run4)
    assert any("duplicate request_id" in p for p in problems)
    assert any("counts[requests]" in p for p in problems)

    run5 = tmp_path / "run5"
    shutil.copytree(REAL_RUN, run5)
    rewrite(run5, lambda recs: recs[1].update(validation_errors=["made_up"]))
    assert any("do not match a recomputation" in p for p in mt.validate_run_dir(run5))


@pytest.mark.skipif(not REAL_RUN.exists(), reason="committed real run not present")
def test_validation_catches_a_failed_request_marked_usable(tmp_path):
    run = copy_real_run(tmp_path)

    def mutate(recs):
        recs[1].update(status="error", error_category="timeout")

    rewrite(run, mutate)
    assert any("failed request marked usable" in p for p in mt.validate_run_dir(run))


@pytest.mark.skipif(not REAL_RUN.exists(), reason="committed real run not present")
def test_validation_catches_manifest_config_tampering(tmp_path):
    run = copy_real_run(tmp_path)
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["config"]["sampling"]["max_tokens"] = 9999
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    assert any("config_hash does not match" in p for p in mt.validate_run_dir(run))


def test_validation_of_a_missing_run_reports_unreadable_files(tmp_path):
    assert any("cannot read run files" in p for p in mt.validate_run_dir(tmp_path / "nope"))
