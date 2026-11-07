"""Deterministic agent stages with typed handoffs.

This is deliberately model-free so the demo runs with no key and its
behaviour is reproducible. Replace a stage's reasoning with an LLM call
via LLMClient below — the handoff schema, gates, and evaluator stay the same.
"""
import re


class LLMClient:
    """Stub. Implement call() with your provider; do not put keys in code."""

    def call(self, prompt):
        raise NotImplementedError(
            "LLMClient.call is a stub. Set up your provider here, or keep "
            "the deterministic stages for the no-key demo."
        )


def _num(telemetry, key):
    for line in telemetry:
        m = re.search(rf"{key}=(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def triage(incident):
    """Safety screen first: telemetry gap or extreme severity before any diagnosis."""
    telemetry = incident["telemetry"]
    gap = _num(telemetry, "telemetry_age_min") or 0
    if gap > 10:
        return {"severity": "UNKNOWN", "safe_to_diagnose": False,
                "reason": f"Telemetry gap {gap} min exceeds 10 min demo threshold"}
    kv = _num(telemetry, "kv_cache_occupancy_pct") or 0
    retrains = _num(telemetry, "cxl_link_retrains_per_hour") or 0
    severity = "HIGH" if kv >= 90 or retrains >= 20 else "MEDIUM" if kv >= 85 or retrains >= 5 else "LOW"
    return {"severity": severity, "safe_to_diagnose": True, "reason": f"kv={kv}% retrains/hr={retrains}"}


def log_analyst(incident):
    anomalies = []
    kv = _num(incident["telemetry"], "kv_cache_occupancy_pct")
    if kv and kv >= 85:
        anomalies.append(f"KV-cache occupancy {kv}% (demo warning threshold 85%)")
    ttft = _num(incident["telemetry"], "time_to_first_token_ms")
    if ttft and ttft > 1000:
        anomalies.append(f"Time-to-first-token {ttft} ms, well above 620 ms baseline in incident")
    retrains = _num(incident["telemetry"], "cxl_link_retrains_per_hour")
    if retrains and retrains >= 5:
        peer = _num(incident["telemetry"], "peer_hosts_retrains_per_hour")
        scope = "isolated to one host" if peer is not None and peer <= 2 else "fleet-wide pattern"
        anomalies.append(f"CXL retrains {retrains}/hr, {scope}")
    return {"anomalies": anomalies}


def spec_comparator(incident):
    cfg = incident.get("config", {})
    mismatches = []
    if cfg.get("kv_offload") == "disabled" and "kv_cache" in " ".join(incident["telemetry"]):
        mismatches.append("KV offload disabled while KV-cache pressure is present")
    if cfg.get("memory_tier") == "cpu_attached_only" and cfg.get("product") == "Leo 2 E-Series":
        mismatches.append("CPU-attached tier only; fabric-attached tier not evaluated for this workload")
    return {"mismatches": mismatches}


def planner(triage_out, analyst_out, comparator_out):
    if not triage_out["safe_to_diagnose"]:
        return {"action": "restore_telemetry", "risk": 0.2,
                "proposal": "Restore COSMOS telemetry for silent hosts first; do not diagnose without data."}
    if any("KV-cache" in a for a in analyst_out["anomalies"]):
        return {"action": "change_memory_tier", "risk": 0.92,
                "proposal": "Evaluate fabric-attached KV-cache offload for this workload in a test rack before any production change."}
    if any("CXL retrains" in a for a in analyst_out["anomalies"]):
        return {"action": "run_link_diagnostics", "risk": 0.35,
                "proposal": "Run link-layer diagnostics on the affected host only; do not change fleet configuration."}
    return {"action": "monitor", "risk": 0.1, "proposal": "Continue monitoring; no anomaly meets the action threshold."}
