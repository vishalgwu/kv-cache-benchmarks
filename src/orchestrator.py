"""Mission orchestrator: typed handoffs, budget cap, approval gate, trace."""
import time
from . import agents
from .gates import Budget, gate_decision
from .retrieval import load_kb, retrieve
from .evaluator import evaluate


def run_mission(incident, approved=False):
    started = time.time()
    budget = Budget()
    trace = {"incident_id": incident["id"], "title": incident["title"], "steps": []}

    def step(name, out):
        if not budget.charge():
            trace["steps"].append({"agent": name, "output": "STOPPED: budget/step cap exceeded", "partial": True})
            raise SystemExit("Budget cap exceeded — run labelled partial.")
        trace["steps"].append({"agent": name, "output": out})

    triage_out = agents.triage(incident); step("Triage", triage_out)
    analyst_out = agents.log_analyst(incident); step("LogAnalyst", analyst_out)
    kb = load_kb()
    query = incident["title"] + " " + " ".join(analyst_out["anomalies"])
    citations = retrieve(query, kb); step("KnowledgeRetriever", [c["id"] for c in citations])
    comparator_out = agents.spec_comparator(incident); step("SpecComparator", comparator_out)
    planner_out = agents.planner(triage_out, analyst_out, comparator_out); step("Planner", planner_out)

    trace.update({"triage": triage_out, "anomalies": analyst_out["anomalies"],
                  "citations": [{"id": c["id"], "title": c["title"]} for c in citations],
                  "mismatches": comparator_out["mismatches"], "planner": planner_out,
                  "gate": gate_decision(planner_out["action"], planner_out["risk"], approved),
                  "latency_s": round(time.time() - started, 3),
                  "simulated_tokens": budget.tokens})
    trace["evaluation"] = evaluate(trace)
    trace["status"] = "WAITING_APPROVAL" if trace["gate"] == "WAITING_APPROVAL" else "COMPLETE"
    if trace["status"] == "WAITING_APPROVAL":
        trace["note"] = "High-risk proposal held at the gate. Re-run with --approve to clear it."
    return trace
