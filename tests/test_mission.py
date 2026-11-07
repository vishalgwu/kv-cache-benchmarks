import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.orchestrator import run_mission

INC = {i["id"]: i for i in json.loads((Path(__file__).parent.parent / "data" / "incidents.json").read_text())}


def test_kv_incident_stops_at_gate():
    t = run_mission(INC["leo-kv-pressure-01"])
    assert t["gate"] == "WAITING_APPROVAL" and t["status"] == "WAITING_APPROVAL"


def test_kv_incident_clears_with_approval():
    t = run_mission(INC["leo-kv-pressure-01"], approved=True)
    assert t["gate"] == "APPROVED" and t["evaluation"]["pass"]


def test_telemetry_gap_abstains():
    t = run_mission(INC["cosmos-telemetry-gap-03"])
    assert t["planner"]["action"] == "restore_telemetry"
    assert t["evaluation"]["abstention_correct"]


def test_link_incident_is_isolated():
    t = run_mission(INC["leo-link-errors-02"])
    assert any("isolated" in a for a in t["anomalies"])
