#!/usr/bin/env python3
"""CLI: python run_mission.py --incident leo-kv-pressure-01 [--approve]"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.orchestrator import run_mission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--incident", help="incident id")
    p.add_argument("--approve", action="store_true", help="clear the approval gate")
    p.add_argument("--list", action="store_true", help="list incidents")
    args = p.parse_args()
    incidents = {i["id"]: i for i in json.loads((ROOT / "data" / "incidents.json").read_text())}
    if args.list or not args.incident:
        print("Incidents:")
        for i in incidents.values():
            print(f"  {i['id']}: {i['title']}")
        return
    if args.incident not in incidents:
        print(f"Unknown incident: {args.incident}"); sys.exit(1)
    trace = run_mission(incidents[args.incident], approved=args.approve)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / "mission_trace.json").write_text(json.dumps(trace, indent=2))
    lines = [f"# Mission report: {trace['title']}", "",
             f"Incident: {trace['incident_id']} | Status: {trace['status']} | Gate: {trace['gate']}",
             f"Latency: {trace['latency_s']}s | Simulated tokens: {trace['simulated_tokens']}", "",
             "## Trace"]
    for s in trace["steps"]:
        lines.append(f"- **{s['agent']}**: {s['output']}")
    lines += ["", "## Citations", str(trace["citations"]), "",
              "## Evaluation", str(trace["evaluation"]), ""]
    if trace.get("note"):
        lines.append(trace["note"])
    (out / "mission_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nSaved: output/mission_report.md and output/mission_trace.json")


if __name__ == "__main__":
    main()
