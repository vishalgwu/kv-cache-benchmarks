# Astera Mission Control — Engineering Diagnostics Copilot

**Independent portfolio concept. Not affiliated with Astera Labs. All incident data is synthetic. Knowledge snippets are paraphrased from Astera Labs public product pages only.**

A hiring-manager-ready demo of exactly what Astera's Applied AI roles ask for:
tool use, handoffs, exception paths, permissions and governance, evaluation,
failure investigation, diagnostics, and documentation/spec comparison — with a
live mission-control trace, not just a chatbot answer.

## The 30-second pitch

An engineer pastes a rack telemetry incident. Mission Control does not just answer.
It triages severity, analyses the log, retrieves only from an approved knowledge
base, compares the config against the spec, proposes a fix, and — if the fix is
risky — stops for human approval. Every step is logged with citations, cost, and
an evaluation score. If evidence is insufficient, it abstains instead of inventing
a diagnosis.

## Run it (no API key, no install beyond Python 3.11+)

```bash
cd astera-mission-control
python run_mission.py --incident leo-kv-pressure-01
python run_mission.py --incident leo-kv-pressure-01 --approve   # clears the high-risk gate
python run_mission.py --incident cosmos-telemetry-gap-03       # abstention demo
python run_mission.py --list
```

Output: `output/mission_report.md` + `output/mission_trace.json`
Open `docs/dashboard.html` in a browser after a run to see the trace rendered
as a mission-control board (it reads the latest trace JSON if you serve the
folder, or just read the markdown report — the report is the portable version).

## What it proves, mapped to the Applied AI postings

| Astera asks for | This project shows it |
|---|---|
| Agentic workflows, tool use, handoffs | Typed LangGraph-style pipeline: triage → log analyst → retriever → spec comparator → planner → evaluator, each with a typed handoff |
| Retrieval over approved knowledge | BM25 retriever over `data/knowledge_base.md` only, with citations per claim |
| Exception paths, permissions, governance | Risk gate at 0.85: write/deploy-class actions wait for approval; budget and step caps hard-stop a run |
| Evaluation, failure investigation | `src/evaluator.py` scores grounding, citation coverage, abstention correctness; `tests/` cover the failure modes |
| Diagnostics and spec comparison | Log anomaly detection + config-vs-spec diff for Leo/COSMOS-style incidents |
| Measurable outcomes | Every run reports latency, simulated token cost, citations found, gate outcome |

## Agents

- **Triage** — severity + safety screen (deterministic rules first, model second)
- **Log Analyst** — parses telemetry, flags anomalies against thresholds
- **Knowledge Retriever** — approved-source retrieval only, no model memory
- **Spec Comparator** — config vs. spec thresholds, lists mismatches
- **Planner** — proposes the lowest-risk next test/fix, with a risk score
- **Evaluator** — independent check: grounded? cited? should it have abstained?

## Safety and honesty rules in this repo

1. Synthetic data only. No real Astera telemetry, specs, or internal documents.
2. No Astera logo, branding, or claim of employment/affiliation.
3. No real money, production system, or deployment action. "Deploy" actions in
   the demo are proposals that stop at the approval gate.
4. Numbers in reports are measured from the demo run (latency, steps, citations),
   or explicitly labelled as simulated token cost — never presented as production SLAs.

## Swap in a real LLM later

`src/agents.py` has a `LLMClient` stub. Set `OPENAI_API_KEY` and implement the
two stub methods to replace the deterministic planner/retriever reasoning with a
model call. The gates, retrieval, evaluator, and trace format do not change —
that is the point. The system is safe with or without a model.

## Next.js dashboard

The live dashboard is in `web/` (Next.js, no API key, no backend). Run it with
`cd web && npm install && npm run dev`, then open http://localhost:3000.
It is a TypeScript port of this same Python engine, so the web trace and the
CLI trace agree.

## Files

- `run_mission.py` — CLI entry point (in this folder, next to README.md)
- `src/` — orchestrator, agents, retrieval, gates, evaluator
- `data/incidents.json` — 3 synthetic incidents
- `data/knowledge_base.md` — approved public-knowledge snippets with sources
- `docs/PITCH_TO_HIRING_MANAGER.md` — short message to send with the repo link
- `docs/dashboard.html` — static mission-control board (reads the trace JSON)
- `tests/test_mission.py` — offline checks, no keys needed
