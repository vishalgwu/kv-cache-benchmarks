# Astera Mission Control — Next.js dashboard

Independent portfolio concept. Not affiliated with Astera Labs.
All incident data is synthetic. Knowledge snippets are paraphrased from
Astera Labs public product pages only.

## Run

```bash
cd web
npm install
npm run dev
# open http://localhost:3000
```

Deploy the `web` folder to Vercel as-is. No API key, no backend, no database:
the mission engine in `lib/mission.ts` is a deterministic TypeScript port of
the Python engine in `../src/`, so the demo is reproducible everywhere.

## How to demo it in 2 minutes

1. Pick the KV-cache pressure incident and click Run mission.
2. Watch the agents run in order, with citations and cost in the top bar.
3. The high-risk proposal stops at the approval gate. Click Approve to clear
   it, or Reject to block it.
4. Switch to the telemetry-gap incident to show the correct behaviour when
   evidence is missing: the system abstains and asks to restore telemetry
   first, instead of inventing a diagnosis.
5. Download the mission report from the Artifacts panel.

## What to say if a hiring manager asks

- "The agents are deterministic in this demo so the trace is reproducible.
  The LLM plug-in point is in the Python version (`src/agents.py`), and the
  gates, retrieval, and evaluator do not change when a model is added."
- "Nothing here deploys or reconfigures a real system. Risky proposals stop
  for a human. That is the governance part of the Applied AI role."
