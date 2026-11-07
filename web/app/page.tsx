"use client";

import { useEffect, useRef, useState } from "react";
import {
  AGENTS, BUDGET_TOKENS, INCIDENTS, RISK_GATE, reportMarkdown, runMission,
  type Trace,
} from "../lib/mission";

export default function Page() {
  const [incidentId, setIncidentId] = useState(INCIDENTS[0].id);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [visible, setVisible] = useState(0);
  const [running, setRunning] = useState(false);
  const [decision, setDecision] = useState<"approved" | "rejected" | null>(null);
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const incident = INCIDENTS.find((i) => i.id === incidentId) ?? INCIDENTS[0];

  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);

  function run() {
    if (timer.current) clearInterval(timer.current);
    const t = runMission(incident, false);
    setTrace(t); setDecision(null); setVisible(0); setRunning(true);
    timer.current = setInterval(() => {
      setVisible((v) => {
        if (v + 1 >= t.steps.length) {
          if (timer.current) clearInterval(timer.current);
          setRunning(false);
          return t.steps.length;
        }
        return v + 1;
      });
    }, 650);
  }

  function approve() {
    const t = runMission(incident, true);
    setTrace(t); setVisible(t.steps.length); setDecision("approved"); setRunning(false);
  }

  function reject() {
    setDecision("rejected");
  }

  function downloadReport() {
    if (!trace) return;
    const blob = new Blob([reportMarkdown(trace)], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "mission_report.md"; a.click();
    URL.revokeObjectURL(url);
  }

  async function copyTrace() {
    if (!trace) return;
    await navigator.clipboard.writeText(JSON.stringify(trace, null, 2));
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  }

  const status = decision === "rejected" ? "REJECTED" : trace?.status ?? "IDLE";
  const gate = decision === "rejected" ? "REJECTED_BY_HUMAN" : trace?.gate ?? "—";
  const shownSteps = trace ? trace.steps.slice(0, visible) : [];

  return (
    <div className="wrap">
      <header className="top">
        <h1>Astera Mission Control</h1>
        <p>Engineering Diagnostics Copilot — independent portfolio demo, synthetic data only, not affiliated with Astera Labs. An engineer submits a telemetry incident; agents triage, analyse, retrieve, compare, and propose, with a human gate on risky actions.</p>
      </header>

      <div className="metrics">
        <div className="metric"><span>Goal</span><strong>Diagnose incident</strong></div>
        <div className="metric"><span>Status</span><strong>{status}</strong></div>
        <div className="metric"><span>Approval gate</span><strong>{gate}</strong></div>
        <div className="metric"><span>Risk gate</span><strong>{RISK_GATE}</strong></div>
        <div className="metric"><span>Simulated tokens</span><strong>{trace ? trace.simulated_tokens : 0} / {BUDGET_TOKENS}</strong></div>
        <div className="metric"><span>Evaluation</span><strong>{trace && visible >= trace.steps.length ? (trace.evaluation.pass ? "PASS" : "CHECK") : "—"}</strong></div>
      </div>

      <h2>Pick an incident</h2>
      <div className="incidents">
        {INCIDENTS.map((i) => (
          <button key={i.id} className={`incident ${i.id === incidentId ? "selected" : ""}`} onClick={() => { setIncidentId(i.id); setTrace(null); setVisible(0); setDecision(null); }}>
            <strong>{i.title}</strong>
            <small>{i.expected}</small>
          </button>
        ))}
      </div>

      <div className="row">
        <button className="primary" onClick={run} disabled={running}>{running ? "Agents running…" : "Run mission"}</button>
        {trace && <button onClick={downloadReport}>Download mission report</button>}
        {trace && <button onClick={copyTrace}>{copied ? "Copied!" : "Copy trace JSON"}</button>}
      </div>

      <div className="grid">
        <aside>
          <div className="card">
            <h2>Agent roster</h2>
            {AGENTS.map((a, idx) => {
              const st = !trace ? "idle" : idx < visible ? "done" : running && idx === visible ? "working" : "idle";
              return (
                <div className="agent" key={a.name}>
                  <span className={`dot ${st}`} />
                  <div><b>{a.name}</b><span>{a.role}</span><br /><span>{a.detail}</span></div>
                </div>
              );
            })}
          </div>
          <div className="card">
            <h2>Input telemetry</h2>
            <div className="telemetry">{incident.telemetry.map((l) => <div key={l}>{l}</div>)}</div>
            <h3>Config</h3>
            <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(incident.config, null, 2)}</pre>
          </div>
        </aside>

        <main>
          <div className="card">
            <h2>Live trace — {incident.title}</h2>
            {!trace && <p className="muted">Click Run mission to watch the agents work in order. No API key is needed; the engine is deterministic so the trace is reproducible.</p>}
            {running && <p><span className="badge warn">Running</span> Agents are working through the pipeline…</p>}
            {shownSteps.map((s, idx) => (
              <div className="step done" key={`${s.agent}-${idx}`}>
                <strong>{idx + 1}. {s.agent}</strong>
                <pre>{JSON.stringify(s.output, null, 2)}</pre>
              </div>
            ))}
            {trace && visible >= trace.steps.length && trace.planner && (
              <p><strong>Proposal:</strong> {trace.planner.proposal} <span className="badge">risk {trace.planner.risk}</span></p>
            )}
          </div>
        </main>

        <aside>
          <div className="card">
            <h2>Approval queue</h2>
            {!trace && <p className="muted">Risky proposals appear here. Nothing deploys or reconfigures without a human.</p>}
            {trace && trace.gate === "WAITING_APPROVAL" && decision !== "approved" && decision !== "rejected" && (
              <>
                <p><span className="badge warn">Waiting</span> {trace.planner.proposal}</p>
                <p className="muted">Risk {trace.planner.risk} is above the {RISK_GATE} gate. This is a proposal for a test rack, not a production change.</p>
                <div className="row">
                  <button className="primary" onClick={approve}>Approve</button>
                  <button className="danger" onClick={reject}>Reject</button>
                </div>
              </>
            )}
            {decision === "approved" && <p><span className="badge ok">Approved</span> Proposal cleared by a human. Still a proposal, not an executed change.</p>}
            {decision === "rejected" && <p><span className="badge bad">Rejected</span> Proposal blocked. The mission ends without the risky action.</p>}
            {trace && trace.gate !== "WAITING_APPROVAL" && <p><span className="badge ok">{trace.gate}</span> No high-risk action in this mission.</p>}
            {trace?.note && decision === null && <p className="muted">{trace.note}</p>}
          </div>

          <div className="card">
            <h2>Citations</h2>
            {!trace && <p className="muted">Retrieved only from the approved public knowledge base.</p>}
            {trace?.citations.map((c) => <p key={c.id}><span className="badge">{c.id}</span> {c.title}</p>)}
            {trace && <><h3>Spec mismatches</h3>{trace.mismatches.length ? trace.mismatches.map((m) => <p key={m}>{m}</p>) : <p className="muted">None found.</p>}</>}
          </div>

          <div className="card">
            <h2>Evaluation</h2>
            {!trace && <p className="muted">Scores after the run finishes.</p>}
            {trace && visible >= trace.steps.length && (
              <>
                <p>Citations found: <strong>{trace.evaluation.citations_found}</strong></p>
                <p>Grounded: <strong>{String(trace.evaluation.grounded)}</strong></p>
                <p>Abstention correct: <strong>{String(trace.evaluation.abstention_correct)}</strong></p>
                <p>Gate correct: <strong>{String(trace.evaluation.gate_correct)}</strong></p>
                <p>Overall: <span className={`badge ${trace.evaluation.pass ? "ok" : "bad"}`}>{trace.evaluation.pass ? "PASS" : "CHECK"}</span></p>
              </>
            )}
          </div>

          <div className="card">
            <h2>Artifacts</h2>
            <p className="muted">Download the markdown report, or copy the full trace JSON to attach to a pull request or a hiring-manager message.</p>
            <div className="row">
              <button onClick={downloadReport} disabled={!trace}>Download report</button>
              <button onClick={copyTrace} disabled={!trace}>Copy trace</button>
            </div>
          </div>
        </aside>
      </div>

      <footer>
        Independent portfolio concept. Synthetic telemetry only. Knowledge paraphrased from Astera Labs public product pages. No real system is changed by this demo, and no Astera affiliation is claimed.
      </footer>
    </div>
  );
}
