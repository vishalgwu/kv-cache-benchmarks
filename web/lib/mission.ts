// Deterministic TypeScript port of ../src/ — same rules, same incidents,
// same 0.85 gate, so the web demo and the Python CLI agree.
export type Incident = {
  id: string;
  title: string;
  telemetry: string[];
  config: Record<string, string>;
  expected: string;
};

export type AgentName =
  | "Triage"
  | "LogAnalyst"
  | "KnowledgeRetriever"
  | "SpecComparator"
  | "Planner"
  | "Evaluator";

export type Step = { agent: AgentName; output: unknown };
export type Citation = { id: string; title: string };
export type Trace = {
  incident_id: string;
  title: string;
  steps: Step[];
  triage: { severity: string; safe_to_diagnose: boolean; reason: string };
  anomalies: string[];
  citations: Citation[];
  mismatches: string[];
  planner: { action: string; risk: number; proposal: string };
  gate: string;
  status: string;
  latency_s: number;
  simulated_tokens: number;
  evaluation: {
    citations_found: number;
    grounded: boolean;
    abstention_correct: boolean;
    gate_correct: boolean;
    pass: boolean;
  };
  note?: string;
};

export const RISK_GATE = 0.85;
export const BUDGET_TOKENS = 5000;

export const AGENTS: { name: AgentName; role: string; detail: string }[] = [
  { name: "Triage", role: "Severity + safety screen", detail: "Deterministic rules first. Blocks diagnosis on a telemetry gap." },
  { name: "LogAnalyst", role: "Telemetry anomaly detection", detail: "Parses KV, latency, and CXL retrain signals against demo thresholds." },
  { name: "KnowledgeRetriever", role: "Approved-source retrieval", detail: "Public Astera product knowledge only. No model memory, no web search." },
  { name: "SpecComparator", role: "Config vs. spec diff", detail: "Lists mismatches between the current config and the retrieved spec." },
  { name: "Planner", role: "Lowest-risk next action", detail: "Proposes one action with a risk score. Risky actions stop at the gate." },
  { name: "Evaluator", role: "Independent check", detail: "Grounded? Cited? Abstention correct? Gate correct?" },
];

const KB = [
  { id: "KB-01", title: "Leo X-Series — fabric-attached memory for agentic AI", text: "Leo X-Series fabric attached memory tier AI inference KV cache agentic workloads Scorpio fabric switches", source: "asteralabs.com/products/leo-smart-memory-controllers/" },
  { id: "KB-02", title: "Leo 2 E-Series — CPU-attached memory expansion", text: "Leo 2 E-Series CXL 3.2 memory expansion PCIe 6 server add DRAM capacity without CPU socket DDR4 DDR5", source: "asteralabs.com/products/leo-smart-memory-controllers/" },
  { id: "KB-03", title: "Leo 2 P-Series — pooled and shared memory", text: "Leo 2 P-Series pooled shared memory hosts rack scale dynamic capacity management disaggregated", source: "asteralabs.com/products/leo-smart-memory-controllers/" },
  { id: "KB-04", title: "COSMOS — fleet telemetry and management", text: "COSMOS connectivity system management optimisation software fleet cloud scale telemetry management resources", source: "asteralabs.com/" },
  { id: "KB-05", title: "Rack-scale AI and long-context inference", text: "long context multi-turn agentic inference KV cache agent context pressure accelerator memory offload fabric attached tier", source: "asteralabs.com/products/leo-smart-memory-controllers/" },
  { id: "KB-06", title: "Intelligent Connectivity Platform", text: "PCIe CXL Ethernet connectivity platform COSMOS scale-up scale-out link physical fleet management layer", source: "asteralabs.com/" },
];

export const INCIDENTS: Incident[] = [
  {
    id: "leo-kv-pressure-01",
    title: "KV-cache pressure on a long-context agentic inference rack",
    telemetry: [
      "host rack-07 gpu-03 kv_cache_occupancy_pct=92",
      "host rack-07 gpu-03 time_to_first_token_ms=1480 baseline_ms=620",
      "host rack-07 gpu-03 tokens_per_sec=41 baseline=58",
      "host rack-07 pooled_memory_latency_us=940 baseline_us=310",
      "host rack-07 context_length_tokens=128000 sessions_active=14",
    ],
    config: { kv_offload: "disabled", memory_tier: "cpu_attached_only", product: "Leo 2 E-Series" },
    expected: "KV-cache pressure; propose evaluating fabric-attached KV offload (high-risk, needs approval)",
  },
  {
    id: "leo-link-errors-02",
    title: "Repeated CXL link retrains on one host",
    telemetry: [
      "host rack-12 cxl_link_retrains_per_hour=27",
      "host rack-12 cxl_link_errors=142",
      "host rack-12 peer_hosts_retrains_per_hour=1",
      "host rack-12 telemetry_age_min=1",
    ],
    config: { memory_tier: "pooled", product: "Leo 2 P-Series" },
    expected: "Isolated host link problem, not fleet-wide; propose link-layer diagnostics only",
  },
  {
    id: "cosmos-telemetry-gap-03",
    title: "Telemetry gap — two hosts silent",
    telemetry: [
      "host rack-03 telemetry_age_min=47",
      "host rack-04 telemetry_age_min=52",
      "host rack-03 last_known_kv_cache_occupancy_pct=unknown",
    ],
    config: { memory_tier: "unknown", product: "unknown" },
    expected: "Abstain on diagnosis for rack-03/rack-04; propose restoring COSMOS telemetry first",
  },
];

function num(telemetry: string[], key: string): number | null {
  for (const line of telemetry) {
    const m = line.match(new RegExp(`${key}=(\\d+)`));
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

function triage(incident: Incident) {
  const gap = num(incident.telemetry, "telemetry_age_min") ?? 0;
  if (gap > 10)
    return { severity: "UNKNOWN", safe_to_diagnose: false, reason: `Telemetry gap ${gap} min exceeds 10 min demo threshold` };
  const kv = num(incident.telemetry, "kv_cache_occupancy_pct") ?? 0;
  const retrains = num(incident.telemetry, "cxl_link_retrains_per_hour") ?? 0;
  const severity = kv >= 90 || retrains >= 20 ? "HIGH" : kv >= 85 || retrains >= 5 ? "MEDIUM" : "LOW";
  return { severity, safe_to_diagnose: true, reason: `kv=${kv}% retrains/hr=${retrains}` };
}

function logAnalyst(incident: Incident) {
  const anomalies: string[] = [];
  const kv = num(incident.telemetry, "kv_cache_occupancy_pct");
  if (kv && kv >= 85) anomalies.push(`KV-cache occupancy ${kv}% (demo warning threshold 85%)`);
  const ttft = num(incident.telemetry, "time_to_first_token_ms");
  if (ttft && ttft > 1000) anomalies.push(`Time-to-first-token ${ttft} ms, well above 620 ms baseline in incident`);
  const retrains = num(incident.telemetry, "cxl_link_retrains_per_hour");
  if (retrains && retrains >= 5) {
    const peer = num(incident.telemetry, "peer_hosts_retrains_per_hour");
    const scope = peer !== null && peer <= 2 ? "isolated to one host" : "fleet-wide pattern";
    anomalies.push(`CXL retrains ${retrains}/hr, ${scope}`);
  }
  return { anomalies };
}

function retrieve(query: string): Citation[] {
  const q = new Set(query.toLowerCase().match(/[a-z0-9]+/g) ?? []);
  return KB.map((s) => {
    const words = new Set(s.text.toLowerCase().match(/[a-z0-9]+/g) ?? []);
    let score = 0;
    q.forEach((w) => { if (words.has(w)) score++; });
    return { score, s };
  })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((x) => ({ id: x.s.id, title: x.s.title }));
}

function specComparator(incident: Incident) {
  const mismatches: string[] = [];
  const joined = incident.telemetry.join(" ");
  if (incident.config.kv_offload === "disabled" && joined.includes("kv_cache"))
    mismatches.push("KV offload disabled while KV-cache pressure is present");
  if (incident.config.memory_tier === "cpu_attached_only" && incident.config.product === "Leo 2 E-Series")
    mismatches.push("CPU-attached tier only; fabric-attached tier not evaluated for this workload");
  return { mismatches };
}

function planner(triageOut: ReturnType<typeof triage>, analystOut: ReturnType<typeof logAnalyst>) {
  if (!triageOut.safe_to_diagnose)
    return { action: "restore_telemetry", risk: 0.2, proposal: "Restore COSMOS telemetry for silent hosts first; do not diagnose without data." };
  if (analystOut.anomalies.some((a) => a.includes("KV-cache")))
    return { action: "change_memory_tier", risk: 0.92, proposal: "Evaluate fabric-attached KV-cache offload for this workload in a test rack before any production change." };
  if (analystOut.anomalies.some((a) => a.includes("CXL retrains")))
    return { action: "run_link_diagnostics", risk: 0.35, proposal: "Run link-layer diagnostics on the affected host only; do not change fleet configuration." };
  return { action: "monitor", risk: 0.1, proposal: "Continue monitoring; no anomaly meets the action threshold." };
}

export function runMission(incident: Incident, approved = false): Trace {
  const started = performance.now();
  const triageOut = triage(incident);
  const analystOut = logAnalyst(incident);
  const citations = retrieve(incident.title + " " + analystOut.anomalies.join(" "));
  const comparatorOut = specComparator(incident);
  const plannerOut = planner(triageOut, analystOut);
  const writeActions = ["deploy", "reconfigure", "restart_host", "change_memory_tier"];
  const gate = writeActions.includes(plannerOut.action) || plannerOut.risk >= RISK_GATE
    ? approved ? "APPROVED" : "WAITING_APPROVAL"
    : "APPROVED_LOW_RISK";
  const steps: Step[] = [
    { agent: "Triage", output: triageOut },
    { agent: "LogAnalyst", output: analystOut },
    { agent: "KnowledgeRetriever", output: citations.map((c) => c.id) },
    { agent: "SpecComparator", output: comparatorOut },
    { agent: "Planner", output: plannerOut },
  ];
  const shouldAbstain = !triageOut.safe_to_diagnose;
  const didAbstain = plannerOut.action === "restore_telemetry";
  const gateCorrect = plannerOut.risk >= RISK_GATE ? gate === "WAITING_APPROVAL" || gate === "APPROVED" : true;
  const evaluation = {
    citations_found: citations.length,
    grounded: citations.length >= 1 || shouldAbstain,
    abstention_correct: shouldAbstain === didAbstain,
    gate_correct: gateCorrect,
    pass: (citations.length >= 1 || shouldAbstain) && shouldAbstain === didAbstain && gateCorrect,
  };
  steps.push({ agent: "Evaluator", output: evaluation });
  const status = gate === "WAITING_APPROVAL" ? "WAITING_APPROVAL" : "COMPLETE";
  return {
    incident_id: incident.id, title: incident.title, steps, triage: triageOut,
    anomalies: analystOut.anomalies, citations, mismatches: comparatorOut.mismatches,
    planner: plannerOut, gate, status,
    latency_s: Math.round(((performance.now() - started) / 1000) * 1000) / 1000,
    simulated_tokens: 500, evaluation,
    note: status === "WAITING_APPROVAL" ? "High-risk proposal held at the gate. Approve or reject it." : undefined,
  };
}

export function reportMarkdown(trace: Trace): string {
  const lines = [
    `# Mission report: ${trace.title}`, "",
    `Incident: ${trace.incident_id} | Status: ${trace.status} | Gate: ${trace.gate}`,
    `Simulated tokens: ${trace.simulated_tokens} / ${BUDGET_TOKENS} | Risk gate: ${RISK_GATE}`, "",
    "## Trace",
    ...trace.steps.map((s) => `- **${s.agent}**: ${JSON.stringify(s.output)}`), "",
    "## Citations", JSON.stringify(trace.citations, null, 2), "",
    "## Evaluation", JSON.stringify(trace.evaluation, null, 2), "",
    "## Planner proposal", trace.planner.proposal, "",
  ];
  return lines.join("\n");
}
