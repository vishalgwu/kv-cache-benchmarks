# Approved knowledge base — public sources only

Paraphrased from Astera Labs public product pages. Use these snippets for
retrieval and citation. Do not invent thresholds that are not listed here.

## [KB-01] Leo X-Series — fabric-attached memory for agentic AI
Leo X-Series creates a fabric-attached memory tier for AI inference, aimed at
KV-cache-heavy agentic workloads. It pairs with Scorpio fabric switches.
Source: https://www.asteralabs.com/products/leo-smart-memory-controllers/

## [KB-02] Leo 2 E-Series — CPU-attached memory expansion
Leo 2 E-Series provides CXL 3.2 memory expansion over PCIe 6, letting a server
add DRAM capacity without another CPU socket. It supports DDR4 and DDR5.
Source: https://www.asteralabs.com/products/leo-smart-memory-controllers/

## [KB-03] Leo 2 P-Series — pooled and shared memory
Leo 2 P-Series enables pooled and shared memory across hosts at rack scale,
with dynamic capacity management. It is the pooled/disaggregated option, not
the single-host expansion option.
Source: https://www.asteralabs.com/products/leo-smart-memory-controllers/

## [KB-04] COSMOS — fleet telemetry and management
COSMOS is Astera's connectivity system management and optimisation software
for managing resources across large fleets at cloud scale. Missing or stale
telemetry therefore limits fleet-wide diagnosis — do not diagnose a host you
have no telemetry for.
Source: https://www.asteralabs.com/

## [KB-05] Rack-scale AI and long-context inference
Long-context, multi-turn agentic inference increases KV-cache and agent-context
pressure on accelerator memory. KV-cache offload to a fabric-attached tier is
the mitigation described for Leo X-Series workloads.
Source: https://www.asteralabs.com/products/leo-smart-memory-controllers/

## [KB-06] Intelligent Connectivity Platform
Astera's platform combines PCIe, CXL, Ethernet and related connectivity
technologies with COSMOS software for scale-up and scale-out connectivity.
A link problem can therefore sit at the physical, link, or fleet-management
layer — check each layer before concluding.
Source: https://www.asteralabs.com/

## Demo thresholds (synthetic, for this portfolio demo only)
- KV-cache occupancy above 85% = pressure warning; above 90% = high severity.
- CXL link retrain count above 5 per hour = investigate; above 20 = high severity.
- Telemetry gap longer than 10 minutes = insufficient evidence for that host.
These thresholds are demo rules written for this repo, not Astera specifications.
