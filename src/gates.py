"""Governance gates. Deterministic, model-independent.

- RISK_GATE = 0.85 (mirrors the mission-control pattern from the X posts,
  but here it is an explicit, testable rule, not a marketing claim).
- Any action in WRITE_ACTIONS, or with risk >= gate, waits for approval.
- Budget and step caps hard-stop a run and label the result partial.
"""

RISK_GATE = 0.85
WRITE_ACTIONS = {"deploy", "reconfigure", "restart_host", "change_memory_tier"}


def gate_decision(action, risk, approved=False):
    if action in WRITE_ACTIONS or risk >= RISK_GATE:
        return "APPROVED" if approved else "WAITING_APPROVAL"
    return "APPROVED_LOW_RISK"


class Budget:
    def __init__(self, max_tokens=5000, max_steps=12):
        self.max_tokens, self.max_steps = max_tokens, max_steps
        self.tokens, self.steps = 0, 0

    def charge(self, tokens=100, steps=1):
        self.tokens += tokens
        self.steps += steps
        if self.tokens > self.max_tokens or self.steps > self.max_steps:
            return False
        return True
