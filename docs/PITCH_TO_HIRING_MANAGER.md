# Short message to send with the repo (do not paste Astera branding into it)

Subject: Applied AI portfolio project — diagnostics mission control

Hi [Name],

I built a small, independent portfolio project inspired by Astera's public
work on rack-scale AI and Leo/COSMOS, using synthetic data only:
[repo link].

It is a mission-control dashboard for engineering diagnostics. An engineer
submits a telemetry incident, and a multi-agent pipeline triages it, analyses
the log, retrieves only from an approved knowledge base, compares config
against spec, and proposes a fix — with a human approval gate for risky
actions and an evaluation score on every run. If evidence is missing, it
abstains instead of guessing.

Why I built it this way: your Applied AI postings emphasise tool use,
handoffs, exception paths, permissions/governance, and evaluation with
measurable outcomes. Those are the parts I wanted to demonstrate, not just
a chatbot demo.

Run it with no API key: `python run_mission.py --incident leo-kv-pressure-01`
The three demo incidents cover KV-cache pressure, an isolated link fault,
and a telemetry gap where abstention is the correct answer.

Thanks for looking — happy to walk through the trace, the gate logic, or
where I would plug in a real model next.

— Vishal Fulsundar
M.S. Data Science, George Washington University
GitHub: https://github.com/vishalgwu
