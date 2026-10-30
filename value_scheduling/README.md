# Value-driven uplink scheduling for collaborative perception on shared Wi-Fi

A separate project from SCooP. It shares no code, data or claims with
`scoop_*` or `task_oriented_comm/`; the SCooP testbed may serve as one
hardware check at the end, nothing more.

**Research question.** When N agents share one Wi-Fi access point,
which pieces of which agents' perception messages should get airtime in
each frame, so that the team's perception is maximized under a deadline?

**Thesis.** Today the codec decides what to send (per link, assuming a
private pipe) and the AP decides who transmits (by queue size, blind to
content). Moving the decision to the AP, which sees every agent's demand
and already schedules the uplink, and letting it select message units by
marginal task gain per unit airtime with cross-agent redundancy removed,
beats both.

| File | What |
|---|---|
| `PLAN_ORIGINAL.md`, `Focused_Plan_Value-Driven_Uplink_Scheduling.docx` | the plan as uploaded on 30 Oct 2026; the reference text |
| `ASSESSMENT.md` | critical assessment of that plan: what holds, what is at risk, what to change |
| `HARDWARE.md` | the AP problem (a commodity Asus AP exposes no uplink scheduler) and the design that answers it |
| `PLAN.md` | the plan with the amendments from the assessment folded in |
| `references.bib` | references; entries marked `confirm` were not checked against a primary text |

Status: planning. No harness, codec or scheduler code yet.
