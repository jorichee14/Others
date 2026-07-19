# Current Thesis — Task-Conditioned Communication for General Collaborative Perception

*Supersedes the framing in `CORE_TASK.md` (dual-link joint routing) and `RESEARCH_DIRECTION.md`
(certification-first). Those remain valid as the actuator and a feature, respectively — this doc
sets the center. Prepared 2026-07-19; prior-art checked against July-2026 literature.*

---

## Problem

Collaborative perception is moving toward **general, multi-task features** — one shared
representation feeding detection, segmentation, tracking, prediction (Scene Completion, STAMP,
foundation-model CP). Generality collides with the bandwidth budget:

> You cannot prune "task-irrelevant" content when you don't know the task, or must serve several.
> Single-task selection (JigsawComm, SComCP) works precisely by discarding what one task doesn't
> need — that mechanism breaks for general features. A feature's transmit-value is now
> **task-dependent and unknown at send time.**

**Open problem:** budget-constrained, link-robust communication of general collaborative features,
**conditioned on the receiver's task demand** — deciding *what to send and over which link* when the
same feature is critical for planning and useless for segmentation. Existing general-CP work solves
how to *learn/interoperate* the representation; nobody solves *what to transmit under a budget* over it.

## Novelty (honest, bounded)

Not the representation — the **communication decision over it**. In order of defensibility:

1. **Task-demand-conditioned selection (core).** A feature unit's value depends on the receiver's
   *task-demand vector*; a detection-trained importance map mis-ranks for segmentation/prediction,
   silently starving those tasks. No CP selector conditions transmission on *which task* the receiver
   runs. (CoSDH's "supply-demand" is *spatial* — blind spots — not *task* demand.)
2. **Generality-vs-efficiency navigation.** A principled way to stay bandwidth-efficient without
   collapsing to one task — the trade-off Scene Completion / STAMP do not address (no budget, no
   task conditioning on transmission).
3. **Certification as a feature (not the center).** A *per-task preservation guarantee*: dropping
   under the budget keeps every requested task within alpha degradation, so task B is not wrecked to
   save bytes for task A. Makes the multi-task selection trustworthy; a component, not the thesis.

**Occupied — cite and stay clear of:** task-agnostic representation learning (Scene Completion, CoRL
2022), model-agnostic interop (STAMP, ICLR 2025), single-task selection/compression (JigsawComm,
V2X-DSC), foundation-model CP (DriveX 2026). **Wedge:** the budgeted, task-conditioned, link-robust
*decision*.

**Honest ceiling:** a real wedge on a contested, fast-moving frontier (foundation-model CP could
encroach). Defensible only if differentiated from STAMP / Scene Completion on two checks — do they
impose a bandwidth budget? do they condition transmission on task demand? (abstracts say no to both;
verify in full text). Their axis is *representation*; yours is *communication under constraint*.

## Approach

- **Representation (borrowed, not the contribution):** a general / task-agnostic shared feature
  (STAMP-style protocol domain or a foundation feature).
- **Per-task value:** for each feature unit `u` and task `t`, estimate `V_t(u)` (value to task `t`);
  the receiver announces a task-demand weight vector `w = {w_t}`.
- **Decision:** select/route units to maximize `sum_t w_t * (task-t utility)` under the bandwidth
  budget. Optional actuator: the dual-link `{drop, best-effort, reliable}` routing from `CORE_TASK.md`,
  so task-critical-for-the-demanded-task content survives link failure.
- **Certificate (feature):** conformal per-task control — guarantee each requested task's degradation
  <= alpha at the chosen budget.

## Validation (gated — each stage can cheaply kill the next)

- **G0 — motivating figure.** At a fixed budget, keep top-K cells by *detection* importance; measure a
  *second* task (segmentation). Show it collapses vs. task-conditioned selection. *If single-task
  importance does not starve other tasks, the center is unmotivated — run this first.*
- **G1 — task-value maps are distinct.** Low cross-task correlation of `V_t` => task conditioning matters.
- **G2 — conditioning wins.** Task-conditioned selection >= best single-task selection across a task mix,
  at equal budget.
- **G3 — certificate holds.** Per-task preservation: each requested task <= alpha degradation.

## Wedge verification — STAMP + Scene Completion (checked 2026-07-19)

Read via official READMEs (raw.githubusercontent) + abstracts + search snippets. NOTE: arxiv /
openreview / mlr are blocked by this session's egress policy, so full method sections were not read —
confidence levels noted.

| Check | Scene Completion / STAR (CoRL'22) | STAMP (ICLR'25) |
|---|---|---|
| Optimized bandwidth budget / adaptive select-what-to-send | No — fixed spatial sub-sampling + temporal mixing ("amortizes comm cost"); not adaptive | No — optimizes interoperability + training scalability (~1MB adapters), not transmit-selection |
| Transmission conditioned on receiver's TASK | **No — explicitly task-agnostic**; one message, det & seg read same completion output, no fine-tune | **No — task-agnostic by design**; "collaborate without being conditioned on specific tasks" |
| Link failure / multi-radio | Not addressed | Not addressed |

**Verdict: the wedge is open.** Both make "task-agnostic" a virtue (one message for all tasks); the
thesis is the counter — under a tight budget, task-agnostic sending is wasteful, condition on task
demand. Position on top of their representation, not against it. STAMP (det + BEV-seg, OPV2V/V2V4Real,
public code) is the ideal G0 testbed. Confidence: task-conditioning = high; budget = medium-high
(confirm no accuracy-vs-bandwidth selection sweep in STAMP full text before submission).

## Immediate next actions

1. Full-text check of STAMP + Scene Completion for (a) bandwidth budget, (b) task-demand conditioning
   of transmission. This sets how much room the wedge actually has.
2. Build G0: reuse the Phase-0 harness with two task heads (detection + segmentation proxy) and show
   single-task importance starves the second task at a fixed budget.
