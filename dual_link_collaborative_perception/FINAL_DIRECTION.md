# Final Research Direction

## Task-Conditioned Token Communication for Collaborative Perception

*The definitive statement. Consolidates and supersedes the framing scattered across
`NOVELTY_ANALYSIS.md`, `THESIS.md`, `RESEARCH_DIRECTION.md`, `PROPOSAL.md`, `CORE_TASK.md`.
Prior art checked against July-2026 literature; the two closest works (STAMP, Scene Completion)
were read in full text.*

---

## 1. The direction in one sentence

> **Collaborative-perception agents should transmit a shared discrete token stream, and each
> downstream task should draw the minimal token subset it needs under a bandwidth budget —
> because a feature's transmit-value is task-dependent, and today's systems send a task-blind message.**

- **Substrate:** tokens (discrete, entropy-codable units) — the modern unit of the field.
- **Core novelty:** the transmit decision is **conditioned on the receiver's task demand**.
- **Feature:** a **per-task preservation guarantee** (each requested task degrades ≤ α at the budget).
- **Extension (Paper 2):** route tokens across **heterogeneous links** so task-critical content
  survives best-effort link failure.

---

## 2. Existing methods and exactly what they lack

| Method | What it does | What it lacks (your opening) |
|---|---|---|
| **Where2comm** (NeurIPS'22) | spatial-confidence selection of critical regions | single fixed task (detection); task-blind unit; no per-task value |
| **JigsawComm** (2025) | learned utility select + top-1 cross-agent redundancy removal | rides an **uncertified** utility proxy; single task; no payload codec; no task-conditioning |
| **HydraCollab** (2026) | sensor gating + intermediate/late hybrid on confidence overlap | every decision inherits Where2comm map error; single task; raw float32 payload; no link model |
| **CoSDH** (CVPR'25) | supply–demand region selection | "demand" is **spatial** (blind spots), **not task** demand; single channel |
| **V2X-DSC** (2026) | Wyner–Ziv conditional codec, VQ + entropy coding | payload compression only; **task-blind**; assumes keep/drop set already chosen |
| **DiffCP** / **InfoCom** | diffusion / info-bottleneck extreme compression | compress the payload; single task; no task-conditioned selection |
| **Scene Completion / STAR** (CoRL'22) | task-agnostic self-supervised shared feature; sub-sampled token transmit | drops tokens by **spatiotemporal redundancy** (task-blind); one message for all tasks; no budget adaptation; no link failure |
| **STAMP** (ICLR'25) | task-/model-agnostic protocol domain via adapter-reverter | **uniform** channel compressor; **task-agnostic by design** (never conditions transmission on task); tests pose noise, **not** link loss |
| **SComCP / task-oriented CP** | compress for one task, discard task-irrelevant | single-task by construction — the opposite of general/multi-task |
| *(ancestors)* layered coding + UEP; multi-RAT diversity | important-on-reliable / enhancement-on-best-effort; parallel radios | fixed codec layers (not content/task-driven); content-blind or duplication-for-reliability |

**The pattern:** every method either (a) optimizes one fixed task, or (b) sends a **task-agnostic**
message (STAR, STAMP), or (c) compresses the payload without touching *what to select per task*.
**None conditions the transmission decision on the receiver's task demand.** Verified open at
full-text level against the two closest (STAR, STAMP).

---

## 3. Your novelty (honest, bounded)

Ranked by defensibility:

1. **Task-conditioned transmit-value (core).** A token's worth depends on which task the receiver
   runs; a detection-trained selector mis-ranks for segmentation/prediction and silently starves them.
   No CP method conditions *what it sends* on the receiver's task. **This is the flag.**
2. **Multi-task budget allocation (what makes it non-trivial).** Under one shared token stream and one
   budget serving several tasks, you cannot just run K selectors — they contend. Allocating a single
   budget across tasks by demand is a real optimization, not relabeling.
3. **Per-task preservation guarantee (differentiating feature).** Certify each requested task degrades
   ≤ α at the budget — certification demoted to a component, not the thesis.

**Not claimed:** general/task-agnostic representation (STAR, STAMP), tokenization/VQ + entropy coding
(V2X-DSC, codebook CP), single-task selection (Where2comm, JigsawComm), two radios / UEP (multi-RAT,
layered coding). Tokens are the **substrate**; task-conditioning is the **novelty**.

**How novel:** a **solid, verified-open, main-track contribution** — comparable in weight to STAMP
itself (a clear "first to do X" on a real problem), **not a paradigm shift.** Ceiling set by execution.

| Dimension | Rating |
|---|---|
| Conceptual originality | 6/10 — new object (task-conditioned decision), but a reachable next step |
| Whitespace / crowding | 6/10 — open vs the flagships; contested frontier (foundation-model CP) |
| Defensibility | 6.5/10 — strong if framed as multi-task budget allocation + guarantee |
| Impact ceiling | 6.5/10 — real deployment relevance; not field-defining |

**Two reviewer attacks and the defense:**
- *"Just run a selector per task."* → It's **joint** allocation under one shared budget where tasks
  compete, plus the guarantee. Frame as budget allocation, not K selectors.
- *"Foundation models make features task-general, so this is moot."* → You optimize the **communication
  decision under constraint**, orthogonal to representation quality. Bigger general features make the
  budget problem **worse**, strengthening you.

---

## 4. Approach

- **Representation (borrowed):** a general shared feature tokenized into discrete tokens (STAMP protocol
  domain / STAR patch tokens / a VQ codebook). *Not the contribution.*
- **Per-task value:** for each token `u` and task `t`, estimate `V_t(u)`; receiver announces a
  task-demand weight vector `w = {w_t}`.
- **Decision:** select the token subset maximizing `Σ_t w_t · (task-t utility)` under the token budget.
- **Guarantee (feature):** conformal per-task control — each requested task's degradation ≤ α at the budget.
- **Extension (Paper 2):** route the selected tokens across `{drop, best-effort, reliable}` links so the
  task-critical subset survives best-effort failure (adds link-robustness — a differentiator vs STAR/STAMP,
  which model no link failure).

---

## 5. Validation (gated — each stage can cheaply kill the next)

- **G0 — the motivating figure (build first).** At a fixed token budget, select top-K tokens by a
  *single-task* (detection) value; measure a *second* task (segmentation). Show it collapses vs.
  task-conditioned selection. *If single-task selection does not starve the other task, the center is
  unmotivated — find out before building anything else.* Testbed: STAMP (det + BEV-seg, OPV2V/V2V4Real,
  public code).
- **G1 — task-value maps distinct.** Low cross-task correlation of `V_t` ⇒ conditioning matters.
- **G2 — conditioning wins.** Task-conditioned selection ≥ best single-task / task-agnostic selection
  across a task mix, at equal budget.
- **G3 — guarantee holds.** Per-task preservation: each requested task ≤ α degradation.
- **G4 (Paper 2) — link robustness.** Under best-effort failure, task-critical tokens on the reliable
  link hold task loss ≤ target; baselines collapse.

---

## 6. Scope discipline

**Paper 1 = task-conditioned token communication under a budget, single link, one differentiator
(per-task guarantee).** Do not stack multi-task + tokens + compression + dual-link + guarantee into one
paper — that reads unfocused. Dual-link routing and payload compression are natural follow-ons, not the
first contribution.

**Next action:** build G0.
