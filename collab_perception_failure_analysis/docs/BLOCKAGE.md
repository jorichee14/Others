# Phase 6 — Geometry-conditioned loss (is lost mail the mail you needed?)

## The claim under test

Every impairment family in Phases 2–5, and every robustness result in the
collaborative perception literature we are aware of, drops messages **independently
of the scene**. The physical objection is that this independence cannot hold:

> The vehicle that occludes an agent's lidar is the vehicle that obstructs its radio.

Occlusion creates the *need* to collaborate; blockage destroys the *ability* to
collaborate; both have the same physical cause. If that coupling is real, then

```
P(message arrives | you needed it)  <  P(message arrives)
```

and every robustness number in the study — ours included — is **optimistic**,
because i.i.d. loss is the best case: it deletes messages uniformly instead of
deleting the ones that would have filled the ego's blind spots.

Phase 6 tests this in two steps. Step 6.1 is model-free and decides whether Step
6.2 is worth running.

---

## Step 6.1 — the audit (model-free, no GPU)

`scripts/run_blockage_audit.py` measures two quantities per (frame, collaborator *j*)
using **labels and geometry only** — no detector, no checkpoint, no propagation
model, so no modelling choice downstream can manufacture the correlation:

| symbol | meaning |
|---|---|
| `B_j` | is a labeled vehicle on the ego↔*j* chord, at clearance *c*? |
| `U_j` | GT boxes **not** ego-visible but visible to *j* — the objects only *j* can reveal |

Visibility is the study's existing definition (≥ `MIN_PTS = 5` of that agent's own
returns inside the box, per `run_phase43.py`), so the numbers are directly
comparable to the published spatial decomposition rather than introducing a second
convention.

Reported:

- `E[U | blocked]` vs `E[U | clear]` — independence predicts these are **equal**
- availability `A = Σ(U·delivered) / Σ U` vs `1 − mean(B)` — independence predicts
  these are **equal**; a negative gap means the lost messages were worth more than
  the kept ones
- point-biserial correlation, blockage base rate, and a per-scenario breakdown so a
  single intersection cannot carry the result

### Clearance is swept, not chosen

`clearance` inflates the blocker box before the chord test and stands in for the
first Fresnel radius (≈1.1 m at 5.9 GHz over 50 m). Hard-coding one value would be
a hidden modelling assumption, so `BlockageTable` stores blocker counts for a whole
grid `(0, 1, 2 m)` in a single pass and the audit reports every column. The verdict
is read at `--decision-clearance` (default 1.0 m).

### Go / no-go, fixed before the numbers

- **NO-GO** if the blockage base rate < `--min-base-rate` (default 0.10) — the
  effect will not survive contact with a detector.
- **NO-GO** if `E[U|blocked] ≤ E[U|clear]` — the common-cause premise is false in
  this data, and the direction dies for the cost of one script.
- **GO** otherwise. This is a genuinely useful negative result if it fails: it says
  the correlation needs real urban geometry (a physical testbed, or a real-radio
  dataset) rather than CARLA.

```bash
python scripts/run_blockage_audit.py \
    --config ~/cpfa/checkpoints/<any>/config.yaml \
    --out ~/cpfa/results/blockage --stride 10
```

Outputs `blockage_audit.json` (per-link records + summary) and
`blockage_audit.md`. Runtime is dominated by point-cloud loading for the
visibility test; stride 10 over the test split is minutes, not hours. The blockage
table itself is yaml-only and caches to `~/.cache/cpfa_blockage/`.

Geometry and decision-statistics self-tests, needing neither dataset nor opencood:

```bash
python scripts/run_blockage_audit.py --selftest
python scripts/test_commchannel.py          # includes the blockage schedule tests
```

---

## Step 6.2 — the matched-PDR sweep

Two new families in `configs/matrix.yaml`:

| family | param | what it varies |
|---|---|---|
| `loss_blocked` | `blockage_p` | P(drop \| chord blocked); geometry decides *which* links |
| `loss_iid_matched` | `loss_p` | i.i.d. control at **equal packet delivery** |

**The matching is the experiment.** `loss_blocked` has a *data-determined* loss
rate — realized loss = `blockage_p` × geometric base rate — so the control arm's
`loss_p` must be set from the measured rate, never guessed. The audit prints a
ready-to-paste block:

```yaml
  loss_iid_matched:
    param: loss_p
    levels: [<-- from run_blockage_audit.py -->]
```

`levels: []` ships empty by design: an unmeasured guess would silently break the
only comparison the family exists to make, and an empty level list produces zero
cells rather than wrong ones.

Every cell now records `channel_stats` with `realized_drop_rate` and
`blocked_rate`, so the matched claim is **verified from the run**, not asserted
from the config.

### The third arm you already have

`loss_burst` (Gilbert-Elliott) is temporal correlation with no geometric content,
and those cells are already banked from Phase 3. That gives a three-way comparison
at matched mean loss:

| arm | correlation structure |
|---|---|
| `loss_iid` / `loss_iid_matched` | none |
| `loss_burst` | temporal only |
| `loss_blocked` | geometric |

If geometric loss hurts more than bursty loss at the same delivery rate, the damage
is attributable to **which** messages were lost, not merely to loss being
correlated. That is a considerably stronger claim than i.i.d.-vs-geometric alone,
and it costs no extra compute.

### Prediction

Same delivery rate, worse AP, with the deficit concentrated in `recall_occluded`
(via `run_phase43.py`) — the exact metric collaboration exists to improve.

---

## Scope and honesty

`commchannel/blockage.py` answers *"is the chord geometrically obstructed"*, **not**
*"what is the path loss"*. There is deliberately no dB, no fading model, and no
blockage→PDR mapping in it. OPV2V gives real vehicle geometry from CARLA but
contains no radio, so this phase establishes **that** geometric correlation matters
and how much it costs at a given drop probability. It does **not** measure how
often real links are obstructed, or by how much they attenuate — that requires
physical measurement (dual-band RSSI/CSI on a real testbed) or a real-radio
dataset. State this distinction in any write-up; the two-stage story is stronger
than implying the simulation measured propagation.
