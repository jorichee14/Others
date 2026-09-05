# MIRC plan: cooperative perception on real indoor data, from one recording

The question: how do OpenCOOD's cooperative detectors, InCoP's CGRF and Where2comm
compare on REAL indoor multi-robot data, clean and under channel impairment, and
what does the answer say about the OPV2V and InCoP simulation results already in
hand?

The constraint: one recording, coop2. No new recording in the near future.

The plan is seven stages. Each produces something the next depends on, each has a
gate, and nothing built early has to be redone when a second recording exists.

## What exists

| Asset | State |
|-------|-------|
| coop2 as an OPV2V-format dataset | 1330 frames, 3 agents, 2 static chairs as GT, validated, `--check` passed |
| Ego | `mobile_2`, the 87-degree depth cart: sees the chairs alone in ~33% of frames |
| Partner | `mobile_1`, the Ouster: sees them in ~95% |
| RSU | `infra_1`, an Arducam radar, ~49 points per frame, 12-18 m away |
| Time split | train 0-94 s (~800 frames, outbound leg); validate 94 s-end (~530 frames, return leg, both link outages) |
| OPV2V arm | complete: frozen Phase 1 table, impairment matrix, attribution analysis, tracking |
| InCoP fork | hosts Where2comm, CoBEVT, V2X-ViT, ERMVP, CGRF, dense-CLC, late; reads OPV2V-format trees; config generator has `opv2v`, `incop`, `mirc` presets |
| Stock OpenCOOD | AttFuse, F-Cooper, V2VNet, CoAlign, early, late; checkpoints per `env/CHECKPOINTS.md` |

## Ground rules that hold across every stage

**One pipeline per number.** Every row in a table comes from one loader, one
evaluator, one config generator. Rows from the InCoP fork and rows from stock
OpenCOOD never sit in the same table without the CoBEVT bridge (Stage 3) having
passed.

**Within-scene, said out loud.** Until a second recording exists, every MIRC number
is "on coop2": one room, two objects, one trajectory pair. Report it that way.
Absolute AP is not compared to the OPV2V table. Degradation curves, NPD and
floor-crossing points are compared, because they are relative and survive a change
of backbone and scene.

**Three seeds, always.** Mean and spread. A one-seed number on 530 frames is a
coin.

**AP@0.5 is the headline.** At IoU 0.7 a 5 cm centre error on a 0.7 m box costs
most of the overlap, so that column is noisy on small objects. Report 0.3 and 0.7
too, and say why 0.7 is noisy rather than read a story into it.

**Time splits only.** Frames are 1.7 cm apart at 10 Hz. A random split is the
training set with a perturbation.

## Stage 0: make the scene measurable

*Status: code done, awaiting the run.*

Reconvert with `mobile_2` as ego and the time split. Validate both splits. Run
`label_static.py --check` on validate.

**Gate.** Ego-only (agent 1) sees each chair in roughly a third of sampled frames;
the partner (agent 2) in nearly all. That gap is the collaboration signal every
later stage measures. If it is not there, nothing downstream can work and the
scene design has to change before any model runs.

## Stage 1: does the pipeline run, and what do the shipped weights do

Three runs, no training.

1. `validate_opv2v.py --with-open3d` on both splits, then `--with-opencood` with an
   InCoP-fork `mirc` config. This builds OpenCOOD's own dataset over the tree and
   reads frames through it. It is the only check that asks OpenCOOD rather than
   re-implementing what it is believed to do.
2. Stock OpenCOOD checkpoints on MIRC validate, unchanged, through
   `run_phase1.py --validate-dir`. Expected: AP 0.000 at every IoU. A car anchor on
   a chair has IoU about 0.06, so no anchor is ever assigned; the result is the
   car-to-chair transfer number and it goes in the write-up as such.
3. InCoP hospital-trained checkpoints on MIRC validate, unchanged. InCoP has a
   `chair` class and 1 m anchors, so this is the honest sim-to-real zero-shot
   number per method. Needs Stage 2's checkpoints if they do not exist yet.

**Gate.** Every run completes without an exception. Whatever the numbers are, they
are recorded.

## Stage 2: the InCoP arm

*This was already planned; MIRC makes it a prerequisite rather than a parallel arm.*

Train the seven methods on InCoP hospital with the `incop` preset, three seeds.
This produces three things at once: the simulated-indoor baseline the study
already wanted, the zero-shot numbers for Stage 1.3, and the initial weights for
Stage 3. Fine-tuning from InCoP weights is a sim-to-real step, which is a clean
story; fine-tuning from OPV2V weights is outdoor-car to indoor-chair, which is an
initialisation and should be named as one.

**Gate.** Clean AP on InCoP validate is sane for every method, and CoBEVT's clean
AP and NPD curve on InCoP agree with the parent study's `cobevt` behaviour in
shape. The InCoP port README already defines this bridge gate.

**Cost.** InCoP's own recipe is roughly one to three hours per model. Seven
methods, three seeds: one to three GPU-days.

## Stage 3: fine-tune on coop2, perfect channel

Every method in the InCoP fork, initialised from its Stage 2 weights, fine-tuned
on MIRC train with the `mirc` preset, same recipe for all, three seeds, evaluated
on MIRC validate.

Recipe: full fine-tune, low learning rate, 10-15 epochs. Early stopping uses the
LAST 10% of train (84-94 s) as a development slice, never validate. Chair-sized
anchors (0.75 x 0.68 x 0.92 m), 0.1 m voxels, +-20 m range, `pos_threshold` 0.40.

OpenCOOD-only methods (AttFuse, F-Cooper, V2VNet, CoAlign, early) run in stock
OpenCOOD with the same `mirc` settings, initialised from their OPV2V weights.
CoBEVT is fine-tuned in BOTH codebases from its respective init.

**Output.** The real-world Phase 1 table: AP@0.3/0.5/0.7, P and R at the operating
point, per method, mean and spread.

**Gates.**
- The ego-only floor sits clearly below the collaborative rows. If it does not,
  Stage 0's gap did not survive training and the reason has to be found before
  anything else.
- CoBEVT's two numbers agree within a stated tolerance. If they agree, rows from
  both codebases may share a table. If they do not, they may not, and the
  disagreement is reported.

**Cost.** Minutes per fine-tune. The whole stage is an afternoon of GPU.

## Stage 4: the memorisation control

Same room, same chair positions on both sides of any time split. A model can
localise from the walls and emit the chairs from memory without using partner
data. Such a model looks IMMUNE to impairment, which reads as "collaboration does
not matter here" and is an artefact.

The test: evaluate every Stage 3 model on validate with the chair points removed
from every agent's cloud. A model that still fires boxes where the chairs always
were memorised the layout.

**Output.** A pass or fail per model. Failed models stay in the Stage 3 table with
a flag; no impairment claim is built on them.

This stage is cheap, decisive, and the reason the plan is defensible with one
scene. Do not skip it.

## Stage 5: impairment sweep on real data

`matrix_incop.yaml` on MIRC validate, three seeds, models that passed Stage 4.
The matrix already rescales pose error and ghost geometry for indoor objects;
ghost boxes take the chair dimensions.

**Pre-registered prediction.** The OPV2V analysis attributed latency damage to
DISPLACEMENT: a stale partner frame places objects where they were, and the error
scales with object speed. MIRC's objects do not move. Under that theory latency
should do almost nothing here, while packet loss, bandwidth collapse and pose
error should degrade fusion much as they did in the indoor simulation. If latency
DOES hurt on static objects, the displacement interpretation is wrong and the
OPV2V result needs re-reading. Either outcome is a finding. Write the prediction
down before running.

**A free natural experiment.** Twelve percent of coop2's frames were lost to three
real link outages during recording (0-1.5 s, 120-125 s, 147-151.5 s). Those are
genuine channel failures, not simulated ones. Detection on outage-adjacent frames
against the simulated burst-loss levels is a calibration of the channel simulator
that no other arm of the study can provide.

**Output.** Degradation curves and floor-crossing points per method per
impairment, on real hardware.

## Stage 6: synthesis across the three arms

| Arm | Scene | Sensors | What differs from the next |
|-----|-------|---------|----------------------------|
| OPV2V | outdoor road, dynamic | simulated | scene dynamics |
| InCoP | indoor, static and slow | simulated | sensor realism |
| MIRC | indoor, static | real | |

OPV2V to InCoP isolates dynamics. InCoP to MIRC isolates sim-to-real. Each
difference in a degradation signature is attributable to exactly one of those,
which is what makes three arms worth more than one. This is the write-up.

## Stage 7: when a second recording exists

Chairs moved, or a different room. Models trained on coop2 alone are evaluated on
it with no fine-tuning. That single number turns every earlier "on coop2" result
into a claim about the method. Nothing above has to be redone to get it, which is
the point of doing it in this order.

## What one recording cannot give

- Cross-scene generalisation. Stage 7 only.
- Scene-level statistics. One scene is n = 1 for any claim about scenes.
- A tracking result. Two static objects have no trajectories.
- Variety in negatives. Every frame has exactly two positives, so precision is
  barely exercised and a detector that fires two boxes a frame looks flawless.

## Decisions still needed from the operator

1. Which OpenCOOD commit or fork the stock rows run on. `env/CHECKPOINTS.md` says
   `31ba160`; the RSU-ordering logic verified on main must be confirmed there.
2. Whether Stage 2 checkpoints exist already, or Stage 2 runs first.
3. Which InCoP scene to pretrain on. Hospital has the `chair` class and a full
   train/validate/test split, so it is the default unless there is a reason.
4. The CoBEVT bridge tolerance. Proposal: clean AP@0.5 within 0.02 and NPD curves
   within seed spread.

## Order of work

| # | Stage | Depends on | Cost |
|---|-------|------------|------|
| 1 | 0: reconvert, check the gap | nothing | minutes |
| 2 | 1.1-1.2: loader check, OpenCOOD zero-shot | 0 | an hour |
| 3 | 2: InCoP arm | nothing, can run alongside 1 | 1-3 GPU-days |
| 4 | 1.3: InCoP zero-shot on MIRC | 2 | an hour |
| 5 | 3: fine-tune, clean table | 0, 2 | an afternoon |
| 6 | 4: memorisation control | 3 | an hour |
| 7 | 5: impairment sweep | 4 | a day |
| 8 | 6: synthesis | 5 | writing |
| 9 | 7: second recording | a recording | — |

Stages 0 through 6 need nothing that does not already exist.
