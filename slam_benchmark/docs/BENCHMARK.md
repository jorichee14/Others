# The coop2 SLAM benchmark

A complete specification: what is asked, of what data, scored how, against which
baselines, and what this recording can and cannot decide.

`docs/TRACK_PLANS.md` carries the per-track method and procedure.
`IMPLEMENTATION.md` carries status. This file is the contract — fix it before
running anything, and change it in a commit of its own if it has to change.

---

## 1. The recording

One sequence, `coop2_20260828`, 156 s, three agents, indoor.

| | `mobile_1` | `mobile_2` | `infra_1` |
|---|---|---|---|
| platform | pushcart | pushcart | static mast, surveyed |
| LiDAR | Ouster, 128 beam, 9.7 Hz | — | — |
| depth | ZED, 32FC1 **metres**, 14.7 Hz, 0.3–12 m | RealSense D4xx, 16UC1 **millimetres**, 27.5 Hz, 0.2–10 m, 87° HFoV | — |
| colour | ZED left, rectified | RealSense colour | Arducam, **raw/distorted**, 10.6 Hz |
| radar | 2 × IWR6843 (not exported) | — | IWR6843, 8.6 Hz, 10¹–10² pts |
| pose frame | `zed_left_camera_optical_frame` | `camera_color_optical_frame` | Arducam optical, fixed |
| role | the LiDAR agent | the **LiDAR-free** agent | external observer |

Scene: an indoor laboratory, 8.5 × 14.3 m spanned by the surveyed boards, 16.6 m
corner to corner, everything within ~15 cm of one height. Three machines, three
clocks, chrony-disciplined, with NTP monitor topics for two of them.

**Two facts that shape every task.** `mobile_1` carries the Ouster *and* the ZED
on one rigid body, which makes a LiDAR-vs-no-LiDAR comparison possible with
nothing else varying. And `mobile_2` has no LiDAR at all, which makes it the
agent the collaborative and infrastructure tasks exist to rescue.

## 2. The reference

**Pseudo ground truth.** GLIM supplies a seed trajectory; stage 01a re-derives
every pose by registering each scan into a frozen reference map, initialising
from the *seed* rather than from the previous scan's result, so error cannot
accumulate along it. Stage 01 then builds the map with per-point deskew,
free-space carving and anchoring to three surveyed ChArUco boards.

| anchor | position in `map` | σ | note |
|---|---|---|---|
| `anchor` | origin | 6.9 mm (3.2 mm loop closure, 526.9 s) | the only one tight enough to resolve centimetre differences |
| `anchor_b` | (8.033, −7.182, −0.102) | 13 mm | its quoted 2.0 mm is within-section and understates it |
| `rs_anchor` | (8.460, −14.267, −0.056) | 15 mm (40 mm max) | the loosest; a residual under 40 mm against it is not evidence |

**`reference.uncertainty_m = 0.015`** — the worst board, not the best. A claim
restricted to the region near `anchor` may use 0.007 and must say so.

### The three tiers

Every trajectory result is reported in all three. Tier 1 alone cannot support an
accuracy claim, because the reference is estimated.

| tier | what | independent? | continuous? |
|---|---|---|---|
| 1 | ATE / RPE against the reference | no — agreement with that pipeline | yes |
| 2 | residual at the surveyed board dwells | **yes** | no, 3 points |
| 3 | revisit residual, reference-free | yes | partly |
| 4 | predicted-vs-observed bearing from `infra_1` | **yes** | **yes** — see S6 |

Tier 4 is the only continuous independent evidence, which is why S6 doubles as
instrumentation for every other task.

---

## 3. Tasks

Six tasks. Each names its input, output, ground truth, and why it is separate
from its neighbours.

### S1 — Trajectory estimation, LiDAR

*Input:* `mobile_1.ouster` (+ IMU if present). *Output:* `map ← sensor` poses at
the sensor's own stamps. *GT:* the reference trajectory for `mobile_1`.
*Why:* the ceiling, and the control for S2 and S4.

### S2 — Trajectory estimation, no LiDAR

*Input:* `mobile_1.zed_rgbd` or `mobile_2.realsense_rgbd`. *Output and GT:* as S1.
*Why:* the benchmark's centre of gravity. `mobile_1.zed_rgbd` is the same rigid
body, motion, clock and reference as S1, so **S1 − S2 on `mobile_1` is the cost
of having no LiDAR with nothing else varying**. `mobile_2` says whether that
generalises off the platform.

### S3 — Map reconstruction

*Input:* as S1/S2. *Output:* a dense cloud in the method's world frame.
*GT:* the anchored map cloud, cropped to the evaluation volume *V*, voxelised at ρ.
*Why separate:* different ground truth, different metrics, and a different
failure mode — a method can track well and map badly. Scored **only** against
`reference_accumulation`, which accumulates raw sweeps along the reference
trajectory and is the ceiling the sequence's own geometry supports.

### S4 — Sensor attribution

*Input:* `mobile_1.ouster`, degraded cell by cell toward the depth camera's
envelope (`configs/ablations.yaml`: FoV, range, beams, rate, density).
*Output:* per axis, the level at which the bridge method's LiDAR result meets its
own RGB-D result. *GT:* as S1.
*Why:* S2 gives a gap; S4 says what the gap was made of. A number becomes
"68% of it was the field of view, not the range."

### S5 — Collaborative SLAM

*Input:* both mobile streams. *Output:* both trajectories **in one frame of the
method's choosing**, plus a joint map. *GT:* the reference pair.
*Metric:* the error in `T_ab(t) = inv(T_wa(t)) · T_wb(t)`, which needs **no
alignment** — a system that lands in an arbitrary frame scores exactly as well as
one that lands in `map`.
*Why:* the agents share a frame only because the reference's anchoring put them
there. A collaborative system must estimate it, and each agent's ATE cannot say
whether it did.

### S6 — Infrastructure-anchored localization

*Input:* one ego stream + `infra_1` detections + the node's surveyed pose.
*Output:* ego poses. *GT:* as S1.
*Variants:* `ego_only` → `+bearing` → `+bearing+range`, the same ego pipeline with
one more evidence source each time.
*Why:* the interesting cell is the LiDAR-free agent — a static observer
constrains bearing well and range poorly, close to complementary to a narrow-FoV
depth camera. Its second use is tier 4 above.

---

## 4. Protocol

**No training.** One sequence; every method is run as published or pretrained,
with indoor re-parameterisation declared per method (default named beside each
changed value). There is no train/test split because there is nothing to fit —
this is an evaluation benchmark for classical and pretrained systems. A learned
entry (DROID-SLAM) is run at its released weights and its training distribution
is stated as a caveat.

**Segments.** Results are reported over the whole 156 s and, where a segment is
defined by the dataset's own paradigm labels, per segment. Segments are for
reading a curve, never for selecting a best window.

**Frozen evaluation constants.** Changing one changes the ranking, so they are
fixed here and read from `configs/coop2.yaml` at run time.

| | value | why |
|---|---|---|
| alignment | `se3` | metric sensors; `sim3` only for a monocular entry, and flagged |
| reference gap | 0.25 s | wider is a dropout; queries inside are dropped, not interpolated |
| RPE deltas | 1 m, 5 m, 10 s | 1 m is noise-dominated, 5 m catches scale — report both |
| map ρ | 0.02 m | at or below the finest F-score threshold; never below the survey's own spread |
| truncation | 0.50 m | with the truncated fraction reported, not hidden |
| F-score | 2, 5, 10 cm | indoor scale; 20–50 cm would put every method at 1.0 |
| volume *V* | **I4, open** | the map cloud's occupied box eroded 0.25 m, confirmed against the cloud |
| revisit | 0.30 m / 20 s | interior closest approach only |

**Frames.** A container writes its own sensor frame in its own world frame and
applies no extrinsic. Every re-expression happens in the evaluator, from
`configs/coop2.yaml`, identically for every method. The Ouster sits 9.3 cm and
~90° from the frame the reference describes; applying that twice, or not at all,
produces a run that completes and is wrong.

**Isolation.** `--network none`, one container per method, image pinned to a
commit, `run.json` records the command, params, replay rate and host.

---

## 5. Baselines

Grouped by what they consume. **Blocks are never merged into one ranking** — and
which block a row lands in comes from the *stream* it ran on, not the method.

| block | method | fusion / representation | loop closure | tasks |
|---|---|---|---|---|
| **LiDAR** | KISS-ICP | scan-to-model ICP, adaptive threshold | no | S1 S3 S4 |
| | RTAB-Map (LiDAR) | ICP odometry + appearance graph | yes | S1 S3 |
| | FAST-LIO2 | iterated-EKF LiDAR-inertial, ikd-Tree | no | S1 S3 *(I1)* |
| | GLIM | LiDAR-inertial, factor graph | yes | **ablation only** — it is the reference's seed |
| **RGB-D** | KISS-ICP (depth) | the **bridge**: same estimator as the LiDAR row | no | S2 S3 |
| | ORB-SLAM3 (RGB-D) | sparse ORB + local BA + DBoW2 | yes | S2 |
| | RTAB-Map (RGB-D) | appearance graph + dense map | yes | S2 S3 |
| | BAD-SLAM | direct BA over surfels | yes | S2 S3 |
| | DROID-SLAM | learned dense BA | yes | S2 |
| **collaborative** | decoupled registration | per-agent SLAM → FPFH/RANSAC → ICP | no | S5 — the **floor** |
| | Swarm-SLAM | decentralised, cross-modal loop closures | yes | S5 |
| | oracle transform | true `T_ab` from the reference | — | S5 — the **ceiling** |
| **references** | `reference_accumulation` | sweeps along the reference trajectory | — | S3 — the **ceiling** |
| | `ego_only` | the S1/S2 run of the same stream | — | S6 — the **floor** |

Every task has a floor and, where one exists, a ceiling. A method that does not
beat its floor has not earned its complexity; a gap to the ceiling that the
method does not close is the part of the problem still open.

---

## 6. Metrics

| task | primary | secondary | mandatory context |
|---|---|---|---|
| S1 S2 | ATE RMSE / median / p90 (mm) | RPE @ 1 m, 5 m, 10 s; rotation RMSE | tracked fraction, alignment mode, observed scale, runtime, peak RSS |
| S3 | Chamfer (mm), F @ 2/5/10 cm | accuracy, completeness | truncated fraction, point count, ρ, *V*, the accumulation ceiling |
| S4 | crossover level per axis | ATE at each level | degenerate fraction, starved fraction per cell |
| S5 | relative `T_ab` error (mm, deg) | constant vs varying split | separation, place overlap, inter-robot match count, solo baselines |
| S6 | ATE per variant | gain over `ego_only` | detection rate, node pose σ, bearing/range residual and bias |

**Instruments**, computed before and independent of any method:

- *geometric observability* — the point-to-plane information matrix of one scan.
  Its weakest eigenvalue names the translation direction the observed surfaces do
  not constrain. Reports **starved** separately from **degenerate**: a sparse
  cloud defeats the normal estimator long before it defeats SLAM.
- *photometric density* — ORB feature count and spread per frame. A textureless
  wall at 2 m is geometrically fine and visually empty; a textured scene at 15 m
  is the opposite, and only one of those is visible to the geometric instrument.

Together they partition every failure into *the geometry went free*, *the image
went blank*, and *neither — the method failed*. Only the third is a statement
about the algorithm.

---

## 7. Reporting rules

Non-negotiable, because each exists to stop a specific wrong reading.

1. **Every trajectory number carries its tracked fraction.** A method that lost
   tracking has not scored badly, it has stopped. An ATE over the 40% it held is
   not comparable to one over 100%.
2. **Every ATE carries its alignment mode and the observed scale**, even when
   scale was not applied. `sim3` on a metric sensor is how a bad trajectory is
   made to look good.
3. **An ATE below 15 mm is daggered**, not ranked. It is indistinguishable from
   the reference, which is a different claim from being the best method.
4. **Blocks are never merged.** LiDAR and RGB-D have different input budgets.
5. **No map score without the accumulation ceiling**; no collaborative number
   without its solo baselines; no S6 variant without `ego_only`.
6. **Runtime and peak RSS on every row.** For a LiDAR-free platform — usually
   LiDAR-free because of cost, power or weight — that column is part of the
   result.
7. **A failure is a result.** Lost tracking, a missing sensor, zero inter-robot
   matches, a sequence that cannot evaluate loop closure: reported with its
   number, never dropped from the table.
8. **A run forced past preflight carries the caveat** on every row it produces.

---

## 8. What this recording cannot decide

Stated here so no result overclaims, and so the next session is scoped from it.

| limit | consequence | fix |
|---|---|---|
| one 156 s sequence, 16.6 m room | long-horizon drift is not measurable; method spread may fall inside the reference's 15 mm | a several-hundred-metre outdoor sequence with the same board anchoring |
| the reference is estimated | tier 1 is agreement, not accuracy | a survey-grade (TLS) prior map — the single gap to Newer College / Oxford Spires / ConSLAM |
| GLIM is the seed | a GLIM row is correlated with the reference | keep it as an ablation, never a peer row |
| possibly no revisit | loop closure may not be evaluable (I6) | route a deliberate revisit, ≥ 2 passes minutes apart |
| agents start 16.3 m apart | S5 may not be runnable (I8) | route both platforms through one shared corridor |
| `infra_1` pose σ unstated | S6 reports relative gain only (I9) | survey it, or state the σ |
| Arducam raw + `camera_info` rate mismatch | infra detection unverified (I10) | confirm frame ids; **put a fiducial on each cart** |
| IWR6843 at 10¹–10² pts/frame | radar is not a SLAM input here | out of scope; extrinsics recorded so they are not lost |
| IMU unknown (I1) | the inertial block may not run | one topic in the next recording |

---

## 9. Results table template

Fixed in advance, so the shape of the output does not become a choice made after
seeing the numbers.

```
S1/S2 — trajectory
| block | method | stream | ATE RMSE | ATE p90 | rot RMSE | RPE 1m | RPE 5m | tracked | scale | runtime | RSS |

S3 — map
| method | stream | acc. | compl. | Chamfer | F@2 | F@5 | F@10 | trunc. | pts | vs ceiling |

S4 — attribution
| axis | crossover level | ATE there | ATE of the RGB-D row | degenerate frac |

S5 — collaborative
| method | rel. trans | rel. rot | constant | varying | matches | map gain | vs floor | vs ceiling |

S6 — infrastructure
| agent | variant | ATE | gain over ego_only | detection rate | bearing bias | range bias |

tier 2 — absolute (all tasks)
| method | agent | anchor | residual | survey σ | verdict |
```

---

## 10. Order of work

```
P0  precheck all three tracks                      minutes, can close S5
P1  export references; reference_accumulation      the S3 ceiling and the sanity check
P2  I1 (IMU?)                                      unblocks the inertial rows
P3  I2 (board windows)                             turns tier 2 on
P4  I3 I4 (map cloud, volume V)                    turns S3 on

S1, S2   ── the headline pair is two runs; nothing blocks it
S3       ── after P1 and P4
S4       ── after S1 and S2
S5       ── after P0's gate, then floor → ceiling → Swarm-SLAM
S6       ── after I9 and I10; tier-4 residual before the fusion variants
```

S2 is the centre of the benchmark and the least blocked thing in it. Its headline
number — one estimator, two sensors, one platform — needs two runs and no open
decision.
