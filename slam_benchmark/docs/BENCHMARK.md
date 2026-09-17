# The coop2 SLAM benchmark

> **Execution order lives in `docs/PLAN.md`** — start there; this document is detail it points into.

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
*Normalisation:* every method's trajectory is fused through one common pipeline
(VDBFusion TSDF at ρ) before scoring, so S3 measures trajectory quality expressed
in geometry rather than the choice of map representation; a native map is a
second column, not the primary one.
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

## 5. Algorithms

Named systems, what each one tests, and why it is in this benchmark rather than
another. Blocks are never merged into one ranking.

### S1 — LiDAR trajectory

The Ouster is 128 beams at 9.7 Hz in a 16.6 m room, so **range is never the
binding constraint here** — the two things that are, are sweep distortion during
in-place rotation (a 103 ms sweep at pushcart turning rate smears azimuthally,
and the smear does not average out) and geometric degeneracy against long flat
lab walls. Methods are chosen to separate those.

| method | what it tests | why on this data |
|---|---|---|
| **KISS-ICP** | scan matching alone, no IMU, no loop closure, no tuning | The anchor. Its published claim is to need no per-dataset tuning, so a poor indoor result is a *finding* rather than a config error. It deskews from the Ouster's per-point `t`. And it is the only entry that also runs on reprojected depth, which makes it the bridge to S2. |
| **FAST-LIO2** | what an IMU buys | Iterated-EKF LiDAR-inertial on an ikd-Tree. The IMU matters here for a specific reason, not a generic one: it supplies the motion model that deskews the sweep during rotation, which is exactly where KISS-ICP's constant-velocity assumption is weakest. ⛔ I1, I5 |
| **LIO-SAM** | what a *backend* buys, with the front end held fixed | Same inertial front end class, plus a factor graph with Scan Context loop closure. FAST-LIO2 vs LIO-SAM is the controlled odometry-vs-global-optimisation pair; FAST-LIO2 vs RTAB-Map is not, because the front ends differ too. This is the pair that replaces the one GLIM's circularity cost us. ⛔ I1 |
| **RTAB-Map (ICP)** | loop closure *without* an IMU | The IMU-free half of the same question, and the pair for KISS-ICP on an identical stream. |
| GLIM | — | **Ablation only.** It is the reference's seed; its errors are correlated with the reference's. Never a peer row. |

*Optional:* **Point-LIO** treats returns as a stochastic process rather than a
sweep, removing the sweep-time assumption entirely — the most principled answer
to the rotation problem above. Add it if FAST-LIO2's deskew turns out to be the
limiting factor. **MOLA-LO** is a current IMU-free alternative to KISS-ICP; run
it only if KISS-ICP under-performs and you need to know whether that is the
estimator or the scene.

### S2 — no-LiDAR trajectory

The centre of the benchmark, and the place to be concrete about what actually
runs. Before any method: the RealSense sees 0.2–10 m through an 87° wedge in a
room 16.6 m across, so **most of the room is out of range most of the time**.
That, not the estimator, is the dominant effect, and every entry below is being
asked how it copes with it.

| method | what it tests | why on this data |
|---|---|---|
| **KISS-ICP** on reprojected depth | the sensor, with the estimator held fixed | The bridge. Its S1 and S2 rows differ *only* in the sensor, on the same rigid body and trajectory. Alone among the entries, it measures the LiDAR's worth rather than one system's response to losing it. |
| **ORB-SLAM3 (RGB-D)** | the sparse-feature standard | Not because it will win, but because its failure modes are diagnostic: it loses tracking on low texture and fast rotation, and a pushcart turning in a lab produces both. Its `tracked_fraction` is a measurement of the *sequence*, not just of the method. |
| **RTAB-Map (RGB-D)** | the practical workhorse | Appearance loop closure plus a dense map. It re-localises rather than dying, so it produces a number where ORB-SLAM3 may not — and it is the RGB-D entry that also enters S3. |
| **BAD-SLAM** | representation vs sensor | Direct bundle adjustment over surfels, geometry and photometry jointly. If it closes most of the gap to S1, the RGB-D disadvantage was the *sparse representation*; if it does not, it was the sensor. That is the single most informative comparison in S2. GPU. |
| **DROID-SLAM** | whether a learned front end changes the answer | Deep dense BA over learned flow. ORB-SLAM3's failures are supposed to be exactly what a learned front end removes. If they are, the classical field understates RGB-D and the track's conclusion changes. GPU, not real time — report that, do not hide it. |

*Optional:* **MASt3R-SLAM** is the current foundation-model entry and is markedly
better than DROID-SLAM on low-texture and wide-baseline cases; add it if
DROID-SLAM is the strongest RGB-D row, to check whether the ceiling is higher
still. **Skip the neural-implicit family** (NICE-SLAM, Co-SLAM, Point-SLAM)
unless S3 specifically needs a neural mapping row: they are tuned on Replica and
ScanNet with near-perfect depth and slow trajectories, their *tracking* is
typically worse than ORB-SLAM3, and on real RealSense depth at walking pace they
are fragile. Including one as a peer would misreport the state of the art.

**Two practical gates before any of this runs.** The ZED publishes 32FC1
**metres** and the RealSense 16UC1 **millimetres** — a factor of 1000, and every
one of these systems completes either way, producing a map of a room a kilometre
across. And the ZED's depth is `depth_registered`, already in the left colour
frame, while the RealSense's sits 59 mm from colour. Both are declared per stream
in `configs/coop2.yaml`; neither has a safe default.

### S3 — map reconstruction

**The map representation is normalised before scoring.** Each method's native
output is a surfel cloud, an ikd-Tree, an accumulated sweep set or a TSDF, and
comparing those directly measures the representation rather than the SLAM. So
every method's *trajectory* is taken, the raw sensor data is fused through one
common pipeline — **VDBFusion** (TSDF, fixed voxel ρ) — and that is what is
scored. S3 then measures trajectory quality as expressed in geometry, with
fusion held constant.

The native map is reported as a **second, separate column** where one exists,
because a method whose own representation beats the common fusion has a result
worth stating — it just is not the same comparison.

| entered in S3 | not entered | why not |
|---|---|---|
| KISS-ICP, FAST-LIO2, RTAB-Map (both), BAD-SLAM | ORB-SLAM3, DROID-SLAM | sparse landmarks and per-keyframe depth are not dense maps; scoring a Chamfer distance against one would be a category error dressed as a number |
| `reference_accumulation` — the **ceiling** | | sweeps fused along the reference trajectory: what the sequence's own geometry supports given perfect poses |

### S4 — sensor attribution

**KISS-ICP only.** The bridge method is the only one whose LiDAR and RGB-D rows
are comparable, so it is the only one whose crossover level means anything. One
sweep per axis, ~12 runs, not 40.

### S5 — collaborative

For a **heterogeneous** pair — one LiDAR agent, one RGB-D agent — the off-the-shelf
field is nearly empty, and saying so is part of the result.

| method | what it tests | why on this data |
|---|---|---|
| **decoupled registration** | the floor | Each agent alone, then one offline registration. Use **TEASER++**, not FPFH+RANSAC: the precheck puts the two paths' overlap at the low end, and TEASER++ is certifiably robust at low inlier ratios where RANSAC degrades. Expect its error almost entirely in the *constant* term — one rigid transform cannot absorb either agent's drift, and that is the shape a joint method must beat in the *varying* term. |
| **Swarm-SLAM** | the only real candidate | Decentralised, with LiDAR *and* RGB-D front ends and sparse inter-robot loop closures. It matches the recording's structure — two machines, two clocks — and it estimates the inter-agent transform rather than being handed it. Log the inter-robot match count: zero matches with a good relative pose means the frames happened to agree. |
| **oracle transform** | the ceiling | The true `T_ab` from the reference. Caps every collaborative row, and separates "the merge was wrong" from "the partner had nothing to add from where it stood". |

*Conditional:* **DiSCo-SLAM** is LiDAR-only, so `mobile_2` would enter as a
reprojected depth cloud — run it only to test whether a Swarm-SLAM failure is
the modality or the system. **Kimera-Multi** and **COVINS-G** are
visual-inertial. ⛔ I1

### S6 — infrastructure-anchored localization

A task, so the algorithms are per stage.

| stage | algorithm | why |
|---|---|---|
| detect, radar | **DBSCAN** per sweep, largest moving cluster | 10¹–10² points: expect a centroid, not a shape. Supplies bearing **and** range. |
| detect, camera | **ArUco / ChArUco on the cart** | The recommendation that decides this task. Detecting a pushcart in a raw, distorted, 10.6 Hz Arducam stream is a research problem whose error would swamp the localization gain being measured. A printed board gives sub-pixel bearing, unambiguous identity, and an identity channel the radar cannot supply. One board per platform, next session. |
| associate | nearest stamp, **after** clock reconciliation | `infra_1` is a third machine with a third clock. Skip this and every residual is a lag. |
| fuse | **GTSAM pose graph**: `BetweenFactor<Pose3>` for ego odometry, `BearingFactor` / `BearingRangeFactor` to a landmark fixed at the surveyed pose | Smooths over sparse detections, degrades gracefully when they stop, and each factor carries its own covariance — so the node's pose σ (I9) enters the estimate rather than being ignored. An ESKF would also work and handles sparsity worse. |

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
