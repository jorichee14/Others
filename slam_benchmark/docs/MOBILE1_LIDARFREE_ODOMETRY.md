# mobile_1 odometry without the LiDAR

A plan for estimating `mobile_1`'s trajectory from the **ZED (image + depth),
the IMU, and the two radars** — and scoring it, with ATE, against the
LiDAR-refined trajectory.

This is track A of `docs/TRACK_PLANS.md` narrowed to one agent and widened in
one direction: it admits the IMU and the radar, which the main benchmark had
scoped out. The reason the widening is allowed here is that the question has
changed. The main benchmark asks *what does an agent lose by having no LiDAR*,
and holds the estimator fixed to answer it. This asks *how good can this agent
get without its LiDAR*, so every sensor it still carries is in scope.

---

## Four facts from the bag metadata that set the plan

These come from the recording's own topic table, and three of them are not yet
reflected in `configs/coop2.yaml`.

**1. There is an IMU, and there are two of them.** This answers **I1**, open
since the project started and blocking half the shortlist.

| topic | msgs | rate |
|---|---|---|
| `/mobile_1/zed/imu/data` | 30067 | **192.4 Hz** |
| `/mobile_1/ouster/imu` | 15137 | 96.9 Hz |
| `/mobile_1/zed/imu/mag` | 7596 | 48.6 Hz |

192 Hz from the ZED's own IMU, rigid to the camera and calibrated to it by the
manufacturer. Every inertial method in the plan below runs on that one. The
Ouster's IMU is a separate chip that merely lives inside the LiDAR housing —
see the open questions, because whether it counts as "using the LiDAR" is the
user's call, not ours.

**2. Only the LEFT image was recorded.** There is no `/mobile_1/zed/right/*`
anywhere in the bag. So **stereo is impossible on this sequence** — no stereo
VO, no stereo-inertial ORB-SLAM3, no re-triangulation of the depth from raw
stereo. The camera arm is strictly RGB-D, or monocular, plus inertial. This is
a property of the recording to fix in the next session (record the right image;
it costs bandwidth and nothing else), and until then it removes the single
strongest classical VIO configuration from the menu.

**3. The incumbent is already in the bag, and nothing in this repo has scored
it.** `/mobile_1/zed/odom` (2292 msgs, 14.7 Hz) is the ZED SDK's own
visual-inertial odometry. `/mobile_1/zed/pose` (2200, 14.1 Hz) is its pose with
spatial memory enabled — i.e. with loop closure. Neither appears in any config
or script here.

These are two complete baselines at **zero compute**, and more to the point they
are *the thing to beat*: "better odometry" means better than what the camera
already produced by itself. Scoring them is an hour of work and it should happen
before a single container is built, because it sets the bar every method below
is measured against and it may reveal that the bar is already high.

**4. The reference is on the LiDAR's clock grid.** `/mobile_1/global_pose` has
**1516 messages at 9.70 Hz — exactly the count and rate of
`/mobile_1/ouster/points`**. That is a clean confirmation that the refined
trajectory is LiDAR-derived, one pose per sweep.

It also contradicts `configs/coop2.yaml`, which records this topic as *"2833
msgs, ~18.1 Hz"*. One of the two is stale — probably the config, written against
an earlier export. Resolve it before quoting anything, since it sets how densely
the reference can be interpolated against a 14.7 Hz estimate.

### One fact that is good news

`configs/coop2.yaml` states that `mobile_1`'s reference trajectory is expressed
in **`zed_left_camera_optical_frame`**, and that `mobile_1.zed_rgbd`'s
`reference_frame_from_sensor` is exactly identity. So a ZED-based estimate is
*already in the reference's frame*: no lever arm, no 90° rotation, nothing for
the evaluator to right-multiply out. On this rig that normally costs several
centimetres of ATE that no alignment can absorb and that looks exactly like
drift. Here it is free — provided the `.txt` really is that frame (see Q1).

And the usual circularity worry runs the *other* way for once. The main
benchmark's difficulty is that the reference was seeded by GLIM, so a LiDAR
method scored against it shares its errors. Nothing in this plan touches the
LiDAR, so **every method here has errors uncorrelated with the reference's**.
This is a cleaner evaluation than the one the main table is doing.

---

## What the number will look like, before we run anything

Worth writing down now so the result can be checked against the expectation
rather than rationalised after it.

The reference's stated uncertainty is **15 mm**, and the main benchmark's
problem is that good LiDAR methods land *inside* that band and cannot be told
apart. A LiDAR-free method on a 156 s indoor lap will land at **5–50 cm** — one
to two orders of magnitude above the floor. So this track is the one the
sequence can actually discriminate, and the daggering rule in
`slambench/traj_metrics.py` will almost never fire here.

The 16.6 m room and 156 s run cut the other way for loop closure: there may be
no revisit at all, in which case RTAB-Map's appearance loop closure never fires
and its row is measuring its odometry. **I6 decides whether half of RTAB-Map is
being evaluated**, and it is answerable from the reference trajectory alone with
`revisit_residual`, with no method and no container.

---

## The methods

Five tiers. Each tier adds exactly one sensor to the tier above it, so a
difference between adjacent tiers is attributable.

### Tier 0 — free, already recorded

| row | input | what it is |
|---|---|---|
| `zed_sdk_odom` | ZED VIO | `/mobile_1/zed/odom`, no loop closure. The incumbent. |
| `zed_sdk_pose` | ZED VIO + spatial memory | `/mobile_1/zed/pose`. What the closed-source system does with loop closure. |

Zero compute. Both are just TUM exports plus `eval_run.py`.

### Tier 1 — depth geometry only, no image, no IMU

| row | method | why |
|---|---|---|
| `kiss_icp.zed_depth` | KISS-ICP on the reprojected depth cloud | The bridge from the main benchmark. Pure geometry: no photometry, no inertial. It is the floor, and it is the row that says how much of the result is the *depth sensor* rather than the camera. |

### Tier 2 — RGB-D, no IMU

| row | method | why |
|---|---|---|
| `rtabmap_rgbd` | RTAB-Map RGB-D | Already configured in `configs/methods/rtabmap_rgbd.yaml`. F2M visual odometry + appearance loop closure + a dense map, so it is the only entry that also scores on the map metrics. The user named it, and it is the right primary. |
| `orbslam3_rgbd` | ORB-SLAM3 RGB-D | Sparse landmarks, local BA, DBoW2. The standard, and its known failure modes — low texture, fast rotation — are exactly what a pushcart turning in a lab produces. |

### Tier 3 — RGB-D + IMU. The actual candidates.

This tier is what "better odometry by using SLAM techniques" most plausibly
means, and it is newly unblocked by fact 1.

| row | method | why |
|---|---|---|
| `orbslam3_rgbd_inertial` | ORB-SLAM3 RGB-D-inertial | Tightly coupled, with ORB-SLAM3's IMU initialisation. The main candidate. Its gap to `orbslam3_rgbd` **is** the IMU's worth, with the estimator held fixed. |
| `openvins` | OpenVINS (MSCKF) | Filter-based rather than optimisation-based. Strong on short sequences, and it does not depend on a good IMU initialisation the way a batch VIO does — which matters on 156 s. |
| `rtabmap_ext_odom` | RTAB-Map, fed an external VIO as its odometry source | RTAB-Map decouples odometry from mapping. Feeding it the best Tier-3 odometry and letting it do loop closure and the dense map is cheap, strong, and the most likely configuration to actually win. |

`vins_fusion` is a reasonable fourth if the above disagree; it is not needed to
answer the question.

### Tier 4 — + radar. Gated, and the framing matters.

**Radar is not a scan matcher here.** `configs/coop2.yaml` is right that ICP on
10¹–10² points indoors is a research contribution rather than a baseline, and
nothing in this plan changes that. What radar supplies is **ego-velocity from
Doppler**: RANSAC + least squares over each sweep's (azimuth, elevation, radial
velocity) triples gives a full 3D body velocity per sweep, which enters an
estimator as a velocity factor. That is the standard technique and it is what
radar-inertial odometry is built from.

The rig is unusually well suited to it. A single radar conditions this poorly —
the velocity component perpendicular to its boresight is weakly observed. There
are **two** radars here, at yaw `-95.2°` and `+24.9°` (from `configs/coop2.yaml`),
so roughly 120° apart, which is close to the configuration people deliberately
build for this.

| row | method | why |
|---|---|---|
| `rio` | radar ego-velocity + IMU | Vision-free. It cannot drift the way vision drifts because it fails for unrelated reasons, so it is the row that survives where the image does not. Expect poor absolute ATE and good *velocity* behaviour. |
| `rgbd_inertial_radar` | Tier-3 winner + radar velocity factor | The integration. The claim being tested. |

**Set the expectation honestly:** on a well-lit 156 s lab lap, vision probably
does *not* fail much, so the radar row may show **no gain in whole-run ATE**.
That is a real result, not a failed experiment, and reporting it as one is rule
8 of `CLAUDE.md`. The way to make it a genuine test rather than a coin flip is
to evaluate it where vision was actually weak: use the two per-frame instruments
that already exist — `slambench/observability.py` for the geometric side and ORB
feature density for the photometric one — to select the sub-segments where the
camera was starved, and report the radar's gain **on those segments** alongside
the whole-run number. That is a targeted claim the sequence can support, where
the whole-run one probably cannot.

---

## The ablations

Small and axis-at-a-time. Roughly 18 runs, not 40. The sensor-set axis is the
headline; the rest are attribution.

**A. Sensor set** (the headline ladder — this *is* tiers 0–4)
`depth only → +image → +IMU → +radar`, and `radar+IMU` off to the side as the
vision-free control.

**B. IMU source** — ZED IMU (192 Hz) vs Ouster IMU (97 Hz), same method.
Tests sensitivity to IMU quality and rate. **Gated on Q2** — whether the Ouster's
IMU is admissible under "don't use the LiDAR".

**C. Loop closure on/off** — RTAB-Map with `Rtabmap/LoopThr` at its default and
at 1.0 (disabled). Isolates loop closure from the odometry front-end. **Only
run if I6 says the route revisits**; otherwise the two rows are identical by
construction and the cell is reported as not evaluable.

**D. Depth max range** — `Vis/MaxDepth` at 3 / 5 / 8 / 12 m. The real-sensor
analogue of the `r3`–`r20` axis already in `configs/ablations.yaml`, except
imposed on a genuine depth camera rather than a decimated LiDAR. It also
cross-checks that grid: the synthetic `realsense_envelope` cell should land near
the real measured one, and if it does not, the synthetic curve must not be
quoted alone.

**E. Radar configuration** — `radar1` only / `radar2` only / both. Directly
tests the two-radar conditioning claim above. Cheap, and it is the only cell
that says whether the second radar earned its place on the cart.

**F. Rate** — depth at 14.7 Hz vs decimated to 5 Hz. What a bandwidth-limited
link would cost, which connects to the sibling collaborative-perception project.

---

## Evaluation

Everything reuses `slambench/` unchanged. No new metric code; the point of that
layer is that a threshold can move and every run re-scores in seconds.

**Alignment — and this choice changes the ranking, so it is made now.** The main
config has `alignment: se3` with `align_n_first: null`, i.e. best-fit over the
whole run. For an *odometry* claim that is the wrong headline: best-fit
systematically flatters a steadily drifting estimate by splitting its error
across the run. Use **`align_n_first: 100`** — anchor on the first ~7 s and
measure drift away from a fixed start — as the headline, and report whole-run
best-fit in the appendix. This answers **I7** for this track specifically.

**The number set per row.** Never ATE alone:

- ATE (trans, rot), anchored alignment
- RPE at 1 m, 5 m, and 10 s — on a 16.6 m room the reference is far more
  trustworthy over 1 m than over the whole run, so RPE@1m is the honest
  discriminator between two methods
- **`tracked_fraction`** beside every ATE. An RGB-D or visual-inertial method
  that loses tracking has not scored badly, it has *stopped*, and an ATE over
  the 40% it held is not comparable to one over 100%. This is the single most
  common way an RGB-D table lies.
- z-drift reported separately from in-plane. RGB-D and inertial estimates fail
  differently in z, and a combined RMSE hides it.
- final-pose drift as a percentage of path length
- runtime / real-time factor. For a platform that is LiDAR-free because of
  cost, power or weight, that column is part of the result.
- tier 3 (revisit residual) always; tier 2 (board dwell) when **I2** is resolved.

**Tier 1 is agreement, not accuracy** — the reference is the offline pipeline's
own output. That caveat is weaker here than in the main table, because these
methods share no sensor with it, but it still holds and the write-up still says
so.

---

## Procedure

```
P0   Radar Doppler gate (Q3) + IMU extrinsic/noise gate (Q4).     minutes / ~1 h
P1   Reference .txt -> TUM in zed_left_camera_optical_frame.      ~1 h
     Verify expected_start [0.697952, -0.062696, 0.193334].
     Resolve the 1516-vs-2833 discrepancy. Compute path length.
P2   I6: revisit count on the reference alone.                    minutes
     Decides whether ablation C exists.
T0   Score /mobile_1/zed/odom and /mobile_1/zed/pose.             ~1 h
     THE BAR. Everything below is measured against it.
T1   kiss_icp on reprojected ZED depth.                           1 run
T2   rtabmap_rgbd, orbslam3_rgbd.                                 2 runs
T3   orbslam3_rgbd_inertial, openvins, rtabmap_ext_odom.          3 runs
T4   rio, rgbd_inertial_radar  [gated on P0].                     2 runs
AB   Ablations B-F on the tier winner only.                       ~10 runs
INS  Per-frame observability + ORB density; select the
     vision-starved segments; re-score T3 vs T4 on them.
```

`T0` before everything. It is an hour, it needs no container, and it is the only
step that can tell you the answer before you build anything — if the ZED SDK's
own VIO is already at 5 cm, the useful question changes from *can we do better*
to *can anything open-source match it*, and the plan above is re-pointed rather
than run as written.

**Deliverable.** One table, tier 0 through tier 4, one row per method, with
tracked-fraction, RPE@1m and runtime beside every ATE, and the ZED SDK rows
drawn as the bar. Then one figure: ATE against sensor set, so the ladder is
visible. Then the radar's gain on the vision-starved segments, reported
separately from the whole-run number and labelled as the narrower claim it is.

---

## Open questions — resolved 2026-09-17

**Q1 — the `.txt` frame. ANSWERED: `zed_left_camera_optical_frame`.** The best
available answer. It is the frame `configs/coop2.yaml` already declares for
`mobile_1`'s reference and the frame `mobile_1.zed_rgbd` maps to with an
identity extrinsic — so a ZED-based estimate is compared **with no transform at
all**. No 13 cm lever arm, no 90° rotation, nothing for the evaluator to
right-multiply out, and none of the silent-drift failure mode that a wrong one
produces. Step P1 shrinks to a format conversion plus the `expected_start`
check against `[0.697952, -0.062696, 0.193334]`.

**Q2 — the Ouster IMU. ANSWERED: it counts as using the LiDAR, so it is
excluded.** Consequences: every inertial row runs on the ZED IMU at 192.4 Hz,
and **ablation B is dropped** — with one admissible IMU there is no source axis
to sweep. Recorded in `configs/coop2.yaml` under `excluded:` so the choice is on
the record as a decision rather than looking like an oversight later.

**Q3 — radar Doppler. ANSWERED: present. Tier 4 is live.** One implementation
detail remains and it is not a gate: the *field name* has to be read off the bag
before anything decodes it, since `velocity`, `doppler` and `v_r` are all in use
across TI driver forks. `ros2opv2v/pointclouds.py` reads `x/y/z/intensity` only,
so that field needs plumbing through.

---

## Q4 — the ZED camera↔IMU extrinsic and noise model

The remaining open item, and it splits cleanly: **the extrinsic is recoverable
and probably already recorded; the noise model is not recoverable from this bag
at all.** Neither blocks the ladder from starting.

### Which camera this is

Nothing in any of these repos records the model, so it is deduced from the topic
table: **a ZED 2 or ZED 2i**. The discriminator is `/mobile_1/zed/imu/mag` — the
ZED Mini and ZED X carry no magnetometer and the original ZED carries no IMU at
all, so a magnetometer plus an IMU narrows it to those two. The rate
corroborates: the ROS 2 wrapper's default sensor rate is 200 Hz and the observed
192.4 Hz is that with a few drops. Confirm against the unit's serial before
quoting it in a paper — the deduction is sound but it is still a deduction.

It matters because both cameras ship with a **factory IMU↔camera calibration**,
which is why the extrinsic half of this question is cheap.

### The extrinsic — three routes, in order of preference

1. **The ZED SDK, which is authoritative and does not depend on the bag.**
   `getSensorsConfiguration().camera_imu_transform` returns it directly. A
   five-line program against the camera, and the answer is the manufacturer's
   own calibration.
2. **`/usr/local/zed/settings/SN<serial>.conf`** — the factory calibration file
   on whichever machine ran the camera. Same numbers, no camera needed.
3. **`/tf_static`.** Three messages sounded alarming when this plan was written;
   it is not. `tf2_msgs/TFMessage` carries an *array* of transforms, and there
   are exactly three static publishers on this bag (`mobile_1`, `mobile_2`,
   `infra_1`). One message per agent with the whole chain inside is the
   expected shape, and the converter already recovered the Ouster and both
   radar extrinsics from it — so the mobile_1 chain is certainly in there. The
   only question is whether `zed_imu_link` was published with it.

**The trap, whichever route.** The wrapper publishes `zed_imu_link` against
`zed_camera_center`, and this benchmark's reference frame is
`zed_left_camera_optical_frame`. The chain between them crosses the ROS-body
(x forward, z up) to optical (z forward, y down) convention change. Get that
rotation wrong and you get a transform that is plausible, silent and wrong —
the same class of defect the track-C precheck already produced once, when
bearings computed in a body frame for an optical pose put a node 1.8 m up at
+70° elevation. Verify the composed transform by checking that the IMU's
gravity vector, rotated into the optical frame, points where the camera's own
pitch says down is.

**If the factory calibration turns out to be the weak link** — visible as a
tightly-coupled row that underperforms its loosely-coupled sibling for no other
reason — the fallback is a Kalibr re-calibration. That needs a target and a
dedicated excitation recording, so it is a next-session item, not a fix for this
bag.

### The noise model — not recoverable here, and that is the answer

Allan variance needs **hours** of static logging. The bias random-walk terms
only emerge at cluster times of 10²–10³ s, which is why `allan_variance_ros` and
`imu_utils` both ask for 2–3 hours minimum. **This recording is 156 seconds.**
A static segment inside it, if one exists at all, bounds the *white noise*
densities loosely and says nothing whatsoever about the random walks — and the
random walks are what a tightly-coupled estimator uses to decide how fast to let
the biases move.

So: **do not try to extract it from coop2.** The fix costs one overnight and no
effort — park the camera on a desk, record `/zed/imu/data` for 3+ hours, run
`allan_variance_ros`. It is a property of the unit, not of the session, so it is
measured once and then serves every recording this lab ever makes. That is the
highest value-per-effort item in this whole plan and it can run tonight, in
parallel with everything else, because it needs nobody present.

**Until then, err high.** The asymmetry is known and it is not symmetric:

| | effect on the estimator |
|---|---|
| noise **over**-stated | distrusts the IMU, degrades gracefully toward vision-only. Safe. |
| noise **under**-stated | over-trusts the IMU, becomes over-confident, diverges. |

Inflating is also standard practice independent of this problem: people inflate
Allan-derived values 2–10× for real operation anyway, because a bench-static
Allan curve under-represents in-motion vibration, temperature drift and
scale-factor error.

Order-of-magnitude starting points for a consumer MEMS IMU of this class —
**a starting point to be replaced, explicitly not a measurement of this unit**:

```
gyroscope     noise density  ~1e-3 .. 3e-3   rad/s/sqrt(Hz)
              random walk    ~1e-5 .. 1e-4   rad/s^2/sqrt(Hz)
accelerometer noise density  ~1e-2 .. 3e-2   m/s^2/sqrt(Hz)
              random walk    ~1e-4 .. 1e-3   m/s^3/sqrt(Hz)
```

Per rule 3 these stay `null` in `configs/coop2.yaml` — the loader refuses rather
than defaults. Per rule 7 they live in the *method* config with their provenance
named beside them, so a poor tightly-coupled result can be checked against the
assumption instead of blamed on the method.

### What this actually changes about the plan: not much

Tier 3 splits by sensitivity, and only half of it is affected.

| row | coupling | needs the noise model? |
|---|---|---|
| `rtabmap_ext_odom` | loose | no — runs today |
| `zed_sdk_odom` / `zed_sdk_pose` | closed, already computed | no |
| `orbslam3_rgbd_inertial` | tight | yes, and its *initialiser* especially |
| `openvins` | tight | yes, but tolerates a static init |

**So the ladder starts now.** The loosely-coupled rows carry the headline until
the overnight log lands; the tight rows run in parallel, each labelled
*datasheet-order noise*, and are re-run once the measurement exists. A tight row
that beats its loose sibling under inflated noise is a *real* result — it won
despite a handicap. One that loses is not yet a result at all, and the table
says so rather than ranking it.

### One risk that is not about the noise model at all

**This trajectory is close to the degenerate case for visual-inertial
initialisation.** A pushcart rolling flat across a lab floor at walking pace is
near-constant-velocity with little rotation about any non-vertical axis, and
constant velocity is exactly the motion that makes accelerometer bias and
gravity direction weakly observable. RGB-D softens it — scale comes from the
depth rather than from the IMU, which is the failure that kills monocular VIO
here — but bias observability still suffers.

The concrete prediction: **ORB-SLAM3's inertial initialisation may simply fail
to converge, or converge to a bad gravity estimate, on this sequence**, and that
would be a property of the *motion*, not of the noise model or the method.
OpenVINS is the hedge, since it can initialise from a static period instead of
requiring excitation.

Two cheap things settle it, both free and both worth doing during P1:

- **Check for a static segment at the start or end of the run**, from the
  reference trajectory alone. It serves double duty: an OpenVINS static
  initialisation, and a loose white-noise bound on the IMU.
- **Compute the rotational excitation** over the run — the spread of angular
  velocity about the two horizontal axes. If it is near zero throughout, say so
  in the write-up *before* reporting any inertial row, because it predicts the
  result rather than explaining it afterwards.

If both come back poor, the recommendation for the next session is one sentence:
**start each run with a few seconds of deliberate handheld excitation** — pitch
and roll the cart, or lift and rock it — before setting off. It costs five
seconds and it is the difference between an inertial arm that initialises and
one that does not.
