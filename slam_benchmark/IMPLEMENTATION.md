# Implementation

Source of truth for progress. Update it in the same commit as the work: flip the
status, fill the Result line with real numbers and paths, append to the log.
Never mark ✅ without checking the "Done when" line.

## Input decisions

These are not implementation details; each one changes what the benchmark can
conclude, and a phase that depends on an unanswered one does not start.

| | decision | status | why it blocks |
|---|---|---|---|
| I1 | **Does the bag carry an IMU?** Topic, rate, and which agent. | ✅ **yes — three of them** | Resolved from the recording's topic table. `mobile_1`: `/mobile_1/zed/imu/data` 192.4 Hz (the default for every visual-inertial entry — rigid and factory-calibrated to the camera) and `/mobile_1/ouster/imu` 96.9 Hz. `mobile_2`: `/mobile_2/imu` 183.3 Hz. Filled into `configs/coop2.yaml`. **Unblocks phase 3 and the whole inertial arm** — but I5 and Q4 (the extrinsics and the noise model) are still open and still gate any *tightly-coupled* row. |
| I2 | **The three board dwell windows** — `scripts/board_visibility.py` (2026-09-19) answers this **geometrically, not by eye**: the reference carries full 6-DoF poses, so projecting each surveyed board into the camera settles in-front-vs-behind, in-FoV-vs-out, range and pixel position automatically. My earlier claim that this needed human eyeballing was wrong — phase0's finder used position and speed only, which was a lazy implementation, not a limitation of the data. Geometry still cannot see the board's **facing** (coop2 surveys position, not normal) or occlusion/blur; the ChArUco detector settles those *and* yields a direct pose measurement, and `../radar_camera_calibration/general_charuco.py` already does detection+PnP. Candidates MEASURED by phase0 (2026-09-19), in `configs/coop2.yaml` beside each anchor, awaiting eye confirmation. Coverage: ~~`mobile_1` never visits `anchor_b`~~ **superseded 2026-09-19**: with the FoV check `mobile_1` does see `anchor_b`, at 4.13 m — position-only search missed it by requiring 2 m. All three boards are available to both agents (six pairs); windows are in `configs/coop2.yaml` under `windows_by_agent`. The agents swap corners over the run. — when each platform stood at `anchor`, `anchor_b`, `rs_anchor`, and at what offset. | ⬜ open | Without them tier 2 is silent and no independent evidence enters the benchmark. The single highest-value open item. |
| I3 | **The anchored map cloud**: path, and confirmation it is in the post-anchor `map` frame, not the pipeline's pre-anchor frame. | ⬜ open | Every map metric needs it. A pre-anchor cloud scores every method against a frame none of them are in. |
| I4 | **The evaluation volume V.** | ⬜ open | Changes the ranking. Proposal: the map cloud's occupied box eroded 0.25 m; confirm against the cloud, do not guess from the trajectories. |
| I5 | **IMU↔sensor extrinsics + noise model.** I1 is yes, so this is live. | ⬜ open, **now blocking** | Neither the extrinsic nor the gyro/accel noise model is published. `/tf_static` carries only **3 messages** for this whole rig, so check what is actually in it rather than assuming the chain is. A guess costs a tightly-coupled estimator far more than a loosely-coupled one, so an unfavourable inertial result under a guess is not a result. Applies to the camera↔IMU chain too — see `docs/MOBILE1_LIDARFREE_ODOMETRY.md` Q4. |
| I6 | **Does the route revisit?** | ⬜ open | If it does not, loop closure cannot be evaluated on this sequence and the RTAB-Map/GLIM entries measure only their odometry. Answer it from tier 3 on the reference trajectory itself — no method needed. |
| I7 | **Which alignment the headline table uses** (`se3` vs `se3` anchored on the first N poses). | ⬜ open (answered for phase 6) | Best-fit ATE and drift-from-a-fixed-start are different claims. Pick one for the headline, report the other in the appendix. Phase 6 answers it for itself: `align_n_first: 100`, because best-fit systematically flatters a steadily drifting *odometry*. The main table still has to choose. |
| I11 | **What frame and format is the LiDAR-refined reference `.txt`?** | ✅ **`zed_left_camera_optical_frame`** | The best available answer: it is the frame the reference already declares and the one `mobile_1.zed_rgbd` maps to with an identity extrinsic, so a ZED estimate compares with **no transform at all**. Phase 6's P1 shrinks to a format conversion plus the `expected_start` check. |
| I12 | **Do the radar clouds carry a Doppler / radial-velocity field?** | ✅ **yes; field is named `doppler`, float32, all three radars** (phase0). Only the sign convention remains. | Both `mobile_1` radars carry radial velocity, so they enter phase 6 as ego-velocity sensors (RANSAC + LSQ per sweep), never as scan matchers. One non-gating detail left: the field's NAME, since `velocity`/`doppler`/`v_r` are all in use across TI driver forks, and `ros2opv2v/pointclouds.py` reads `x/y/z/intensity` only. |
| I13 | **The ZED camera↔IMU extrinsic.** | ✅ **resolved 2026-09-19** — the wrapper's published static TF on the cart supplied the per-unit factory value: `zed_left_camera_frame→zed_imu_link`, 23.1 mm lever arm, **1.290° total misalignment** (rpy −0.972, −0.063, +0.847) — precisely the per-unit rotation a nominal transform would have missed (~0.22 m/s² gravity leak per rotation). Composed to `IMU.T_b_c1` in both tight configs; sanity-checked (imu +z → optical −y). Not in the bag's tf_static and not in the `.conf` — the cart's live TF was the route. Remaining verification: gravity check against a static segment of the run. | Three routes, cheapest first: the SDK's `getSensorsConfiguration().camera_imu_transform`; the factory `SN<serial>.conf`; or `/tf_static` — whose 3 messages are the *expected* shape, one array per static publisher, not a shortfall. The trap is the body↔optical convention change between `zed_camera_center` and `zed_left_camera_optical_frame`. |
| I14 | **The ZED IMU noise model.** | ⬜ **not recoverable from coop2** | Allan variance needs 2-3 h of static logging for the random-walk terms to emerge; this bag is 156 s. Fix costs one overnight with nobody present and is a property of the unit, so it is measured once and serves every future recording. Until then the tightly-coupled rows carry inflated order-of-magnitude values **in their method config**, labelled — erring high, because over-stated noise degrades gracefully toward vision-only while under-stated noise diverges. |
| I15 | **Is there enough rotational excitation to initialise a VI estimator?** | ⬜ open, free to answer | A pushcart rolling flat at walking pace is near the degenerate case: constant velocity makes accelerometer bias and gravity weakly observable. Predicts, rather than explains, an ORB-SLAM3 inertial-init failure. Answerable from the reference trajectory alone during P1, together with whether the run contains a static segment. |
| I8 | **Did the two agents ever occupy the same place?** (track B precondition B1) | ✅ **yes — 100% overlap, closest approach 0.00 m** (phase0) | Decides whether collaborative SLAM is runnable on this sequence at all. Answerable today from the reference trajectories with `precheck_tracks.py`. Starts are 16.3 m apart in a 16.6 m room. |
| I9 | **`infra_1`'s pose uncertainty.** | ⬜ open | An infrastructure-anchored fix cannot be better than its anchor, and coop2 states the pose to six decimals with no uncertainty anywhere. Until it is a number, track C reports only relative improvement. |
| I10 | **Does the Arducam's `camera_info` belong to the Arducam?** | ⬜ open | It publishes at 27.5 Hz against a 10.6 Hz image stream — the signature of a `camera_info_manager` on its own timer, which can carry a different `frame_id`. Verify before projecting anything into that image. |

## Phases

The benchmark specification is `docs/BENCHMARK.md` (tasks S1-S6, protocol,
baselines, metrics, reporting rules, limits). Per-track algorithms and procedures
are in `docs/TRACK_PLANS.md`. The phases below carry status only.

### Phase 0 — evaluator ✅
Pure-Python evaluation layer with self-tests, plus an end-to-end smoke test on
synthetic data.
*Done when:* `scripts/test_slambench.py` and `scripts/smoke_e2e.py` both pass
with no bag, no ROS and no network.
*Result:* 32/32 self-tests pass; smoke test passes 5/5 assertions end to end
through the real `eval_run.py` and `aggregate.py`. Two defects found and fixed
by these tests before any real data touched them: `rotation_angle` lost half its
precision near identity (arccos of the trace is vertical there, so a 1e-16
rotation error read as 1e-8 rad), and `revisit_residual` counted
end-of-trajectory clamping as drift — on a single-lap route that alone produced
13 spurious "revisits" at 18 cm.

### Phase 1 — the reference, and whether it holds ⬜
Export both agents' trajectories to TUM. Run `reference_accumulation` and LOOK
at the cloud. Compute tier 3 on the reference itself (answers I6).
*Done when:* both TUM files pass the `expected_start` check; the accumulated
cloud is sharp enough to read the room's planes; I6 is answered with a number.
*Blocks on:* nothing. Start here.
*Result:*

### Phase 2 — the IMU-free block ⬜
KISS-ICP and RTAB-Map (LiDAR) on `mobile_1.ouster`; KISS-ICP and RTAB-Map
(RGB-D) and ORB-SLAM3 on both depth streams.
*Done when:* every cell has a trajectory, a tracked-fraction, and a runtime;
`results/coop2.md` builds.
*Blocks on:* I3, I4 for the map half — the trajectory half runs without them.
*Result:*

### Phase 3 — the inertial block ⬜
FAST-LIO2 and GLIM on `mobile_1.ouster`.
*Blocks on:* I1, I5. If I1 is "no IMU", this phase is closed as not applicable
and that goes in the write-up as a property of the recording.
*Result:*

### Phase 2b — track A, sensor-constrained ⬜
The ablation sweep over `configs/ablations.yaml`, KISS-ICP and RTAB-Map (LiDAR),
each cell reported with its per-frame degenerate fraction; then the validation
that the `realsense_envelope` cell lands near the real `mobile_2` result.
*Done when:* one curve per method per axis, with the observability floor shaded,
and the synthetic-vs-real envelope check answered either way.
*Blocks on:* A1 (run the precheck with a sample scan; a cell already degenerate
there is an observability floor, not a method failure). Trajectory half needs
nothing else.
*Result:*

### Phase 2c — track B, collaborative ⬜
Swarm-SLAM and decoupled registration on the pair, scored on the inter-agent
transform with the constant/varying split, plus the map gain against each
agent's solo run.
*Done when:* both methods have a relative-pose row, an inter-robot loop closure
count, and a map gain read against the solo baseline.
*Blocks on:* **B1 — did the agents ever share a place.** Answer it first with
`precheck_tracks.py --track collaborative`; it needs no method and no bag
decoding. Below ~10% overlap this phase does not run and the finding is about
the recording.
*Result:*

### Phase 4 — the static observer (track C) ⬜
`infra_1` is at a surveyed pose, watching the room, on its own clock. That makes
it a source of evidence about the mobile agents' trajectories that is
independent of both the pipeline and the boards — the only *continuous*
independent tier available. Detect each platform in the infrastructure radar and
camera, and compare the bearing (and, where the radar supports it, the range) to
what each method's trajectory predicts.
*Done when:* one figure of predicted-vs-observed bearing per method, with the
infrastructure pose's own uncertainty on it.
*Blocks on:* Phase 2. Speculative in payoff, cheap in effort, and the only route
to a continuous check that does not come from the thing being tested.
*Result:*

### Phase 6 — pseudo ground truth for the sensor-constrained agent ⬜
**T0 measured (2026-09-19), se3-aligned, translation channels:**
| row | ATE RMSE | median | drift | scale | revisit |
|---|---|---|---|---|---|
| `zed_sdk_odom` (m1) | **2.43 m** | 2.55 m | 8.55% | **1.091** | 139 mm |
| `zed_sdk_pose` (m1) | 2.27 m | 2.34 m | 7.76% | 1.100 | 138 mm |
| `cuvslam_odom` (m2) | **0.183 m** | 0.132 m | **0.73%** | 0.977 | 252 mm |

**Provenance answer (user, 2026-09-19): mobile_2's `global_pose` IS cuVSLAM** +
anchoring. The 0.18 m row is therefore **self-agreement** and is daggered; what
it actually measures is the anchoring step's own effect (−2.3% scale, ~18 cm
shape). mobile_2's only independent evidence is boards + infra + co-observation
with mobile_1 — the paper's motivating gap is now a recorded fact, not a
suspicion. Corollary hypothesis for infra round 2: mobile_1's bearing residual
is +0.11° against a LiDAR-grade reference while mobile_2's is +9° against the
cuVSLAM-derived one — a systematic +9° at 6–8 m is ~1 m lateral, i.e. plausibly
a **yaw error in mobile_2's anchoring** rather than (or on top of) a clock
offset. infra_diag v2's per-agent scan separates them: a clock offset moves with
dt, an anchoring yaw error does not.

Readings: the "6 m" is actually **2.4 m se3-aligned with a +9% scale error** —
and the scale error vindicates the reflective-wall diagnosis under a corrected
mechanism: specular surfaces produce *valid-looking but wrong* depth (virtual
images behind the wall), which `depth_health`'s validity metric cannot see. Wrong
depth inflates scale; missing depth was never the story. ZED's loop closure
(`zed_pose`) recovers almost nothing (2.43→2.27 m), so the failure is in the
odometry, not correctable by relocalisation. `cuvslam_odom` at 18 cm / 0.73%
drift is excellent — **caveat: mobile_2's `global_pose` provenance is unknown; if
that pipeline itself consumed cuVSLAM, the 18 cm is self-agreement, not
corroboration. Establish provenance before celebrating.** All three ATE *rot*
values sit at 113–118°, i.e. the ~120° body↔optical corner rotation: the
orientation channels are frame-convention artifacts, not measurements — exporters
compare body-frame orientations against an optical-frame reference. Translation
columns only, until orientation conventions are reconciled.

**INFRA VALIDATED (2026-09-19) — residuals collapsed, chain is trustworthy.**
Re-run on the corrected pose:

| agent | az median | az sd | range median | range sd | rngNEAR |
|---|---|---|---|---|---|
| `mobile_1` | +0.28° | 3.64° | **−0.12 m** | 0.58 m | −0.01 m |
| `mobile_2` | +0.16° | 3.51° | **+0.16 m** | 0.53 m | +0.01 m |

From −5.5 m to ±0.15 m, and from ~9° to ~0.2°. `rngNEAR` ≈ ±0.01 m says the
carts are detected exactly where predicted. The *wrong* pose hypotheses now
carry the ~−6 m residual — the signature to want. **No significant clock offset
survives**: bearing at dt=0 is already sub-degree, and range is *worse* at the
scan's nominal optimum, so the scan was chasing noise (the tool now says so
rather than advising a wider scan).

**Radar ceiling is a duty cycle, not a veto.** 65 058 returns: p99.9 = 9.30 m,
MAX 9.30 m, 0.34% beyond 9 m, **none beyond 12 m** — a hard chirp-config limit.
With the corrected pose the carts sit 2–13.5 m out, so the node supplies a
factor only while an agent is inside ~9.3 m. Association still found a usable
detection in **305/328 and 313/328 sweeps (93%/95%)**, so the duty cycle is
high. `infra_diag` now reports the in-range fraction of each reference
trajectory; quote it beside whatever the `+infra` rung buys.

**Bonus from the TF dump:** `/tf` carries `map → mobile_1/zed_left_camera_optical_frame`
and `map → mobile_2/camera_color_optical_frame` **with quaternions** — start
poses matching `expected_start` to 2–4 cm. These give the map→optical
*orientation* convention directly, which is the lever for reconciling the
113–118° ATE rotation artifact in T0.

**Root cause — the config's mast pose was simply wrong.**
The user supplied the published `map → arducam_optical_frame` quaternion. Against
it the config's `static_world_pose` is **off by 5.84 m and 9.34°** — which is
precisely the −5.5 m range and ~9° bearing residual phase0 measured against
*both* agents. Corrected in `configs/coop2.yaml` and, per rule 4, in
`../rosbag_to_opv2v/configs/mirc_coop2.yaml` in the same commit:
`[-5.508181, -2.590753, 1.998232] → [0.1832321194, -3.850593738, 1.684896384]`,
rpy `(-117.6730, 0.7379, -127.6503) → (-114.7303, 1.3671, -118.7566)`.
The cam→radar half **verified identical** to its published quaternion (0.0000°),
so that half was always right.

Consequences: the detections at 2.5–7.5 m **were the carts** all along — with the
true pose, predicted ranges land at 2.1/4.3/4.2/8.8/13.5 m across the room. The
radar's range envelope is **not** in question and the `+infra` factor class is
**live**. My round-2 claim that "the composed rotation is wrong" is **retracted**:
that test compared elevation in the *radar* frame against elevation in the *map*
frame, which differ legitimately for a tilted sensor — an invalid test that
pointed at the right chain for the wrong reason.

**The tell, worth keeping:** the old value carried a comment in the converter
config arguing that x = −5.5 m, *outside* the surveyed-board box, "is correct and
not a frame error." The true x is +0.183, **inside** it. A number that needs an
argument for why it looks wrong usually is wrong. Every `infra_1` number the
converter has ever emitted inherits this error and needs regenerating.

**Infra diagnosis round 1:** the DECLARED convention (`cam_fwd/ext_fwd`) wins
and `mobile_1`'s bearing is essentially perfect (+0.11°) — the geometry chain is
right in azimuth. The mystery narrows to: range −5.5 m systematic (flat under
±2 s of clock shift), and `mobile_2` bearing +9° that improves toward the scan
edge (its own clock offset vs infra, plausibly larger than ±2 s — the machines'
NTP topics exist to check against). infra_diag v2 adds the `rngNEAR`
discriminator (is *anything* returned at the predicted range on the right
bearing? — separates association-grabs-near-clutter from genuine geometry
error), a per-agent bearing-only clock scan, and a raw sweep dump.

**Goal restated 2026-09-17.** The objective is a defensible pseudo-GT for
`mobile_2`, which has no LiDAR and therefore no reference. `mobile_1` is the
*instrument*, not the subject: it carries both sensors, so a LiDAR-free
construction is built there and **certified against the held-out LiDAR**, and
that error distribution transfers to `mobile_2` with a stated platform-transfer
assumption. See `docs/PSEUDO_GT.md` for the related work, the gap, the factor
graph and the V0–V5 protocol. The roster in
`docs/MOBILE1_LIDARFREE_ODOMETRY.md` is now the set of candidate backbones and
the ablation that prices each factor class, not a ranking.

#### (former framing) mobile_1 odometry without the LiDAR
The user-facing track: estimate `mobile_1`'s trajectory from the **ZED, the IMU
and the two radars only**, and score ATE against the LiDAR-refined reference.
Plan, method ladder, ablation grid and open questions:
`docs/MOBILE1_LIDARFREE_ODOMETRY.md`.

Differs from phase 2b in what it is asking. 2b holds the estimator fixed to ask
*what does an agent lose without a LiDAR*; this asks *how good can this agent get
without one*, so the IMU and the radar are in scope where the main benchmark
scoped them out.

Two things make it more tractable than the main table. Methods here will land at
5-50 cm against a reference good to 15 mm, so the reference can discriminate
easily — where in the LiDAR block the methods crowd the floor. And no method here
shares a sensor with the reference, so for once the errors are uncorrelated.

*Done when:* the tier 0-4 table builds with tracked-fraction, RPE@1m and runtime
beside every ATE, and the radar's contribution is reported both whole-run and on
the vision-starved segments.
*Blocks on:* nothing, as of 2026-09-17. I11 and I12 are answered and the
reference needs no transform at all. I13/I14 gate the **tightly-coupled** rows
of tier 3 (`orbslam3_rgbd_inertial`, `openvins`) and nothing else — the loose
rows and tiers 0-2 and 4 run today. The IMU-source ablation (B) is **dropped**:
the Ouster's IMU counts as using the LiDAR, so there is one admissible IMU and
no axis to sweep.
*Roster:* 16 method configs, all loading and all validated against the streams
they declare. Eight clear `run_method.py`'s preflight **today**
(`zed_sdk_odom`, `zed_sdk_pose`, `kiss_icp`, `rtabmap_rgbd`, `orbslam3_rgbd`,
`rtabmap_rgbd_imu`); five are held by a `null` the config says cannot be
guessed, which is the refusal working rather than a gap. `openvins` is
`status: deferred` with its reversal condition recorded, not in the roster.
*BAR MEASURED 2026-09-17: `/mobile_1/zed/odom` lands ~6 m from the reference* —
~36% of the room diagonal, caused by stereo depth dropout on reflective walls.
Not drift, a failure. It re-orders the roster by how much each entry leans on
depth, and it makes the reference able to separate everything (the 15 mm
crowding worry is gone). Step 1 is now `scripts/depth_health.py`.

*SCOPE:* the roster is a **menu**, not a run list. The plan's `START HERE`
section names the core five (`zed_sdk_odom`, `rtabmap_rgbd`, `rtabmap_rgbd_imu`,
`orbslam3_rgbd_inertial`, `infra_anchored`); everything else is conditional and
is added when a result demands it. Sixteen rows on a 156 s lap is a survey this
sequence cannot support.
*Start with:* T0 — score `/mobile_1/zed/odom` and `/mobile_1/zed/pose`, which are
already in the bag and which nothing in this repo had noticed. One hour, no
container, and it sets the bar the rest of the phase is measured against.
*Result:*

### Phase 5 — what this sequence cannot decide ⬜
Write up the limits with numbers rather than hedges: the fraction of the method
spread that falls inside the reference's uncertainty, the revisit count, the
path length. Scope the follow-up recording from that (see README, "Which
environment").
*Result:*

## Progress log

| date | phase | what changed |
|---|---|---|
| 2026-09-14 | 0 | Evaluator, configs, container contract, docs. 32 self-tests + e2e smoke green. |
| 2026-09-19 | 6 | **Windows measured for all six agent×board pairs**, and running them exposed a defect in the evaluator itself. `absolute_check` compared the platform's mean position against the board's surveyed position — but the platform parks ~0.7 m away *looking* at the board, so that residual is the **standoff**, not an error; against a 7 mm sigma it would have read "resolved above the survey's uncertainty" on a perfect estimate, every time. Exactly the "completes, validates, and is geometrically wrong" failure rule 3 exists to prevent, sitting in the scorer. Rewritten: tier 2 now takes `observations` (the board's position in the camera frame per frame, from ChArUco+PnP) and computes `est_pose @ p_board_cam` against the survey — a true absolute error — with a surveyed `standoff` as a weaker fallback, and **refuses** on a window alone. The old unit test asserted the standoff residual (`< 0.35 m`), i.e. it was pinning the defect in place; replaced with three tests covering refusal, a perfect observed estimate (0 mm), a 3 cm biased one (3 cm, correctly above sigma), out-of-window observations, and the standoff path. The e2e smoke test now synthesises detections so it exercises the real path. **Correction to an earlier entry:** `mobile_1` *does* see `anchor_b` — at 4.13 m, which position-only search missed by requiring 2 m. All three boards are available to both agents; the `mobile_1`/`anchor_b` pair is the weakest (a board at 4 m is small in 960×540, so its PnP will be the noisiest of the six). |
| 2026-09-19 | 6 | Wrote `scripts/board_visibility.py` after the user pushed back, correctly, on my claim that I2 needed human eyeballing. The reference trajectory carries full 6-DoF poses; phase0's dwell finder used only position and speed, so it could not separate "parked facing the board" from "drove slowly past" — but that was my implementation being lazy, not the data being insufficient. Projecting each surveyed board through the pose into the camera settles it: behind-vs-in-front, inside-vs-outside the image, range, pixel position, and contiguous windows, with intrinsics read from the bag's own `camera_info`. Synthetic test drives the case position+speed cannot decide: same position, same speed, camera rotated 180° — projection marks the facing half visible and the turned-away half not. What geometry genuinely cannot settle is the board's **facing** (the survey gives position, not normal) plus occlusion and blur, so the output says "candidate" and points at the ChArUco detector, which settles both definitively and returns a pose measurement as a bonus. |
| 2026-09-19 | 6 | **Infra chain validated end-to-end.** Re-run on the corrected pose: range residual −5.5 m → **−0.12 m / +0.16 m**, bearing ~9° → **+0.28° / +0.16°**, and the nearest-return check at ±0.01 m confirms the carts are detected where predicted; the wrong hypotheses now carry the −6 m error. No clock offset survives (bearing at dt=0 already sub-degree, range worse at the scan optimum — tool updated to call that noise rather than advise a wider scan). The radar's 9.30 m ceiling (p99.9 = MAX, nothing past 12 m) is a **duty cycle** rather than a veto: 93%/95% of sweeps still associated, and `infra_diag` now reports each trajectory's in-range fraction so the `+infra` rung can be priced honestly. The TF dump also surfaced `map → *_optical_frame` quaternions for both agents (start poses matching `expected_start` to 2–4 cm) — the lever for reconciling T0's 113–118° rotation artifact. |
| 2026-09-19 | 6 | **Infra resolved: the config's mast pose was wrong by 5.84 m / 9.34°**, exactly the residual phase0 measured on both agents. Fixed in both configs (rule 4, same commit); cam→radar verified identical to its quaternion. The detections were the carts; the range-envelope worry is closed and `+infra` is a live factor class. Retracted my round-2 "composed rotation is wrong" claim — that test compared radar-frame elevation against map-frame elevation, which differ legitimately for a tilted sensor. **Every `infra_1` number the converter emitted inherits the bad pose and must be regenerated**, including anything in the sibling perception study. |
| 2026-09-19 | 6 | T0 scored and infra round 1 analysed (see phase 6 header table). infra_diag upgraded to v2 with the rngNEAR discriminator — verified both ways on synthetic data: cart-at-range found despite same-bearing near clutter (rngNEAR≈0 while the az-pick misleads), and a genuine geometry error shows in rngNEAR too. Per-agent clock scan added because the round-1 scan shifted both agents by one dt while each machine has its own clock — m1's bearing is already centred while m2's improves monotonically toward the scan edge. |
| 2026-09-19 | 6 | Built `scripts/infra_diag.py` to chase the −5.4 m infra residual mechanically: it enumerates the four pose-direction hypotheses (node pose forward/inverted × cam→radar extrinsic forward/inverted), scores each against BOTH reference trajectories with a loose association, ranks them by how completely they fix both agents at once (a frame error is common to the agents; a per-agent fix is a coincidence), and if even the winner leaves a systematic residual, scans the clock offset before anyone blames the survey. The synthetic truth test caught a design flaw before the bag could: the planned ROS-vs-optical axes dimension is **unidentifiable** with point-cloud detections, because prediction and detection get the same relabelling and the residual cancels — planted-truth recovery kept picking an arbitrary axes label until the dimension was removed. All four pose conventions now recover exactly from planted synthetic truth. If nothing associates under any hypothesis, the script says the survey itself is the problem, which is a different and worse finding. |
| 2026-09-19 | 6 | **I13 resolved.** The user pulled the wrapper's static TF on the cart: `zed_left_camera_frame→zed_imu_link`, t=[−0.0020, −0.0231, 0.0002] m, q_xyzw=[−0.008474, −0.000611, 0.007383, 0.999937]. That is a **1.290° per-unit misalignment** with a 23.1 mm lever arm — the exact magnitude that justified refusing a nominal transform (≈0.22 m/s² of gravity leaking on every rotation). Composed with tf_static's frame→optical rpy(−90,0,−90) into `IMU.T_b_c1` (imu←left-optical) and filled into both tightly-coupled configs; self-check passes (imu +z maps to optical −y, off-axis components ≈ the stated misalignment). Neither the bag's tf_static nor the `SN*.conf` carried it — the cart's live TF did, which is worth remembering as the recovery route. The tight rows are now held by **I14 alone** (four noise terms); the PyPI `pyzed` package turned out to be an impostor (3 kB, unrelated), so the SDK factory-noise route goes through `get_python_api.py` on a machine with the real SDK, or the overnight Allan log decides it. |
| 2026-09-19 | 6 | **Phase 0 ran on the real bag.** Both references export and pass the start check (32.4 mm / 44.7 mm); the 1516-count is confirmed and the config's 2833 was stale; paths are 25.8 m / 24.7 m. **I2**: dwell candidates for every board measured and written into the config — `mobile_1` never visits `anchor_b`, so its graph gets two boards; the agents swap corners. **I8/B0: 100% place overlap, closest 0.00 m** — the inter-agent factor class is fully live. **I12 closed**: the Doppler field is named `doppler` on all three radars. **I13 inverted**: the ZED IMU chain is NOT in `/tf_static` — my earlier "3 messages is the expected shape, probably in there" was wrong, and worse, phase0's detector printed a false YES by matching `os_imu`; detector fixed to require zed+imu, verdict corrected to SDK/`SN*.conf`. **The depth-dropout hypothesis failed measurement**: ZED depth is median 85.5% valid with ZERO starved frames (RealSense 72.4%, also zero) — so the 6 m ZED odometry error CANNOT be explained by depth dropout, the RGB-D arm is not structurally handicapped, and the reflective-wall re-ordering weakens; the 6 m now needs a proper aligned score before any further diagnosis. **Infra chain AMBER**: visibility is fine (73%/56%) but the range residual is −5.4 m systematic on BOTH agents and bearing medians are +3.0°/+8.6° — something in the chain (node pose or extrinsic direction, the check's own frame math, or the third clock) is wrong, and no infra factor enters any graph until it is found; phase0's lenient bearing verdict tightened. **cuVSLAM first look**: path length within 0.5% of the mobile_2 reference — promising, unmeasured; registered as `configs/methods/cuvslam_odom.yaml` with the read-its-score caveats. Two more phase0 defects fixed from the run: tf_static duplicates now deduped. |
| 2026-09-19 | 6 | Phase 0 turned into tooling, since the bag never enters this container: `scripts/phase0.py` runs sections A–G of stage 0 in one command against the bag (references→TUM with the start refusal, the four recorded baselines→TUM, board dwell **candidates** for I2, the full `/tf_static` dump with the ZED-IMU chain flagged for I13, gate B0, `infra_1` visibility + gated bearing/range residuals with the gating caveat stated, and the radar field layouts for I12). Pure helpers (`xyz_of`, `find_dwells`, `gated_residual`, `pose_topic_to_traj`, `fields_of`) are all covered by synthetic tests — padding+NaN clouds, a constructed dwell, azimuth wraparound, unsorted/duplicate stamps. **Fixed a crash-on-first-message bug in `export_reference.py`**: it unpacked `iter_messages` as `(topic, msg, stamp_ns)` where the reader yields `(topic, stamp_ns, msg)` — it had never met a real bag. Registered from the topic table that `mobile_2` carries its own recorded VIO (**Isaac cuVSLAM**: `tracking/odometry`, `vo_pose`, + `vo_pose_covariance`) — the tier-0 incumbent for the constrained agent, exported by phase0 alongside the ZED rows. |
| 2026-09-18 | 6 | Answered the structural question the plan had been skirting: **is the SLAM work GT production or dataset benchmark? Both, behind a wall.** Added "The two roles" to `PLAN.md`. On `mobile_1` the GT is the LiDAR pipeline, so every camera/IMU/radar method is uncorrelated with it and all of them are legal benchmark rows. On `mobile_2` the GT *is* the anchored construction, so SLAM is Role A first (one frozen backbone inside the graph) and Role B second (everything else scored against the result) — and the backbone that fed the GT is **excluded from the peer table**, exactly the `correlated_with: [glim]` rule one level down; when `mobile_2`'s reference is released its config declares `correlated_with: [<backbone>]`. The escape hatch is measured rather than argued: stage 3c's consensus check quantifies how much the GT moves under a backbone swap, and if that is below the GT's own certified error bar the reference is anchor-determined and the daggered row becomes readable. Also named the third role — the certification run on `mobile_1` is neither GT production nor benchmark but **GT validation**, the methods contribution that licenses the released reference. |
| 2026-09-18 | 6 | Added `splat_slam` (RGB-only Gaussian-splatting SLAM, conditional) in two roles and explicitly NOT a third: not a consensus backbone, because its learned-flow tracking shares a failure family with `mast3r_slam` and two learned entries agreeing through a shared prior would certify nothing — it may replace MASt3R in the learned slot, never sit beside it. Role 1: the B6 reconstruction/novel-view task (Oxford Spires' radiance-field benchmark as precedent, with reflective walls as the stressor theirs lacks). Role 2: the **consumer test** — train the same splat map from pseudo-GT poses and from reference poses on `mobile_1`; the held-out-view rendering gap is what pseudo-GT error costs a downstream user, and the relative version runs on `mobile_2` with no reference at all. RGB-only on purpose: an RGB-D splat variant would inherit exactly the depth dropout this phase routes around. |
| 2026-09-17 | 6 | Consolidated everything into **`docs/PLAN.md`** — five stages with gates, deliverables and costs, declared the file that wins when documents disagree. Stage 0 is a day plus one unattended overnight and produces every input the later stages consume (reference TUM, dropout covariates for both platforms, board windows, the V0 ceiling, the infra residual check, the B0/visibility gates, the cam↔IMU extrinsic, the IMU log, the Doppler field name and sign). Folded the three pending protocol items into `PSEUDO_GT.md` V3/V5: the **block bootstrap** (≈1516 poses are a handful of independent failure episodes, so per-pose error bars overstate confidence by an order of magnitude), the **covariate error model** (the transferred object is a function fitted on covariates observable on both platforms, validated leave-one-segment-out, with interpolation-vs-extrapolation stated), and **backbone consensus** (three backbones with independent failure modes through the same anchored graph — the one uncertainty estimate computable on `mobile_2`, and the project's own seed_independence_check applied one level down). V5 reframed as a falsification test of V3's prediction rather than a formality. Paper tasks fixed at B1–B5 with B2 the headline; S1 demoted to a control row, S3 to an appendix. |
| 2026-09-17 | 6 | **Goal restated: the deliverable is a pseudo-GT for the sensor-constrained agent, not a better odometry.** `mobile_1` is the instrument, not the subject — it carries a LiDAR and a depth camera on one body, so a LiDAR-free construction is certified there against the **held-out** LiDAR and the resulting error distribution transfers to `mobile_2`, which has no reference at all. Searched the literature and wrote `docs/PSEUDO_GT.md`. The survey-prior recipe (Newer College, Oxford Spires ~1-2 cm, ConSLAM, Hilti) always requires the evaluated platform to **carry** the LiDAR; the pseudo-GT-from-a-stronger-SLAM recipe (PIN-SLAM on KITTI 11-21) asserts rather than certifies its substitute; and the GT-uncertainty work that asks the right question (RTS-GT, MoCap2GT, Boxi, Vaidis et al.) applies it to instrumented references on the platform being measured. Nobody builds a pseudo-GT for an agent that **lacks** the reference sensor and certifies its error bar on a twin that carries both. Two consequences reshape the plan: a pseudo-GT is **offline**, so batch optimisation, loop closure and absolute anchors are in scope where an online odometry could not use them, and the sixteen-method survey collapses to **six runs of one optimiser with different factor sets**. **Corrected earlier advice:** I had written that the agent-to-agent rows cannot help because `mobile_2` is the weaker platform — true for improving `mobile_1`, backwards for this goal, since `mobile_1` is LiDAR-grade and is therefore the *strong* partner when `mobile_2` is the subject. Inter-agent anchoring moves from marginal to primary, which also promotes the B0 shared-corridor gate and makes I2 (board dwell windows) load-bearing rather than merely valuable. |
| 2026-09-17 | 6 | **The bar is 6 m.** `/mobile_1/zed/odom` lands ~6 m from the reference on a 16.6 m room — not drift, a failure, and the cause is the room: reflective walls make the ZED's stereo depth drop out. Re-ordered the roster by how much each entry leans on depth, which is the axis that now matters: `kiss_icp` (pure depth geometry) becomes a *measurement of the handicap* rather than a contender; the inertial rows gain, because bridging an intermittent sensor is what inertial fusion is for; and `mast3r_slam`, `radar_inertial_odometry` and `infra_anchored` are promoted because none of them inherits the failure. **Corrected a prediction that was wrong and would have misled:** `rgbd_inertial_radar` said to expect no whole-run gain because "vision does not fail much on a well-lit lab lap" — vision fails here, so the vision-starved segments its claim was scoped to exist on this sequence and no new recording is needed to test it. Wrote `scripts/depth_health.py`, which is now step 1: it reports valid-depth fraction and, more usefully, the **starved stretches** with their lengths and start times, since a uniform 30% dropout and a four-second blackout break a tracker differently and only the second explains 6 m. Decode and run-length logic tested on synthetic 32FC1/16UC1 frames, including that an unknown encoding refuses rather than guessing a factor of 1000. One upside worth recording: a 6 m bar means nothing crowds the reference's 15 mm floor, so every row in this phase will be legible. |
| 2026-09-17 | 6 | Added a `START HERE` section to the plan, because the roster had grown across several passes into something that reads as a run list and is not one. The order is now explicit: score the ZED SDK's own odometry against the reference FIRST — one hour, no container — because that number decides whether there is a gap worth closing at all, and every method choice before it is a guess about an unmeasured quantity. Then a **core five** whose four differences are each attributable to one change. The other eleven are marked conditional with the condition named. Also recorded the two next-session items that are worth more than any row on the list: record the right image (one setting; restores the whole stereo family) and log the IMU overnight (measured once, serves every future recording). |
| 2026-09-17 | 6 | Roster extended to **16**: `mast3r_slam` and a collaborative tier 5. MASt3R-SLAM pairs with `orbslam3_mono_inertial` — both monocular, neither sees the depth, one takes geometry from an IMU and the other from a learned prior — and it is the only entry that could beat the depth sensor where the sensor is weakest (past 12 m, and on textureless surfaces a prior does not need texture for). Reported twice, sim3 and depth-anchored se3, because monocular-with-priors is not reliably metric and the gap between them *is* the scale error. Tier 5 split by mechanism rather than listed flat: `infra_anchored` (5a) is the only **absolute** anchor a LiDAR-free `mobile_1` has, since `infra_1` is static and surveyed, and it inverts that agent's track-C role from control to subject; the agent-to-agent rows (5b) cannot rescue absolute accuracy because `mobile_2` is the weaker platform, and are labelled as answering the common-frame question instead. One real finding on the way: **dropping the LiDAR makes collaboration easier** — track B's pair needs cross-modal place recognition, this one is RGB-D to RGB-D, so `swarm_slam_nolidar` doubles as B4's diagnostic for free. Kimera-Multi is out of tier 5 for the same no-right-image reason that thinned tier 3. **Gate fix:** the rule-3 params-null check only scanned the top level, so `source_runs: {mobile_1: null}` passed it — a dict is not `None`. Made it recurse and name the path; only one config's behaviour changed (my own), and a regression test drives the real `preflight` over the real configs and was verified to fail when the gate is reverted. |
| 2026-09-17 | 6 | Roster finalised at **11 entries** and the plan doc's method section rewritten around it, with a per-entry justification: each has to answer a question no other entry can and pair with something that isolates one variable. The ORB-SLAM3 trio closes a depth x IMU 2x2 on one codebase, so `#7-#5` is the IMU's worth and `#7-#8` is the depth's, neither of which any mix of separate systems can subtract. `rtabmap_rgbd_imu` added as the loosely-coupled inertial row that needs **neither** I13 nor I14, so the inertial claim is never hostage to the overnight static log. Recorded the distinction between the two kinds of hold: `orbslam3_*_inertial` and `radar_inertial_odometry` wait on missing *information*, while `rtabmap_ext_odom` and `rgbd_inertial_radar` wait on a *result* — filling their `*_source` early would mean choosing the winner before measuring it. BAD-SLAM and DROID-SLAM stay in track A and out of phase 6: both need a GPU and neither answers a question the eleven leave open. Ablation B struck from the grid in the doc too. |
| 2026-09-17 | 6 | Replaced `openvins` with `orbslam3_mono_inertial` in the tier-3 mono slot. Two reasons, and the first is the repo's own rule: OpenVINS varies the estimator *and* the sensor set against every RGB-D-inertial row at once, which is exactly the comparison the README refuses and the reason KISS-ICP is the bridge. The mono-inertial number is better had by varying the depth input alone, on the same container and the same calibration — which also yields a 2x2 on one codebase (depth on/off × IMU on/off) that no mix of systems can give. Second, the role first claimed for OpenVINS here was wrong: it was added as the hedge against an ORB-SLAM3 inertial-init failure because it can initialise statically, but static init recovers gravity and biases and **not scale**, and monocular scale needs translational excitation — so on this motion it is the first entry to fail, not the fallback. Kept as `status: deferred` with the reversal condition (worth running as *stereo*-inertial once the right image exists, where it is metric without the IMU and the filter-vs-optimiser question is finally clean). |
| 2026-09-17 | 6 | Method roster written: 11 configs for phase 6, all loading and all validated against their declared streams. Six pass preflight today; five are correctly held by a `null` (the camera↔IMU extrinsic, the four IMU noise terms, the radar Doppler field name and sign, and the two `*_source` run directories that are results rather than settings). No new gate code was needed — the existing params-null check in `run_method.py` already enforces rule 3 for exactly these. Two pre-existing blockers cleared on the way: `orbslam3_rgbd`'s `DepthMapFactor` is derivable from the stream's own `depth_scale` rather than a value needing declaration, and a modality mismatch of my own making (the ZED SDK rows claimed `rgbd_inertial` against an `rgbd` stream — they use the manufacturer's IMU internally but consume no IMU stream from us). Roster shape is set by the no-right-image finding: the tightly-coupled RGB-D-inertial field is thin, so ORB-SLAM3 `IMU_RGBD` is the only real peer entry and OpenVINS enters as *monocular*-inertial in its own block, with a scale caveat rather than as a peer. |
| 2026-09-17 | 6 | Q1-Q3 answered. Reference `.txt` is `zed_left_camera_optical_frame`, so it compares with **no transform** — the 13 cm lever arm that would have looked exactly like drift is not in play. The Ouster IMU **counts as using the LiDAR** and is excluded, dropping ablation B. Radar clouds **do** carry Doppler, so tier 4 is live as an ego-velocity arm. Q4 answered and split: the camera↔IMU extrinsic is recoverable three ways and probably already in `/tf_static` (3 messages is the expected shape — one array per static publisher — not a shortfall), while the noise model is **not recoverable from a 156 s bag** and wants one unattended overnight static log. Neither blocks the ladder; they gate the tightly-coupled tier-3 rows only. Flagged a risk that is not about either: flat constant-velocity pushcart motion is near-degenerate for visual-inertial initialisation, which predicts an ORB-SLAM3 inertial-init failure rather than explaining one afterwards. |
| 2026-09-16 | 6 | Plan for LiDAR-free `mobile_1` odometry (`docs/MOBILE1_LIDARFREE_ODOMETRY.md`). Four facts read off the recording's topic table, three of which the configs did not carry: **I1 resolved — the bag has three IMUs**, the ZED's at 192 Hz, which unblocks the entire inertial arm; **only the left image was recorded**, so stereo and stereo-inertial are impossible on this sequence; **the ZED SDK's own VIO and its loop-closed pose are already in the bag** and were referenced nowhere, so the incumbent had never been scored; and `global_pose` is 1516 msgs at 9.70 Hz — exactly one per LiDAR sweep — against the 2833/18.1 Hz this repo recorded. |
| 2026-09-15 | 0 | Tracks A/B/C: observability, sensor ablation, collaborative and infrastructure metrics, precheck and collaborative evaluator. 59 self-tests green. Three defects caught by running the precheck on a fixture: bearings were computed in a body frame for a pose describing an optical one (a node 1.8 m above read the agents at +70 deg elevation); observability returned zeros for a starved normal estimate, indistinguishable from degenerate geometry, on exactly the density axis the track exists to measure; and the beam-decimation cells read as degenerate for the same reason until the voxel size was made to scale with density. |
