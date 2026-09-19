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
| I2 | **Board dwell windows — MEASURED, 5 of 6 pairs usable.** `mobile_1`: `anchor` 6.47 s @ 0.72 m (clamped to the pipeline's `departure_t`), `rs_anchor` 26.60 s @ 0.75 m, `anchor_b` **none** (never closer than 2.48 m → 0.5 px/bit). `mobile_2`: `anchor` 6.70 s @ 0.66 m (161 poses), `anchor_b` 6.00 s @ 0.84 m, `rs_anchor` 11.30 s @ 0.81 m. All gated on four conditions at once — in frame, facing the board's +x normal, ChArUco detectable at that range, and slow. In `configs/coop2.yaml` under `windows_by_agent`. **The observations already exist** — the mapping pipeline emits per-frame board-derived camera poses in map (`zed_cam_in_map.tum`, 112 rows @ 15 Hz, std 1.98 mm; `realsense_cam_in_map.tum`, 92 rows, 6.93 mm). Verified against the session anchor's fused pose (0.76 mm) and `/tf` orientation (1.21°). Point `observed_poses_by_agent` at them; **no detector run needed**. | (superseded) |
| ~~I2~~ | ~~**The three board dwell windows**~~ — `scripts/board_visibility.py` (2026-09-19) answers this **geometrically, not by eye**: the reference carries full 6-DoF poses, so projecting each surveyed board into the camera settles in-front-vs-behind, in-FoV-vs-out, range and pixel position automatically. My earlier claim that this needed human eyeballing was wrong — phase0's finder used position and speed only, which was a lazy implementation, not a limitation of the data. Geometry still cannot see the board's **facing** (coop2 surveys position, not normal) or occlusion/blur; the ChArUco detector settles those *and* yields a direct pose measurement, and `../radar_camera_calibration/general_charuco.py` already does detection+PnP. Candidates MEASURED by phase0 (2026-09-19), in `configs/coop2.yaml` beside each anchor, awaiting eye confirmation. Coverage: **`mobile_1` has TWO usable boards, not three** — and my intermediate 'correction' saying otherwise was wrong. Sequence: phase0 (position only) said `mobile_1` never visits `anchor_b`; the FoV check found it *in frame* at 4.13 m and I called that a correction; the board **designs** then settled it — an 18 cm board with 15 mm markers spans 2.0 px per marker at 4.13 m, which no detector finds. In frame ≠ detectable. Usable pairs: `mobile_1` × {`anchor`, `rs_anchor`}, `mobile_2` × {`anchor`, `anchor_b`, `rs_anchor`}. Windows in `configs/coop2.yaml` under `windows_by_agent`. The agents swap corners over the run. — when each platform stood at `anchor`, `anchor_b`, `rs_anchor`, and at what offset. | ⬜ open | Without them tier 2 is silent and no independent evidence enters the benchmark. The single highest-value open item. |
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

**STAGE 1 RIG BUILT (2026-09-19) — six cells, all six pass preflight.**
`docs/PLAN.md` stage 1 is three backbones with independent failure modes, each
on both platforms. The containers are now buildable and the harness feeds them
correctly; nothing has been *run* — there is no Docker daemon in this
environment, so execution is the user's step.

| backbone | image | fails when | m1.zed_rgbd | m2.realsense_rgbd |
|---|---|---|---|---|
| `kiss_icp` | `slambench/kiss-icp:humble` | depth drops out | ✅ preflight | ✅ preflight |
| `rtabmap_rgbd_imu` | `slambench/rtabmap:humble` | low texture, low excitation | ✅ preflight | ✅ preflight |
| `mast3r_slam` | `slambench/mast3r-slam:humble` (GPU) | prior out of distribution | ✅ preflight | ✅ preflight |

**Four defects found while wiring it, each of which produces a run that
completes and is wrong — the class rule 3 exists for:**

1. **An inertial method was never played its IMU.** `preflight` built
   `SLAM_TOPICS` from the *stream's* `*_topic` keys only, and the container sees
   exactly `ros2 bag play --topics $SLAM_TOPICS`. So `rtabmap_rgbd_imu` would
   have run IMU-starved and its number would have read as "the IMU did not
   help" — the precise claim the row exists to test. `/tf_static` was missing
   too (RTAB-Map will not start RGB-D odometry without it). `/tf` is
   deliberately still excluded: it carries the recording pipeline's own
   `map→sensor` estimate, and replaying that into a method is at best a tf
   conflict and at worst a method reading its answer off its input.
   Regression test verified to fail when the fix is reverted.
2. **`SLAM_LAUNCH` was expanded at image-build time.** A Dockerfile `ENV`
   substitutes `${...}` when the image is built, so `Dockerfile.kiss-icp`'s
   `topic:=${SLAM_CLOUD_TOPIC}` baked in the **empty string** and the node would
   have started subscribed to nothing. Replaced by a per-image
   `/opt/slambench/launch.sh`.
3. **KISS-ICP's own launch file would have discarded this project's tuning.**
   `odometry.launch.py` hard-codes `max_range=100`, `min_range=5`,
   `voxel_size=1.0` and says in a comment that they are "not exposed through the
   launch system" — the 100 m automotive defaults, on a 16.6 m room, while
   `kiss_icp.yaml` claimed the indoor ones. A run tuned differently from its own
   manifest is rule 7's exact failure. The node is now run directly and every
   declared parameter is passed.
4. **MASt3R-SLAM would have produced a trajectory on a fabricated clock.** Its
   folder loader numbers frames `index / 30.0`; the recording is 14.7 Hz. Those
   numbers are not times. `bag_to_frames.py` writes the real stamps out and
   `restamp_tum.py` inverts the map exactly — or refuses, because a
   resynchronisation that "mostly works" is a time offset nobody finds later.
   (Naming the folder `tum` to reach its TUM loader was the tempting shortcut
   and is worse: that loader applies hard-coded Freiburg intrinsics.)

**Two things resolved by reading implementations rather than choosing values:**

* `mast3r_slam.params.image_size` was `null` and blocked preflight. It is a
  **hard-coded attribute** of MASt3R-SLAM's `MonocularDataset` (=512, the
  resolution its checkpoint is named for) with no CLI or config key exposing it.
  Recorded as 512 with that provenance; there was never a knob.
* RTAB-Map's `frame_id` decides which frame the poses come out in. It is now
  **read off the bag** — the first depth-image header — rather than declared, so
  the container emits the sensor frame it was given and the evaluator applies
  `reference_frame_from_sensor` exactly once. A `base_link` there would have
  emitted body poses that look fine and are wrong by the camera's lever arm.

**New container-side pieces, all self-tested without ROS, bag, GPU or network**
(`scripts/test_slambench.py`, **73 passed**, 11 new): `depth_to_cloud.py` (the
pinhole bridge that lets the geometry-only backbone eat a depth camera; refuses
when the declared `depth_scale` contradicts the image encoding — metres vs
millimetres is the factor of 1000 that "produces a map of a room 1000 m
across"), `sniff_frame.py`, `params_to_args.py` (slash keys to RTAB-Map, plain
keys to ROS; `Grid/3D: "True"` would have parsed as **false**),
`bag_to_frames.py` (honours `step`, so a row-padded image does not come out
sheared), `restamp_tum.py`.

**Next, and it needs a machine with Docker:** build the four images, run the six
cells, score the `mobile_1` pair against the reference. Gate: at least one
backbone tracks ≥ 90% of the run on each platform.

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
*Phase 1 design note (2026-09-19):* asked why RTAB-Map isn't run without depth.
It cannot be — its visual odometry needs depth or stereo, and there is no mono
mode; depth-free RTAB-Map *is* `rtabmap_ext_odom`, riding someone else's
odometry. But the question is the right one given T0's **+9.1% scale error**,
and the roster already carries the depth-free answers: `mast3r_slam` (learned
prior) and `orbslam3_mono_inertial` (scale from the IMU). What was missing is
the measurement that decides whether depth should be used at all, so
`scripts/depth_vs_lidar.py` now projects Ouster points into the ZED depth image
on the same rigid body and fits `zed = a·lidar + b`. If `a ≈ 1.09` the depth
carries T0's error and the construction should be built depth-free (or the
depth rescaled); if `a ≈ 1.00` the depth is sound and the fault is the ZED's
tracker. Per-range-band ratios separate a scale error from a constant offset.
Verified on synthetic data: planted +9% scale and +5 cm offset both recover
exactly, and the band table distinguishes them (scale is flat across bands,
offset decays toward 1.0).

*Depth-vs-LiDAR, first run (2026-09-19) — and the tool's own verdict was
wrong, twice over.* It printed "the depth is SOUND" from a median ratio of
0.9862. Two problems. **(a) Sampling:** 744,922 of 744,972 pixel pairs landed
in the 1–2 m band — essentially the floor ahead of the cart, seen by both
sensors every frame — while `depth_health` puts the depth image's *median* at
2.75 m. The pairing sampled one surface, not the sensor. Now stratified: capped
per band per frame. **(b) The verdict itself:** the fit (`a=0.779, b=+0.282`)
disagreed violently with the median (0.9862), and I first called that a
lever-arm artefact. Wrong — 1→2 m is a factor of two and has ample lever arm.
The disagreement is the **real finding**: the depth error is
**range-dependent**, over-reading **+6% at 1 m** and under-reading **−8% at
2 m**, and the median ratio sat near 1.0 only because it is the trend's value
at the median sampled range. So *neither* hypothesis survives — the depth is
not a clean +9% scale, and it is not sound. The verdict logic now names the
range-dependent case explicitly (verified on synthetic data: it fires on a
sloped error, stays quiet on a pure scale or a pure offset) and withholds any
verdict while fewer than three bands have samples. **The ranges that dominate a
trajectory scale error, 3–8 m, remain unmeasured.**

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
| 2026-09-19 | 6 | **The fail-fast worked, and caught the next layer.** With the images finally rebuilt in order, the new entrypoint refused within 3 s: *"the recorder did not start -- refusing to replay the bag into nothing"*. Cause: the `--ros-args -p use_sim_time:=true` override added to fix the previous crash is rejected by the recorder's own `parse_args()` — argparse exits 2 on flags it does not know, another first-line death. `parse()` now uses `parse_known_args` and hands the remainder to `rclpy.init(args=...)`. Unit-tested without ROS. 87 self-tests green. |
| 2026-09-19 | 6 | **The recorder fix never reached the container — my rebuild instructions were wrong.** After the `use_sim_time` fix, the running container's log had no `recorder` line at all, neither `up` nor `did not start`, which the new entrypoint prints within 3 s. `slambench/rtabmap:humble` is `FROM slambench/base:humble`, and Docker snapshots the base's layers into the method image at *its* build time; rebuilding base alone changes nothing in it. I told the user to rebuild base and run. Every run since the fix used rtabmap's stale copy of the crashing recorder. `docker/build.sh` now builds every image in dependency order, and the README says: always the script, never one image. |
| 2026-09-19 | 6 | **ROOT CAUSE: the recorder has been dead on arrival since the first run.** `Recorder.__init__` called `self.declare_parameter("use_sim_time", True)`, but rclpy's `Node.__init__` **already declares it** (`TimeSource.attach_node`, rclpy/time_source.py:82), so the call raises `ParameterAlreadyDeclaredException` and the node dies on construction. The entrypoint backgrounds it with `&`, which is perfectly happy with a process that exits immediately — so **every run replayed 156 s of bag into a dead recorder** and produced a database with no trajectory. I had been attributing the missing `.tum` files to the interrupts; the interrupts were real but they were not this. Declaring it would not have worked anyway: sim time is switched on by a parameter **override at launch**, which the entrypoint now passes to both nodes. The entrypoint also **checks the recorder is alive and has opened its output before replaying**, since three minutes is far too long to discover this at the end. Two regression tests, both verified to fail on revert — and the first version of the redeclaration test passed for the wrong reason, matching the explanatory comment rather than code, so it now parses the AST instead of grepping. 86 self-tests green. |
| 2026-09-19 | 6 | **The deliverable no longer depends on a clean exit — and seven stale containers were probably the dropped frames.** `docker stop -t 60` on the hung run still produced only `rtabmap.db`, *and no `INTERRUPTED` marker*, which localises it exactly: that image had the trap but not the bounded waits, so `interrupted()` blocked in `shutdown()` on the method's `wait`, the recorder was never signalled, and the `touch` after it never ran. `record_tum.py` now writes `trajectory.tum` **on every optimised-graph message** via a temp-file rename, so a completed run survives being SIGKILLed — a kill mid-write leaves the previous good graph, not half of the new one — and flushes the odometry every 50 poses (~3 s) instead of every 200. **`docker ps` showed SEVEN containers still running**, one per earlier attempt, all hung in shutdown. Seven concurrent RTAB-Map instances saturate the CPU, and a starved inertial subscriber is precisely what emits *"We didn't receive IMU newer than previous image/scan"* — so CPU contention is now the leading explanation for the ~580 otherwise-unexplained dropped frames, testable by one run on an idle machine. They could not have interfered via DDS: `--network none` gives each its own network namespace. |
| 2026-09-19 | 6 | **Bounded the container shutdown.** The replay completed (last frame at bag +156.0 s, the bag ends at +156.3) and then the entrypoint stopped returning: the shutdown did `kill -INT` followed by an **unbounded** `wait`, so one node that ignores SIGINT turns a finished 156 s replay into a run that never ends — and by that point every pose is already computed, so the cost of hanging is the whole run for nothing. `stop_pid` now polls for a grace period and escalates to SIGKILL: 20 s for the method, bridge and transform publishers, **45 s for the recorder**, which is stopped last and gets the longest grace because its clean exit *is* the deliverable. Verified both branches — a process that ignores SIGINT is killed at the limit, a well-behaved one returns immediately rather than waiting the timeout out. |
| 2026-09-19 | 6 | **The run directory was empty because the run was interrupted, and the harness then kept a STALE manifest.** `KeyboardInterrupt` landed in `run_method.py` at `subprocess.run`, so `docker run` forwarded it to PID 1 and the entrypoint — which had no trap — died on the spot: the recorder never reached `finish()`, and three minutes of replay produced only `rtabmap.db` (written live by RTAB-Map) plus a `run.json` from an earlier run. A run directory describing a *different* run is worse than an empty one. Fixed at both ends: the entrypoint traps INT/TERM, runs the same shutdown path, and drops an `INTERRUPTED` marker so no table quotes it as complete (rule 8 — a partial trajectory is a result; nothing at all is not); `run_method.py` catches the interrupt and still writes the manifest. **IMU coverage measured:** starts **+0.94 s** and ends **−2.92 s** against the image stream — ~14 frames at the head, ~43 at the tail, **~57 of the ~637 missing frames.** The other ~580 are unexplained, so `odom_health` now parses the bag stamp out of each dropped-frame message and histograms it: clustered at the ends means coverage, spread through the middle means frames lost at run time. Verified on synthetic logs of both shapes. |
| 2026-09-19 | 6 | **`rtabmap_rgbd_imu` × `mobile_1` PASSES the stage-1 gate — with a denominator caveat I had to fix first.** Full run: **1641/1651 tracked (99.4% of frames PROCESSED)**, median 290 inliers, p05 133, **one** automatic reset, one 0.6 s gap at 82.0 s and 99.9% clean before it. `Odom/ResetCountdown: 8` did exactly its job: the same 82 s event that previously ended the run cost 9 frames. **But 1651 is not 2288.** The estimator only ever saw ~72% of the colour/depth pairs — the rest were dropped upstream by *"We didn't receive IMU newer than previous image/scan"*, which `odom_health` was not counting, so those frames vanished from the denominator instead of appearing as a loss. A tracked fraction that quietly shrinks its own denominator is the most flattering error a benchmark can make, so the tool now counts that message, reports **processed vs. whole-run** separately, and gates on the run when `--expected-frames` is given. The honest pair is **99.4% of processed / ~72% of the run**, and the shortfall is far larger than the ~58 frames the IMU's 3.9 s tail gap explains — cause still open. **`timing.json` is missing** from the run directory, so the trajectory may not have been written: unresolved. |
| 2026-09-19 | 6 | **The IMU stream is shorter than the image stream, and it costs frames at the tail.** `/mobile_1/zed/imu/data` spans 152.35 s; the depth stream spans 156.21 s. With `wait_imu_to_init`, RTAB-Map requires an inertial sample newer than each image, so once the IMU ends every remaining image is dropped — *"We didn't receive IMU newer than previous image/scan ... The previous image/scan is dropped!"*, once per frame to the end of the bag. `bag_probe` now measures coverage **at both ends** rather than printing two spans: two spans say nothing about alignment, and 152 vs 156 could overlap perfectly or miss by 4 s at either end. It reports the head and tail gaps and names the consequence in frames. **This is a property of the recording, not of the method** — the shortened evaluation window gets reported; those frames are not counted as lost tracking. Add "stop the IMU last" to the next-session list. |
| 2026-09-19 | 6 | **First `rtabmap_rgbd_imu` measurement, and the cause is a DEFAULT, not the room.** `odom_health`: 697 frames, **186 tracked (26.7%)**, but inliers *while tracking* are **median 245, p05 95** — vision in this room is strong, not marginal. The loss is a single event at **12.7 s** that ran 510 frames (34.7 s) and **never recovered**, because RTAB-Map's `Odom/ResetCountdown` defaults to **0**, documented as "a value of 0 disables auto-reset". So one bad moment ends the run by design and every frame after it is lost for free. Declared as 8 frames (~0.54 s at 14.9 Hz) in both RTAB-Map configs with the published default named, per rule 7. A reset is a **discontinuity, not a repair** — it resumes from the last good pose with large covariance, so the motion across it was never observed — and `odom_health` now counts resets and reports the tracked fraction *before the first loss* separately, since a run that is perfect for 12 s then dead otherwise reads like one that was mediocre throughout. Also confirmed benign: the 158 `does not exist` warnings are the mapping node's cosmetic `odom` lookup, and the 9 `Dropping imu data` are startup only, before the static publisher came up. **This run was interrupted (~47 s of a 156 s bag), so 26.7% is a partial-run number** and the re-run must be allowed to finish. |
| 2026-09-19 | 6 | **`publish_tf_odom:=true` was wrong, for the opposite reason I checked.** I turned it on reasoning that nothing else publishes `odom`, so no conflict. Wrong half of the edge: a tf frame has exactly ONE parent, and the bag's `tf_static` already gives `zed_left_camera_optical_frame` one (`zed_left_camera_frame`). Publishing `odom → zed_left_camera_optical_frame` **re-parents** it and deletes the static link — and with it the only camera→IMU path. The tell is exact: *"Lookup would require extrapolation into the future ... from [zed_imu_link] to [zed_left_camera_optical_frame]"*. Two statically-related frames can never extrapolate; that they did proves the path was routing through a dynamic edge. Reverted, with the real reason recorded. The mapping node's "cannot look up odom" warning is cosmetic — it also subscribes to the odometry topic, which is how it built a 58-node map in the prior run. **Open, and not guessed at: odometry lost tracking at bag time ~851.7 s (~42 s in)** — `quality=0`, `Not enough inliers 0/15`, guesses of 12–15 m in a 16.6 m room. Whether the tf break caused that or merely followed it is undetermined; the re-run decides. `scripts/odom_health.py` added so the answer is a number: tracked fraction, lost-tracking **stretches** with start times (scattered zeros are ridden through, one long gap is where the trajectory leaves the room), warning counts by cause, and the stage-1 ≥90% gate stated rather than implied. 83 self-tests green. |
| 2026-09-19 | 6 | **RTAB-Map tracks on `mobile_1` — and the run was about to score the wrong trajectory.** With the IMU edge published and `sync: EXACT`, `rgbd_odometry` runs clean: quality 275–315 inliers, 5–9 mm / 11–16 mrad std dev, ~35 ms per update, map growing (WM=58). No IMU warnings, no sync warnings. **Two defects the log exposed.** (a) I had set `publish_tf_odom:=false` out of caution about tf conflicts; nothing else publishes `odom` (the bag's `/tf` is deliberately not replayed), and the *mapping* node needs that edge to place its nodes — hence `"odom" passed to lookupTransform ... does not exist` on every frame. Wrong caution, now `true`. (b) **`SLAM_POSE_TOPIC` was `/rtabmap/odom`, the INCREMENTAL estimate**, while the method config declares `loop_closure: true`. Recording that would have put a pure-drift number in a row claiming loop closure. The optimised graph (`/rtabmap/mapPath`) is now `trajectory.tum`; the odometry is kept beside it as `odometry.tum`, which is free and is exactly the drift floor the closing row has to beat, and `timing.json` records which source was used. 82 self-tests green. |
| 2026-09-19 | 6 | **`bag_probe` settled both unknowns, and two of my readings were wrong.** (1) **Colour and depth on `mobile_1` are stamped BYTE-IDENTICALLY** — 2288/2288, |dt| max 0.00 ms. The 67 ms I saw was not a property of the recording: it was `approx_sync:=true`, my own default, mispairing frames one period apart (67.01 ms) *despite a perfect partner existing*. An approximate matcher can mispair; an exact one cannot. Sync mode is now a measured per-stream field in the config — `exact` for `mobile_1`, `approx` with a 5 ms interval for `mobile_2` (0/4295 exact but 10 µs median, i.e. same capture, not bit-identical). (2) **`/tf_static` carries 7 edges in 2 disconnected trees** — the ZED chain and the Ouster chain — and nothing else: no RealSense frames, no `infra_1`, no `zed_imu_link`. So `mobile_2`'s geometry is not recoverable from the bag at all. `mobile_2.imu.sensor_frame` filled in from the header (`camera_imu_optical_frame`), `color_frame` added to both streams, and the colour←depth 59 mm edge is now published into the container for `mobile_2` from the config — without it that offset lands on every point. **The config's claim that loosely-coupled rows do not need the camera↔IMU extrinsic is falsified and corrected**: RTAB-Map refuses to initialise gravity without it and then *continues*, so the row does not fail — it becomes its own IMU-free control. Preflight now refuses, which blocks `rtabmap_rgbd_imu` on `mobile_2` until `rs-enumerate-devices -c` is run once. Stage 1 is therefore **five runnable cells and one named blocker**, with `rtabmap_rgbd` covering `mobile_2`'s feature+depth slot meanwhile. 80 self-tests green. |
| 2026-09-19 | 6 | **RTAB-Map ran, and named two real properties of the recording.** (a) `zed_imu_link` is **not in the bag's `tf_static`**, so every IMU sample was dropped and the inertial row would have completed as its own IMU-free control — a number reading 'the IMU did not help' with no IMU in it. The transform is not lost: it is I13, already resolved in `configs/coop2.yaml` from the wrapper's published TF, so the container now publishes that one edge itself (rule 4's number, not a re-derived one; it travels in `run.json`). (b) **colour and depth stamps are ~67 ms apart — one whole frame at 14.7 Hz — with the sign alternating.** That is not jitter, and no sync interval fixes a pairing that was never recorded. `scripts/bag_probe.py` added to settle both from the bag itself: it dumps the static TF tree as connected components (tf resolves through any path, so 'can these two frames be transformed' is 'are they in one component'), and reports the signed colour/depth gap against the frame period. **Correction:** my first test asserted the interleaved case shows an *alternating* sign. It does not — a uniform offset gives a constant sign; the flips come from jitter straddling the tie. Magnitude against the frame period is the discriminator, and the tool's verdict already used it. 77 self-tests green. |
| 2026-09-19 | 6 | **First real container run: `set -u` vs ROS.** `rtabmap_rgbd_imu` on `mobile_1.zed_rgbd` exited 1 at `/opt/ros/humble/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable` — `set -euo pipefail` killed every script before a line of ours ran, because setup.bash reads that variable with no default. Guarded in all four image scripts (`set +u` around the source, straight back after); the strictness is kept everywhere else, since an unset `SLAM_*` expanding to nothing is exactly how a container runs on starved input. A structural test now walks every `docker/**/*.sh` and fails on an unguarded source, so the next image cannot repeat it — verified to fail on revert. 74 self-tests green. |
| 2026-09-19 | 6 | **Stage 1 rig built; four completes-and-is-wrong defects caught.** An inertial method was never played its IMU (SLAM_TOPICS came from the stream's own keys only, and the container sees nothing else); `SLAM_LAUNCH` expanded at image-BUILD time so kiss-icp subscribed to the empty string; kiss-icp's launch file hard-codes the 100 m automotive tuning and would have silently overridden this project's indoor values; MASt3R-SLAM's folder loader fabricates `index/30.0` timestamps on a 14.7 Hz recording. Each fixed, each with a regression test — the IMU one verified to fail on revert. `mast3r_slam.image_size` resolved by reading the implementation (hard-coded 512, no knob exists), RTAB-Map's `frame_id` now sniffed off the first depth header rather than declared. All six stage-1 cells pass preflight; 73 self-tests green. Nothing run yet — no Docker here. |
| 2026-09-19 | 6 | **Tier 2 closed without a detector run.** `zed_cam_in_map.tum` turned out to be **112 per-frame rows at 15.0 Hz**, not a single fused pose — board-derived camera poses in the map frame, sitting 0.76 mm from the session anchor's fused value with orientation agreeing with `/tf map→zed_left_optical` to 1.21°. So `absolute_check` gained an `observed_poses` mode that scores the estimate against the *pipeline's own detections*, which is strictly better than reimplementing its board conventions — a reimplementation that had already gone wrong once on the +x/+z normal. It reports a rotation residual beside the translation one, refuses on disjoint stamps, and the existing `observations`/`standoff`/refusal paths are unchanged. **`cfg.anchors()` is now per-agent**, resolving `windows_by_agent` and `observed_poses_by_agent`; `mobile_1` correctly yields 2 anchors and `mobile_2` 3. **Side finding that clears T0's mystery:** the board-derived orientation matching `/tf` to 1.21° means the reference's orientation convention is sound, so T0's 113–118° ATE rotation lives in the *estimates*, not the reference. |
| 2026-09-19 | 6 | **All board windows measured; 5 of 6 agent×board pairs usable.** `mobile_2` gets all three boards — and the detectability gate earned its keep here: the old longest-window rule had picked `anchor` at 1.36 m (1.19 px/bit, undetectable) when the same board has a **161-pose window at 0.66 m** (2.44 px/bit); the gate moved it. `anchor_b` is `mobile_2`'s thinnest at 3.4% of the run clearing both gates, one 6 s window. `rs_anchor` deliberately **not** clamped to the pipeline's realsense `departure_t` — that marks the end of the *static* dwell it fused 92 views over, while the later frames are still facing, in range and slow, so they are valid observations that merely aren't static; clamp only if the detector's reprojection degrades past that point. (`mobile_1`'s `anchor` window *is* clamped, because there the overshoot ran past the dwell into motion.) |
| 2026-09-19 | 6 | **`mobile_1` board windows final.** With facing fixed: `anchor` 39.5% facing → best window 92 poses at 0.72 m, 1.90 px/bit, **clamped to the pipeline's own `departure_t`** (1787899810.524 — end of the static dwell it fused 112 views over at 1.98 mm; my geometric window ran 2.6 s past it). `rs_anchor` 31.4% facing → 267 poses at 0.75 m, 1.87 px/bit, the best pair on either agent. **`anchor_b` confirmed dead for `mobile_1`**: in frame 23.9% of the run but never closer than 2.48 m, where a 15 mm marker is 0.5 px/bit. So `mobile_1` has two usable boards. `mobile_2`'s windows in the config are marked **stale/provisional** — they were picked by the old longest-window rule before the detectability and facing gates existed, and its `anchor` pick sits at 1.36 m (1.19 px/bit, undetectable) when closer windows with more poses exist. |
| 2026-09-19 | 6 | **Facing test was rejecting 100% of views — wrong normal axis.** The first run with `--anchor-frame` returned `facing: 0.0%` on every board, which contradicted a known fact: the pipeline detected `anchor` 112 times (std 1.98 mm) from the ZED at `[0.698, −0.063, 0.193]`. I had assumed `board_axes: "ros"` puts the outward normal on **+z**; dotting each board axis against the direction to the camera that actually saw it gives **+0.96 on +x** and only +0.27 on +z, on *both* boards independently. Normal is **+x**. Fixed, with the two known-good detections pinned as a regression test (and the old +z choice asserted to fail, since that is what produced the 0%). Also: `anchor_frame_coop2.json` carries only the boards `03_anchor` used (`anchor`, `rs_anchor`), so `anchor_b` has no orientation there — the script now takes a comma-separated list of anchor-frame files and names any board left without one instead of silently skipping its facing test. |
| 2026-09-19 | 6 | **Board designs and the pipeline config landed, and they overturn two things.** (1) The boards are SMALL — `anchor`/`anchor_b` are 18×14 cm with 15 mm DICT_4X4 markers, `rs_anchor` 22.5×17.5 cm with 18 mm DICT_5X5 — so *range*, not field of view, decides usability: a 15 mm marker spans 11.4 px at 0.72 m (which the pipeline detected: 112 views, std 1.98 mm) and **2.0 px at 4.13 m**, which nothing detects. So **`mobile_1`/`anchor_b` is dead** and my "correction" claiming all three boards work for both agents was wrong — phase0's original position-only verdict was right, for the wrong reason. `board_visibility` now models pixels-per-bit and ranks windows by pose count among *detectable* ones rather than by duration, which had preferred `mobile_2`/`anchor` at 1.36 m (undetectable) over the same board at 0.66 m with 161 poses. (2) The boards now have **orientations** (`anchor_frame_coop2.json`, `board_axes: ros`, +z out of the face), closing my "geometry cannot see facing" caveat — `--anchor-frame` computes incidence and rejects oblique/behind views. **(3) The most consequential finding: board independence is per agent.** `mobile_1`'s published reference is track `mobile_1_lidar`, type `lidar_icp`, with no boards — so boards ARE independent of it. `mobile_2`'s is `traj_mobile_2_rs_odom_icp_boards.tum`, whose recipe is in its name: cuVSLAM odometry + depth ICP + **board detections**. The boards were consumed to build `mobile_2`'s reference and cannot check it. `PSEUDO_GT.md`'s V5 falsification test must therefore rest on `infra_1` and co-observation with `mobile_1` for that agent. |
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
