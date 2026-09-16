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
| I2 | **The three board dwell windows** — when each platform stood at `anchor`, `anchor_b`, `rs_anchor`, and at what offset. | ⬜ open | Without them tier 2 is silent and no independent evidence enters the benchmark. The single highest-value open item. |
| I3 | **The anchored map cloud**: path, and confirmation it is in the post-anchor `map` frame, not the pipeline's pre-anchor frame. | ⬜ open | Every map metric needs it. A pre-anchor cloud scores every method against a frame none of them are in. |
| I4 | **The evaluation volume V.** | ⬜ open | Changes the ranking. Proposal: the map cloud's occupied box eroded 0.25 m; confirm against the cloud, do not guess from the trajectories. |
| I5 | **IMU↔sensor extrinsics + noise model.** I1 is yes, so this is live. | ⬜ open, **now blocking** | Neither the extrinsic nor the gyro/accel noise model is published. `/tf_static` carries only **3 messages** for this whole rig, so check what is actually in it rather than assuming the chain is. A guess costs a tightly-coupled estimator far more than a loosely-coupled one, so an unfavourable inertial result under a guess is not a result. Applies to the camera↔IMU chain too — see `docs/MOBILE1_LIDARFREE_ODOMETRY.md` Q4. |
| I6 | **Does the route revisit?** | ⬜ open | If it does not, loop closure cannot be evaluated on this sequence and the RTAB-Map/GLIM entries measure only their odometry. Answer it from tier 3 on the reference trajectory itself — no method needed. |
| I7 | **Which alignment the headline table uses** (`se3` vs `se3` anchored on the first N poses). | ⬜ open (answered for phase 6) | Best-fit ATE and drift-from-a-fixed-start are different claims. Pick one for the headline, report the other in the appendix. Phase 6 answers it for itself: `align_n_first: 100`, because best-fit systematically flatters a steadily drifting *odometry*. The main table still has to choose. |
| I11 | **What frame and format is the LiDAR-refined reference `.txt`?** | ⬜ open | Blocks phase 6 entirely. `zed_left_camera_optical_frame` drops straight in; `os_lidar` needs the 13 cm / 90° transform applied first, and skipping it is a systematic error that no alignment absorbs and that looks exactly like drift. |
| I12 | **Do the radar clouds carry a Doppler / radial-velocity field?** | ⬜ open | Minutes to answer with `inspect_bag.py`, which already dumps field layouts. Decides whether radar can contribute ego-velocity to phase 6 at all. If the driver published only `x,y,z,intensity`, radar-inertial odometry is not available on this recording and that is a finding about the recording. |
| I8 | **Did the two agents ever occupy the same place?** (track B precondition B1) | ⬜ open | Decides whether collaborative SLAM is runnable on this sequence at all. Answerable today from the reference trajectories with `precheck_tracks.py`. Starts are 16.3 m apart in a 16.6 m room. |
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

### Phase 6 — mobile_1 odometry without the LiDAR ⬜
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
*Blocks on:* **I11** (the `.txt`'s frame) outright. **I12** gates tier 4 only.
I5/Q4 gate the tightly-coupled rows of tier 3, not the loosely-coupled ones.
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
| 2026-09-16 | 6 | Plan for LiDAR-free `mobile_1` odometry (`docs/MOBILE1_LIDARFREE_ODOMETRY.md`). Four facts read off the recording's topic table, three of which the configs did not carry: **I1 resolved — the bag has three IMUs**, the ZED's at 192 Hz, which unblocks the entire inertial arm; **only the left image was recorded**, so stereo and stereo-inertial are impossible on this sequence; **the ZED SDK's own VIO and its loop-closed pose are already in the bag** and were referenced nowhere, so the incumbent had never been scored; and `global_pose` is 1516 msgs at 9.70 Hz — exactly one per LiDAR sweep — against the 2833/18.1 Hz this repo recorded. |
| 2026-09-15 | 0 | Tracks A/B/C: observability, sensor ablation, collaborative and infrastructure metrics, precheck and collaborative evaluator. 59 self-tests green. Three defects caught by running the precheck on a fixture: bearings were computed in a body frame for a pose describing an optical one (a node 1.8 m above read the agents at +70 deg elevation); observability returned zeros for a starved normal estimate, indistinguishable from degenerate geometry, on exactly the density axis the track exists to measure; and the beam-decimation cells read as degenerate for the same reason until the voxel size was made to scale with density. |
