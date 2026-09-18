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
| I2 | **The three board dwell windows** ⚠ *now load-bearing: the only absolute evidence with an independent error bar, and both V0 and V5 of the pseudo-GT protocol need it* — when each platform stood at `anchor`, `anchor_b`, `rs_anchor`, and at what offset. | ⬜ open | Without them tier 2 is silent and no independent evidence enters the benchmark. The single highest-value open item. |
| I3 | **The anchored map cloud**: path, and confirmation it is in the post-anchor `map` frame, not the pipeline's pre-anchor frame. | ⬜ open | Every map metric needs it. A pre-anchor cloud scores every method against a frame none of them are in. |
| I4 | **The evaluation volume V.** | ⬜ open | Changes the ranking. Proposal: the map cloud's occupied box eroded 0.25 m; confirm against the cloud, do not guess from the trajectories. |
| I5 | **IMU↔sensor extrinsics + noise model.** I1 is yes, so this is live. | ⬜ open, **now blocking** | Neither the extrinsic nor the gyro/accel noise model is published. `/tf_static` carries only **3 messages** for this whole rig, so check what is actually in it rather than assuming the chain is. A guess costs a tightly-coupled estimator far more than a loosely-coupled one, so an unfavourable inertial result under a guess is not a result. Applies to the camera↔IMU chain too — see `docs/MOBILE1_LIDARFREE_ODOMETRY.md` Q4. |
| I6 | **Does the route revisit?** | ⬜ open | If it does not, loop closure cannot be evaluated on this sequence and the RTAB-Map/GLIM entries measure only their odometry. Answer it from tier 3 on the reference trajectory itself — no method needed. |
| I7 | **Which alignment the headline table uses** (`se3` vs `se3` anchored on the first N poses). | ⬜ open (answered for phase 6) | Best-fit ATE and drift-from-a-fixed-start are different claims. Pick one for the headline, report the other in the appendix. Phase 6 answers it for itself: `align_n_first: 100`, because best-fit systematically flatters a steadily drifting *odometry*. The main table still has to choose. |
| I11 | **What frame and format is the LiDAR-refined reference `.txt`?** | ✅ **`zed_left_camera_optical_frame`** | The best available answer: it is the frame the reference already declares and the one `mobile_1.zed_rgbd` maps to with an identity extrinsic, so a ZED estimate compares with **no transform at all**. Phase 6's P1 shrinks to a format conversion plus the `expected_start` check. |
| I12 | **Do the radar clouds carry a Doppler / radial-velocity field?** | ✅ **yes — tier 4 is live** | Both `mobile_1` radars carry radial velocity, so they enter phase 6 as ego-velocity sensors (RANSAC + LSQ per sweep), never as scan matchers. One non-gating detail left: the field's NAME, since `velocity`/`doppler`/`v_r` are all in use across TI driver forks, and `ros2opv2v/pointclouds.py` reads `x/y/z/intensity` only. |
| I13 | **The ZED camera↔IMU extrinsic.** | ⬜ open, gates tightly-coupled rows only | Three routes, cheapest first: the SDK's `getSensorsConfiguration().camera_imu_transform`; the factory `SN<serial>.conf`; or `/tf_static` — whose 3 messages are the *expected* shape, one array per static publisher, not a shortfall. The trap is the body↔optical convention change between `zed_camera_center` and `zed_left_camera_optical_frame`. |
| I14 | **The ZED IMU noise model.** | ⬜ **not recoverable from coop2** | Allan variance needs 2-3 h of static logging for the random-walk terms to emerge; this bag is 156 s. Fix costs one overnight with nobody present and is a property of the unit, so it is measured once and serves every future recording. Until then the tightly-coupled rows carry inflated order-of-magnitude values **in their method config**, labelled — erring high, because over-stated noise degrades gracefully toward vision-only while under-stated noise diverges. |
| I15 | **Is there enough rotational excitation to initialise a VI estimator?** | ⬜ open, free to answer | A pushcart rolling flat at walking pace is near the degenerate case: constant velocity makes accelerometer bias and gravity weakly observable. Predicts, rather than explains, an ORB-SLAM3 inertial-init failure. Answerable from the reference trajectory alone during P1, together with whether the run contains a static segment. |
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

### Phase 6 — pseudo ground truth for the sensor-constrained agent ⬜
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
