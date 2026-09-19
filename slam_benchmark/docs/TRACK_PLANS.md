# Track plans

One plan per track: the algorithms, the procedure, the gates, and what each step
produces. `IMPLEMENTATION.md` carries the status; this file carries the method.

Read the shared prerequisites first — all three tracks depend on them, and two
of the three can be blocked by a question that costs an afternoon to answer.

---

## Shared prerequisites

| | what | cost | blocks |
|---|---|---|---|
| P0 | `precheck_tracks.py` on both reference trajectories + one sample scan | minutes | everything; it can close track B outright |
| P0b | Confirm the ZED and RealSense depth scales (32FC1 metres vs 16UC1 mm) per stream | minutes | every RGB-D method in track A — a factor of 1000, and the run completes either way |
| P1 | Export both references; run `reference_accumulation`; **look at the cloud** | ~1 h | every map metric in every track |
| P2 | Resolve I1 (IMU present?) | minutes | the inertial arm of track A |
| P3 | Resolve I2 (board dwell windows) | ~1 h | tier 2 everywhere — the only independent evidence |
| P4 | Resolve I3/I4 (map cloud frame, evaluation volume) | ~1 h | map scores in tracks A and B |

P0 first, always. It is the only step that can tell you a track is not runnable
before you spend a week on it.

---

## Track A — SLAM without a LiDAR

### The question

What does an agent lose by having no LiDAR, which property of the LiDAR that
loss is made of, and is a given failure the method's or the scene's?

### Why coop2 can answer it cleanly

`mobile_1` carries the Ouster **and** the ZED on one rigid body. So the headline
comparison is `mobile_1.ouster` against `mobile_1.zed_rgbd`: one platform, one
motion, one clock, one reference, and the only difference is whether the agent
had a LiDAR. Almost no dataset offers that pair — the usual LiDAR-vs-RGB-D
comparison is across datasets, or against *predicted* depth. Here the depth is
measured and the trajectory is identical.

`mobile_2` is the real LiDAR-free platform and checks that the conclusion
generalises off that one body. It is also the agent tracks B and C are trying to
rescue, so its number here is the baseline both of them are read against.

### Algorithms

| role | algorithm | why this one | runs today |
|---|---|---|---|
| **bridge** | **KISS-ICP** on both sides | The only entry that runs on the Ouster *and* on the reprojected depth. Its two rows differ in the sensor and nothing else, so it — alone — measures the LiDAR's worth. Every other method varies estimator and modality at once. | ✅ |
| sparse | **ORB-SLAM3 (RGB-D)** | The standard. Sparse ORB landmarks, local BA, DBoW2 loop closure. Its known failure modes — low texture, fast rotation — are exactly what a pushcart turning in a lab produces, so it sets the realistic floor. | ✅ |
| dense, practical | **RTAB-Map (RGB-D)** | Appearance loop closure plus a dense map, so it is the only RGB-D entry that scores on the map metrics as well as the trajectory. | ✅ |
| dense, ceiling | **BAD-SLAM** | Direct bundle adjustment over surfels, geometry and photometry jointly. Says how much of the RGB-D disadvantage is the *sparse representation* rather than the sensor. GPU. | ✅ |
| learned | **DROID-SLAM** | Deep dense BA over learned flow. Answers whether ORB-SLAM3's failures survive a learned front-end — if they do not, the RGB-D gap is smaller than the classical field suggests and the track's conclusion changes. GPU, not real time. | ✅ |
| control | KISS-ICP, RTAB-Map on `mobile_1.ouster` | Taken from the main table, not re-run. The ceiling. | ✅ |

Nothing here needs an IMU, so **this track runs today in full** — it does not
wait on I1. If an IMU turns out to exist, add ORB-SLAM3 stereo-inertial as a
second sparse row: it tests whether inertial sensing closes the gap that the
missing LiDAR opened, which is the cheapest realistic fix for a LiDAR-free
platform.

### Two instruments, because there are two ways to fail

A geometric observability metric cannot see a photometric failure. A textureless
white wall at 2 m is geometrically well-conditioned and visually empty; a
richly-textured scene at 15 m is the opposite. Report both per frame:

- **geometric** — `slambench/observability.py`, the point-to-plane information
  matrix of the depth cloud. Its weakest eigenvalue names the translation
  direction the 87° wedge stopped constraining.
- **photometric** — ORB feature count and spatial spread per frame. The standard
  proxy, and the one that explains a tracking loss that geometry says should not
  have happened.

Together they partition every RGB-D failure into *the depth geometry went free*,
*the image went blank*, and *neither — the method failed*. That third category
is the only one that is a statement about the algorithm.

### Procedure

```
# 1. the headline: one estimator, two sensors, one platform
for stream in (mobile_1.ouster, mobile_1.zed_rgbd):
    traj = kiss_icp(stream)
    ate[stream], rpe[stream], tracked[stream] = eval_run(...)
gap = ate[mobile_1.zed_rgbd] - ate[mobile_1.ouster]      # the cost of no LiDAR

# 2. the RGB-D field, on the same stream
for m in (orbslam3_rgbd, rtabmap_rgbd, badslam, droid_slam):
    run on mobile_1.zed_rgbd and on mobile_2.realsense_rgbd

# 3. the instruments, per frame, before and independent of any method
geo[f]  = observability(depth_cloud[f], voxel="auto")
phot[f] = orb_feature_count_and_spread(image[f])

# 4. attribution: what was the gap MADE of?
#    degrade the Ouster toward the depth camera until the bridge's result meets it
for axis in (fov, range, density):
    for level in grid[axis]:
        a = ate[kiss_icp on ablate(ouster, level)]
        if a >= ate[mobile_1.zed_rgbd]:
            crossover[axis] = level      # this much of the LiDAR was the difference
            break
```

Step 4 is what the ablation grid is for. It is not the track; it is the
decomposition that turns "RGB-D was 3.2x worse" into "and 68% of that was the
field of view, not the range".

### Plan

1. **A0 — gate: is the ZED depth usable over the run?** Its range is 0.3–12 m
   against the Ouster's 20+, in an 87° wedge. If depth returns nothing over long
   stretches, the comparison is against a sensor that was not working rather than
   one that is weaker. `precheck_tracks.py --track no_lidar`.
2. **A1 — the headline pair.** KISS-ICP on `mobile_1.ouster` and
   `mobile_1.zed_rgbd`. Two runs. This is the number the whole track exists for,
   and it is available on day one.
3. **A2 — the RGB-D field.** Four methods × two streams. Report
   `tracked_fraction` beside every ATE: an RGB-D method that loses tracking has
   not scored badly, it has stopped, and an ATE over the 40% it held is not
   comparable to one over 100%.
4. **A3 — generalisation.** Does the `mobile_2` ordering match the
   `mobile_1.zed_rgbd` ordering? If it does, the conclusion is about the sensor
   class. If it does not, it is about the platform, and say so.
5. **A4 — attribution.** The crossover sweep of step 4 above. One sweep per axis,
   bridge method only — about 12 runs, not 40.
6. **A5 — the runtime column.** KISS-ICP and ORB-SLAM3 run real time on a CPU;
   DROID-SLAM does not, on a GPU. For a LiDAR-free platform — which is usually
   LiDAR-free because of cost, power or weight — that column is part of the
   result, not an appendix.

**Deliverable.** One table: the same estimator on both sensors of one platform,
the RGB-D field beneath it, `mobile_2` beneath that, with tracked-fraction and
runtime beside every row. Then one figure: the ablation level at which the
LiDAR's result meets the depth camera's, per axis.

## Track B — collaborative SLAM, two mobile agents

### The question

Can two agents — one with a LiDAR, one with only a depth camera — establish and
hold a common frame, and what does the pair recover that neither has alone?

### Algorithms

| role | algorithm | why this one | runs today |
|---|---|---|---|
| **primary** | **Swarm-SLAM** | Decentralised: per-agent front-ends (LiDAR *and* RGB-D), sparse inter-robot loop closures found by exchanging descriptors, then distributed pose-graph optimisation. The only entry that takes this heterogeneous pair as-is, and it estimates the inter-agent transform rather than being given it. | ✅ |
| **floor** | **decoupled registration** | Each agent alone (from the main table), then one offline global registration: FPFH features → RANSAC → ICP refine. No communication, no inter-robot loop closure, no joint optimisation. | ✅ |
| **ceiling** | **oracle transform** | Give the true inter-agent transform from the reference and fuse the two maps. Not a method — it is the upper bound on collaboration gain. | ✅ |
| conditional | DiSCo-SLAM | Scan-context place recognition, LiDAR-only — so `mobile_2` would have to enter as a reprojected depth cloud. Degraded mode; run only if Swarm-SLAM's cross-modal matching fails and you need to know whether the failure is the modality or the system. | ~ |
| conditional | Kimera-Multi / COVINS-G | Visual-inertial. | ⛔ I1 |

The floor and the ceiling are both mandatory. Without the floor, a joint result
cannot be shown to be doing anything; without the ceiling, a small map gain
cannot be separated into *the merge was wrong* and *the partner had nothing to
add from where it stood*.

### Procedure

```
# gate, from the reference trajectories alone
if place_overlap(ref_a, ref_b, radius = weaker_sensor_range).fraction < 0.10:
    STOP. Not runnable on this sequence; fix the route, not the radius.

# floor
T0    = ransac_fpfh(map_a, map_b, voxel=0.10)        # global, no initial guess
T_hat = icp_refine(map_a, map_b, T0, max_corr=0.20)
score(est_a, T_hat . est_b)

# primary
traj_a, traj_b, map_joint, n_matches = swarm_slam(stream_a, stream_b)
score(traj_a, traj_b)                                 # in ITS own frame; no alignment

# ceiling
T_true = inv(ref_a[0]) . ref_b[0]                     # from the reference
map_oracle = fuse(map_a, map_b, T_true)

# the metric, for each of the three
E(t)        = inv(T_ab_ref(t)) . T_ab_est(t)          # gauge-free, no alignment
constant    = best-fit rigid, gauge fixed on agent a  -> merge / place recognition
varying     = residual after removing it              -> relative drift
map_gain    = completeness(joint) - completeness(solo), vs the oracle's gain
```

### Plan

1. **B0 — gate: did they share a place?** `precheck_tracks.py --track collaborative`.
   Needs only the two reference TUMs. Below ~10% overlap the track does not run
   and the finding is about the recording.
2. **B1 — the floor.** Costs one registration; the single-agent runs already
   exist. Expect its error to sit almost entirely in the **constant** term — a
   single rigid transform cannot absorb either agent's drift. That is the shape
   a joint method has to beat in the *varying* term.
3. **B2 — the ceiling.** Oracle transform, joint map, map gain. This number caps
   every collaborative row.
4. **B3 — Swarm-SLAM**, logging the inter-robot loop closure count on every run.
   Zero matches with a good relative pose means the two frames happened to
   agree — report the count or the number is unreadable.
5. **B4 — the cross-modal question.** If Swarm-SLAM finds no inter-robot matches,
   that is the headline finding of the track: *LiDAR-to-RGB-D place recognition
   did not fire on this sequence*. Confirm it is the modality and not the system
   by running the LiDAR-only pair (`mobile_1.ouster` against `mobile_1.zed_rgbd`
   reprojected, same platform) — if matching works there and not across agents,
   the cause is the viewpoint gap, not the code.

**Deliverable.** A three-row table — floor, Swarm-SLAM, ceiling — each with the
gauge-free relative error, the constant/varying split, the inter-robot match
count, and the map gain against each agent's solo run.

---

## Track C — infrastructure-anchored localization

### The question

Given messages from a node at a known pose, how much better can an agent
localize than it can alone — and does that rescue the constrained agent?

### Algorithms

Track C is a task, not a system, so the algorithms are per stage.

| stage | algorithm | notes |
|---|---|---|
| detect (radar) | DBSCAN over `/infra_1/radar/points_all`, per sweep, keep the largest moving cluster | 10¹–10² points per frame: expect a centroid, not a shape. Gives **bearing and range**. |
| detect (camera) | fiducial first, detector second | See the recommendation below — this stage dominates the error budget. Gives **bearing** only. |
| associate | nearest-stamp within `association_max_dt_s`, **after** clock reconciliation | `infra_1` is a third machine with a third clock. Get this wrong and every residual is a lag. |
| fuse | **pose graph with unary factors on a fixed landmark** | Ego odometry as between-factors; each detection as a bearing (and, from radar, range) factor to a landmark at the surveyed pose, with its own covariance. Optimise with GTSAM/g2o. Standard, and it degrades gracefully when detections are sparse. |
| baselines | `ego_only` → `+bearing` → `+bearing+range` | The same ego pipeline with one more evidence source each time, so the difference between rows is the infrastructure and nothing else. |

**The recommendation that matters: put a fiducial on each cart.** An ArUco or
ChArUco board on the mast turns infrastructure detection from a research problem
into a solved one — sub-pixel bearing, unambiguous identity, no detector to
train or tune. Without it you are detecting a pushcart in a raw, distorted,
10.6 Hz Arducam stream, and that detector's error will dominate whatever
localization gain you are trying to measure. It costs one printed board per
platform in the next session, and it also gives track C an identity channel the
radar cannot supply.

### Procedure

```
# C2 first — instrumentation, cheaper and it validates everything else
pred = predict_observation(traj, infra_pose, axes="optical")   # NOT "ros"
res  = observation_residual(pred, det_stamps, det_azimuth[, det_range])
# a bearing residual with consistent SIGN  -> infra pose or extrinsic error
# a range residual growing with distance   -> the agent drifting along the line of sight

# C1 — the benchmark task
for variant in (ego_only, +bearing, +bearing_range):
    graph = pose_graph(ego_odometry_factors)
    if variant != ego_only:
        for each detection d:
            graph.add(BearingFactor(landmark=infra_pose, sigma=sigma_az))
            if variant has range:
                graph.add(RangeFactor(landmark=infra_pose, sigma=sigma_r))
    traj = optimise(graph)
    ate[variant] = eval vs reference
report ate[+bearing] - ate[ego_only]   # the gain, per agent
```

### Plan

1. **C0 — resolve I9 and I10.** The node's pose uncertainty, and whether the
   Arducam's `camera_info` belongs to the Arducam. Until I9 is a number, C1
   reports *improvement over ego-only* and not an absolute accuracy.
2. **C1 — C2 before C1.** The instrumentation use is cheaper, needs no fusion
   code, and validates the node's pose before you build anything on it. Run it
   against the reference trajectory: if the predicted bearings do not match the
   detections, the fault is the node's pose or the clock, and nothing downstream
   is worth building yet.
3. **C2 — detection.** Radar first (DBSCAN, gives range too), camera second.
   Report detection rate per agent; a variant evaluated on 12% of frames is a
   different experiment from one on 90%.
4. **C3 — the fusion, three variants, both agents.** `mobile_2` is the cell that
   matters — a static observer constrains bearing well and range poorly, close
   to complementary to a narrow-FoV depth camera's weakness. `mobile_1` is the
   control: the agent that already localizes well should gain little, and if it
   gains a lot, the gain is coming from the reference rather than the node.
5. **C4 — the crossover with track A.** Re-run the best infrastructure variant on
   two or three *ablated* `mobile_1` cells. The claim worth testing: infrastructure
   support extends the usable sensor envelope, and it should extend it furthest
   in exactly the cells where observability says the geometry went free.

**Deliverable.** A per-agent table of ATE for the three variants with detection
rate beside each, plus the bearing/range residual plot that validates the node.

---

## Sequencing across tracks

```
P0 precheck ──┬─> B0 gate ──> B1 floor ──> B2 ceiling ──> B3 Swarm-SLAM ──> B4
              │
              ├─> A0 gate ──> A1 pair ──> A2 field ──> A3 generalise ──> A4 attribution
              │                                  │
P1 reference ─┴─> P3 boards ──> tier 2 live      └──────────> C4 crossover
              └─> P4 volume ──> map metrics           ↑
                                                      │
                  C0 (I9,I10) ──> C1 residual ──> C2 detect ──> C3 fusion
```

Track A runs **in full** on what exists today: no entry in it needs an IMU, a
second agent, or infrastructure detection. Its headline number — one estimator,
two sensors, one platform — is two runs away. Start there while P3 and C0 are
being resolved, and keep B0 as the first thing you run regardless: it is
minutes, and it decides whether track B is a week of work or a sentence in the
paper.
