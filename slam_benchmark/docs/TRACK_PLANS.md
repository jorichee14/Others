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
| P1 | Export both references; run `reference_accumulation`; **look at the cloud** | ~1 h | every map metric in every track |
| P2 | Resolve I1 (IMU present?) | minutes | the inertial arm of track A |
| P3 | Resolve I2 (board dwell windows) | ~1 h | tier 2 everywhere — the only independent evidence |
| P4 | Resolve I3/I4 (map cloud frame, evaluation volume) | ~1 h | map scores in tracks A and B |

P0 first, always. It is the only step that can tell you a track is not runnable
before you spend a week on it.

---

## Track A — sensor-constrained SLAM

### The question

At what point does the sensor stop supporting the task, and when it does, is the
failure the method's or the geometry's?

### Algorithms

| role | algorithm | why this one | runs today |
|---|---|---|---|
| **primary** | **KISS-ICP** | Scan-to-model point-to-point ICP with an adaptive correspondence threshold and constant-velocity motion prediction. No IMU, no loop closure, no per-dataset tuning — so a degraded cell degrades for one reason, and the same estimator runs on every cell including the depth streams. | ✅ |
| second | **RTAB-Map (LiDAR)** | Adds appearance/proximity loop closure over an ICP odometry front-end. Paired with KISS-ICP on the same cell it isolates **whether loop closure buys back what constraint takes away** — the single most useful pair in this track. | ✅ |
| inertial arm | **FAST-LIO2** | Tightly-coupled iterated-EKF LiDAR-inertial with an ikd-Tree map. The scientifically interesting question in track A: *does an IMU restore the translation direction a narrow FoV stopped observing?* It should, and the observability metric predicts exactly where. | ⛔ I1, I5 |
| instrument | **observability** (not SLAM) | Point-to-plane information matrix of one scan. Runs before any method, on every cell. | ✅ |

The inertial arm is the payoff of this track, not an optional extra. An IMU
constrains the directions geometry does not, so the prediction is sharp and
falsifiable: FAST-LIO2's ATE should stay flat across exactly the cells where
KISS-ICP's degrades *and* observability says the free direction is one the IMU
observes. If I1 comes back "no IMU", the track still runs — it just cannot make
that claim, and the recording needs an IMU next time.

### Procedure

```
# offline, no SLAM system, minutes
for cell c in ablations.yaml:
    for frame f in ouster_stream:
        P    = ablate(cloud[f], c)                  # SENSOR frame, before any pose
        o[f] = observability(P, voxel="auto")       # no map, no reference
    deg[c]     = mean(o[f].degenerate())            # geometry stopped constraining
    starved[c] = mean(o[f].starved())               # too sparse to judge — NOT the same
    surfaces[c]= median(o[f].n_points)

# the sweep
for cell c, method m:
    traj, map = run(m, ablated_stream(c))
    ate[c,m], rpe[c,m], tracked[c,m] = eval_run(...)

# the curve
plot x = constraint level, y = ATE[c,m], one line per method per axis,
     with deg[c] shaded underneath.
     A method's line is only interpretable where the shading is thin.
```

### Plan

1. **A0 — gate: precheck.** `precheck_tracks.py --track sensor_constrained --cloud <one sweep>`.
   Cells already degenerate on a single scan are the observability floor. Either
   stop the grid before them or report them as the floor — never as method failures.
2. **A1 — gate: the harness is inert.** Run KISS-ICP on the `full` cell and check
   it reproduces the main table's KISS-ICP number **exactly**. The ablation path
   must be bitwise inert when it removes nothing; if it is not, every cell below
   is measuring the harness. (Same discipline as `commchannel` in the sibling
   study, and for the same reason.)
3. **A2 — one axis at a time.** FoV, then range, then beams, then rate, then
   density. Five sweeps × 2 methods × ~4 levels ≈ 40 runs plus controls.
4. **A3 — the envelope validation.** Compare the `realsense_envelope` cell on
   `mobile_1` against the real `mobile_2.realsense_rgbd` run. If they diverge,
   the ablation is missing something the real sensor has — depth dropout at
   range, a noise model, rolling shutter — and the synthetic curve must not be
   quoted on its own. Report the gap either way; it is a result about how far
   synthetic degradation can stand in for a real sensor.
5. **A4 — the inertial arm**, if I1 is yes: FAST-LIO2 across the FoV axis only,
   against the observability prediction.

**Deliverable.** One figure: ATE vs constraint level, per axis, with the
observability floor shaded and the real `mobile_2` result marked as a point on
the FoV/range axes.

---

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
              ├─> A0 gate ──> A1 inert ──> A2 sweep ──> A3 envelope ──> A4 (I1)
              │                                  │
P1 reference ─┴─> P3 boards ──> tier 2 live      └──────────> C4 crossover
              └─> P4 volume ──> map metrics           ↑
                                                      │
                  C0 (I9,I10) ──> C1 residual ──> C2 detect ──> C3 fusion
```

Track A is the one that runs furthest on what exists today: no IMU needed, no
second agent, no infrastructure detection. Start there while P3 and C0 are being
resolved, and keep B0 as the first thing you run regardless — it is minutes, and
it decides whether track B is a week of work or a sentence in the paper.
