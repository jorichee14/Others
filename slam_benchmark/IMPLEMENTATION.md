# Implementation

Source of truth for progress. Update it in the same commit as the work: flip the
status, fill the Result line with real numbers and paths, append to the log.
Never mark ✅ without checking the "Done when" line.

## Input decisions

These are not implementation details; each one changes what the benchmark can
conclude, and a phase that depends on an unanswered one does not start.

| | decision | status | why it blocks |
|---|---|---|---|
| I1 | **Does the bag carry an IMU?** Topic, rate, and which agent. | ⬜ open | Decides whether FAST-LIO2, GLIM and every VIO entry can run at all. `run_method.py` refuses inertial methods until `streams.*.imu.present` is set. |
| I2 | **The three board dwell windows** — when each platform stood at `anchor`, `anchor_b`, `rs_anchor`, and at what offset. | ⬜ open | Without them tier 2 is silent and no independent evidence enters the benchmark. The single highest-value open item. |
| I3 | **The anchored map cloud**: path, and confirmation it is in the post-anchor `map` frame, not the pipeline's pre-anchor frame. | ⬜ open | Every map metric needs it. A pre-anchor cloud scores every method against a frame none of them are in. |
| I4 | **The evaluation volume V.** | ⬜ open | Changes the ranking. Proposal: the map cloud's occupied box eroded 0.25 m; confirm against the cloud, do not guess from the trajectories. |
| I5 | **IMU↔LiDAR extrinsic**, if I1 is yes. | ⬜ open | Not published in the bag. A guessed one costs FAST-LIO2 more than KISS-ICP, so an unfavourable LIO result under a guess is not a result. |
| I6 | **Does the route revisit?** | ⬜ open | If it does not, loop closure cannot be evaluated on this sequence and the RTAB-Map/GLIM entries measure only their odometry. Answer it from tier 3 on the reference trajectory itself — no method needed. |
| I7 | **Which alignment the headline table uses** (`se3` vs `se3` anchored on the first N poses). | ⬜ open | Best-fit ATE and drift-from-a-fixed-start are different claims. Pick one for the headline, report the other in the appendix. |

## Phases

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

### Phase 4 — the static observer ⬜
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
