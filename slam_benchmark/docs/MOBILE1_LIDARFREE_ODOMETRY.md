# mobile_1 odometry without the LiDAR

> **Execution order lives in `docs/PLAN.md`** — start there; this document is detail it points into.

> **GOAL CHANGED 2026-09-17 — read `docs/PSEUDO_GT.md` first.**
>
> The objective is **a pseudo ground truth for the sensor-constrained agent**,
> not a better odometry for `mobile_1`. `mobile_1` is the *instrument*: it
> carries a LiDAR and a depth camera on one body, so a LiDAR-free construction
> can be built there and **certified against the held-out LiDAR**, then applied
> to `mobile_2`, which has no LiDAR and therefore no reference at all.
>
> That changes what this document is. Its roster stops being competitors and
> becomes **candidate backbones plus the ablation that prices each factor
> class**; a pseudo-GT is *offline*, so batch optimisation, loop closure and
> absolute anchors are all in scope where an online odometry could not use them.
> The method notes below remain valid — read them as inputs to the construction
> in `PSEUDO_GT.md`, not as a ranking to produce.

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

## START HERE — the bar is measured, and it is 6 m

**`/mobile_1/zed/odom` lands ~6 m from the reference.** That is ~36% of the
room's diagonal, so it is not drift — it is a failure. The cause is known and it
is the room: **the walls are reflective, so the ZED's stereo depth drops out.**

This is the single most important fact in this document and it **re-orders
everything below**. The roster was written for a well-behaved depth camera. It
is not one here.

### What a 6 m bar changes

**Good news first.** A 6 m bar means there is enormous room to improve and the
reference can separate anything. The earlier worry — that methods would crowd
the 15 mm uncertainty floor and be indistinguishable — is gone. Every row in
this phase will be legible.

**Then the re-ordering.** Depth dropout is not a uniform handicap. It splits the
roster by *how much each entry leans on depth*:

| leans on depth | entry | expect |
|---|---|---|
| **totally** | `kiss_icp` (depth→cloud) | worst. It is pure depth geometry with no fallback. Now a *measurement of how bad the depth is*, not a contender. |
| **heavily** | `rtabmap_rgbd`, `orbslam3_rgbd` | structurally handicapped. Their features need depth to be metric. |
| **partly** | `rtabmap_rgbd_imu`, `orbslam3_rgbd_inertial` | the IMU **bridges** dropouts. This is exactly the failure inertial fusion exists for, so the IMU is worth far more here than in a well-lit room. |
| **not at all** | `orbslam3_mono_inertial`, `mast3r_slam` | **promoted.** No depth dependence. MASt3R's learned prior predicts geometry on the reflective, textureless surfaces where stereo returns nothing — the case it is best at is the case this room produces. |
| **not at all** | `radar_inertial_odometry`, `rgbd_inertial_radar` | **promoted, and the earlier prediction was wrong** — see below. |
| **not at all** | `infra_anchored` | **promoted.** An external observer at a surveyed pose does not care what the agent's own sensors are doing. |

### A prediction in this document is now wrong, and it matters

`rgbd_inertial_radar.yaml` says to expect *no whole-run gain*, on the grounds
that "on a well-lit 156 s lab lap, vision does not fail much". **Reflective walls
mean vision does fail, and probably a lot.** So:

* the vision-starved segments the radar claim was scoped to **exist on this
  sequence** — the claim no longer needs a new recording to be testable;
* the radar arm moves from *probably shows nothing* to *plausibly the thing that
  rescues the run*, because mmWave does not care about optical reflectivity;
* `slambench/observability.py` and the ORB-density instrument stop being
  optional attribution and become the tools that **locate the failure**.

### Step 1 — quantify the dropout before choosing anything

Not "is the depth bad" but **"for how long at a stretch was it absent"**, because
those break a tracker differently. A uniform 30% dropout is noise a system rides
through; a four-second blackout is where the trajectory leaves the room and the
6 m comes from.

```bash
python3 scripts/depth_health.py --bag <bag> \
    --topic /mobile_1/zed/depth/depth_registered \
    --range 0.3 12.0 --out runs/depth_health_mobile_1.csv
```

It reports the valid-depth fraction per frame, the starved stretches with their
lengths and start times, and writes a CSV. **Cross the longest stretches against
where the ZED odometry diverged.** If they coincide, the 6 m is explained and
the ranking above is confirmed rather than assumed.

Then read it:

| result | consequence |
|---|---|
| starved **< 10%**, uniform | depth is not the whole story; something else broke the ZED tracker too, and it is worth finding before building on it |
| starved **> 20%**, episodic | the RGB-D arm is structurally handicapped. Run the promoted rows and report the RGB-D ones as the *measurement of the handicap* |

### Step 2 — the revised core five

Reordered by robustness to the failure that is actually present:

| | method | why now |
|---|---|---|
| 1 | `rtabmap_rgbd_imu` | IMU bridges dropouts; runs **today**, no calibration |
| 2 | `orbslam3_rgbd_inertial` | tight coupling pays most when one sensor is intermittent |
| 3 | `mast3r_slam` | no depth dependence; a prior needs no texture — the reflective-wall case |
| 4 | `radar_inertial_odometry` → `rgbd_inertial_radar` | immune to optical reflectivity |
| 5 | `infra_anchored` | absolute, and immune to the agent's sensors entirely |

`rtabmap_rgbd`, `orbslam3_rgbd` and `kiss_icp` still run — as the rows that
**quantify the depth handicap**, which is now a finding rather than a baseline.

Also free, and worth an hour: score **`/mobile_1/zed/pose`** (the loop-closed
topic). If spatial memory recovers much of the 6 m, that tells you how much of
the failure is recoverable by loop closure alone, on the same data.

### The 6 m is a better paper than 5 cm would have been

A dataset where a depth camera fails on reflective indoor surfaces, and where
inertial, radar and infrastructure evidence measurably rescue it, is a stronger
contribution than one where RGB-D quietly works. The failure is the result.
Report the dropout statistics beside every RGB-D row so the handicap is visible
rather than inferred.

### The two next-session items are unchanged and still worth more than any row

1. **Record the right image** — one setting; restores the entire stereo family.
2. **Log the ZED IMU overnight, 3 h** — measured once, serves every recording.
   And now more urgent: the IMU carries far more weight in the revised plan.

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

## The methods — the final roster

Sixteen entries. The test each one had to pass to be here: **it answers a question
no other entry can**, and it forms a pair with something else in the list that
isolates one variable. An entry that only adds a name to the table was cut — a
benchmark of unrelated systems answers nothing, which is the same argument the
main table uses to justify KISS-ICP as its bridge.

| # | tier | method | input | coupling | LC | today |
|---|---|---|---|---|---|---|
| 1 | 0 | `zed_sdk_odom` | ZED VIO, recorded | closed | – | ✅ |
| 2 | 0 | `zed_sdk_pose` | ZED VIO + spatial memory | closed | ✓ | ✅ |
| 3 | 1 | `kiss_icp` | ZED depth → cloud | none | – | ✅ |
| 4 | 2 | `rtabmap_rgbd` | RGB-D | none | ✓ | ✅ |
| 5 | 2 | `orbslam3_rgbd` | RGB-D | none | ✓ | ✅ |
| 6 | 3 | `rtabmap_rgbd_imu` | RGB-D + IMU gravity | loose | ✓ | ✅ |
| 7 | 3 | `orbslam3_rgbd_inertial` | RGB-D + IMU | tight | ✓ | I13/I14 |
| 8 | 3 | `orbslam3_mono_inertial` | colour + IMU | tight | ✓ | I13/I14 |
| 9 | 3 | `mast3r_slam` | colour only, learned prior | none | ✓ | GPU + `image_size` |
| 10 | 3 | `rtabmap_ext_odom` | winner + LC + map | loose | ✓ | needs 1–9 |
| 11 | 4 | `radar_inertial_odometry` | 2× radar Doppler + IMU | tight | – | field name/sign |
| 12 | 4 | `rgbd_inertial_radar` | tier-3 winner + radar | loose | ✓ | needs 7–11 |
| 13 | **5a** | `infra_anchored` | winner + infra_1 factors | loose | – | needs 1–12, I9 |
| 14 | 5b | `decoupled_registration_nolidar` | m1 ⊕ m2, offline merge | none | – | needs the winner |
| 15 | 5b | `swarm_slam_nolidar` | m1.zed ⊕ m2.realsense | none | ✓ | ✅ (gate B0) |
| 16 | 5b | `covins_g` | centralised, any VIO | loose | ✓ | conditional on 15 |

`openvins` is `status: deferred`, not in the roster. See the end of this section.

### Why each one is here

**1–2, the ZED SDK rows — because the incumbent has to be on the table.**
"Better odometry" is a comparative claim and it needs something to be better
than. This is the camera's own visual-inertial output, computed on the cart at
record time, and an open-source entry that loses to the sensor's firmware has
improved nothing. They cost no compute and no calibration. Their **pair** is a
free loop-closure ablation on a system nobody had to build: same camera, same
run, one feature toggled — which also answers I6 from the baselines alone. They
stay out of the peer ranking because closed source cannot be ablated: they set
the bar without ever explaining it.

**3, KISS-ICP — because it separates the depth sensor from the camera.**
It consumes the depth as a point cloud and ignores the image entirely, so it is
the floor that says how much of any result belongs to the *range sensor* rather
than to vision. It is also the bridge to the main benchmark: the same estimator
already runs on `mobile_1.ouster`, so its two rows differ in the sensor and
nothing else, and that is the only clean LiDAR-vs-depth number on the sequence.

**4–5, the RGB-D field — because dense and sparse fail differently.**
RTAB-Map carries appearance loop closure and a dense map, so it is the only
tier-2 entry that scores on the map metrics at all. ORB-SLAM3 is the sparse
standard, and its documented failure modes — low texture, fast rotation — are
precisely what a pushcart turning in a lab produces, so it sets the realistic
floor rather than an optimistic one. Two systems, because the RGB-D result
should not rest on one implementation's tuning.

**6, RTAB-Map + IMU gravity — because it separates "the IMU helped" from "tight
coupling helped".** Those are two different claims that a single inertial row
cannot distinguish, and that papers routinely conflate. A gravity constraint is
a small, well-understood intervention: it bounds roll and pitch drift and does
almost nothing for yaw or scale. Paired with #4 it gives the loosely-coupled
answer to the same question #7 asks tightly. It also needs neither the
camera↔IMU extrinsic nor the noise model, which is why it runs today and can
carry the inertial headline while I13/I14 are open.

**7–8, the ORB-SLAM3 inertial pair — because this is where the 2×2 closes.**
Together with #5 they give three cells of a depth × IMU grid on *one codebase*,
with identical parameters and identical calibration:

| | no IMU | + IMU |
|---|---|---|
| **RGB-D** | `orbslam3_rgbd` (#5) | `orbslam3_rgbd_inertial` (#7) |
| **mono** | not run — no scale at all | `orbslam3_mono_inertial` (#8) |

`#7 − #5` is what the IMU buys. `#7 − #8` is what the depth buys. No mix of
different systems can produce either subtraction, because each would vary the
estimator at the same time. #8 costs nothing extra: same container, same
parameters, one sensor mode changed.

ORB-SLAM3 is the tight entry rather than something stronger on paper because
the recording leaves no alternative — see *the thin field* below.

**9, MASt3R-SLAM — because it is the other answer to the scale question.**
It pairs with #8: both are monocular and neither sees the depth, but one takes
its geometry from an IMU and the other from a learned 3D prior. That pair asks,
at its extreme, where a LiDAR-free agent should get scale and geometry from.
It also attacks the depth camera's *actual* weakness rather than a generic one —
the ZED stops at ~12 m in an 87° wedge and degrades on textureless surfaces, and
a prior needs no texture to produce a pointmap. So it is the one entry that
could plausibly beat the depth sensor where the sensor is worst, which makes a
loss here as informative as a win. Dense output, so unlike ORB-SLAM3 it also
scores on the map metrics. It supersedes `droid_slam` for this phase;
`droid_slam` stays in track A, where the question is about learned front-ends in
general rather than about this platform's scale. **Report it twice** — sim3, and
se3 after fixing one global scale against the ZED depth — because monocular with
learned priors is not reliably metric and the gap between the two *is* the scale
error.

**9, RTAB-Map on external odometry — because it is probably the winner and it is
nearly free.** RTAB-Map decouples odometry from mapping, so this row inherits
whatever front-end won tiers 0–3 and adds loop closure, graph optimisation and a
dense map on top. No new estimator, no new calibration. One reading rule: if its
source is `zed_sdk_odom`, it measures what open-source loop closure adds to
closed-source odometry — a real result, but not an open-source odometry result,
so the source goes in the row label and never just the method name.

**10, radar-inertial — because it fails for reasons vision does not.**
It consumes ego-velocity, not geometry: `v_r = -d · v_radar` per static return,
three non-collinear directions determine the body velocity, RANSAC rejects the
movers, and the lever arm comes off with the IMU's own angular rate. Two radars
~120° apart in yaw is close to what people deliberately build for this, since
one unit observes the component across its boresight only weakly. Its ATE will
be the worst in the table and that is not the point — velocity integrates, so
heading error is unbounded and there is no loop closure here. Read it on short
RPE and on velocity error against the reference's own differentiated velocity.

**11, the fusion — because it is the actual proposition.** Loose on purpose: a
velocity factor inside ORB-SLAM3 means modifying ORB-SLAM3, while a pose graph
outside composes three things that already work, a week cheaper. If it pays,
the tight version becomes justified work instead of speculative work.


### Tier 5 — collaborative, and which half of it actually helps

Two mechanisms, and they are not equally useful for *this* goal. Say which is
which or the tier reads as decoration.

**5a, `infra_anchored` — the one with an absolute mechanism.** `infra_1` is
static, surveyed, on its own clock, watching the room from 2 m up. Every other
entry in this phase is an odometry whose error grows without bound; a fixed
observer at a *known* pose, seen periodically, is precisely the thing that stops
it growing. Nothing else available to a LiDAR-free `mobile_1` is absolute.

The geometry suits this agent unusually well. A single static observer
constrains **bearing** well and **range** poorly — error along its line of sight
is what it sees worst — and a narrow-FoV depth camera's weakness is close to
complementary. The radar gives bearing *and* range; the Arducam gives bearing
only; running both separates what the range measurement was worth.

Note the role reversal: `mobile_1` was track C's *control*, the agent that
already localizes well and should therefore gain little. Without its LiDAR it is
no longer that agent, so here it is the subject.

Two things to do before any fusion code exists, both free:
- **The residual check (C2 before C1).** `predict_observation()` and
  `observation_residual()` in `slambench/collab.py` already do it against the
  reference, with no detector. A bearing residual with a *consistent sign* means
  `infra_1`'s pose or extrinsic is wrong and nothing built on it is worth
  building; a range residual *growing with distance* is the agent drifting along
  the line of sight, which is what this row exists to catch. Pass
  `axes="optical"` — `"ros"` returns plausible bearings wrong by 90°.
- **The visibility gate.** Is `mobile_1` in view often enough? Answerable from
  the reference trajectory and the surveyed pose alone. A variant scored on 12%
  of frames is a different experiment from one on 90%, so detection rate goes
  beside every ATE.

**5b, the agent-to-agent rows — a different question, honestly labelled.**
`mobile_2` is the **weaker** platform: a narrower-FoV RealSense, and no LiDAR
either. You cannot anchor yourself to something drifting worse than you are, so
these rows will not rescue `mobile_1`'s absolute accuracy and must not be read
as if they might. What they *can* do is bound drift through co-observation — an
inter-robot loop closure is a constraint, and joint optimisation distributes
error across both trajectories — and answer whether two agents can hold a common
frame at all, which is track B's question.

One genuinely interesting consequence: **dropping `mobile_1`'s LiDAR makes
collaboration easier.** Track B pairs the Ouster against a RealSense, so its
inter-robot loop closures must match descriptors *across modalities* — the hard
part, and the most likely source of zero matches. Here both agents are RGB-D and
the matching problem is the ordinary one. That makes `swarm_slam_nolidar` a
diagnostic for track B as well: if matching fires on this pair and not on the
LiDAR pair, the cause is the modality gap and not the system — exactly the check
B4 specifies, obtained for free.

`decoupled_registration_nolidar` is the floor and is not optional: without it a
joint result cannot be shown to be doing anything a single offline registration
would not have done. `covins_g` is conditional on #15 firing — if decentralised
matching finds nothing, a centralised back-end on the same descriptors will not
either, and the two rows would report one failure twice.

**Gate B0 can close all of 5b.** The agents start 16.3 m apart in a 16.6 m room.
Below ~10% place overlap these rows do not run, and the finding is about the
*recording* — fixed by routing both platforms through one shared corridor next
session, not by widening the radius until the number looks acceptable.

### The thin field, and why it is not a shortlisting failure

Only the left image was recorded. That removes every stereo-inertial system —
Kimera-VIO, MSCKF-VIO, VINS-Fusion stereo — and leaves **ORB-SLAM3's
`IMU_RGBD` as the only well-maintained tightly-coupled RGB-D-inertial option**.
The tier-3 tight slot is one system because the recording admits one, not
because the search stopped early. The same constraint reaches tier 5: **Kimera-Multi
needs stereo and is out for the same reason**, which is worth recording because it
is the obvious name to reach for there. That is a finding about the recording, and the
fix is one bandwidth setting next session.

`openvins` was in an earlier draft of this roster and was cut for two reasons.
It varies the estimator *and* the sensor set against every RGB-D-inertial row at
once, which is exactly the comparison this benchmark refuses. And the role first
claimed for it was wrong: it was the hedge against an ORB-SLAM3 inertial-init
failure on the grounds that it can initialise statically — but static init
recovers gravity and the biases, **not scale**, and monocular scale needs
translational excitation. On near-constant-velocity pushcart motion it is the
first entry to fail, not the fallback. Its config is kept at
`status: deferred` with the reversal condition: worth running as *stereo*-
inertial once the right image exists, where it is metric without leaning on the
IMU and the filter-versus-optimiser question finally becomes clean.

BAD-SLAM and DROID-SLAM stay in track A's plan and out of this one. Both need a
GPU and neither answers a question the eleven above leave open; the LiDAR-free
question here is about sensor sets and coupling, not about dense or learned
front-ends.

### What is held, and by what

Five of the eleven do not start yet, and `run_method.py` refuses them rather
than defaulting — the existing `params`-null check already enforces rule 3 for
exactly these values, so no new gate code was needed.

| held | by | resolvable |
|---|---|---|
| `orbslam3_rgbd_inertial`, `orbslam3_mono_inertial` | `IMU.T_b_c1` + the four noise terms | I13 today; I14 needs one overnight log |
| `radar_inertial_odometry` | `doppler_field`, `doppler_sign`, noise terms | field name in minutes; sign off one known-motion segment |
| `mast3r_slam` | `image_size` + a GPU | today, once the resize is recorded |
| `rtabmap_ext_odom`, `rgbd_inertial_radar`, `infra_anchored`, `decoupled_registration_nolidar`, `covins_g` | `*_source` / `ego_source` | not a value — a *result*. Held by design until the single-agent rows rank. |
| `infra_anchored` (additionally) | `infra_pose_sigma_*` | I9. Until it is a number this row reports *improvement over ego-only*, never an absolute. |

The last row is the important distinction: those two are not blocked on missing
information, they are blocked on the experiment not having run yet. Filling them
early would mean choosing the winner before measuring it.

## The ablations

Small and axis-at-a-time. Roughly 16 runs, not 40. The sensor-set axis is the
headline; the rest are attribution.

**A. Sensor set** (the headline ladder — this *is* tiers 0–4)
`depth only → +image → +IMU → +radar`, with `radar+IMU` off to the side as the
vision-free control and `mono+IMU` as the no-depth control.

**~~B. IMU source~~ — DROPPED.** The Ouster's IMU counts as using the LiDAR
(Q2), so there is one admissible IMU and no axis to sweep.

**C. Loop closure on/off** — RTAB-Map with `Rtabmap/LoopThr` at its default and
at 1.0 (disabled). Isolates loop closure from the odometry front-end. **Only if
I6 says the route revisits**; otherwise the two rows are identical by
construction and the cell is reported as not evaluable. The `zed_sdk_odom` /
`zed_sdk_pose` pair answers this for free before any container runs.

**D. Depth max range** — `Vis/MaxDepth` at 3 / 5 / 8 / 12 m. The real-sensor
analogue of the `r3`–`r20` axis already in `configs/ablations.yaml`, except
imposed on a genuine depth camera rather than a decimated LiDAR. It also
cross-checks that grid: the synthetic `realsense_envelope` cell should land near
the real measured one, and if it does not, the synthetic curve must not be
quoted alone.

**E. Radar configuration** — `radar1` only / `radar2` only / both. Directly
tests the two-radar conditioning claim. The cheapest cell in phase 6 and the
only one that says whether the second unit earned its place on the cart.

**F. Rate** — depth at 14.7 Hz vs decimated to 5 Hz. What a bandwidth-limited
link would cost, which connects to the sibling collaborative-perception project.

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
P3   B0 place overlap (both reference TUMs) + infra_1 visibility.  minutes
     Either can close tier 5's two halves before any work starts.
T0   Score /mobile_1/zed/odom and /mobile_1/zed/pose.             ~1 h
     THE BAR. Everything below is measured against it.
T1   kiss_icp on reprojected ZED depth.                           1 run
T2   rtabmap_rgbd, orbslam3_rgbd.                                 2 runs
T3a  rtabmap_rgbd_imu  [no unknowns -- runs with T2].            1 run
T3b  orbslam3_rgbd_inertial, orbslam3_mono_inertial [I13/I14].    2 runs
T3c  rtabmap_ext_odom  [needs the T0-T3b ranking].                1 run
T3d  mast3r_slam  [GPU; report sim3 AND depth-anchored se3].       1 run
T4   rio, rgbd_inertial_radar  [gated on P0].                     2 runs
T5a  infra_anchored: residual check FIRST, then 3 variants.       3 runs
     The residual check needs no detector and no fusion code.
T5b  decoupled_registration_nolidar (floor), swarm_slam_nolidar,
     covins_g [conditional].                       2-3 runs, gate B0
AB   Ablations C-F on the tier winner only (B is dropped).        ~9 runs
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
| `zed_sdk_odom` / `zed_sdk_pose` | closed, already computed | no |
| `rtabmap_rgbd_imu` | loose (gravity constraint) | **no — runs today, and carries the inertial headline** |
| `rtabmap_ext_odom` | loose | no; held only on which source wins |
| `orbslam3_rgbd_inertial` | tight | yes, and its *initialiser* especially |
| `orbslam3_mono_inertial` | tight | yes, and it has no depth to fall back on |

**So the ladder starts now.** The loosely-coupled rows carry the headline until
the overnight log lands; the tight rows run in parallel, each labelled
*datasheet-order noise*, and are re-run once the measurement exists.
`rtabmap_rgbd_imu` is the row that makes this tolerable: a loosely-coupled
inertial result that needs neither unknown, so the inertial claim is never
hostage to the overnight log. A tight row
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
