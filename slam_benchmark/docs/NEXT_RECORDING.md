# The next recording — spec

Every requirement below comes from a **measured** failure in `coop2_20260828`,
named beside it. This is claim 4 of the paper turned into a shopping list, and
the paper's strongest experiment if it is executed as written.

---

## The design principle: DIVERSITY OF SOURCES, not count of landmarks

The construction's error bar needs cross-validation, and cross-validation needs
more than one thing to hold out. There are two ways to get that:

**Many landmarks of one kind.** HortiMulti / Poly-TagSLAM (2026) surveys **35**
AprilTag landmarks and leaves each one out in turn, recovering held-out
positions at 5.9 cm mean error. It works, and it is the closest published
certification of a fiducial-anchored ground truth.

**Few landmarks, several KINDS.** This is the design here, and the argument for
it is one this project already makes one level down: *three similar RGB-D
backbones would agree through a shared failure mode and the check would certify
nothing.* The same holds for absolute sources. **Leave-one-out over homogeneous
landmarks cannot see a failure common to all of them** — a systematic survey
offset, a detector bias, a lighting regime. Every held-out tag is predicted
beautifully and the trajectory is wrong together.

So the requirement is **sources that fail for unrelated reasons**:

| source | precision | covers | fails when |
|---|---|---|---|
| surveyed board | ~8 mm | the dwell windows | too far, too oblique, too small, blur |
| static surveyed observer | decimetre | whenever in range | occluded, out of range, clutter |
| peer robot with a reference sensor | its own grade | whenever co-visible | not co-visible, mis-association |
| prior-map registration | cm | wherever the map overlaps | degenerate geometry, depth dropout |
| backbone SLAM | relative only | always | (holds shape; never absolute) |

Nothing in that table fails for the same reason as anything else. That is what
makes leave-one-**source**-out mean something that leave-one-landmark-out
cannot, and it is also why n=2 boards stopped being fatal: the lever exists
when one board is held out of two, not when one SOURCE is held out of four.

**It also attacks the error directly rather than only measuring it.** `coop2`'s
199 mm lives almost entirely in the 118-second gap where no board is visible.
Read the coverage column again: the union of these sources covers most of that
gap.

**Design rule: at least three absolute source kinds, with their coverage in
time deliberately overlapping only partly.** Full overlap wastes them; disjoint
coverage leaves gaps.

---

## The camera arm: one rig, N cameras, one held-out LiDAR

**Mount every camera and the LiDAR on a single rigid body.** Not two robots.

`coop2` certified the procedure on a ZED and applied it to a RealSense, which
confounds the camera with the route, the room corner, the reference and the
platform. A reviewer will say so, and they will be right. One rig removes every
confound at once: all cameras traverse the same trajectory past the same boards
at the same instant, and **one held-out LiDAR certifies all of them
simultaneously**.

That makes camera-agnosticism a within-subject controlled experiment rather
than an argument, and it tests the transfer mechanism head-on: each camera has
its own backbone drift rate, measurable with no ground truth at all. Same
coverage, different drift rates. **Does the coverage law predict each camera's
error from its own drift rate?** That is the paper's central claim, measured.

**Minimum useful set:** the same D455 as `mobile_2` (removes the camera
difference entirely, so the existing result transfers with no assumption), plus
one deliberately different camera — different resolution, different depth
principle, ideally different field of view. A third is better. A global-shutter
mono and a rolling-shutter phone-class camera would span the space usefully.

**Non-negotiable:** the LiDAR is on the same rigid body as every camera, and
its extrinsic to each camera is calibrated and recorded. Without that the LiDAR
certifies nothing.

---

## Six fixes, each from a measured failure

### 1. Boards sized for the WORST camera, with 2x margin

**The failure.** `anchor_b` returned **0 detections in 179 frames** — 178 saw no
ChArUco corners at all. Not a tuning miss: the same board design at `anchor`
read 44 of 48 from the same camera in the same session. At its only observed
range, 0.84 m, its 15 mm markers span **~1.9 px per bit** in 848x480. The ZED
read 1.9 px/bit successfully at `anchor`; the RealSense did not. **1.9 px/bit
is the edge, and the edge is camera-dependent.**

**The rule.** Size for **>= 4 px per bit at the maximum range you intend to
detect from, on the lowest-resolution camera on the rig.** For a DICT_4X4
marker (4 data bits + 1 border each side = 6 bits across):

```
marker_size_m  >=  4 * 6 * range_m / fx_px  =  24 * range_m / fx_px
```

At fx ~ 425 px (D455, 848x480) and a 2 m working range that is **11 cm
markers**; at 4 m, **23 cm**. `coop2`'s were **15 mm**. These are A3/A2 boards,
not postcards. Print them large and the whole detection problem disappears.

**Check it before recording, not after:** `scripts/board_visibility.py`
computes px/bit from the bag's own `camera_info`. Run it on a 30-second test
recording and read the number.

### 2. Anchors at BOTH ENDS of the route

**The failure.** Holding out one of two boards left the chain anchored at one
end. The solve moved **7.26 m** while reporting convergence at whitened
residuals of 0.01 / 0.08. A near-zero residual with a large move is an
under-determined system, not a good fit.

**The rule.** The route **starts and ends** at a board dwell. Non-negotiable.
An unanchored stretch at an END is a lever; an interior gap is a bridge held
from both sides and is a different, milder thing.

Note this is also a warning to the nearest paper in the literature
(Marker-Constrained Pose-Graph Correction, 2026), which anchors with a single
relocated marker.

### 3. MANY gap lengths in one run — this is the headline experiment

**Why.** The coverage law (`THEORY_AND_VALIDATION.md` §2) predicts error from
the gap between anchors. `coop2` has exactly **one** interior gap, 118 s. One
point does not measure a curve, which is why the law currently has to be tested
by masking public datasets instead.

**The rule.** Lay boards so the route passes them at **deliberately varied
intervals**: roughly 10 s, 20 s, 40 s, 80 s, 160 s between sightings, in one
continuous run. Five or six boards, each visited at least twice.

That yields many (gap length, observed error) pairs **from real fiducial
detections with real detection failures**, on a route whose truth is known
everywhere by the held-out LiDAR. **It measures the coverage law natively**,
and it is evidence the masking experiments cannot produce, because masked mocap
has no detection failures, no incidence limits and no blur.

The two experiments are complementary and the paper needs both: masking tests
the estimator across many sequences; this tests the estimator **and the front
end** on one.

### 4. Rotation about more than one axis

**The failure.** Cumulative rotation per camera optical axis was
**[140, 1094, 101] degrees** — the cart yaws and does essentially nothing else.
The correlation matrix is rank one (singular values 44.238 / 0.014 / 0.005), so
**no method on that recording can self-calibrate its camera-IMU extrinsic.**
Every inertial rung has to be *given* the extrinsic, and MoCap2GT's own Fig. 4
reports the same failure under insufficient rotational excitation.

**The rule.** Include deliberate **pitch and roll** excitation — a handheld
segment, a ramp, or a pan-tilt head. Target comparable cumulative rotation on
all three axes. Two minutes of it is enough.

### 5. A static segment, and an overnight static IMU log

**The failures.** Two, both still open. The IMU noise model is `null` and
blocks the preintegration rung entirely, because Allan variance needs hours and
the bag is 156 s. And the gravity-direction check — the open question behind
`mobile_1`'s IMU making RTAB-Map *worse* — needs a stationary stretch to read
the accelerometer cleanly.

**The rule.** (a) **60 seconds stationary** at the start and again at the end of
the run. (b) Separately, leave each IMU logging on a desk **overnight** and run
`allan_variance_ros` once. It is a property of the unit, measured once, and it
serves every future recording.

### 6. If multi-agent: make them actually co-observe

**The failure.** `coop2`'s agents have 100% *place* overlap but swap corners,
so there is no useful simultaneous co-observation and the inter-agent factor
class is unavailable. (Strictly: simultaneity was never directly measured —
place overlap is a different statistic — but no inter-agent factor was ever
usable.)

**The rule.** If a second platform is recorded, route the two through a
**shared corridor at the same time**, in view of each other, for at least
30 seconds. Otherwise record one rig and drop the multi-agent claim.

---

## Also worth capturing

| item | why |
|---|---|
| **the prior map cloud**, exported and kept | `map_cloud` is null in `coop2` and it is the difference between the subject platform having one independent check and two. Export it as a file at recording time |
| **a static surveyed observer** with its pose uncertainty *stated* | `infra_1`'s pose is given to six decimals with no uncertainty. State the survey's sigma, or the check is bounded by an unknown |
| **camera-IMU extrinsics per camera**, from the SDK | recovering these cost multiple failed runs on `coop2`; one of them existed only in the cart's live tf and not in the bag |
| **board poses surveyed with a stated sigma** | `coop2`'s boards were repeatable to 3-13 mm but that was inferred afterwards rather than declared |
| **right camera image**, if any camera is stereo | `coop2` recorded only the left image, which killed the entire stereo and stereo-inertial family |

---

## What this recording buys, in paper terms

1. **Camera-agnosticism as a controlled experiment**, not an argument — the single biggest weakness in the current result.
2. **The coverage law measured natively**, on real detections, with several gap lengths instead of one.
3. **The transfer mechanism tested directly**: same route, same coverage, different drift rates. Does the law predict each camera's error?
4. **The D455 arm removes the transfer assumption entirely** for `mobile_2`'s existing result — same camera, certified.
5. **Two currently-blocked rungs unblocked**: IMU preintegration (noise model), and camera-IMU self-calibration (multi-axis rotation).

Estimated cost: one day of recording plus one unattended overnight. That is
cheap for what it closes.

---

## Before you record

Do a **30-second test recording** and run:

```bash
python3 scripts/bag_probe.py   --config <cfg> --stream <cam>.rgbd     # clocks, tf tree
python3 scripts/board_visibility.py --config <cfg> --bag <bag> \
    --agent <rig> --anchor-frame <json>                               # px/bit, facing
python3 scripts/depth_health.py --bag <bag> --topic <depth>           # dropout, starved stretches
```

Each of these caught a defect on `coop2` **after** the recording was finished,
when it was too late to fix. Twenty minutes now against a repeat session later.
