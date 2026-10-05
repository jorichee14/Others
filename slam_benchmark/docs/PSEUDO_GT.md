> Execution order and gates live in `docs/PLAN.md`; this file carries the argument.

# Pseudo ground truth for a sensor-constrained agent

The goal, restated: **`mobile_2` has no LiDAR, so there is no LiDAR-derived
reference for it.** Every trajectory claim about the constrained agent — and
every collaborative-perception result that depends on its pose — currently
rests on nothing. This document is about building a pseudo ground truth for
that agent, and, more importantly, about **stating its error bar honestly**.

`mobile_1` is not the subject. It is the **instrument**: the one platform that
carries a LiDAR *and* a depth camera on one rigid body, so a LiDAR-free
pseudo-GT can be built there and checked against the LiDAR-derived one. That
check is what converts a pseudo-GT from an assertion into a measurement with an
uncertainty.

---

## What the literature does, and where it stops

### The dominant recipe: register a mobile LiDAR into a survey-grade prior map

This is the state of the art for GT without motion capture, and `configs/coop2.yaml`
already names it as the construction the coop2 reference imitates.

- **Newer College** (Ramezani et al., 2020) — centimetre-accurate GT by
  registering LiDAR scans against a TLS prior map.
- **Oxford Spires** (Tao et al., IJRR 2025/26) — the refined version: offline ICP
  of motion-undistorted LiDAR clouds into a merged TLS map, using an offline
  VILENS with unlimited compute per frame. **~1–2 cm**, with the error sources
  named (LiDAR noise, ICP matching). Its localisation benchmark then scores
  LiDAR SLAM, LIVO and LiDAR-BA methods against that.
- **ConSLAM** — the same idea with periodic survey-grade scans on a construction
  site.
- **Hilti SLAM Challenge** — total station plus TLS, millimetre-scale.

**Where it stops: the recipe requires the evaluated platform to carry the
LiDAR.** Every one of these builds GT *for the sensor that produced it*. None of
them has the problem "the agent I need a reference for has no LiDAR at all."

### The pseudo-GT-from-a-better-SLAM recipe

Where no GT exists, people substitute the output of a stronger system and say so:

- **KITTI sequences 11–21** carry no official poses, so recent work adopts
  **PIN-SLAM** output as pseudo-GT, justified on the grounds that LiDAR geometric
  constraints do not drift the way vision does in texture-sparse scenes.
- Various datasets derive pseudo-GT from LiDAR-inertial fusion and then
  **validate it by comparing the reconstructed map against a geospatial map** —
  i.e. the validation is on the *map*, not on the trajectory.

**Where it stops: the substitute is asserted, not certified.** The justification
is qualitative ("LiDAR drifts less"), and the error bar is usually absent. One
survey result puts it bluntly: GT from LiDAR-SLAM relies on relative trajectory
estimation, which makes its accuracy hard to quantify, and errors can exceed ten
metres.

### The GT-uncertainty literature — the closest work to the right question

A small, recent body of work treats the reference itself as an estimate with an
uncertainty:

- **Uncertainty analysis for GT trajectories with robotic total stations**
  (Vaidis et al.) — Monte Carlo propagation to a 6-DoF uncertainty, and the
  observation that GT pose uncertainty *is rarely computed at all* although
  evaluation has become precise enough to need it.
- **RTS-GT** — robotic total stations as an independent GT source, with
  inter-run consistency and absolute accuracy assessed rather than assumed.
- **MoCap2GT** (2025) — mocap plus IMU fusion, explicitly motivated by *errors in
  the official GT of public datasets*, i.e. the reference being wrong is a known,
  published problem.
- **Boxi** (2025) — validates a reference system by checking that its reported
  uncertainty bounds actually match observed error across conditions.

**This is the right instinct and the closest precedent.** But it is applied to
instrumented references — total stations, mocap — on the platform being measured.

---

## The gap

> Nobody builds a pseudo-GT **for an agent that lacks the reference sensor**, and
> certifies its error bar on a twin platform that carries both.

That is the shape of what coop2 supports, and it is a held-out-sensor argument in
the statistical sense rather than a metaphor:

```
CERTIFY on mobile_1            APPLY to mobile_2
-------------------            -----------------
has Ouster AND ZED             has RealSense only
                    
build pseudo-GT WITHOUT        build pseudo-GT by the
the LiDAR                      SAME construction
        |                              |
compare against the            no LiDAR to compare against
LiDAR-derived reference                |
        |                              |
   error distribution  ----------> quote THAT as its error bar
```

The LiDAR is **held out** — used to score the construction, never as an input to
it. What you learn on `mobile_1` is the error distribution of a procedure, and a
procedure's error transfers to a second platform in a way a single trajectory's
does not.

**The honest caveats, stated now rather than discovered by a reviewer:**

1. The transfer assumes the two platforms' constrained sensors fail *similarly*.
   They do not exactly: a ZED with reflective-wall dropout is not a RealSense at
   27.5 Hz with a narrower FoV. So the transferred bound is an **estimate with a
   stated platform-transfer assumption**, and the assumption is testable — run
   the construction on both and compare the *shape* of the residuals, not just
   their size.
2. coop2's own reference is a pipeline output, not a survey. It is good to a
   stated 15 mm against the boards, and every number here inherits that.
3. One 156 s sequence gives one sample of the error distribution. Two more
   sequences would make it a distribution rather than a point.

---

## What a pseudo-GT is allowed to be, and why that changes the roster

**A pseudo-GT is offline.** This is the single most important consequence and it
is easy to miss after several passes of thinking about odometry. An online SLAM
system is causal, real-time and single-pass. A pseudo-GT construction is none of
those. It may use:

- the whole sequence at once, including the future
- unlimited compute per frame (Oxford Spires' GT does exactly this)
- multiple passes, re-initialising from the previous pass
- every absolute anchor available, whether or not an online agent could use it
- loop closure and global batch optimisation as a matter of course

So the question stops being *which SLAM system is most accurate* and becomes
**what is the best offline estimator you can build from everything the
constrained agent's session recorded**. That is a different question and it has a
different answer.

### The construction

A single batch pose graph over the constrained agent's trajectory, with every
evidence source coop2 carries as a factor class:

| factor | source | what it does |
|---|---|---|
| **odometry** | the agent's best RGB-D / visual-inertial result | the backbone; shape between anchors |
| **IMU preintegration** | 192 Hz ZED IMU / 183 Hz `mobile_2` IMU | bridges the dropouts that break the backbone |
| **radar ego-velocity** | 2× IWR6843 Doppler | bounds velocity where vision is starved — immune to reflective walls |
| **board anchors** | the three surveyed boards | **absolute**, 3–15 mm, the only evidence with an independent error bar |
| **infrastructure** | `infra_1` detections, surveyed static pose | **absolute** and *continuous*, from neither the pipeline nor the agent |
| **inter-agent** | co-observation with `mobile_1` | **`mobile_1` is LiDAR-grade — for `mobile_2` it is the STRONG partner** |
| **loop closure** | within the agent's own run | removes accumulated drift globally |

**Correction to earlier advice in this project.** I previously wrote that the
agent-to-agent rows cannot help, because `mobile_2` is the weaker platform. That
is right *for improving `mobile_1`* and **backwards for this goal**. The subject
is now `mobile_2`, and `mobile_1` carries a LiDAR — so inter-agent anchoring runs
from the strong platform to the weak one and becomes a **primary** pseudo-GT
mechanism rather than a marginal one. `swarm_slam_nolidar` and
`decoupled_registration_nolidar` were scoped as common-frame experiments; for
pseudo-GT the relevant construction is the *oracle* direction — anchor `mobile_2`
to `mobile_1`'s reference-grade poses wherever the two co-observe.

### What the roster is now for

The sixteen entries stop being competitors and become **candidate backbones plus
the ablation that prices each factor class**. The deliverable is not a ranking.
It is:

> *"A pseudo-GT for a LiDAR-free agent built from {odometry + IMU + boards +
> infrastructure + inter-agent} is accurate to X cm (95%), measured by holding
> out the LiDAR on a twin platform. Dropping the infrastructure factors costs Y;
> dropping the inter-agent factors costs Z."*

That sentence is the paper. Every row in the roster exists to supply one number
in it.

---

## The protocol

**V0 — the ceiling, and it is free.** Before building anything, compute how well
`mobile_1`'s *LiDAR-derived* reference agrees with the board survey. That is the
best case: no LiDAR-free construction can beat the reference it is scored
against. It also answers whether the 15 mm figure holds in practice. Needs the
board dwell windows — **I2, still open, and now the highest-value hour in the
project** rather than merely the highest-value one in the benchmark.

**V1 — the backbone.** Pick the best LiDAR-free odometry on `mobile_1` from the
revised core five. Its ATE against the LiDAR reference is the **no-anchors
baseline**: what a pseudo-GT would be worth with the agent's own sensors alone.
Given the 6 m ZED result, expect this to be poor, and that is the point — it is
the number every factor class has to improve on.

**V2 — add factor classes one at a time, on `mobile_1`, LiDAR held out.**
Each addition is priced against the LiDAR reference:

```
backbone
backbone + IMU
backbone + IMU + radar
backbone + IMU + radar + boards
backbone + IMU + radar + boards + infrastructure
backbone + IMU + radar + boards + infrastructure + inter-agent   <- the full construction
```

Six runs of the same optimiser with different factor sets. **This is the
experiment**, and it is smaller than the sixteen-method survey it replaces.

**V3 — characterise, do not just rank.** Three parts, and each fixes a way the
transferred bound would otherwise be hollow.

*(a) The distribution, with honest sample size.* Percentiles and the worst
stretch, never one RMSE — a pseudo-GT with a 5 cm median and a 2 m worst case is
unusable for what pseudo-GT is used for. And the error bar on those percentiles
comes from a **block bootstrap over time segments**, not per-pose statistics:
trajectory errors are heavily autocorrelated, so ~1516 poses are effectively a
handful of independent failure episodes, and a per-pose bound would overstate
confidence by an order of magnitude.

*(b) The covariate model — what actually transfers.* Do not transfer a scalar
("8 cm on mobile_1, so 8 cm on mobile_2"); the two platforms' inputs are not
exchangeable and the claim would be unfounded. Fit error as a function of
covariates **observable on both platforms**: depth-valid fraction, longest
starved stretch (`scripts/depth_health.py` supplies both), anchor visibility
fraction, inter-agent co-observation rate, time since the last absolute factor.
Validate the fit leave-one-segment-out within mobile_1. The transferred object
is this *function*; evaluating it at mobile_2's covariates is the prediction,
and reporting mobile_2's covariates against mobile_1's observed range says
whether that prediction is interpolation or extrapolation.

*(c) Backbone consensus — the uncertainty that needs no reference.* Run the full
construction on **three backbones chosen for independent failure modes** —
geometric (`kiss_icp` on depth), feature-inertial (`orbslam3_rgbd_inertial` /
`rtabmap_rgbd_imu`), learned-prior (`mast3r_slam`). Their spread through the
same anchored graph is evidence about whether the solution is determined by the
anchors and the geometry or by the estimator — and it is computable on
`mobile_2`, where nothing else is. Diversity is the design requirement: three
similar RGB-D systems would agree through a shared failure mode and the check
would certify nothing. This is the project's own `seed_independence_check`
(kiss_icp.yaml) applied one level down.

**V4 — apply to `mobile_2` and quote the transferred bound.** Same construction,
no LiDAR to check against, error bar from V3 with the platform-transfer
assumption stated.

**V5 — the independent check on `mobile_2`, framed as a test of V4's
prediction.** Not optional, because V4's bound is transferred rather than
measured. `mobile_2`'s pseudo-GT must be consistent with the board survey at its
own dwell windows and with `infra_1`'s observations — the only evidence about
`mobile_2` that does not come from the construction itself. Run it as a
falsification test, not a formality: V3(b) *predicts* the residual distribution
at those checkpoints; V5 observes it. Consistent → the transfer is corroborated
by independent evidence and the paper says "predicted X cm, corroborated by N
independent residuals". Inconsistent → **the transfer failed, which is a result**
— report it as one, with the covariate comparison that says why.

---

## What this changes about the recording

Two items already in `docs/MOBILE1_LIDARFREE_ODOMETRY.md` get more urgent,
and one is new.

1. **Board dwell windows (I2).** Was "the highest-value hour in the benchmark".
   It is now load-bearing: the boards are the only absolute evidence with an
   independent error bar, and both V0 and V5 need them.
2. **`infra_1`'s pose uncertainty (I9).** An anchor's uncertainty propagates
   directly into the pseudo-GT's. Currently stated nowhere.
3. **NEW — route the agents through a shared corridor next session.** Inter-agent
   anchoring is a primary mechanism here, not a curiosity, and starting 16.3 m
   apart in a 16.6 m room may leave it with nothing to work with. The B0 gate was
   a nice-to-have for track B; for pseudo-GT it decides whether one of the
   strongest factor classes exists at all.

---

## Sources

- Oxford Spires GT methodology: <https://github.com/ori-drs/oxford_spires_dataset/wiki/Ground-Truth>
- Oxford Spires Dataset (IJRR): <https://doi.org/10.1177/02783649251369905>
- Newer College GT: <https://ori-drs.github.io/newer-college-dataset/ground-truth/>
- Newer College Dataset: <https://arxiv.org/pdf/2003.05691>
- Uncertainty analysis for GT trajectories with robotic total stations: <https://arxiv.org/pdf/2308.01553>
- RTS-GT dataset: <https://arxiv.org/pdf/2309.11935>
- MoCap2GT: <https://arxiv.org/pdf/2507.12920>
- Boxi (reference-system design/validation): <https://arxiv.org/pdf/2504.18500>
- S3E collaborative SLAM dataset: <https://arxiv.org/html/2210.13723v5>
- CU-Multi multi-robot dataset: <https://arxiv.org/pdf/2509.19463>
