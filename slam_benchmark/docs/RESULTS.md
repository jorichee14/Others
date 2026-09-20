> Every number here is reproducible from the repo: `scripts/rescore.py` then
> `scripts/aggregate.py` for the benchmark, `scripts/construction_table.py` for
> the construction. Provenance for each run is in its own `run.json`.
> The argument is in `docs/PSEUDO_GT.md`, the design in `docs/STAGE2_GRAPH.md`,
> the chronology in `IMPLEMENTATION.md`.

# Results

Measured on `coop2_20260828`: two agents through one indoor room, 146–152 s,
26–31 m of path. `mobile_1` carries an Ouster LiDAR and a ZED 2i on one rigid
body; `mobile_2` carries a RealSense D455 and nothing else. Three surveyed
ChArUco boards are fixed in the room.

**The LiDAR is held out throughout.** It scores; it never feeds. That is what
makes `mobile_1` an instrument for certifying a procedure that `mobile_2`
cannot certify for itself.

---

## 0. The measuring instruments, checked first

A benchmark is only as good as the thing it measures against, so these were
validated before any method was scored.

**The board detector** (`scripts/board_detect.py`) reproduces the mapping
pipeline's own detections to **8.1 mm / 0.66°** on the ZED and **8.3 mm /
0.58°** on the RealSense — at or below each board's own scatter (2.0 and
6.9 mm). Distortion is handled at ChArUco corner interpolation, not at PnP;
handling it at PnP instead gave 34.2 mm, and undistorting the image gave
21.2 mm with *worse* reprojection.

**The board frame convention** was settled by measurement, not by reading the
survey's prose. All eight planar conventions (4 spins × 2 chiralities) were
re-solved on the same 90 detections and scored against the pipeline's file:
the winner sits at 6.0 mm, the runner-up at 288 mm. Two earlier guesses read
from prose gave 412 mm and 1449 mm.

**The reference trajectories, against the boards directly**: `mobile_1`
0.88° / 10.0 mm, `mobile_2` 1.38° / 54.0 mm.

**Does the route revisit?** Yes, on both agents — 280 interior closest
approaches on `mobile_1` and 140 on `mobile_2`, each within 0.30 m and at
least 20 s apart, measured on the *references* so it is a property of the
route. Loop closure is therefore evaluable here, and RTAB-Map's graph rows
are not odometry under another name. The separations themselves (medians 133
and 256 mm) are **not** a drift bound: they are censored by the 0.30 m search
radius, and a hand-pushed cart failing to retrace its line is
indistinguishable from reference drift without registering the two passes'
sensor data.

**Board coverage.** Four of six board × agent pairs carry observations:
`anchor` and `rs_anchor` on both agents. `anchor_b` carries none — see §4.

---

## 1. Benchmark (paper task B1)

Three backbones with independent failure modes, on both platforms. Every
tier-1 number is quoted with its tier-2 companion, because ATE against a
reference is agreement, not accuracy.

### `mobile_1` — reference is LiDAR-derived, uncertainty 15 mm

| method | ATE RMSE | rot RMSE | board residual | span |
|---|---:|---:|---:|---:|
| `rtabmap_rgbd_imu`, front-end | 372–481 mm | 5.8–7.1° | 467–577 mm (`anchor`), 350–548 mm (`rs_anchor`) | 99% |
| `rtabmap_rgbd_imu`, loop-closed graph | 366–451 mm | 3.4–4.7° | 421–499 mm (`anchor`) | **78%** |
| `mast3r_slam` | 1908 mm | 1.91° | 3536 mm | **64%** |
| `kiss_icp` | 2.2–6.7 m | 28–165° | 3.3–9.5 m | 99–100% |

**The two tiers agree on magnitude and on ranking.** That corroborates the
reference and is what licenses `mobile_1` as the instrument for §2.

`mast3r_slam` recovers shape but not size: its estimate is **37.1% too
large**, and removing the scale by a sim3 fit drops the ATE from 1908 mm to
85 mm. Quoting the 85 mm alone would be reporting a scale model of the room
as an accuracy.

`kiss_icp` fails outright — a 126 m path for a 25.8 m route — and the board
residuals confirm the failure independently of the reference.

### `mobile_2` — reference is cuVSLAM-derived and **correlated** with the methods

| method | ATE RMSE | board @ `anchor` | board @ `rs_anchor` | span |
|---|---:|---:|---:|---:|
| `rtabmap_rgbd`, front-end | — | **39 / 51 mm** | 95 / 118 mm | — |
| `rtabmap_rgbd`, graph | 135 / 149 mm | 55 / 87 mm | 94 / 123 mm | 97 / 95% |
| `mast3r_slam` | **120 mm** | 61 mm | 201 mm | **89%** |
| `kiss_icp` | 5008 mm | 2639 mm | 7379 mm | 100% |
| `rtabmap_rgbd_imu` | no result — see §4 | | | |

**Two findings here, and both are about the benchmark rather than the methods.**

**(a) `mobile_2`'s ATE cannot rank methods.** By ATE, MASt3R wins (120 <
135). By the surveyed 15 mm marker, RTAB-Map wins at `rs_anchor` by a factor
of two (94–123 vs 201) and is level-or-better at `anchor` (39–87 vs 61).
Neither board reproduces the ATE ranking. `mobile_2`'s reference is an
image-based VIO product, as is MASt3R, and a method correlated with its
reference is flattered by it. Contrast `mobile_1`, where the two tiers agree.

**(b) The comparison was not over the same run.** MASt3R's 120 mm covers
**89%** of the sequence; RTAB-Map's 135 mm covers **97%**. The conventional
coverage statistic reads 0.97–1.00 for both and cannot show this, because it
asks what fraction of the *estimate* found a reference — which stays near 1
for an estimate that simply stops early. A `span` column was added and every
row below 95% is marked.

### Two structural facts about the backbones

**RTAB-Map's loop-closed graph does not cover the end of `mobile_1`'s run.**
Every graph export ends 1787899928.08–928.75; the cart parks at `rs_anchor` at
928.753. RTAB-Map adds a graph node on *motion*, so a stationary robot
produces none. Graph rows are therefore scored over ~124 s and front-end rows
over ~151 s, and the surveyed board at that end is unreachable for any graph
output **by construction**.

**MASt3R keyframes are bursty**, not merely sparse: 151 keyframes over 142 s,
but clustered while the view changes and then quiet. This matters in §2.

---

## 2. The construction (paper task B2 — the headline)

A batch pose graph over the backbone's own relative motion plus surveyed-board
absolute priors. The architecture is deliberately the literature's — EuRoC's
estimator (2016), Vicon2GT (2020), Kalibr's MoCap branch, MoCap2GT (RA-L
2026) — with **board anchors substituted for a motion-capture system**.
Design and the novelty argument: `docs/STAGE2_GRAPH.md`.

One factor is structurally new. Those methods have absolute pose at 100 Hz
*continuously*; here it exists in **two windows totalling ~24 s of 146–152**,
so a backbone odometry factor carries the other two minutes. The sparsity is
the problem, not an inconvenience.

### V1 — the wiring gate

Odometry factors only. It must reproduce the backbone it was given, or nothing
downstream is trustworthy.

| agent | V1 ATE | backbone | board residuals |
|---|---:|---:|---|
| `mobile_1` | 372 / 377 / 411 mm | 372–481 mm | **identical to the digit** |
| `mobile_2` | 130 mm | 135 mm | **identical to the digit** |

Tier 2 matches its source exactly — 350.3 / 3.77°, 409.1 / 9.06°, 466.8 /
6.28° and so on appear unchanged on both rows. The graph adds nothing it was
not given.

### V2c — with the board anchors

**Scored unaligned**, in the surveyed map frame. An se3 fit to the reference
before scoring would remove the very placement the anchors provide, and reuse
the reference to do it. V1 is in the backbone's own frame and is aligned like
any backbone.

| agent | V2c (unaligned) | V1 (fitted **to** the reference) | rot | RPE 1 m |
|---|---:|---:|---:|---:|
| `mobile_1` | **228 / 254 / 254 / 291 mm** | 372 / 377 / 411 mm | 3.1–4.9° vs 4.5–7.2° | 132–176 vs 91–96 mm |
| `mobile_2` | 278 mm | 130 mm | 4.50° | 212 vs 177 mm |

**On `mobile_1`: 254 mm in the map frame with no use of the reference, against
377 mm for the same trajectory after being fitted to the reference.** The
anchors buy ~12 cm *and* remove the need for an oracle alignment. A pseudo-GT
that must be fitted to ground truth before it scores well is not a pseudo-GT.

**The trade is real and belongs beside the headline.** RPE at 1 m worsens from
~94 to ~154 mm: the anchors bend the trajectory to satisfy global constraints,
and local smoothness pays for it.

### Why the certification holds

**The declared weights do not carry the result.** Across a 4× range of
per-edge sigma the ATE moves **1.1 mm**:

| σ | 5.5 mm / 0.112° | 11.1 mm / 0.223° | 22.1 mm / 0.446° |
|---|---:|---:|---:|
| ATE | 290.4 mm | 291.2 mm | 290.1 mm |

The sigmas are derived from each run's RPE, which was scored against the
LiDAR — so without this sweep the LiDAR ATE could be dismissed as circular.
With it, the LiDAR ATE is independent certification.

**Leave-one-board-out cannot certify a two-board construction**, and that is a
property of the data rather than a gap:

| build | ATE | move | independent board residual |
|---|---:|---:|---:|
| `anchor` only | 1163 mm | **7261 mm** | `rs_anchor` 1719 mm |
| `rs_anchor` only | 771 mm | 178 mm | `anchor` 1201 mm |

Holding either board out leaves a **lever**: a chain anchored at one end,
where a rotation inside the remaining board's own 1° sigma becomes metres of
displacement 25 m along the path. The `anchor`-only build moved 7.26 m while
reporting convergence, a cost of 4.7, and whitened residuals of 0.01 / 0.08.
**A near-zero whitened residual is not a good fit**; with a large move it is
the signature of an under-determined system whose solver returned one member
of a family. The builder now warns on that conjunction.

Both leave-one-out numbers measure the *one-board* construction, not the
two-board one that is reported. With n = 2 there is no leave-one-out, so the
LiDAR is the certification of record.

**The RPE column shows the mechanism.** Both single-board builds sit at 94 mm
— exactly V1's value — because one board *transports* the trajectory rigidly
without bending it. The full build bends it, and the bending is where the
12 cm comes from.

### What conditions the graph: coverage in time, not board count

| agent | leading | trailing | widest interior gap |
|---|---:|---:|---:|
| `mobile_1` | ~0 s | ~0 s | 118.2 s |
| `mobile_2` | 0.0 s | 8.5 s | 125.1 s |

Both agents are long **bridges** held at both ends, not levers. `mobile_2`'s
trailing stretch moved **29 mm** against 142 mm in the anchored part — the
tail moved *less* than the middle. An unanchored stretch at an **end** is the
risk; an interior gap is bridged from both sides and is not the same thing.

### Transfer to `mobile_2`

V2c scores **278 mm against the cuVSLAM reference** where V1 scores 130 mm.
**This is disagreement, not error.** `mobile_2`'s reference is cuVSLAM-derived
and the RGB-D backbone shares its sensors and front-end, so 130 mm is
agreement between two correlated estimates. The construction moved 128 mm
median *toward the survey and away from cuVSLAM* — and on `mobile_1`, where a
held-out LiDAR exists, that same move was verified correct (377 → 254 mm).
So 278 mm measures how far cuVSLAM sits from the surveyed boards.

**The caveat is stated rather than resolved.** `mobile_1`'s backbone was badly
wrong at the boards (421–577 mm) and the construction had much to fix;
`mobile_2`'s was already good (39–95 mm) and it still moved 128 mm median,
most of that in the 125 s bridge where nothing measures it.

---

## 3. What could not be measured, and why

Reported with their numbers rather than omitted.

**`mobile_2`'s uncertainty cannot be priced by backbone consensus.** The plan
calls for running the construction on several independent front-ends and
taking their spread as a reference-free error bar. `mobile_2` has exactly one
front-end dense enough: `kiss_icp` is 5 m off, and `mast3r_slam` contributes
only **6 board observations** across its 151 keyframes (5 at `anchor`, 1 at
`rs_anchor`) against 321 for the RGB-D backbone. Six priors cannot anchor a
trajectory.

**Three of six factor-ablation rungs are unavailable on this recording**, each
for a specific measured reason:

| rung | status |
|---|---|
| V1, V2c | measured, above |
| V2b radar ego-velocity | feasible, not run |
| V2a IMU preintegration | **blocked** — the factor's covariance needs an IMU noise model, and Allan variance requires hours of static logging against a 156 s recording |
| V2d infrastructure | **blocked** — `infra_1`'s pose is stated to six decimals with no uncertainty anywhere, and an anchored fix cannot be better than its anchor |
| V2e inter-agent | **unavailable** — no simultaneous co-observation; the agents swap corners rather than share one |

**No method may self-calibrate its camera–IMU extrinsic on this sequence.**
Cumulative rotation per camera optical axis is **[140, 1094, 101]°** — the
cart yaws about one axis and does essentially nothing else, as a wheeled
platform on a flat floor does. The correlation matrix is rank one (singular
values 44.238 / 0.014 / 0.005). Rotation is not scarce (median 3.77 °/s, 61%
of intervals above 2 °/s); it is all on one axis, and one axis cannot
determine the other two. Any estimator that solves for its own extrinsic
meets the same deficiency, so every inertial rung must be *given* the
extrinsic.

**`anchor_b` is below the RealSense's resolving power.** A detector run
returned **0 of 179 frames**; 178 saw no ChArUco corners at all and the median
over the window was 2 of 48. This is not a tuning miss — the identical board
design at `anchor` reads 44 of 48 from the same camera in the same session.
At its only observed range, 0.84 m, its 15 mm markers span ~1.9 px per bit in
an 848×480 frame. `mobile_1` never gets closer than 2.48 m. The board
contributes to neither agent, so coop2 has three surveyed boards and each
robot gets two.

**`rtabmap_rgbd_imu` produced no result on `mobile_2`.** Four attempts. Three
distinct faults were found and fixed, each of which lets a run complete with
no inertial data in it: a null camera↔IMU extrinsic; a raw IMU stream with no
orientation quaternion, which RTAB-Map requires and does not compute; and the
extrinsic written in a schema its consumer does not read, so no transform was
ever published. After all three, the odometry still never initialised.

---

## 4. Reporting rules these results impose

1. **Never quote a tier-1 number alone.** On `mobile_2` the two tiers
   disagree on ranking; on `mobile_1` they agree. Which of those is true is
   not visible from tier 1.
2. **Quote the span.** Two ATEs over different fractions of a run are not
   comparable, and the conventional coverage statistic will not tell you.
3. **A tier-2 residual at a board the construction consumed is
   self-agreement.** 0.7 mm at `anchor` is 7 mm / √86 — the standard error of
   the observations it was fitted to. Only a held-out board or the LiDAR is
   evidence.
4. **State the alignment.** A construction anchored to a survey must be scored
   unaligned, or the fit hands it back the reference it is being measured
   against.
5. **A near-zero whitened residual is not a good fit.** With a large move it
   means the system is under-determined.

---

## 5. Claim, precisely

**Not novel:** the estimator. Factor graph + absolute-pose factors + a linear
initialiser is four papers deep. Framing the contribution that way invites
them all as prior art, and MoCap2GT must be cited regardless.

**Novel:**

1. **Pseudo-GT with no motion-capture volume** — three printed boards and a
   survey, with absolute evidence that is *sparse and intermittent* rather
   than continuous. The sparsity is the hard part.
2. **A certified error figure that does not use the reference**: 254 mm in
   the map frame, beating a backbone that was fitted to ground truth, with a
   sigma sweep showing the weights do not carry it.
3. **Transfer to an agent with no reference at all**, with a falsification
   test that can come back negative.
4. **A catalogue of what a dataset must contain** for this to work: board
   coverage at both ends of the route rather than board count; two boards
   minimum, since one is a lever; markers sized for the camera that must read
   them; rotation about more than one axis if anything is to self-calibrate.
