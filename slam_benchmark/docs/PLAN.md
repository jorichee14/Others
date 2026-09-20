# THE PLAN — read this file only

One goal, five stages, each with a gate. Everything else in `docs/` is detail
this file points into. When this plan and another document disagree, this file
wins and the other document gets fixed.

**Goal:** a certified pseudo ground truth for the sensor-constrained agent
(`mobile_2`), for the MIRC dataset paper. `mobile_1` (LiDAR + ZED on one body)
is the instrument that certifies the procedure; the LiDAR is held out — it
scores, it never feeds. Full argument: `docs/PSEUDO_GT.md`. Paper tasks B1–B5:
defined at the end of this file.

**Known so far:** ZED SDK VIO lands ~6 m off (reflective walls → depth
dropout). The bag has IMUs (192 Hz ZED), radar Doppler, no right image.
Reference is in `zed_left_camera_optical_frame`, no transform needed.

---

## The two roles of SLAM in this project — and the wall between them

The same algorithms appear twice, in different jobs, and mixing the jobs makes
the paper circular. This section is the wall.

| | **Role A: ground-truth production** | **Role B: dataset benchmark** |
|---|---|---|
| purpose | produce the *released* reference trajectories | the tasks (B1–B5) users of the dataset compete on |
| SLAM is | an **ingredient** — one frozen backbone inside the anchored graph | the **subject** — the baselines in the tables |
| runs | stages 0–4, offline, once | after the GT is certified, scored against it |
| method count | 3 backbones (consensus), 1 chosen and frozen | the roster menu, as many as B1 wants |

**Per agent, the split is different, and this is the part to hold on to:**

* **`mobile_1`** — its GT is the *LiDAR* pipeline. No camera, IMU or radar
  method is inside it, so **every** such method is uncorrelated with the
  reference and every one can be a benchmark row. That is why `mobile_1`'s
  benchmark table is clean and rich, and why the certification experiment is
  legal there at all.
* **`mobile_2`** — its GT *is* the anchored construction, which contains one
  camera backbone. So on `mobile_2`, SLAM is **first Role A, then Role B**:
  build and freeze the construction, then score everything else against it.

**The circularity rule, inherited and extended.** `configs/coop2.yaml` already
refuses to rank GLIM against a GLIM-seeded reference (`correlated_with:
[glim]`; GLIM runs as an ablation, never a peer row). The same rule extends one
level down: **the backbone that feeds `mobile_2`'s pseudo-GT is excluded from
`mobile_2`'s B1 peer table.** When that reference is released, its config
declares `correlated_with: [<backbone>]` and the evaluator treats it exactly as
it treats GLIM today. Its row may still be *printed* — daggered, in the
ablation block — never ranked.

**The escape hatch is measured, not argued.** The backbone-consensus check
(stage 3c) quantifies how much the construction's output moves when the
backbone is swapped. If it moves less than the GT's own certified error bar,
the GT is *anchor-determined* — the correlation is demonstrably weak, the paper
can say so with a number, and the daggered row becomes readable. If it moves
more, the exclusion is doing real work and stays.

**And the third role, which is neither.** Running the LiDAR-free construction
on `mobile_1` against the held-out LiDAR is not GT production (the LiDAR GT
already exists) and not a benchmark task (nothing competes) — it is **GT
validation**, and its output is the error bar published *with* the `mobile_2`
reference. It is the methods-section contribution that licenses Role A's
product for Role B's use.

So, in one line: **on `mobile_1` the SLAM work is benchmark; on `mobile_2` it
is ground truth first and benchmark second, with the producer barred from the
race it made the track for.**

---

## Stage 0 — foundations. ~1 day + 1 unattended overnight. No containers.

Everything here is hours or minutes, needs no method, and every later stage
consumes its outputs. Do it in this order.

| # | item | cost | output |
|---|---|---|---|
| 0.1 | Reference `.txt` → TUM; `expected_start` check; resolve the 1516-vs-2833 count; path length | 1 h | the reference the whole plan scores against |
| 0.2 | `depth_health.py` on `mobile_1` ZED **and** `mobile_2` RealSense | 30 min | dropout covariates for both platforms; where the 6 m came from |
| 0.3 | **I2 — board dwell windows**, both agents, read off the run once | 1 h | the only absolute evidence with independent error bars. Load-bearing. |
| 0.4 | **V0 — reference vs boards** at those windows | 30 min | the ceiling; does the 15 mm claim hold |
| 0.5 | **Infra residual check** — `predict_observation` vs reference, `axes="optical"` | 2 h | validates `infra_1`'s pose and the third clock before anything is built on them |
| 0.6 | **Gates:** B0 place overlap + `infra_1` visibility fraction, from reference TUMs | 15 min | decides whether inter-agent and infra factor classes exist |
| 0.7 | Score `/mobile_1/zed/pose` (loop-closed SDK) | 30 min | how much of the 6 m loop closure alone recovers |
| 0.8 | **I13 — cam↔IMU extrinsic** from SDK / `SN*.conf` / `tf_static`; gravity-direction sanity check | 1 h | unblocks the tightly-coupled backbones |
| 0.9 | **Start the 3 h static IMU log** (I14) tonight | 0 effort | the noise model, once, forever |
| 0.10 | Radar Doppler **field name + sign** (sign off one known-motion segment) | 30 min | unblocks the radar factor class |

**Gate out of stage 0:** boards readable (0.3/0.4) and infra residuals sane
(0.5). If either fails, fix it before stage 1 — every anchor class hangs on
them. B0 failing only removes the inter-agent class; the plan survives.

## Stage 1 — backbones (paper task B1). ~1 week, mostly container builds.

Three backbones, chosen for **independent failure modes**, not quality — their
disagreement is the uncertainty estimate that works where no GT exists
(`docs/PSEUDO_GT.md`, "backbone consensus"):

| backbone | uses | fails when |
|---|---|---|
| `kiss_icp` (zed depth) | geometry only | depth drops out |
| `rtabmap_rgbd_imu` → `orbslam3_rgbd_inertial` once I13/I14 land | features + depth + IMU | low texture, low excitation |
| `mast3r_slam` | image + learned prior | prior out of distribution |

Exactly one entry per failure family — `splat_slam` shares the learned family
with `mast3r_slam` (learned-flow tracking, image-only), so it may *replace*
MASt3R in that slot but never sit beside it in the consensus: two learned
entries agreeing through a shared prior would certify nothing.

Run each on `mobile_1.zed_rgbd` (LiDAR held out) and on
`mobile_2.realsense_rgbd`. Score vs the reference on `mobile_1`; hold the
`mobile_2` runs for stages 3–4.

**GATE PASSED 2026-09-19 on `rtabmap_rgbd_imu` × `mobile_1`: 97.3% of the run
tracked, 0 resets, 95 optimised nodes. Loop-closed ATE 365 mm / drift 1.52% /
scale 1.057 vs 408 / 1.43% / 1.064 for the front-end alone — the residual is a
+5.7% scale bias that loop closure cannot touch, i.e. the depth. Getting there
took one line (`qos:=1`; RTAB-Map's default subscriptions are best-effort under
rmw_fastrtps and silently lost two thirds of the frames) and is recorded in
`IMPLEMENTATION.md`. RTAB-Map is not reproducible run to run (366–451 mm over identical runs), so the cell is reported as median [min, max] via `scripts/spread.py`: **N=4, loop-closed 381 mm [366, 451], drift 1.60% [1.54, 1.90], scale 1.055 [1.048, 1.062]; front-end alone 394 mm [372, 481].** Cell filled. `kiss_icp` × `mobile_1` (2026-09-20, offline, every frame): tracks 100%, **ATE 2.24 m, scale 1.40** — passes the gate on count, fails on quality; the geometry-only row is a failure result on this depth. mobile_2 cells (2026-09-20, gate on count only, scores held for stages 3–4): `kiss_icp` 4300/4300; `rtabmap_rgbd` 4281/4300 with the container registering the RealSense depth into the colour camera. `mast3r_slam` × `mobile_1` (2026-09-20): 38 keyframes, **sim3 85 mm / 0.44% drift**, scale 0.729 — monocular, so sim3-only and no metric scale. **Stage 1 grid complete.** Remaining cells unchanged below.**

**MEASURED 2026-09-19 — the grid is five cells and one named blocker, not six.**
`scripts/bag_probe.py` on the bag: `/tf_static` carries **7 edges in 2
disconnected trees** — the ZED chain and the Ouster chain — and nothing else. No
RealSense frames, no `infra_1` frames, and no `zed_imu_link`. Consequences:

| cell | state |
|---|---|
| `kiss_icp` × both | runs |
| `rtabmap_rgbd_imu` × `mobile_1` | runs — the camera↔IMU edge is I13, published into the container from the config |
| `rtabmap_rgbd_imu` × `mobile_2` | **blocked**: `mobile_2.imu.extrinsic_from_camera` is null and the bag cannot supply it. Recover with `rs-enumerate-devices -c`, once, like I13 |
| `rtabmap_rgbd` × `mobile_2` | runs — the IMU-free half of the pair carries that platform's feature+depth slot meanwhile |
| `mast3r_slam` × both | runs |

Two properties of the recording, both now in `configs/coop2.yaml` per stream:
`mobile_1`'s colour and depth are stamped **byte-identically** (2288/2288), so
that stream must be synchronised exactly — an approximate matcher paired its
frames a full 67 ms period apart on the first run. `mobile_2`'s are 10 µs apart
(0/4295 exact), so it needs approximate sync with a tight interval. Report tracked-fraction and the per-frame
instruments (observability, ORB density) beside every ATE.

**Deliverable:** the B1 table — SDK rows, three backbones, two platforms.
**Gate:** at least one backbone tracks ≥ 90% of the run on each platform.
If none does, the paper's B1 finding is "online estimation fails here" and
stage 2 leans harder on anchors — the plan bends, it does not break.

## Stage 2 — the construction (paper task B2 core). ~2 weeks. The main build.

One batch pose graph (GTSAM/iSAM2), run six times with growing factor sets —
this replaces the sixteen-method survey:

```
V1  backbone only                        <- the no-anchor floor
V2a + IMU preintegration
V2b + radar ego-velocity                 [needs 0.10]
V2c + board anchors                      [needs 0.3]
V2d + infrastructure bearing/range      [needs 0.5]
V2e + inter-agent (mobile_1 co-observation)  [needs 0.6 pass]
```

Each run certified against the held-out LiDAR on `mobile_1`. The radar
ego-velocity front-end is ~100 lines of numpy (RANSAC over Doppler triples),
tested on synthetic sweeps first; the infra fusion reuses `slambench/collab.py`.

**Deliverable:** the factor-ablation ladder — what each evidence class buys, in
centimetres. This one table prices B3 and B4 as side effects.

## Stage 3 — certification (paper task B2 proof). ~1 week. Statistics, no new runs.

1. **Error distribution**, not an RMSE: percentiles, worst stretch, **block
   bootstrap over time segments** (trajectory errors are autocorrelated; ~1516
   poses are ~a handful of independent failure episodes).
2. **Covariate model:** fit error vs {dropout fraction, longest starved
   stretch, anchor visibility, time since last absolute factor} on `mobile_1`;
   validate leave-one-segment-out. This is what transfers — a function, not a
   scalar.
3. **Backbone consensus:** run the full construction (V2e) on all three
   backbones; their spread is the reference-free uncertainty, usable on
   `mobile_2`.

**Deliverable:** "the construction is accurate to X cm (95%), and its error is
predicted by these covariates" — with the fit quality shown.

## Stage 4 — transfer to `mobile_2` + paper. ~1–2 weeks.

1. **Freeze** the construction (no re-tuning — retuning voids the bound).
2. Apply to `mobile_2`; predict its error from the covariate model; report
   `mobile_2`'s covariates against `mobile_1`'s observed range (interpolation
   vs extrapolation, stated).
3. **The test:** `mobile_2`'s board-dwell and infra residuals must be
   consistent with the prediction. Consistent → corroborated. Not →
   the transfer failed, which is also a result and goes in the paper as one.
3b. **The consumer test (optional, GPU):** train the same Gaussian-splat map
   (`splat_slam`) from the pseudo-GT poses and from the reference poses on
   `mobile_1`; the held-out-view rendering gap is what the pseudo-GT's error
   costs a downstream user — utility, not geometry. On `mobile_2` the relative
   version needs no reference: better poses render sharper.
4. Write B1–B5 (task definitions below), the two reporting rules (every number
   in all tiers; every GT-adjacent number with an error bar), and the limits
   section with the next-session spec.

---

## The paper's five tasks (what stages feed what)

| task | what | fed by |
|---|---|---|
| B1 | constrained-agent trajectory, online | stage 1 |
| **B2** | **pseudo-GT construction + certification — the headline** | stages 2–4 |
| B3 | infrastructure-anchored localization | 0.5 + V2d |
| B4 | strong-to-weak inter-agent anchoring | 0.6 + V2e (report the gate either way) |
| B5 | depth-dropout robustness, natural stressor | 0.2 + stage 3 covariate model |

| B6 *(appendix, GPU)* | dense reconstruction + novel-view synthesis (`splat_slam`; reflective walls as the stressor Spires lacks) | stage 4.3b |

S1 (LiDAR trajectory) is a control row inside B1; S3 (maps) folds into B6.

## Next recording session (fixed cost, huge return — schedule it now)

1. Right ZED image (one setting — restores the stereo family)
2. Route both agents through one shared corridor (makes B4/V2e real)
3. 5 s of deliberate excitation at each start (VI initialisation)
4. NTP monitor on `mobile_1`; fiducials on the carts (track C's recommendation)
5. **Stop the IMU last.** Measured 2026-09-19: `/mobile_1/zed/imu/data` spans
   152.35 s against the image stream's 156.21 s, and a front-end that needs an
   inertial sample newer than each frame drops every image past the IMU's last
   one — ~58 frames here, at the end of the run. Start the inertial topics
   first and stop them last, on every agent.

## Standing rules

- The 16-method roster is a **menu** (`MOBILE1_LIDARFREE_ODOMETRY.md`); nothing
  from it runs unless a stage above calls for it.
- Refuse rather than default (`CLAUDE.md` rule 3) — the preflight nulls are the
  plan's gates, not obstacles.
- Every result lands in `IMPLEMENTATION.md` phase 6 in the same commit.
