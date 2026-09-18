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
`mobile_2` runs for stages 3–4. Report tracked-fraction and the per-frame
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

## Standing rules

- The 16-method roster is a **menu** (`MOBILE1_LIDARFREE_ODOMETRY.md`); nothing
  from it runs unless a stage above calls for it.
- Refuse rather than default (`CLAUDE.md` rule 3) — the preflight nulls are the
  plan's gates, not obstacles.
- Every result lands in `IMPLEMENTATION.md` phase 6 in the same commit.
