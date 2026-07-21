# Radar–Camera Extrinsic Calibration — How This Method Compares

This document positions `radar_camera_calib.py` against the published literature,
identifies concrete improvements, and lists references.

Short version: your method sits in the **target-based, single-correspondence**
family (ChArUco board + trihedral corner reflector), but its estimator is closer
in spirit to the newest **uncertainty-aware** target-based tools than to the
classic ones. Its two genuinely distinctive pieces are (1) using a **metric
ChArUco board pose** instead of a checkerboard centre + depth/click, and (2)
**jointly solving the apex offset (MAP)**. Where it is behind the frontier is
**initialization robustness, temporal calibration, and a targetless cross-check**.

---

## 1. The landscape (three families)

### A. Target-based, single correspondence point (your family)

A fiducial that both sensors can localize. The camera gives a full pose (or a
point); the radar gives one bright return (a trihedral corner reflector, chosen
for its high, stable RCS and compact size). You move the rig to N poses and solve
a point-set registration for `T_cam_radar`.

| Work | Target | Camera cue | Radar model / solver | Notes |
|---|---|---|---|---|
| **Peršić et al. 2019** | Styrofoam triangle, checkerboard front / trihedral back | board pose | reprojection + **RCS** cost, 2-step opt | radar–lidar–camera; RCS used to reject bad returns |
| **Domhof et al. 2019/2021 (TU Delft)** | 1.0×1.5 m styrofoam, 4 holes + centre trihedral | circle centres | strongest return, joint 3-sensor opt | open-source ROS tool; ~0.33° rotation |
| **Cheng et al. 2023** | corner reflector | checkerboard centre | RANSAC + Levenberg–Marquardt | "flexible & accurate" target-based 3D radar–camera |
| **Cao et al. 2025 (3DUPnP)** | corner reflector | PnP correspondences | **3D-uncertainty PnP**, models spherical-coord noise + de-biases the coordinate transform | *closest peer to your estimator* |
| **This tool** | **ChArUco board + trihedral** | **full metric board pose** | **measurement-space ML** in (range, az, el) weighted by per-axis σ; Huber + σ-gating; **joint apex-offset MAP**; covariance/observability | metric board pose (no depth topic); offset solved |

### B. Targetless / motion-based (no fiducial)

Recover the extrinsic from natural-scene motion — align trajectories, or match
radar ego-velocity to camera ego-motion.

| Work | Signal | Solver | Recovers |
|---|---|---|---|
| **Peršić et al. 2020** | moving-object tracks + RCS | trajectory sync | yaw only; needs known translation |
| **Wise et al. 2021 / 2023** | radar **ego-velocity** + unscaled camera pose | **continuous-time** batch NLS | full 3D extrinsic **+ time offset** |
| **Schöller et al. 2019** | scene features | deep net | rotation auto-calib |
| **Cheng & Cao 2023** | common features of RD-A data + image | deep feature matching | online extrinsic |
| **Trajectory Alignment 2025** | moving-object trajectories | alignment | full extrinsic, targetless |

### C. Learning-based (end-to-end)

RC-AutoCalib (2025), RLCNet (2025), CLRNet (2026), CalibRefine (2025) — regress or
iteratively refine the extrinsic from raw data. Flexible/online, but need training
data, generalize poorly across rigs, and give no interpretable covariance. Not a
fit for a one-rig lab calibration where you want a trustworthy number.

---

## 2. Where your method is strong (vs the field)

1. **Metric camera cue.** Almost every target-based paper localizes the camera
   side with a *checkerboard centre* (a single point) plus a hand-click or a depth
   topic on the bare reflector — noisy and specular. You use the **full ChArUco
   board pose**, which is sub-mm and automatic, and derive the apex from a rigid
   offset. This is a real, defensible upgrade and worth stating explicitly as your
   contribution. (README §"Why the camera side uses the board, not depth".)

2. **Anisotropic, measurement-space likelihood.** Classic tools (Domhof, early
   Peršić) minimize isotropic Cartesian error — provably biased because radar
   cross-range error grows as `range·σ_az`. You minimize residuals in
   (range, az, el) weighted by real per-axis σ. Only the newest work (Cao 2025,
   3DUPnP) does something equivalent. You are on the current frontier here.

3. **Joint apex-offset estimation (MAP).** No target-based paper I found *solves*
   the reflector offset; they all measure it by hand and trust it. Making `a`
   observable through board tilt and regularizing it toward a prior is genuinely
   uncommon and directly attacks the phase-center-wander problem you documented.

4. **Honest observability reporting.** Per-DOF covariance, planar/2-D detection,
   LOO-CV, condition number, signed per-axis bias, and a pass/fail VERDICT. Most
   open tools report a single RMS. This is better engineering practice than most
   of the literature.

5. **Robustness.** Huber + iterative σ-gated rejection, background subtraction,
   Doppler gating, camera-range gating, min-SNR strict capture. Comparable to or
   better than the RANSAC+LM of Cheng 2023.

---

## 3. Where it is behind — and how to improve it

Ordered by expected payoff for *your* live-convergence blocker (azimuth spread /
under-constrained rotation) and for scientific defensibility.

### 3.1 Add a targetless motion-based cross-check (highest value)
Your rotation is under-constrained because a hand-held board can't span wide
azimuth. **Wise et al.'s continuous-time, ego-velocity approach** needs *no wide
azimuth* — it uses the radar's Doppler ego-velocity against camera ego-motion, and
it also recovers the **time offset** you currently assume away. Even if you keep
the target method as primary, running a targetless solve as an independent check
(or as a rotation prior) would break your current deadlock and make the result far
more convincing. → *Wise 2021/2023; Peršić 2020.*

### 3.2 Estimate temporal offset (you currently assume sync)
You pair via `message_filters` ApproximateTime. With a **moving** reflector
(continuous/sweep mode), even 30–50 ms of radar-vs-camera latency puts the two
"same physical point" observations at different places → a systematic azimuth
residual that looks like a rotation error. Add a scalar time-offset parameter
(estimate it, or at least sweep it and watch residual). This is standard in
continuous-time calibrators and is likely contributing to your `residual ≈ 2.3σ`.

### 3.3 Use multiple reflectors → instant spatial diversity per pose
Your rotation needs non-collinear points; a single reflector forces you to *move*
to get them, and wide azimuth by hand is hard. Mounting **3+ trihedrals at known
board-frame offsets** gives 3+ non-collinear correspondences from a *single* pose
(each is a separate `a_k`), which constrains rotation immediately and de-risks the
"clustered poses → weak rotation" failure. This is the cheapest hardware fix and
turns the collinearity problem into a non-issue. (Compare CalTag's multi-tag idea.)

### 3.4 Model the spherical→Cartesian bias in the Kabsch init (Cao 2025)
3DUPnP's headline point is that converting noisy `(r, az, el)` to Cartesian has a
**non-zero mean** (the transform is nonlinear), so Cartesian least-squares is not
just badly-weighted but *biased*. Your final solve lives in measurement space and
sidesteps this — good — but your **Kabsch initialization is Cartesian and biased**,
which matters when the ML solve can fall into a nearby basin. Two options:
(a) de-bias the init the way Cao does, or (b) prefer the extrinsic-prior init (you
already support it) whenever a prior exists. → *Cao et al. 2025.*

### 3.5 Global / certifiable initialization instead of Kabsch
Kabsch → local NLS can land in a wrong basin (you saw pitch bouncing −47↔−72°).
A **semidefinite-relaxation / certifiable** point-registration init (à la the
certifiable-perception line, or Wise's globally-optimal treatment) gives a
provably-good starting point and would kill the "small local 1σ hiding a bad
basin" risk you flagged in the sprint notes.

### 3.6 Fold RCS / phase-center into the model (Peršić)
You documented the large trihedral's radar phase center sitting ~7 cm behind the
geometric apex, range/aspect-dependent. Peršić's RCS-aware cost models exactly this
kind of return-quality/position effect. At minimum, weight each correspondence by
its SNR/RCS (down-weight glancing, low-RCS returns) rather than a hard min-SNR
gate; better, model a small aspect-dependent phase-center shift. → *Peršić 2019.*

### 3.7 Report against a shared benchmark / baseline
For a paper or a convincing write-up, run the **CPnP/3DUPnP baseline** and the
**Domhof open-source tool** on your data and tabulate. Your current numbers
(|t|≈18 cm, 3-D≈165 mm) mean little without a baseline column; Cao 2025 explicitly
benchmarks against CPnP, and that's the comparison a reviewer will want.

### 3.8 Hardware (already in your sprint notes, restated for completeness)
Higher camera resolution + a larger (~100 mm) ChArUco unlock range diversity; a
smaller/point-like reflector reduces phase-center wander. These gate accuracy more
than any solver change right now.

---

## 4. One-paragraph "related work" you can drop into a report

> Target-based radar–camera calibration typically pairs a checkerboard with a
> trihedral corner reflector and minimizes an isotropic Cartesian registration
> error [Domhof 2021; Cheng 2023], optionally augmented with radar-cross-section
> consistency to reject poor returns [Peršić 2019]. Recent work models the radar's
> anisotropic spherical-coordinate noise explicitly and de-biases the coordinate
> transform in an uncertainty-aware PnP [Cao 2025]. Targetless alternatives recover
> the extrinsic from motion — aligning object trajectories [Peršić 2020; Zhang 2025]
> or matching radar ego-velocity to camera ego-motion in a continuous-time
> formulation that also resolves the time offset [Wise 2021/2023] — and end-to-end
> networks regress the extrinsic directly [RC-AutoCalib 2025]. Our method is
> target-based but replaces the checkerboard-centre-plus-depth cue with a full
> metric ChArUco board pose, performs maximum-likelihood estimation in the radar's
> measurement space, and additionally solves the reflector apex offset via a MAP
> formulation, reporting per-DOF observability throughout.

---

## 5. References

**Target-based**
1. J. Peršić, I. Marković, I. Petrović. "Extrinsic 6DoF calibration of a
   radar–LiDAR–camera system enhanced by radar cross section estimates evaluation."
   *Robotics and Autonomous Systems*, 2019.
   https://www.researchgate.net/publication/329446130
2. J. Domhof, J. F. P. Kooij, D. M. Gavrila. "An Extrinsic Calibration Tool for
   Radar, Camera and Lidar." *ICRA* 2019; extended "A Joint Extrinsic Calibration
   Tool for Radar, Camera and Lidar," *IEEE T-IV* 2021 (TU Delft, open-source).
   https://research.tudelft.nl/en/publications/a-joint-extrinsic-calibration-tool-for-radar-camera-and-lidar/
3. X. Cheng et al. "3D Radar and Camera Co-Calibration: A Flexible and Accurate
   Method for Target-based Extrinsic Calibration." 2023.
   https://www.scribd.com/document/841325213/
4. C. Cao, X. Wang, W. Xi, H. Zhang, W. Chen, J. Wang. "A 4D Radar Camera Extrinsic
   Calibration Tool Based on 3D Uncertainty Perspective N Points (3DUPnP)."
   arXiv:2507.19829, 2025. https://arxiv.org/abs/2507.19829

**Targetless / motion-based**
5. E. Wise, J. Peršić, C. Grebe, I. Petrović, J. Kelly. "A Continuous-Time Approach
   for 3D Radar-to-Camera Extrinsic Calibration." arXiv:2103.07505, ICRA 2021.
   https://arxiv.org/pdf/2103.07505 (extended AIM 2023 with ego-velocity).
6. J. Peršić et al. "Online multi-sensor calibration based on moving object
   tracking." *Advanced Robotics*, 2020.
   https://www.researchgate.net/publication/345092954
7. X. Cheng, Y. Cao. "Online Targetless Radar-Camera Extrinsic Calibration Based on
   the Common Features of Radar and Camera." *NAECON* 2023. arXiv:2309.00787.
   https://arxiv.org/abs/2309.00787
8. "Targetless Radar–Camera Calibration via Trajectory Alignment." 2025.
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12737237/

**Learning-based**
9. "RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network."
   arXiv:2505.22427, 2025. https://arxiv.org/pdf/2505.22427
10. "CLRNet: Targetless Extrinsic Calibration for Camera, Lidar and 4D Radar Using
    Deep Learning." arXiv:2603.15767. https://arxiv.org/html/2603.15767

**Surveys / context**
11. "Radar-Camera Fusion for Object Detection and Semantic Segmentation in
    Autonomous Driving: A Comprehensive Review." arXiv:2304.10410, 2023.
    https://arxiv.org/pdf/2304.10410
12. "4D Millimeter-Wave Radar in Autonomous Driving: A Survey." arXiv:2306.04242.
    https://arxiv.org/pdf/2306.04242
13. "CalTag: Robust calibration of mmWave Radar and LiDAR using backscatter tags."
    arXiv:2408.16867 (multi-tag idea, relevant to §3.3).
