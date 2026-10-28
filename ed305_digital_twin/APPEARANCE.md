# Appearance of the Gaussian map: plan

Gaussian-LIC2 and LetsGo (LOD-LiDAR-GS) both render worse than we want. This is the
plan to fix that **without loosening the geometry**, because the geometry is what the
twin is for: Sionna RT meshes, collaborative-perception ground truth, and the
few-centimetre check in [PLAN.md](PLAN.md) phase 1.

Rule that orders everything: **measure the problem before buying a prior.** Several
causes look identical in a screenshot and only one is helped by generation. The cheap
non-generative fixes come first; each phase can end the plan.

```
0 measure ─► 1 cheap fixes ─► 2 provenance ─► 3 2D repair ─► 4 distillation ─► 5 RF consequence
     │             │                              (figures)      (the research bit)
     │             └─ gap closed? stop here ◄──────────┘
     └─ which cause? ──────────────────────────────────────────────────────────────
```

Setup assumed here: one camera, LiDAR, IMU. Not the 16 ED305 Basler cameras.

## The systems

- **Gaussian-LIC2** ([arXiv 2507.04004](https://arxiv.org/abs/2507.04004)):
  continuous-time LiDAR-inertial-camera SLAM, incremental, real-time, fills LiDAR
  gaps with a zero-shot depth model blended against sparse LiDAR depth.
- **LetsGo / LOD-LiDAR-GS** ([arXiv 2404.09748](https://arxiv.org/abs/2404.09748),
  ACM TOG 43(6)): Polar rig (fisheye + LiDAR + IMU) → LIV-SLAM poses → Poisson mesh →
  points resampled from the mesh → multi-resolution Gaussians, one **independent**
  Gaussian model per level, LOD chosen at render time by projected depth.

Both already suppress floaters — LIC2 by construction, LetsGo with its unbiased depth
regularizer at `λ_depth = 0.8`. So the remaining complaint is more likely **blur than
wisps**, and phase 0 exists to establish which before anything generative is bought.

## The causes

| # | cause | symptom | fixed by |
|---|---|---|---|
| 1 | init points come from a **Poisson mesh**, which the paper says fails on reflective and transparent surfaces | wrong geometry exactly on glass and polished floors, and `λ_depth = 0.8` then *pulls Gaussians to the wrong depth there* | confidence-weighted depth regularizer (phase 1) |
| 2 | optimization budget (LIC2's real-time cap) | overall blur, primitives never split or pruned | more iterations offline (phase 1) |
| 3 | single-trajectory viewpoint coverage | fine along the path, wispy off it | generation (phase 4) |
| 4 | **auto exposure and auto ISO**, by design in the capture protocol | brightness jumps between frames, baked into the Gaussians | per-image appearance embeddings (phase 1) |
| 5 | LoD: independent per-level models, level chosen by projected depth | blur at distance, popping at transitions | level selection and inter-level consistency, **not** generation |
| 6 | **fisheye, 180°×180°**, against a pinhole rasterizer | blur and warping worsening toward the image edge | correct projection (phase 1) |
| 7 | `SH degree 2`, opacity reset disabled, densification delayed to 75k | weak view-dependent effects; floaters that never get reset away | three config flags (phase 1) |

Causes 5, 6 and 7 are not generation problems. Cause 6 is a prime suspect with a
single 180° fisheye and nothing generative will fix it.

## 0. Measure it: the four-pass capture is already the test set

LetsGo's scanning protocol collects each scene **four times — forward-facing,
backward-facing, side-left and side-right**. That is a free off-trajectory benchmark
with real ground truth: **train on the forward and backward passes, evaluate on the
side-left and side-right passes.** The side passes see the same surfaces from genuinely
different angles, which is exactly the viewpoint shift that exposes cause 3.

**Random every-Nth frame holdout is not a substitute.** At 30 FPS and ~1 m/s,
neighbouring frames are nearly identical, so it measures interpolation and will report
that everything is fine.

| step | do | gives |
|---|---|---|
| 0.1 | Train on forward + backward. Evaluate on side-left + side-right. | the true off-trajectory set |
| 0.2 | Also hold out a **contiguous segment** of a training pass, not scattered frames. | a mild in-distribution control |
| 0.3 | Report PSNR / SSIM / LPIPS for each, and again masked to regions with no direct LiDAR return. | the gap table |
| 0.4 | **Pin the LOD level** before measuring anything (see below). | otherwise you measure level selection |
| 0.5 | Attribute the rest: re-run with 10x iterations (cause 2); correlate error with per-frame exposure (cause 4); compare finest vs coarse level (cause 5); plot error against radial distance from the fisheye centre (cause 6); check error against block boundaries (30% expansion overlap). | per-cause attribution |

Geometry ground truth needs no second camera: the LiDAR map gives point-to-plane error
with one camera exactly as well as with sixteen, at the rig's stated 1–1.5 cm accuracy.
So the phase-4 check that geometry did not move is unaffected by the single-camera
setup — only the appearance test set is weaker.

**Pin the LOD level first.** Selection is `L(d) = clamp(⌊L^(1−d/d_max)⌋, 0, L−1)` with
`d_max` taken from the maximum depth projected onto the training viewpoints. `d_max` is
therefore scene-dependent and inherited from the garage-scale captures; at different
scene scale the level chosen for a given viewing distance changes completely. If a
forced finest level looks right, the "appearance problem" is the selector and the fix
is one constant, not a research project.

Done when: a table attributing the error to causes 1-7 at a pinned LOD level, with the
off-trajectory gap quantified in LPIPS.

## 1. The cheap fixes, no generation

Offline for v1, so the real-time budget that constrains LIC2 does not bind. Ordered by
cost; several are single flags.

| step | do | cause |
|---|---|---|
| 1.1 | `SH degree 2 → 3`. Garage floors and vehicle glass are specular; degree 2 cannot represent that, and the optimizer pays for it with blur. | 7 |
| 1.2 | Re-enable opacity reset. It is the standard 3DGS mechanism for killing floaters and LetsGo disables it, on the assumption that LiDAR init plus the depth term make it unnecessary. One flag, one run. | 7 |
| 1.3 | **Confidence-weight the depth regularizer.** `λ_depth = 0.8` is high, and the Poisson mesh it supervises against is wrong on glass and reflective surfaces by the paper's own admission. Down-weight it per-pixel where LiDAR return is sparse, where returns disagree across passes, or where the mesh was interpolated rather than measured. | 1 |
| 1.4 | Per-image appearance embedding (latent + small MLP, NeRF-W / VastGaussian style). The capture protocol uses auto exposure and auto ISO deliberately, so cross-frame appearance variation is guaranteed, not incidental. | 4 |
| 1.5 | Fisheye: either a projection model the rasterizer handles natively, or undistort and accept the peripheral resolution loss — but measure which, from the 0.5 radial error plot. | 6 |
| 1.6 | Offline refinement pass with no iteration cap, densification and pruning on. | 2 |
| 1.7 | Mip-Splatting 3D smoothing filter. | 5 |
| 1.8 | Re-run the phase-0 table. | — |

Done when: the phase-0 table is re-run. **If the off-trajectory gap is acceptable, the
plan stops here** and phases 3-5 are not built. On present evidence 1.1-1.4 are the
most likely to account for most of it.

## 2. Provenance, before any generation touches the map

Every Gaussian carries where it came from. This is the rail that makes generation safe
to add, so it comes first — and note that **LetsGo's map is already partly inferred
before any diffusion model is involved**, because its init points are resampled from a
Poisson reconstruction, not from raw returns.

| value | means |
|---|---|
| `lidar` | a direct LiDAR return supports this region |
| `mesh` | Poisson-interpolated, or LIC2's zero-shot depth filling a LiDAR gap |
| `generated` | created or moved under a diffusion prior (phase 4) |

| step | do |
|---|---|
| 2.1 | Set at initialization from the point's origin; propagate through cloning and splitting. |
| 2.2 | Render masked by provenance (measured only; invented highlighted). |
| 2.3 | Mesh / Sionna export filters or flags everything that is not `lidar`. |

**LoD aggregation rule.** Each level is its own Gaussian model seeded from its own
downsampled point cloud, so provenance is set per level at init and needs no
child-to-parent rule — but a coarse level must not be allowed to report `lidar` for a
region whose finer levels are `mesh` or `generated`. Combine pessimistically when
comparing levels.

Done when: a render showing measured and inferred regions in different colours, an
export that can exclude the inferred ones, and the rule holding at every level.

## 3. Post-render 2D repair, for figures only

A single-step diffusion model cleans the rendered image after rasterization
(Difix3D+, [GSFix3D](https://gsfix3d.github.io/), [GSFixer](https://arxiv.org/pdf/2508.09667)).
It never touches the Gaussians, so geometry cannot be harmed.

| step | do |
|---|---|
| 3.1 | One checkpoint, over the phase-0 off-trajectory renders. |
| 3.2 | Side-by-side in the phase-0 table, reported separately: it is a 2D post-process, not a better map. |

Slides, figures and demos. **Never** as input to anything metric, to labelling, or to
Sionna. Known failure: sharper output with ghosting, unreliable far from observed views.

## 4. Distillation with an asymmetric gradient mask

The research contribution. Render virtual views off the real trajectory, repair them
with the diffusion model, feed them back as extra supervision — but let that
supervision reach **appearance parameters only**.

| parameter group | supervised by |
|---|---|
| position, scale, rotation, opacity | LiDAR, the depth regularizer, and the measured photometric loss |
| SH coefficients, appearance embedding | diffusion-repaired pseudo-views allowed |

**Both systems keep explicit per-Gaussian parameters** — LetsGo stores position, SH and
opacity per primitive alongside its LOD attribute, with no shared latent feature and no
MLP decoder — so this is a gradient mask on optimizer parameter groups, a few lines in
the optimizer setup. (Had LetsGo been anchor-based with MLP-decoded attributes, as
Scaffold-GS and Octree-GS are, the mask would be impossible without first splitting the
anchor feature into geometry and appearance halves with a `stop_grad` between them.
Worth knowing if the LoD backbone is ever swapped.)

**What LoD does change is inter-level consistency.** Each resolution level "operates
independently during optimization", so a pseudo-view loss applied at one level does not
reach the others, and the levels drift apart in appearance — which shows up as *worse*
popping at transitions, i.e. making cause 5 worse while fixing cause 3.

| step | do |
|---|---|
| 4.1 | Pseudo-view sampler: virtual poses offset from the training passes in height and laterally, toward the side-pass poses but **not at them** (those are the test set). |
| 4.2 | Distil under the existing random-resolution-level scheme so every level is supervised, not only the finest. Add an inter-level appearance consistency term: a coarse level's colour should match the correct downsample of the fine level's. |
| 4.3 | Gradient mask on parameter groups; confidence weighting on the pseudo-view loss (down-weight far from any real view and where there is no LiDAR support). |
| 4.4 | Ablate: (a) no diffusion, (b) diffusion into all parameters, (c) diffusion into appearance only. |
| 4.5 | For each: appearance on the phase-0 off-trajectory set, geometry error point-to-plane against the LiDAR map, **and** level-transition popping, so 4.2 is shown to have worked. |

Expected and publishable: (b) improves appearance and degrades geometry, (c) improves
appearance and holds geometry inside the LiDAR's own 1–1.5 cm. If (c) does not hold
geometry, the mask is not enough and the honest answer is that generation stays at
phase 3.

Done when: that ablation table exists, with geometry error and popping on every row.

## 5. What it costs the radio

Appearance repair that invents geometry gives a confident, wrong channel. Quantify it
rather than asserting it.

| step | do |
|---|---|
| 5.1 | Export two meshes from the same map: all surfaces, and `lidar` only. |
| 5.2 | Sionna RT on both: CIR per link and a radio map, same Tx/Rx poses. |
| 5.3 | Report the difference (path loss, delay spread, strongest-path geometry). |

Done when: a number for how much the inferred surfaces would have changed the channel
prediction. This is what justifies the provenance channel to a reviewer, and it is the
concrete form of our edge over single-capture world models.

## Later, separately

Feed-forward generative reconstruction (pixelSplat / MVSplat / DepthSplat / GS-LRM /
VGGT-family backbones) replaces per-scene optimization with a network that predicts
Gaussians. It is the Echo-3 direction, a much larger bet, and by default it discards
the LiDAR metric authority that is our advantage. A LiDAR-conditioned feed-forward
reconstructor would be a real contribution, but it is not v1 and not this plan.

## Not in scope

Relighting, dynamic objects in the Gaussian map (people and robots are tracked, not
reconstructed), block-boundary seams beyond measuring them in 0.5, and anything that
needs the live uplink of [DATAFLOW.md](DATAFLOW.md).

## Gates

| after | stop if |
|---|---|
| 0 | the gap is dominated by causes 1, 2, 4, 5, 6 or 7 — then phase 1 is the whole job |
| 1 | the off-trajectory gap is acceptable |
| 4.5 | masked distillation cannot hold geometry; generation stays at phase 3 |
