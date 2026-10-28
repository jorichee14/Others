# Appearance of the Gaussian map: plan

Gaussian-LIC2 and LoD-3DGS both render the room worse than we want. This is the
plan to fix that **without loosening the geometry**, because the geometry is what
the twin is for: Sionna RT meshes, collaborative-perception ground truth, and the
few-centimetre check in [PLAN.md](PLAN.md) phase 1.

Rule that orders everything: **measure the problem before buying a prior.** Four
different causes look identical in a screenshot, and only one of them is helped by
generation. The cheap non-generative fixes come first; each phase can end the plan.

```
0 measure ─► 1 cheap fixes ─► 2 provenance ─► 3 2D repair ─► 4 distillation ─► 5 RF consequence
     │             │                              (figures)      (the research bit)
     │             └─ gap closed? stop here ◄──────────┘
     └─ which of the four causes? ─────────────────────────────────────────────────
```

## The four causes, and what each one needs

| # | cause | symptom | fixed by |
|---|---|---|---|
| 1 | no Gaussian seeds where LiDAR has no return (glass, dark surfaces, beyond range) | blur / holes in specific materials | LIC2's zero-shot depth already; then generation |
| 2 | real-time optimization budget per keyframe | overall blur, primitives never split or pruned | **more iterations offline** (phase 1) |
| 3 | single-trajectory viewpoint coverage | looks fine along the drive, wispy off it | generation (phase 4) |
| 4 | exposure / white-balance drift, motion blur | brightness jumps between frames, smeared texture | appearance embeddings (phase 1) |

LoD-3DGS's blur is a **fifth, separate** thing: coarse levels are aggregates, so they
are blurry by construction, and levels pop at transitions. That is anti-aliasing and
level blending (Mip-Splatting-style 3D smoothing filter), not generation. Keep it out
of this plan except as phase 0's control.

## 0. Measure it: the fixed cameras are the test set

The 16 Basler cameras are at calibrated poses metres away from any robot path. So:
**train the Gaussian map on robot data only, and evaluate the renders against the
real Basler images at their calibrated poses.** That is a true off-trajectory
novel-view benchmark with real ground truth, which nobody with only a handheld
capture can build.

| step | do | output |
|---|---|---|
| 0.1 | Needs `configs/cameras.yaml` from PLAN phase 1 (calibrated extrinsics in the room frame) and the clock from phase 2, so robot frames and Basler frames can be paired in time. | prerequisite |
| 0.2 | Record one session: robot drives ED305, all 16 cameras recording, PTP timestamps. | paired set |
| 0.3 | Build the map from robot data only. Render at each Basler pose at matching timestamps. | renders vs real |
| 0.4 | Report PSNR / SSIM / LPIPS **split three ways**: robot held-out frames (on-trajectory), Basler cameras (off-trajectory), and Basler cameras masked to regions with no LiDAR return. | the gap table |
| 0.5 | Attribute: re-run offline with 10x iterations (cause 2); correlate error with per-frame exposure (cause 4); compare LoD finest vs coarse level (cause 5). | per-cause attribution |

The **on- minus off-trajectory gap** is the number this whole plan is trying to
close. Cause 3 is the only one that shows up as a large gap with everything else
controlled.

Done when: a table attributing the appearance error to causes 1-5, with the
off-trajectory gap quantified in LPIPS.

## 1. The cheap fixes, no generation

We work offline for v1 (live is digital twin v2), so the real-time budget that
constrains Gaussian-LIC2 does not bind here. Take the poses and the LiDAR-seeded
map from LIC2, then refine offline with no iteration cap.

| step | do | expect |
|---|---|---|
| 1.1 | Offline refinement pass on the finished map: full iteration budget, densification and pruning enabled. | most of cause 2 |
| 1.2 | Per-keyframe appearance embedding (latent + small MLP, NeRF-W / VastGaussian style) so exposure changes stop being baked into the Gaussians. | cause 4 |
| 1.3 | Mip-Splatting 3D smoothing filter. | scale aliasing, and cause 5 for LoD |
| 1.4 | Re-run the phase-0 table. | how much gap is left |

Done when: the phase-0 table is re-run. **If the off-trajectory gap is acceptable,
the plan stops here** and phases 3-5 are not built.

## 2. Provenance, before any generation touches the map

Every Gaussian carries where it came from. This is the rail that makes generation
safe to add, so it comes first.

| value | means |
|---|---|
| `observed` | LiDAR return, or triangulated from two or more real views |
| `interpolated` | LIC2's zero-shot depth model filling a LiDAR gap |
| `generated` | created or moved under a diffusion prior (phase 4) |

LIC2 already distinguishes LiDAR depth from predicted depth when it seeds
Gaussians; the work is carrying that label through the optimizer instead of
discarding it, and keeping it across splits and clones.

| step | do |
|---|---|
| 2.1 | Add the channel, set it at seeding, propagate through densification. |
| 2.2 | Render masked by provenance (show only measured; highlight invented). |
| 2.3 | Mesh / Sionna export filters or flags non-`observed` surfaces. |

Done when: a render of the map showing measured and invented regions in different
colours, and an export that can exclude the invented ones.

## 3. Post-render 2D repair, for figures only

A single-step diffusion model cleans the rendered image after rasterization
(Difix3D+, GSFix3D, GSFixer). It never touches the Gaussians, so the geometry
cannot be harmed.

| step | do |
|---|---|
| 3.1 | One checkpoint, run over the phase-0 off-trajectory renders. |
| 3.2 | Side-by-side in the phase-0 table. Report it separately: it is a 2D post-process, not a better map. |

Use for slides, figures and demos. **Never** as input to anything metric, to
labelling, or to Sionna. Known failure: sharper output with ghosting, and
unreliable far from observed views.

Done when: the comparison exists and is labelled as post-processing.

## 4. Distillation with an asymmetric gradient mask

The research contribution. Render virtual views off the real trajectory, repair
them with the diffusion model, feed them back as extra supervision — but let that
supervision reach **appearance parameters only**.

| parameter group | supervised by |
|---|---|
| position, scale, rotation, opacity | LiDAR and the measured photometric loss only |
| SH coefficients, appearance embedding | diffusion-repaired pseudo-views allowed |

Implementation is a gradient mask on optimizer parameter groups, plus a pseudo-view
sampler and a confidence weight per pseudo-pixel (down-weight far from any real
view, and where there is no LiDAR support).

| step | do |
|---|---|
| 4.1 | Pseudo-view sampler: virtual poses offset from the real path in height and laterally, toward the Basler poses but **not at them** (those are the test set). |
| 4.2 | Gradient mask on parameter groups; confidence weighting on the pseudo-view loss. |
| 4.3 | Ablate: (a) no diffusion, (b) diffusion into all parameters, (c) diffusion into appearance only. |
| 4.4 | For each: appearance on the phase-0 off-trajectory set, **and** geometry error — point-to-plane against the LiDAR map, plus the PLAN D2 held-out triangulated point. |

The expected and publishable result: (b) improves appearance and degrades geometry,
(c) improves appearance and holds geometry inside the few-centimetre bound. If (c)
does not hold geometry, the mask is not enough and the honest answer is that
generation stays at phase 3.

Done when: that ablation table exists, with geometry error reported for every row.

## 5. What it costs the radio

Appearance repair that invents geometry gives a confident, wrong channel. Quantify
it rather than asserting it.

| step | do |
|---|---|
| 5.1 | Export two meshes from the same map: all surfaces, and `observed` only. |
| 5.2 | Sionna RT on both: CIR per link and a radio map, same Tx/Rx poses. |
| 5.3 | Report the difference (path loss, delay spread, strongest-path geometry). |

Done when: a number for how much the invented surfaces would have changed the
channel prediction. This is what justifies the provenance channel to a reviewer,
and it is the concrete form of our edge over single-capture world models.

## Later, separately

Feed-forward generative reconstruction (pixelSplat / MVSplat / DepthSplat / GS-LRM
/ VGGT-family backbones) replaces per-scene optimization with a network that
predicts Gaussians. It is the Echo-3 direction, it is a much larger bet, and by
default it discards the LiDAR metric authority that is our advantage. A
LiDAR-conditioned feed-forward reconstructor would be a real contribution, but it
is not v1 and it is not this plan.

## Not in scope

LoD level-transition popping beyond phase 1.3, relighting, dynamic objects in the
Gaussian map (people and robots are tracked, not reconstructed), and anything that
needs the live uplink of [DATAFLOW.md](DATAFLOW.md).

## Gates

| after | stop if |
|---|---|
| 0 | the gap is dominated by causes 2, 4 or 5 — then phase 1 is the whole job |
| 1 | the off-trajectory gap is acceptable |
| 4.4 | masked distillation cannot hold geometry; generation stays at phase 3 |
