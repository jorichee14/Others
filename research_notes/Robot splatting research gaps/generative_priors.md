# Generative priors and diffusion models for filling unobserved regions in Gaussian splatting / radiance fields, and separating measured from generated content

Scope: 2023 to Oct 2026. Context: ground robots mapping a ~6000 m2 building with LiDAR. A low, forward-facing camera leaves about 20% of surfaces (mostly ceiling) barely observed. The only GPU is one RTX 3080 (12 GB).

**Method and evidence-level caveat (read first).** In this session arxiv.org, alphaxiv.org, research.nvidia.com and radiancefields.com could not be fetched: the fetches failed with a DNS error because those hosts are outside the session's allowed network domains. As a result, **almost every finding below comes from search-engine snippets or summaries of abstracts and project pages, not from full-text reading.** Items marked **[snippet]** were seen only that way. Items marked **[background, not re-verified]** come from the researcher's prior knowledge of well-known papers and were not confirmed this session. The report writer should treat specific numbers (PSNR, FID, timings) as unverified unless they are cross-checked. To read the full texts, the session's network settings must allow arxiv.org, alphaxiv.org, research.nvidia.com and radiancefields.com.

## Diffusion "fixers" and artifact repair (Difix3D+, GSFix3D, GSFixer, ConFixGS, 3DGS-Enhancer and similar)

### Takeaway
"Fixers" are image or video diffusion models fine-tuned to turn degraded 3DGS/NeRF renders into clean images. The fixed novel views are then distilled back into the 3D model as pseudo-ground-truth with a photometric loss, usually progressively. They repair floaters, blur and holes in under-constrained regions, but they are view-space, appearance-only priors with no notion of metric geometry. Single-step image fixers such as Difix (an SD-Turbo-class model) are plausible on a 12 GB card. Video-DiT fixers such as GSFixer (CogVideoX) are probably not.

### Cited Findings
- **Difix3D+ (Wu et al., NVIDIA, CVPR 2025 oral, Best Paper Award candidate; arXiv 2503.01774).** Difix is a single-step image diffusion model fine-tuned from Stable Diffusion. It removes artifacts from rendered novel views in regions the 3D representation leaves under-constrained. It has two roles: (1) during reconstruction it cleans pseudo-training views that are then distilled back into 3D; (2) at inference it acts as a real-time neural post-render enhancer. Reported result: an average 2x improvement in FID over baselines while staying 3D-consistent. It works with both NeRF and 3DGS. **[snippet]** — [NVIDIA project page](https://research.nvidia.com/labs/toronto-ai/publication/2025_cvpr_difix3d); [CVPR open access](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_DIFIX3D_Improving_3D_Reconstructions_with_Single-Step_Diffusion_Models_CVPR_2025_paper.html); [CVPR oral page](https://cvpr.thecvf.com/virtual/2025/oral/35364)
- Difix model card: "not guaranteed to fix 100% of the image artifacts". Use is governed by the NVIDIA License; a third-party mirror describes it as research and non-commercial only. NVIDIA notes that a commercially available successor, "Fixer", has been released. **[snippet]** — [Hugging Face nvidia/difix](https://huggingface.co/nvidia/difix); [third-party mirror](https://app.alphaneural.io/models/z-x-j/difix)
- Difix3D+ mechanism details: conditioning on a reference (training) view; a "progressive 3D update" that renders views gradually farther from the training cameras, fixes them and adds them to training; single-step inference fast enough for real-time post-processing (the researcher recalls about 76 ms per image on an A100). **[background, not re-verified]** — [arXiv 2503.01774](https://arxiv.org/abs/2503.01774)
- **GSFix3D (arXiv 2508.14717, Aug 2025).** It fine-tunes a latent diffusion model ("GSFixer" inside the GSFix3D pipeline, not to be confused with the separate GSFixer paper below) to repair novel views of 3DGS reconstructions. Random-mask augmentation teaches it to inpaint missing regions. Repaired images are fed back into 3DGS with a photometric loss. It is compared against DIFIX and RTG-SLAM on **ScanNet++ and Replica (indoor)**. Code is public, but the repo was nearly empty when the search snippet was taken. **[snippet]** — [arXiv PDF](https://arxiv.org/pdf/2508.14717); [GitHub](https://github.com/GSFix3D/GSFix3D); [papers.cool](https://papers.cool/arxiv/2508.14717)
- **GSFixer (Yin et al., arXiv 2508.09667; ICML 2026 poster).** A reference-guided video restoration model built on a DiT video diffusion model (CogVideoX). It conditions on DINOv2 2D semantic features and VGGT 3D geometric features from the input reference views to restore novel views rendered from low-quality sparse-view 3DGS. The restored frames are distilled back into 3DGS. It adds a reference-guided trajectory sampling strategy and introduces the DL3DV-Res benchmark. It claims state of the art on 3DGS artifact restoration and sparse-view reconstruction. **[snippet]** — [arXiv PDF](https://arxiv.org/pdf/2508.09667); [ICML 2026](https://icml.cc/virtual/2026/poster/63372); [HF README](https://huggingface.co/flow666/GSFixer/blob/main/README.md)
- **ConFixGS (arXiv 2605.09688, May 2026).** It fixes *feed-forward* 3DGS in driving scenes (Waymo, nuScenes, KITTI). It generates diffusion-enhanced pseudo-targets for novel views, cross-checks them against support views by reprojection, and uses the resulting confidence maps to weight refinement, which suppresses inconsistent hallucinations. Reported gain: up to +3.68 dB PSNR. **[snippet]** — [arXiv PDF](https://arxiv.org/pdf/2605.09688)
- **3DGS-Enhancer (Liu, Zhou, Huang; NeurIPS 2024; arXiv 2410.16266).** It recasts 3D-consistent image restoration as temporally consistent video generation, using a video diffusion prior to restore novel-view features that then fine-tune the initial 3DGS. It targets unbounded scenes with sparse input views. **[snippet]** — [arXiv HTML](https://arxiv.org/html/2410.16266v1); [NeurIPS](https://neurips.cc/virtual/2024/poster/95333)
- Related fixers seen only in titles: FixingGS (training-free score distillation for 3DGS enhancement, arXiv 2509.18759). **[snippet: title only]** — [arXiv HTML](https://arxiv.org/html/2509.18759v1)

### Inferences
- **Distillation pattern common to all of them:** render a novel view from the current 3DGS, fix it with a diffusion model, add it as a pseudo-target with a photometric (sometimes confidence-weighted) loss, and repeat, moving progressively away from the training cameras. None of the snippets says that any fixer uses metric depth or LiDAR as a hard constraint during distillation, so the fixed appearance can pull geometry wherever it is photometrically convenient.
- **GPU (inference, not sourced):**
  - Difix is a single-step SD-derived model, so it should run on 12 GB at moderate resolution.
  - GSFixer (CogVideoX DiT), 3DGS-Enhancer (video diffusion) and GEN3C (Cosmos-7B) are very likely beyond 12 GB without heavy offloading or quantization.
  - No paper's VRAM figures were retrieved (see Gaps).
- **Indoor relevance:** GSFix3D (ScanNet++/Replica) is the closest of the fixers to indoor robot captures. ConFixGS's reprojection-confidence idea transfers most directly to "keep generated content consistent with measured views".

### Gaps
- VRAM and GPU-hour figures for Difix3D+, GSFix3D, GSFixer and 3DGS-Enhancer were not retrieved, because full texts were blocked.
- No source found that tests any fixer on ceilings or on building-scale robot trajectories.
- Exact Difix3D+ PSNR/LPIPS tables and its limitations section were not read.

## Multi-view and video generative priors for sparse or incomplete captures (ReconFusion, CAT3D, ViewCrafter, GEN3C, ReconX, NVS-Solver, 2025-2026 work), and which handle indoor building-scale trajectories

### Takeaway
The field moved from per-scene 2D diffusion distillation (ReconFusion, about 1 hour per scene) to multi-view diffusion (CAT3D, minutes) to 3D-conditioned video diffusion (ViewCrafter, ReconX, GEN3C), in which a point cloud or 3D cache rendered along the camera path conditions the generator. Geometry-guided indoor variants exist (G4Splat with planar priors, MVRoom with layouts, Rein3D with panoramas), but **every method found is evaluated at room scale or on short clips**. None was found to handle a multi-room building captured along a long robot trajectory.

### Cited Findings
- **ReconFusion** takes about 1 hour per scene. **CAT3D** (multi-view diffusion) cuts this to a few minutes. CAT3D was evaluated on 3-, 6- and 9-view reconstruction and outperforms baselines by a larger margin on harder datasets (CO3D, mip-NeRF 360). **[snippet]** — [CAT3D arXiv 2405.10314](https://arxiv.org/pdf/2405.10314)
- **ViewCrafter (arXiv 2409.02048)** conditions a video diffusion model on point clouds built from the input images. Another paper reports that with small input overlap ViewCrafter "generates many unreasonable contents or ghosting artifacts". The ReconX supplement adds that when DUSt3R fails (for example, mispredicting planar geometry), ViewCrafter "can be significantly compromised". **[snippet]** — [ViewCrafter](https://arxiv.org/pdf/2409.02048); [ReconX](https://arxiv.org/html/2408.16767v4)
- **ReconX (arXiv 2408.16767).** It builds a global point cloud from sparse views, encodes it as a 3D structure condition, and uses it to guide a video diffusion model. One reported figure: PSNR 24.57, against 18.83 for MVSplat and 17.62 for pixelSplat. **[snippet]** — [ReconX HTML](https://arxiv.org/html/2408.16767v1)
- **GEN3C (NVIDIA, CVPR 2025; arXiv 2503.03751).**
  - Mechanism: it builds a persistent "3D cache" by predicting depth for seed images or earlier generated frames and unprojecting it into point clouds. The cache is rendered along the user's camera path and fed into the video diffusion model, which gives more precise camera control than conditioning on raw camera parameters.
  - Released model: GEN3C-Cosmos-7B. The model card lists Ampere, Hopper and Blackwell GPUs and Linux, describes it as a model for researchers, and warns that outputs "may not be entirely visually accurate". **[snippet]** — [project page](https://research.nvidia.com/labs/toronto-ai/GEN3C); [HF model card](https://huggingface.co/nvidia/GEN3C-Cosmos-7B); [arXiv](https://arxiv.org/pdf/2503.03751)
- **NVS-Solver (ICLR 2025)** is a training-free method that modulates a pretrained video diffusion model's (SVD) score with warped input views. **[background, not re-verified]** — no source retrieved this session.
- **G4Splat (Ni, Chen et al., ICLR 2026; arXiv 2510.12099).**
  - Core idea: accurate geometry is the prerequisite for useful generative completion. It uses planar structure in indoor scenes to derive scale-accurate geometric constraints, which improve visibility masks, novel-view selection and multi-view consistency during video-diffusion inpainting.
  - Evaluation: Replica, ScanNet++ and DeepBlending, plus Mip-NeRF 360 in the OpenReview version. It outperforms baselines, especially in unobserved regions.
  - Ablation: the generative prior alone gives limited geometry gains and blurry unseen areas. It also handles single-view and unposed-video inputs.
  - **[snippet]** — [arXiv](https://arxiv.org/pdf/2510.12099); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007768); [OpenReview PDF](https://openreview.net/pdf/b60a6180eda7d6c25e55daf8272250755abe4e62.pdf)
- **Indoor scene generation and completion (2024-2026), all room-scale:**
  - GenRC (2024) completes rooms from sparse RGB-D via panorama inpainting into a mesh and is training-free.
  - MVRoom (Dec 2025) uses layout-conditioned iterative multi-view diffusion, then fits 3DGS.
  - Rein3D (Apr 2026) produces a 360-degree indoor scene from a single panorama using 3DGS plus video-diffusion priors with global refinement.
  - L3DG (2024) is a latent 3D Gaussian diffusion model that scales to room-scale generation.
  - **[snippet]** — [GenRC](https://cdn.amazon.science/c3/34/df11431a43958dbeb4003512f888/genrc-generative-3d-room-completion-from-sparse-image-collections.pdf); [MVRoom](https://arxiv.org/html/2512.04248v1); [Rein3D](https://www.alphaxiv.org/abs/2604.10578); [L3DG](https://portal.fis.tum.de/de/publications/l3dg-latent-3d-gaussian-diffusion/)
- Other 2025 scalable generators: StarGen (spatiotemporal autoregression with video diffusion for scalable, controllable scene generation) and "You See it, You Got it" (3D creation learned from pose-free videos). **[snippet: titles only]** — [StarGen](https://arxiv.org/pdf/2501.05763); [See3D](https://arxiv.org/pdf/2412.06699)
- **Extrapolation-specific:** ViewExtrapolator refines renders with video diffusion priors and argues LPIPS suits extrapolation better than PSNR "because the task is more generative than regressive". **[snippet]** — [arXiv 2411.14208](https://arxiv.org/pdf/2411.14208)

### Inferences
- Methods conditioned on point clouds or depth caches (ViewCrafter, ReconX, GEN3C, G4Splat) are the natural place to inject robot LiDAR. Their documented failure mode is wrong geometry from DUSt3R or monocular depth, which metric LiDAR would remove.
- **Building scale:**
  - GEN3C's persistent 3D cache is the most promising design for long trajectories, but it has no published building-scale evaluation.
  - G4Splat's planar priors fit ceilings well.
  - For a low forward camera, the ceiling problem is extreme extrapolation (steep upward viewing angles), the regime where ViewCrafter-style methods hallucinate.

### Gaps
- No paper found that evaluates generative completion on a multi-room, building-scale robot trajectory (more than about 1000 m2).
- Compute numbers for CAT3D, ViewCrafter and ReconX were not retrieved.
- NVS-Solver details were not verified.

## World models that turn captures into 3D worlds (SpAItial Echo-3, World Labs Marble, others) and feed-forward large reconstruction models

### Takeaway
Commercial "world models" (SpAItial Echo-3, World Labs Marble) now accept real walkthrough captures and output Gaussian splats, blending capture with learned priors. **No public technical report was found for either**, so it is unknown how much of the output is measured and how much is generated, or whether provenance is exposed. Feed-forward reconstructors (AnySplat, VGGT-based) give fast geometry but interpolate within the views they are given and do not complete unseen regions.

### Cited Findings
- **SpAItial Echo-3 (launched 27 Oct 2026).**
  - What it does: a capture-based product, separate from the text/image Echo line. It turns iPhone walkthrough video into explorable, **metric-scale** 3D worlds and "combines the capture with what it has learned about how real spaces look".
  - Workflow: an iPhone companion app uploads scans for cloud processing. Enterprise API access is offered.
  - Exports: SPZ, SOG, meshes, normals and depth. Launch materials do not say which exports each tier offers.
  - Benchmarks are vendor-reported; commentary advises treating them as a starting point pending independent evaluation.
  - **[snippet]** — [Radiance Fields](https://radiancefields.com/spaitial-echo-3-turns-iphone-scans-into-3d-worlds); [Runtime Wire](https://runtimewire.com/article/spaitial-echo-3-iphone-3d-world-capture)
- **SpAItial Echo-2 (Apr 2026)** uses 3DGS as its rendering primitive and predicts semantic segmentation masks (walls, floors, chairs, tables) for object-level edits. The original Echo generated a 3D-consistent world from text or an image. **[snippet]** — [Radiance Fields Echo-2](https://radiancefields.com/spaitial-ai-announces-new-model-echo-2); [Auganix](https://www.auganix.org/?p=18368)
- **World Labs Marble.**
  - Inputs: text, images or a short walkthrough video, with one primary input type per run.
  - Output: .spz Gaussian splats of about 2M splats at full resolution, or a lighter ~500k version for real-time playback.
  - Reported limits: better for interiors and sets than for exteriors; environments, not animations.
  - World Labs has raised a reported $1B round. Sources are secondary blogs.
  - **[snippet]** — [Scenario help](https://help.scenario.com/articles/4774055514-marble-by-world-labs-the-essentials); [BDTechTalks](https://bdtechtalks.substack.com/p/what-to-know-about-world-labs-marble); [Radiance Fields launch](https://radiancefields.com/world-labs-formally-launches-marble-a-generative-world-model); [World Labs blog "bigger better worlds"](https://www.worldlabs.ai/blog/bigger-better-worlds)
- **AnySplat (arXiv 2505.23716).** It is a feed-forward model: a geometry transformer encodes unconstrained, unposed images and decodes them into Gaussians and camera poses, with zero-shot evaluation on ARKitScenes, ScanNet++ and BlendedMVS. An optional post-optimization step helps with dense inputs. **[snippet]** — [arXiv HTML](https://arxiv.org/html/2505.23716v2); [HF papers](https://huggingface.co/papers/2505.23716.md)
- The AnchorSplat authors (2026) argue that most feed-forward methods "only interpolate within limited view ranges", which restricts large-scale reconstruction and extrapolation to unseen viewpoints. **[snippet]** — [AnchorSplat](https://www.alphaxiv.org/abs/2604.07053.md)
- ConFixGS (above) explicitly targets the artifacts of feed-forward 3DGS, which shows that feed-forward output still needs a generative fixer. **[snippet]** — [arXiv](https://arxiv.org/pdf/2605.09688)

### Inferences
- Echo-3 is the closest commercial analogue to "capture plus generative completion at metric scale" for indoor spaces. It is a cloud service, not a local model, and nothing found says whether it marks captured versus generated regions or accepts LiDAR input. iPhone LiDAR may be used, but that is unconfirmed.
- Building scale for Echo-3 and Marble is unknown. Marble's ~2M-splat cap suggests room or apartment scale, not 6000 m2 (inference).

### Gaps
- No technical report, method description or failure analysis was found for Echo-3 or Marble.
- No Google DeepMind (e.g., Genie-3) capture-to-3D information was retrieved this session.
- AnySplat's own limitations section was not read.

## Uncertainty and provenance: observed vs hallucinated content, generation constrained by metric LiDAR, keeping generated content off measured surfaces

### Takeaway
Two lines are converging but not yet joined:
- **Classical radiance-field uncertainty** (FisherRF, Bayes' Rays, PH-Dropout, photometric-uncertainty 3DGS) estimates where the model is under-constrained.
- **2025-2026 hallucination-aware generative reconstruction** (HAD at CVPR 2026, OracleGS at WACV 2026, ConFixGS, GA-GS) masks or down-weights unreliable generated pixels or primitives during training.

LiDAR enters only as per-Gaussian geometric confidence (GeomGS) or as outdoor LiDAR-scan generation (LiDAR-GS++). No work was found that uses metric LiDAR as a hard constraint on generated content **and** exports an explicit per-surface observed/generated label.

### Cited Findings
- **FisherRF (Jiang, Lei, Daniilidis; arXiv 2311.17874; ECCV 2024).** Fisher information over radiance-field parameters supports active view selection and pixel-wise uncertainty. The Fisher matrices are highly sparse, so computing the trace costs about as much as back-propagation. View selection runs at 70 fps with the 3DGS backend. **[snippet]** — [arXiv](https://arxiv.org/abs/2311.17874v1); [NSF PAR](https://par.nsf.gov/biblio/10565454-fisherrf-active-view-selection-mapping-radiance-fields-using-fisher-information)
- **Bayes' Rays (Goli et al., CVPR 2024).** It is described as treating uncertainty as allowable volumetric variation to find under-constrained regions, focused on spatial (depth) uncertainty caused by a lack of views and tightly coupled to neural volumetric representations. The paper itself was not retrieved; the description comes from a 2026 paper's related-work section. **[snippet, secondary]** — [Predictive Photometric Uncertainty in GS, arXiv 2603.22786](https://arxiv.org/pdf/2603.22786)
- **PH-Dropout (arXiv 2410.05468).** "Occlusions and missing views inherently limit the epistemic knowledge" a model can acquire; GS models lack a systematic approach to full epistemic UQ. **[snippet]** — [arXiv](https://arxiv.org/html/2410.05468v1)
- **Modeling uncertainty for Gaussian Splatting** (arXiv 2403.18476). **[snippet: title only]** — [arXiv](https://arxiv.org/html/2403.18476v1)
- **HAD: Hallucination-Aware Diffusion Priors for 3D Reconstruction (Liu et al., CVPR 2026; arXiv 2605.16873).**
  - Problem: diffusion priors add invented content ("aliens") that is not in the observed regions.
  - Mask: a learned branch predicts a per-pixel hallucination score, trained to regress the per-pixel error between the diffusion-augmented image and ground-truth novel observations, then thresholded into a binary mask.
  - Use: masked pixels are excluded from the 3DGS loss on augmented views. It reduces hallucination without retraining the prior and claims state-of-the-art NVS.
  - **[snippet]** — [CVPR 2026 open access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_HAD_Hallucination-Aware_Diffusion_Priors_for_3D_Reconstruction_CVPR_2026_paper.html); [supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Liu_HAD_Hallucination_Aware_Diffusion_CVPR_2026_supplemental.pdf); [alphaXiv](https://www.alphaxiv.org/abs/2605.16873)
- **GA-GS (generation-assisted GS, arXiv 2604.04331).** It uses diffusion video inpainting for occluded regions and treats generated content as "inherently less reliable than real observations", applied through a learnable **per-primitive** reliability weighting. **[snippet]** — [arXiv](https://arxiv.org/pdf/2604.04331v1)
- **OracleGS (Topaloglu, Li, Niemeyer, Navab, Tekalp, Tombari; WACV 2026).** Propose-and-validate: a 3D-aware diffusion model proposes novel views, then an MVS model's attention maps mark where generated content has multi-view support. That uncertainty weights the 3DGS loss, filtering hallucinations while keeping plausible completions in under-constrained regions. Evaluated on Mip-NeRF 360 and NeRF Synthetic. **[snippet]** — [MCML publication](https://mcml.ai/publications/tln+26/); [KU research hub](https://research.hub.ku.edu.tr/entities/publication/b7f03b9d-3740-4089-b0f6-8e86ca6b52c2)
- **4DGS in the Wild (Kim, Lim, Han; NeurIPS 2024).** Uncertainty-aware regularization finds regions with few observations and selectively applies diffusion and depth-smoothness priors only there. **[snippet]** — [NeurIPS](https://neurips.cc/virtual/2024/poster/96899)
- **DAV-GSWT (arXiv 2602.15355).** It combines diffusion priors with uncertainty-driven active view sampling for terrain Gaussian Wang tiles. The listings show inconsistent dates. **[snippet]** — [alphaXiv](https://www.alphaxiv.org/abs/2602.15355)
- **GeomGS (arXiv 2501.13417).** LiDAR-guided 3DGS for robot localization. Each Gaussian gets a **geometric confidence score** from its distance to the nearest LiDAR point (asymmetric sigmoid), optimized jointly under probabilistic distance constraints. No generative component. **[snippet]** — [arXiv HTML](https://arxiv.org/html/2501.13417v1); [ar5iv](https://ar5iv.labs.arxiv.org/html/2501.13417)
- **LiDAR-GS++ (AAAI 2026; arXiv 2511.12304).** Diffusion priors for LiDAR Gaussian reconstruction on outdoor urban roads. A controllable LiDAR generation model conditioned on coarsely extrapolated renderings produces extra geometry-consistent scans for under-fitted regions. **[snippet]** — [arXiv](https://arxiv.org/pdf/2511.12304); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37289)
- **Gau-Occ (arXiv 2603.22852).** A point-wise local diffusion model densifies sparse LiDAR in occluded or unobserved regions for outdoor occupancy prediction. **[snippet]** — [arXiv](https://arxiv.org/pdf/2603.22852)
- **Gaussian-LIC2.** LiDAR-inertial-camera GS-SLAM whose depth completion initializes Gaussians in LiDAR-blind areas (no diffusion). **[snippet]** — [arXiv 2507.04004](https://www.arxiv.org/pdf/2507.04004)
- **Real-time LiDAR GS-SLAM (arXiv 2607.04127).** A covariance-derived geometry score drives pruning and densification, and LiDAR confidence cues feed back into tracking. **[snippet]** — [arXiv](https://arxiv.org/pdf/2607.04127)

### Inferences
- **Two granularities of "provenance" exist today:**
  1. **Per-pixel training masks** (HAD, ConFixGS, OracleGS). These decide what is learned from generated views, but they are not stored as a persistent label in the final model.
  2. **Per-primitive reliability weights** (GA-GS). These are the closest to a stored observed/generated tag, but they are learned and not tied to sensor evidence.
- Pairing GeomGS-style per-Gaussian LiDAR distance confidence with GA-GS-style per-primitive weighting and HAD-style masks is an obvious, unpublished combination.
- "Keeping generated content off measured surfaces" maps onto two pieces:
  - **Freeze or lock** Gaussians supported by LiDAR and multi-view photometry.
  - **Restrict** diffusion pseudo-targets to visibility masks of unobserved regions. G4Splat already uses geometry-derived visibility masks, so this part is partly done.

### Gaps
- No source found that produces an exported per-surface or per-Gaussian observed/generated provenance map for downstream users.
- Full texts of Bayes' Rays and CG-style uncertainty papers were not retrieved. "CG-Splat-uncertainty", named in the brief, was not found under that name.

## Robotics use: generative completion for robot-built maps, digital twins and simulation; fidelity concerns for engineering / RF simulation

### Takeaway
Robotics real-to-sim pipelines use 3DGS for appearance and meshes for physics. They validate fidelity by correlating simulated and real policy success, and they flag occlusion and noisy poses as threats to geometric accuracy. Generative completion has not been adopted or evaluated in them. RF digital-twin work uses 3DGS either as a radio-field model or as a geometry source for ray tracing (with LiDAR refinement). **No work was found that assesses what generated surfaces do to RF or engineering simulation.**

### Cited Findings
- **RoboSimGS:** 3DGS for photorealistic appearance plus mesh primitives for interactive objects. Multimodal LLMs infer articulation and physical properties, which is an unverified, generative-inference step. **[snippet]** — [alphaXiv 2510.10637](https://www.alphaxiv.org/overview/2510.10637); [arXiv](https://arxiv.org/abs/2510.10637v1)
- **Splatting Physical Scenes (arXiv 2506.04120):** "real robot data suffers from occlusions, noisy camera poses, dynamic scene elements", which can undermine geometric accuracy. 3DGS is merged with physics-ready meshes. **[snippet]** — [arXiv](https://arxiv.org/html/2506.04120v2)
- **Real-to-sim policy evaluation with GS soft-body simulation (arXiv 2511.04665):** "rendering and dynamics fidelity are both crucial to the trustworthiness of simulation-based evaluation". Simulated rollouts correlate strongly with real success rates. **[snippet]** — [arXiv](https://arxiv.org/pdf/2511.04665)
- **RF-3DGS (Zhang et al., 2024):** adapts 3DGS to "radio radiance fields" (received power in dB) and needs position-tagged measurements, for example from robot- or drone-mounted receivers. **[snippet]** — [TU Berlin book chapter](https://platform.commit.tu-berlin.de/en/book/rfi/chapter/ch26/section/s02)
- **Wideband frequency-embedded 3DGS (2025):** a dataset of 50,000 power-angular-spectrum samples from 1 to 100 GHz in 6 indoor environments; SSIM 0.70 on unseen frequencies. **SwiftWRF:** deformable 2DGS for wireless spectra. **PropSplat (May 2026):** map-free, 5.38 dB RMSE versus 5.87 dB for WRF-GS+. **[snippet]** — [arXiv 2505.20714](https://arxiv.org/abs/2505.20714v2); [arXiv 2506.12787](https://arxiv.org/pdf/2506.12787v2); [PropSplat](https://radiancefields.com/papers/propsplat-map-free-rf-field-reconstruction-via-3d-gaussian-propagation-splatting)
- **RFDT-Channel (May 2026):** an RF digital twin for 28 GHz indoor ray tracing. A COLMAP, 3DGS and SuGaR pipeline produces an initial mesh, which is refined with LiDAR and then fed to Sionna RT. **[snippet]** — [Emergent Mind 2605.08035](https://www.emergentmind.com/papers/2605.08035)
- **GaussTwin:** position-based dynamics and Cosserat rods for physics, with GS rendering providing visual correction. **[snippet]** — [arXiv 2603.05108](https://arxiv.org/pdf/2603.05108)

### Inferences
- For RF ray tracing, the ceiling is a dominant reflector indoors, at roughly 28 GHz and in many-bounce paths. A hallucinated ceiling (wrong height, material or fixtures) would bias path loss, but no study quantifying this was found.
- The RFDT-Channel pipeline (3DGS, then mesh, then LiDAR refinement, then ray tracing) is the closest pipeline to the user's setting. It contains no generative step and no provenance.

### Gaps
- No paper found on the effect of diffusion-generated geometry or appearance on RF or engineering simulation accuracy.
- No robotics map-building paper found that uses generative completion and reports a fidelity audit.

## Verified research gaps (each checked against the closest existing work)

### Takeaway
- **(a)** LiDAR-constrained generation with explicit per-surface provenance: **OPEN as an integrated system**; partly closed in pieces.
- **(b)** Whole-building consistent generative completion: **OPEN**.
- **(c)** Evaluating generated content against real held-out captures from another robot or session: **partly closed** for outdoor driving (EUVS), **OPEN** for indoor robots.
- **(d)** Generative completion's impact on RF / engineering simulation: **OPEN**.
- **(e)** Fixer pipelines on 12 GB consumer GPUs for building scale: **largely open** (inference).

All verdicts rest on snippet-level searches; see the caveat at the top.

### Cited Findings
- **Gap (a): metric-LiDAR-constrained generation plus per-surface observed/generated provenance. Verdict: OPEN (integration), partly closed (components).**
  - Searches run: "LiDAR-constrained diffusion completion Gaussian splatting indoor"; "LiDAR geometry-preserving diffusion ... per-Gaussian confidence generated observed label"; "mark hallucinated regions ... provenance".
  - The search results themselves reported that no single paper combines these elements.
  - Closest work:
    - G4Splat: indoor geometry-guided diffusion inpainting with visibility masks; planar priors, not LiDAR; no exported provenance. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007768)
    - HAD: per-pixel hallucination masks used during training. — [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_HAD_Hallucination-Aware_Diffusion_Priors_for_3D_Reconstruction_CVPR_2026_paper.html)
    - GA-GS: per-primitive reliability for generated content. — [arXiv 2604.04331](https://arxiv.org/pdf/2604.04331v1)
    - GeomGS: per-Gaussian LiDAR confidence, no generation. — [arXiv 2501.13417](https://arxiv.org/html/2501.13417v1)
    - LiDAR-GS++: LiDAR-domain diffusion, outdoor. — [arXiv 2511.12304](https://arxiv.org/pdf/2511.12304)
- **Gap (b): generative completion consistent across a whole building, not per view or per room. Verdict: OPEN.**
  - Search run: "generative scene completion large indoor building multi-room consistent 3DGS diffusion 2025 2026".
  - The results covered single rooms or panoramas only, and the search summary explicitly said no result addresses connected multi-room buildings.
  - Closest work:
    - GEN3C's persistent 3D cache, designed for long-range consistency but not evaluated at building scale. — [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C)
    - MVRoom (layout-conditioned iterative generation). — [arXiv 2512.04248](https://arxiv.org/html/2512.04248v1)
    - Rein3D (global refinement of a 360-degree indoor scene). — [alphaXiv 2604.10578](https://www.alphaxiv.org/abs/2604.10578)
    - SpAItial Echo-3 (commercial walkthrough-to-world; scale and method undisclosed). — [Runtime Wire](https://runtimewire.com/article/spaitial-echo-3-iphone-3d-world-capture)
- **Gap (c): evaluating generated content against real held-out captures from another robot or session. Verdict: PARTLY CLOSED (driving), OPEN (indoor robots and ceilings).**
  - The **EUVS (Extrapolated Urban View Synthesis) benchmark** is built from public AV datasets with multiple traversals, vehicles and cameras. It reports a 28% PSNR drop for 3DGS on far out-of-distribution views and finds that 3DGS overfits to training views. — [papers.cool 2412.05256](https://papers.cool/arxiv/2412.05256); [alphaXiv](https://www.alphaxiv.org/overview/2412.05256)
  - Other held-out / extrapolation evaluations:
    - FillGS: trains on 2 cameras and tests on both interpolation and extrapolation views; argues standard benchmarks mainly test interpolation. — [arXiv 2607.29284](https://arxiv.org/pdf/2607.29284)
    - A scene-extrapolation benchmark dataset (DL3DV, SpatialVID, Kubric-4D clips; masked PSNR). Unreviewed. — [HF dataset](https://huggingface.co/datasets/luuuulinnnn/scene-extrapolation-benchmark)
    - The driving free-form-trajectory generative-prior work. — [arXiv 2412.01717](https://arxiv.org/pdf/2412.01717)
    - ViSE. — [arXiv 2510.18341](https://arxiv.org/html/2510.18341v1)
  - HAD trains its hallucination predictor against ground-truth novel observations, which is relevant as a supervision signal but is not a cross-session evaluation protocol. — [alphaXiv](https://www.alphaxiv.org/abs/2605.16873)
  - No indoor multi-session or multi-robot benchmark scoring generated (versus observed) regions specifically was found.
- **Gap (d): fidelity of generated surfaces for RF / engineering simulation. Verdict: OPEN.**
  - Closest work: RFDT-Channel, a 3DGS-derived, LiDAR-refined mesh for 28 GHz ray tracing with no generation. — [Emergent Mind](https://www.emergentmind.com/papers/2605.08035)
  - Robotics real-to-sim fidelity validation via policy-success correlation. — [arXiv 2511.04665](https://arxiv.org/pdf/2511.04665)
- **Gap (e): building-scale fixing on a single 12 GB GPU. Verdict: largely open (inference only).**
  - The released video-generative priors target data-center GPUs: GEN3C-Cosmos-7B lists Ampere, Hopper and Blackwell. — [HF](https://huggingface.co/nvidia/GEN3C-Cosmos-7B)
  - GSFixer builds on CogVideoX. — [arXiv](https://arxiv.org/pdf/2508.09667)
  - Difix is single-step and SD-based. — [HF](https://huggingface.co/nvidia/difix)
  - No source gives memory footprints.

### Inferences
- **Most defensible novel contribution for the user's setting:**
  - Use the robot's metric LiDAR map (including sparse ceiling returns, plus planar ceiling priors as in G4Splat) as a hard geometric scaffold.
  - Let a single-step fixer (Difix-class, which fits 12 GB) or a geometry-conditioned generator paint appearance only in visibility-masked, unobserved regions.
  - Store a persistent per-Gaussian provenance tag combining LiDAR support, photometric observation count and FisherRF-style uncertainty. This differs from GA-GS's learned weight.
  - Validate the generated regions against a second robot or session's held-out capture, ideally a deliberately upward-looking pass, following an EUVS-style protocol adapted indoors.
- Gaps (a) and (c) together are the strongest pitch: no paper found both labels generated content persistently and measures it against independent real captures.

### Gaps
- Because the full texts were blocked, some 2025-2026 papers may close parts of (a) or (b) without the search snippets saying so. Before claiming novelty, re-check full texts of G4Splat, GA-GS, HAD and OracleGS, plus a direct arXiv search for "provenance" and "generated region" in GS papers.
- Not checked: whether Difix3D+ or its commercial successor "Fixer" offers a confidence or provenance output.
- No source found on whether Echo-3 or Marble expose captured-versus-generated masks; this remains unknown.
