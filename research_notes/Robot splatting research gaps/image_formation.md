# Robust image formation for Gaussian splatting / radiance fields from moving platforms (motion blur, rolling shutter, auto-exposure/HDR), with emphasis on sensor-informed methods (2023-2026)

Source-access note for the report writer: arxiv.org, ar5iv.labs.arxiv.org and papers.cool did not resolve from this research session (network allow-list), so no paper full text was opened. Every finding below comes from web-search result summaries and snippets of the linked pages (arXiv abstract/HTML pages, CVF/ECVA proceedings pages, project/GitHub pages, Zenodo, radiancefields.com and similar aggregators). All of it should be read as **snippet-level**. Where a claim comes only from a secondary aggregator (radiancefields.com, themoonlight.io, alphaXiv overview, emergentmind) it is marked "(secondary)". Numbers should be checked against the PDFs before publication.

Context assumed: ground robots mapping a ~6000 m2 building with LiDAR-inertial odometry (GLIM), RealSense D4xx and ZED cameras with auto-exposure, NeST-Splatting (2D surfels + hash-grid texture) initialised from LiDAR.

## Q1. Learned-trajectory deblurring for 3DGS: what each method models, sub-poses, cost, limits

### Takeaway
Almost all 3DGS deblurring methods (2024-2026) learn the intra-exposure camera motion from images alone. They parameterise it as linear/spline/Bezier/ODE trajectories sampled at about 5-21 virtual sharp sub-frames and average the renders, or they learn blur kernels or Gaussian offsets with small networks. Cost grows roughly linearly with the number of sub-frames. The stated limitations are dependence on SfM/COLMAP initialisation under blur, poor handling of complex or long-exposure motion with low-order trajectories, and an assumed fixed or known exposure window. Evaluation is mostly on Deblur-NeRF-style object/room scenes.

### Cited Findings
**BAD-Gaussians (Zhao et al., ECCV 2024)**
- Models the camera trajectory within the exposure time with a spline, renders a sequence of virtual sharp images by projecting the Gaussians, averages them to synthesise the blurred input, and jointly optimises Gaussians and trajectory. It is described as the first 3D-GS motion-deblur framework. — [arXiv 2403.11831](https://arxiv.org/pdf/2403.11831); [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3633_ECCV_2024_paper.php)
- Uses n = 10 virtual camera poses per blurry image as an accuracy/speed trade-off, and its ablation saturates as n grows (the table values were garbled in the snippet). — [arXiv 2403.11831](https://arxiv.org/pdf/2403.11831)
- Linear interpolation between two poses worked best on Deblur-NeRF synthetic data (Cozy2room 34.68 dB linear vs 33.74 dB cubic B-spline). Cubic B-spline was better on real data, which the authors attribute to longer exposure times. — [ECCV supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03633-supp.pdf)
- Uses the BAD-NeRF re-rendered version of the Deblur-NeRF synthetic set, which has more virtual sharp images per blurry frame. — [arXiv 2403.11831](https://arxiv.org/pdf/2403.11831)
- Inherits 3D-GS's need for a COLMAP sparse-point initialisation and notes that recovering accurate poses from blurred multi-view images with COLMAP is challenging. No explicit limitations section was found in the snippets. — [arXiv 2403.11831](https://arxiv.org/pdf/2403.11831)

**Deblur-GS (Chen & Liu, I3D 2024)**
- Reconstructs a sharp 3DGS from camera-motion-blurred images, treating blur as joint optimisation of the camera trajectory and the time sampling. The official code offers Linear, Spline and Bezier trajectory models, with a configurable Bezier order. The full paper text was not seen. — [GitHub Chaphlagical/Deblur-GS](https://github.com/Chaphlagical/Deblur-GS); [GitHub v6p/Deblur-GS](https://github.com/v6p/Deblur-GS)

**DeblurGS (Oh et al., arXiv 2404.11358)**
- Jointly optimises the latent 6-DoF camera motion of each blurry image and the sharp 3DGS scene. "Sub-frame alignment parameters" control the discrete sampling intervals along the continuous trajectory. — [arXiv 2404.11358](https://arxiv.org/html/2404.11358v2)
- Gaussian Densification Annealing prevents Gaussians being created at wrong locations early in training, while camera motion is still imprecise. It works by adjusting the densification threshold (secondary). — [arXiv 2404.11358](https://arxiv.org/abs/2404.11358v2); [themoonlight review (secondary)](https://www.themoonlight.io/review/deblurgs-gaussian-splatting-for-camera-motion-blur)
- The authors claim it is the only method that restores high-quality sharp scenes from the noisy SfM poses obtained on existing blur datasets. — [arXiv 2404.11358](https://arxiv.org/abs/2404.11358v2)

**Deblurring 3D Gaussian Splatting (Lee et al., ECCV 2024)**
- A small MLP takes each Gaussian's position, rotation and scale and manipulates its covariance (defocus) or predicts several auxiliary sets of Gaussian offsets that are rendered and averaged (camera motion). It is presented as the first real-time 3DGS deblurring method. — [arXiv 2401.00834](https://arxiv.org/pdf/2401.00834)
- The MLP is used only in training. Inference renders the plain 3DGS, so there is no test-time cost. v3 treats mean and covariance differently to model spatially varying blur. — [arXiv 2401.00834v3](https://arxiv.org/html/2401.00834v3)
- No explicit camera trajectory is modelled. Blur is absorbed into per-Gaussian offsets.

**BAGS (Peng et al., ECCV 2024)**
- The authors observe that 3DGS overfits to blurred inputs and does worse than NeRF methods. They propose a Blur Proposal Network (BPN) that estimates a per-pixel convolution kernel from spatial position, view embedding, and colour/depth variation, plus a quality mask of blurred regions. — [arXiv 2403.04926](https://arxiv.org/pdf/2403.04926); [ECCV poster](https://eccv2024.ecva.net/virtual/2024/poster/557)
- A small three-layer CNN extracts RGBD features from the rendered colour and depth. A coarse-to-fine kernel optimisation helps when the initial sparse point cloud is poor. It is "blur-agnostic", covering motion, defocus and downsampling, and has no camera-motion model. — [arXiv HTML](https://arxiv.org/html/2403.04926v1); [Kavli preprint summary (secondary)](https://preprints.kavlimeetings.org/2024/03/07/all/theo-phys/ucas/192462/)

**MBA-SLAM (Wang, Zhao, Zhang, Zhao, Liu; arXiv 2411.08279; cited as TPAMI 2025)**
- A motion-blur-aware tracker plus a NeRF or 3DGS mapper that models the physical blur formation in both tracking and mapping. It builds on MBA-VO (ICCV 2021), BAD-NeRF (CVPR 2023) and BAD-Gaussians. — [arXiv 2411.08279v2](https://arxiv.org/html/2411.08279v2)
- The TPAMI 2025 venue comes from a citation in Unblur-SLAM, not from the TPAMI page. — [arXiv 2603.26810](https://arxiv.org/pdf/2603.26810)

**BARD-GS (Lu et al., CVPR 2025)**
- Dynamic scenes from handheld monocular video. Blur is decomposed into camera-motion deblurring and object-motion deblurring. The authors note that prior dynamic-GS deblurring handled only camera-induced blur. They also release a real-world dynamic motion-blur dataset. — [arXiv 2503.15835](https://arxiv.org/html/2503.15835v1); [CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/32769)
- Used as a baseline in later work, e.g. MSCD-GS (CVPR 2026) on the Stereo Blur dataset. — [CVPR 2026 supp](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Liao_MSCD-GS_Motion-Separated_Cooperative_CVPR_2026_supplemental.pdf)

**EBAD-Gaussian (Deng et al., arXiv 2504.10012)**
- Fuses event streams with RGB and optimises Gaussian parameters and camera poses within the exposure time. Scope is static scenes on synthetic and real datasets. — [arXiv 2504.10012](https://arxiv.org/html/2504.10012v1)
- One listing reports +14% SSIM on Extreme-GoPro and +18% SSIM on Extreme-KITTI over the next best method. Attribution to EBAD-Gaussian was unclear in the snippet, so this is unverified. Its predecessor EBAD-NeRF found that linearly interpolated BA is worse than unconstrained BA under complex motion. — [arXiv 2406.14360](https://arxiv.org/pdf/2406.14360)
- Related event-based variants: EaDeblur-GS ([arXiv 2407.13520](https://arxiv.org/abs/2407.13520v2)), BeSplat for a single blurry image plus events ([arXiv 2412.19370](https://arxiv.org/html/2412.19370v1)), and DeblurSplat, SfM-free with events ([arXiv 2509.18898](https://arxiv.org/pdf/2509.18898)).

**Splat-based 3D Scene Reconstruction with Extreme Motion-blur (Jang et al., KAIST; ICCV 2025; arXiv 2607.16926)**
- Targets dim light with long exposures, where COLMAP fails. Input is RGB-D. Consecutive frames are aligned by optical flow and ICP, then camera poses and Gaussian positions are refined for depth alignment. Camera motion during exposure is modelled through the start and end poses of each frame. Dense Gaussians are initialised from depth maps, not SfM. — [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Splat-based_3D_Scene_Reconstruction_with_Extreme_Motion-blur_ICCV_2025_paper.pdf); [alphaXiv](https://www.alphaxiv.org/abs/2607.16926); [project](https://vclab.kaist.ac.kr/iccv2025p2)
- Introduces a new RGB-D extreme-blur dataset and compares against MonoGS, DROID-SLAM and ElasticFusion. Code is at github.com/KAIST-VCLAB/gs-extreme-motion-blur. The arXiv ID dates to July 2026 even though the paper is ICCV 2025, probably late posting. — [radiancefields (secondary)](https://radiancefields.com/papers/splat-based-3d-scene-reconstruction-with-extreme-motion-blur); [ICCV poster](https://iccv.thecvf.com/virtual/2025/poster/2496)

**Other learned-trajectory variants seen (2024-2026)**
- CRiM-GS: continuous camera trajectory via neural ODEs plus an adaptive distortion-aware transform for nonlinear effects "such as rolling shutter". — [arXiv 2407.03923](https://arxiv.org/abs/2407.03923v2)
- BSGS: bi-stage camera-motion deblurring that uses 21 sub-poses (from a snippet that contrasted it with BAD-Gaussians). — [arXiv 2510.12493](https://arxiv.org/pdf/2510.12493)
- GeMS: efficient GS for extreme motion blur. Title only was seen. — [arXiv 2508.14682](https://arxiv.org/pdf/2508.14682)
- Deblur-SLAM (Girlanda, Rozumnyi, Pollefeys, Oswald, 2025): RGB SLAM that recovers sub-frame trajectories with a deformable 3DGS map, monocular depth, and a DROID-style tracker. Numbers were not seen. — [arXiv 2503.12572](https://arxiv.org/html/2503.12572v1)
- Unblur-SLAM, dense neural SLAM for blurry inputs. — [arXiv 2603.26810](https://arxiv.org/pdf/2603.26810)
- Deblur4DGS: casts within-exposure dynamics as exposure-time estimation, with an exposure regulariser. — [arXiv 2412.06424](https://arxiv.org/html/2412.06424v1)
- MoBGS: infers a per-frame exposure time from estimated latent poses. — [arXiv 2504.15122](https://arxiv.org/html/2504.15122v4)
- RoMo-GS (2026, arXiv 2610.24605): online robot 3DGS that integrates each projected Gaussian's contribution **analytically over the exposure window** instead of averaging discrete sub-frames. It jointly refines pose, velocity and Gaussians, and adds Hessian-based pruning. Reported quality matches or beats offline deblurring at ~15-150x speed with ~20% fewer Gaussians than an online baseline. Inputs come from handheld, legged-robot and robot-arm platforms, camera-based only. All of this is from a secondary summary. — [radiancefields (secondary)](https://radiancefields.com/papers/romo-gs-efficient-online-3d-gaussian-splatting-for-robot-mapping-via-exposure-integrated-motion-blur); [alphaXiv listing](https://www.alphaxiv.org/abs/2610.24605)

### Inferences
- Cost model: discrete-sample methods (BAD-Gaussians n=10, BSGS 21) multiply rasterisation cost per training iteration by n. Lee et al. pays only at training time. RoMo-GS's analytic exposure integration is the first sign of escaping the n-sample cost.
- None of the learned-trajectory methods in Q1 uses an IMU or odometry prior. The trajectory is a free variable fit to photometric loss, which is the ambiguity sensor priors could remove.
- None of these methods works on 2D surfels / 2DGS or on hybrid surfel + hash-grid texture representations (see Q5, gap d).

### Gaps
- Exact per-method numbers (PSNR on Deblur-NeRF real/synthetic), training times, and the explicit limitations sections of BAD-Gaussians, DeblurGS, Deblur-GS, BAGS, MBA-SLAM and EBAD-Gaussian could not be read because full text was inaccessible.
- How many offset sets Lee et al. uses for motion blur was not found.
- RoMo-GS details rest on one secondary summary. The primary paper was not seen.

## Q2. Sensor-informed methods (IMU / VIO / LiDAR timing) for blur and rolling shutter

### Takeaway
A small but growing set of methods inject sensor data into image formation. Gaussian Splatting on the Move uses VIO velocities plus known exposure and readout times. Lv et al. 2025 use visual-inertial BA for rolling-shutter timing on Aria. TRGS-SLAM (2026) uses an IMU with B-splines for blurry rolling-shutter thermal images on robots. SplatAD and NeuRAD use per-ray/per-point timestamps for camera and LiDAR rolling shutter in driving. 3DGUT makes rolling shutter representable in splatting. No paper was found that drives blur, rolling shutter and exposure from a **LiDAR-inertial** trajectory on a ground robot.

### Cited Findings
**Gaussian Splatting on the Move (Seiskari et al., ECCV 2024; Spectacular AI / ETH / Aalto / Tampere)**
- Models motion blur and rolling shutter in 3DGS using a screen-space approximation. Camera pose is non-static during exposure. Linear and angular velocities are initialised from VIO (smartphone IMU) and jointly optimised with poses. No extra MLPs are used. — [arXiv 2403.13327](https://arxiv.org/html/2403.13327v3); [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08967.pdf)
- L2 penalties keep velocities near VIO estimates (secondary). Reported synthetic gain on rolling-shutter compensation for "Cozyroom" is 19.21 -> 35.84 PSNR (secondary, alphaXiv overview). — [alphaXiv overview](https://alphaxiv.org/overview/2403.13327v3)
- Stated limitation in the repo: simultaneous blur and RS compensation is only possible with **known readout and exposure times**. On plain video it can run in blur OR RS mode. Velocities can be initialised to zero without IMU. — [GitHub spectacularai/3dgs-deblur](https://github.com/spectacularai/3dgs-deblur); [ecosyste.ms](https://science.ecosyste.ms/projects/59083)
- Dataset: 11 short handheld scenes on Samsung S20 FE, Pixel 5 and iPhone 15 Pro recorded with Spectacular Rec. Metadata contains device-reported exposure times and Android RS readout times. COLMAP intrinsics are relatively inaccurate under strong RS. — [Zenodo 10848124](https://zenodo.org/records/10848124)
- Results with CVR preprocessing were mixed across phones (best on Pixel 5, worse on S20 and iPhone). An average PSNR of 28.59 on the smartphone set comes from a third-party summary and is unverified. — [arXiv 2403.13327](https://arxiv.org/pdf/2403.13327)

**Photoreal Scene Reconstruction from an Egocentric Device (Lv et al., Meta RL Research, SIGGRAPH 2025, arXiv 2506.04444)**
- Argues that pipelines assuming frame-rate 6DoF VIO poses with a global-shutter model miss pixel motion on a rolling-shutter head-worn camera. Uses visual-inertial bundle adjustment (VIBA) to calibrate RS timing and get a continuous trajectory. — [arXiv 2506.04444](https://arxiv.org/abs/2506.04444v1)
- Builds a physical image formation model in GS covering rolling shutter, lens shading, exposure and gain / sensor dynamic range. It applies to common GS variants. Reports about +1 dB PSNR from VIBA and about +1 dB more from the image formation model. Evaluated on Project Aria indoor/outdoor and validated on Meta Quest 3. Code, datasets and a recording profile are released. — [arXiv 2506.04444](https://arxiv.org/abs/2506.04444v1); [Project Aria page](https://www.projectaria.com/photoreal-reconstruction/)
- Motion blur was not mentioned in the snippets seen. Whether exposure and gain come from device metadata or are optimised was not confirmed.

**TRGS-SLAM (Carmichael & Skinner, U. Michigan, arXiv 2603.20443, Mar 2026)**
- 3DGS thermal-inertial SLAM for uncooled microbolometers with motion blur, rolling shutter and fixed-pattern noise. Components: model-aware thermal rendering, B-spline trajectory optimisation with a two-stage IMU loss, view-diversity opacity resetting, and drift correction. — [alphaXiv](https://www.alphaxiv.org/abs/2603.20443); [arXiv HTML](https://arxiv.org/html/2603.20443v1)
- Claims accurate tracking on fast, high-noise real thermal data where all other tested SLAM fails. Offline refinement gives restoration competitive with prior work that needed GT poses. Claims to be the first thermal SLAM to consider rolling shutter. Project page: umautobots.github.io/trgs_slam. — [radiancefields (secondary)](https://radiancefields.com/papers/trgs-slam-imu-aided-gaussian-splatting-slam-for-blurry-rolling-shutter-and-noisy-thermal-images)

**Driving / LiDAR-timing methods**
- SplatAD (Hess et al., CVPR 2025): first 3DGS method rendering both camera and LiDAR for dynamic driving in real time. Models camera and LiDAR rolling shutter (LiDAR sweeps up to 100 ms), LiDAR intensity and ray drop. Reports up to +2 PSNR NVS and +3 PSNR reconstruction, about 10x faster than NeRF-based methods. Built on gsplat. — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Hess_SplatAD_Real-Time_Lidar_and_Camera_Rendering_with_3D_Gaussian_Splatting_CVPR_2025_paper.html); [arXiv 2411.16816](https://arxiv.org/html/2411.16816v1)
- NeuRAD (Tonderski et al., CVPR 2024): its ablations show that rolling shutter, LiDAR ray drop and beam divergence are essential for sensor realism. Per-ray timestamps with interpolated sensor poses (secondary). Appearance embedding dim 16, per-sensor embeddings (secondary). — [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.pdf); [supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Tonderski_NeuRAD_Neural_Rendering_CVPR_2024_supplemental.pdf)
- 3DGUT (NVIDIA, CVPR 2025): replaces EWA with the Unscented Transform. Rolling shutter is handled by transforming each sigma point with its own extrinsic at its capture time, and the authors note RS is unclear to represent in EWA. RS-aware UT approximates Monte-Carlo references well. Code: nv-tlabs/3dgrut. — [arXiv 2412.12507](https://arxiv.org/html/2412.12507v2); [project](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
- A 2026 follow-up generalises this to a per-pixel capture time linear in image coordinates, τ(u,v)=τ_start+uτ_u+vτ_v, and treats spherical LiDAR in 3DGUT. — [arXiv 2602.05617](https://arxiv.org/pdf/2602.05617)

**Rolling-shutter NeRFs (learned)**
- URS-NeRF (Xu et al., ECCV 2024): estimates poses and velocities to model RS formation from **unordered** images, removing USB-NeRF's sequential-video requirement. Uses coarse-to-fine optimisation with RS epipolar constraints to fix poses stuck in local minima. Built on Tri-MipRF. — [arXiv 2403.10119](https://arxiv.org/abs/2403.10119v2); [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05236.pdf)
- USB-NeRF parameterises the trajectory as a continuous cubic B-spline in SE(3). — [liner review (secondary)](https://liner.com/review/usbnerf-unrolling-shutter-bundle-adjusted-neural-radiance-fields)

**LiDAR-inertial(-visual) Gaussian SLAM: geometry and pose, mostly not image formation**
- Gaussian-LIC (2024): LiDAR-inertial-camera 3DGS SLAM. Explicitly models sky and **varying camera exposure** for outdoor scenes, and initialises from LiDAR plus triangulated points. The exact formulation was not seen. — [arXiv 2404.06926](https://arxiv.org/pdf/2404.06926)
- Gaussian-LIC2 (Jul 2025): continuous-time fusion of LiDAR-inertial data with photometric constraints from the Gaussian map, plus a zero-shot depth model for dense depth. Snippets showed no blur or RS handling. — [arXiv 2507.04004](https://arxiv.org/abs/2507.04004v1)
- GS-LIVO (HKUST): IESKF odometry with a global Gaussian map plus a sliding window, real-time on Jetson Orin NX. Exposure handling was not mentioned in snippets. — [arXiv 2501.08672](https://arxiv.org/html/2501.08672v1)
- FAST-LIVO2 (not splatting): estimates **inverse exposure time in the ESIKF state** in real time. — [arXiv 2408.14035](https://arxiv.org/html/2408.14035v2); [themoonlight (secondary)](https://www.themoonlight.io/review/fast-livo2-fast-direct-lidar-inertial-visual-odometry)
- VIGS-SLAM (Zhu et al., ETH/Stuttgart, Dec 2025): tightly coupled visual-inertial 3DGS SLAM. Motivated by visual-only GS SLAM degrading under blur. Evaluated on EuRoC, RPNG AR Table, UTMM and FAST-LIVO2 data. Uses the IMU for tracking robustness, **not** for a blur image-formation model per the snippets. — [arXiv 2512.02293](https://arxiv.org/html/2512.02293v1)

### Inferences
- The sensor-informed methods split into two families. In "IMU/VIO -> intra-exposure motion" (Seiskari, TRGS-SLAM, Lv) the sensor supplies velocity or continuous trajectory for blur and RS. In "LiDAR-camera SLAM -> geometry/pose only" (Gaussian-LIC/LIC2, GS-LIVO, LVI-GS, VIGS-SLAM) the sensor does not enter the image-formation model. Driving simulators (SplatAD, NeuRAD) use timestamps for RS but target outdoor cars.
- A LiDAR-inertial continuous-time trajectory (GLIM-style) could supply per-row and per-exposure-sample poses directly, replacing the learned splines of Q1. No paper doing this for deblurring was found.

### Gaps
- Whether SplatAD or NeuRAD model camera motion blur (not just RS) was not visible in the snippets.
- Whether Lv et al. also model motion blur, and whether exposure and gain are read from Aria metadata, was not confirmed.
- LVI-GS was named in one search result but its details were not retrieved.

## Q3. Exposure and appearance: per-image embeddings, ISP models, HDR, and the use of exposure metadata

### Takeaway
The dominant approaches are learned per-image corrections: GLO/appearance embeddings, per-view affine or colour matrices, per-view bilateral grids, tone curves, and learned CRF/exposure networks. These absorb auto-exposure and ISP effects but do not transfer to novel views without an extra controller. Methods that use **actual exposure-time metadata** are a minority: HDR-GS conditions its tone mapper on exposure time, Seiskari and Lv use device-reported timing, and Casual3DHDR and MoBGS estimate unknown exposure time. Only Casual3DHDR explicitly ties blur extent and brightness to the same exposure variable.

### Cited Findings
- **Bilateral Guided Radiance Field Processing** (Wang, Wang, Gong, Xue; SIGGRAPH/TOG 2024): per-view 3D bilateral grids, as locally affine ISP approximations, are optimised jointly with the radiance field. View-dependent ISP effects (exposure, tone) are absorbed rather than baked in, which reduces floaters. A low-rank 4D bilateral grid lifts a 2D edit to 3D. The repo says to **disable the per-view grid for test views**. Built on Zip-NeRF; a 3DGS variant exists in gsplat (not verified here). — [arXiv 2406.00448](https://arxiv.org/html/2406.00448v2); [GitHub bilarf](https://github.com/yuehaowang/bilarf)
- **Luminance-GS** (Cui, Chu, Harada; CVPR 2025): per-view colour-matrix mapping plus a view-adaptive tone curve, jointly optimised with 3DGS for low-light, overexposure and varying exposure. Real-time rendering. Assumes white ambient light. Luminance-GS++ (arXiv 2602.18322) adds pixel-wise residual refinement and handles colour temperature and mixed degradations. — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Luminance-GS_Adapting_3D_Gaussian_Splatting_to_Challenging_Lighting_Conditions_with_CVPR_2025_paper.html); [arXiv 2602.18322](https://arxiv.org/pdf/2602.18322)
- **HDR-GS** (Cai et al., NeurIPS 2024): a dual-dynamic-range Gaussian point cloud. An MLP tone mapper maps log HDR colour plus **log exposure time** to LDR, and parallel HDR/LDR rasterisation allows exposure control at render time. Reports +3.84 dB LDR and +1.91 dB HDR over the best NeRF method, ~1000x faster inference, ~6.3% of the training time. Requires multi-exposure inputs with known exposure times. — [arXiv 2405.15125](https://arxiv.org/html/2405.15125v4); [NeurIPS poster](https://neurips.cc/virtual/2024/poster/95805)
- **Casual3DHDR / CasualHDRSplat** (Gong, Zhao, Li et al.; ACM MM 2025; WU-CVGL): uses auto-exposure video as an HDR source. Per-frame exposure times are often unavailable in video, and AE brightness shifts disrupt SfM. A one-stage physical model fits a continuous SE(3) trajectory and jointly optimises poses, **exposure times**, CRF and the HDR 3DGS. Notes that motion blur and brightness are both governed by exposure time and constrains them jointly. — [arXiv 2504.17728](https://arxiv.org/html/2504.17728v2)
- **PPISP** (NVIDIA, arXiv 2601.18336, Jan 2026): physically plausible ISP made of exposure offset, chromatic vignetting, linear colour correction and non-linear CRF. It separates camera-intrinsic effects (vignetting, CRF) from per-frame settings (exposure, white balance), which earlier modules conflated. A **controller predicts per-frame ISP parameters for novel views**, like AE/AWB. Ablation novel-view PSNR: full 24.62; without exposure 23.33; without vignetting 24.08; without colour correction 24.27; without CRF 24.36. Integrated into 3DGRUT as USD shaders. — [arXiv 2601.18336](https://arxiv.org/html/2601.18336); [radiancefields (secondary)](https://radiancefields.com/nvidia-announces-ppisp-for-radiance-fields)
- **AERGS-SLAM** (Zhou, Hui, Liu; CVPR 2026): stereo 3DGS SLAM with a camera-exposure network that recovers the CRF to map per-image radiance to RGB, giving exposure-controlled GS. Adds an illumination-robust geometric feature for localisation and initialisation. Evaluated on public and self-collected data. Code: zzy-2021/AERGS-SLAM. — [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_AERGS-SLAM_Auto-Exposure-Robust_Stereo_3D_Gaussian_Splatting_SLAM_CVPR_2026_paper.html)
- **GS-CPR**: a 4-layer MLP maps the query image's luminance histogram to a 3x3 colour matrix plus bias. Exposure is inferred from pixels, not metadata. — [arXiv 2408.11085](https://arxiv.org/pdf/2408.11085)
- **IBGS**: borrows the exposure of the nearest source view. One cited paper notes that per-view affine colour transforms **cannot generalise to novel views**. — [arXiv 2511.14357](https://arxiv.org/pdf/2511.14357)
- **DAVIGS** (decoupled appearance): appearance modules handle only mild changes such as AE and AWB. — [arXiv 2501.10788](https://arxiv.org/pdf/2501.10788)
- **RadSplat**: claims to handle lighting and exposure variation, using Zip-NeRF as a prior. One search summary claimed RadSplat uses exposure duration and ISO, but a follow-up search could **not** confirm this, so treat it as unverified. — [arxiv-vanity 2403.13806](https://www.arxiv-vanity.com/papers/2403.13806)
- **Gaussian-LIC** models varying camera exposure (see Q2). FAST-LIVO2 estimates exposure time online for odometry (see Q2). — [arXiv 2404.06926](https://arxiv.org/pdf/2404.06926); [arXiv 2408.14035](https://arxiv.org/html/2408.14035v2)
- **P2GS** (arXiv 2605.16925): physical-prior-guided GS for photometrically consistent urban reconstruction. Title only was seen. — [arXiv 2605.16925](https://arxiv.org/pdf/2605.16925)
- **Gaussian in the Dark** (arXiv 2408.09130): view synthesis from inconsistent dark images. Title only was seen. — [arXiv 2408.09130](https://arxiv.org/pdf/2408.09130)

### Inferences
- Per-view learned corrections (NeRF-W-style embeddings, bilateral grids, Luminance-GS curves) are unconstrained by the sensor and are disabled at test time. Off-trajectory renders therefore have an "average" appearance, unless a controller (PPISP) or nearest-view rule (IBGS) is added.
- For a ground robot with RealSense and ZED cameras, per-frame exposure and gain are likely available from driver metadata. This is unverified: the search found no paper that uses RealSense or ZED exposure metadata for splatting. A physically parameterised model (PPISP-like, with exposure fed from metadata and vignetting/CRF shared per camera) would be a straightforward fit, with blur length driven by the same exposure value.

### Gaps
- No source was found that compares metadata-driven against learned exposure compensation head-to-head.
- No paper was found that uses RealSense or ZED per-frame exposure metadata in NeRF or GS (the search returned nothing relevant).
- NeRF-W details were not re-verified here; standard knowledge is that it uses per-image appearance and transient embeddings.

## Q4. Evaluation practice: robot-captured building-scale data? Off-trajectory sharpness?

### Takeaway
Deblurring and RS methods are evaluated almost entirely on Deblur-NeRF synthetic/real scenes (object- or room-scale), BAD-NeRF re-renders, smartphone handheld clips, RGB-D blur sets, ExBluRF, or driving logs. Test views are typically on or near the training trajectory. The one robot/large-scale benchmark with explicit off-trajectory evaluation, Oxford Spires, uses global-shutter cameras and does not stress blur, RS or AE. It reports that radiance fields overfit to training poses.

### Cited Findings
- Deblur-NeRF real set: 20 scenes. Sharp reference images were captured with a still camera at the same exposure settings, close to the blurry training views. — [Deblur-NeRF CVPR 2022 supp](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Ma_Deblur-NeRF_Neural_Radiance_CVPR_2022_supplemental.pdf)
- BAD-Gaussians evaluates on the BAD-NeRF version of the Deblur-NeRF synthetic set plus the Deblur-NeRF real scenes (Ball, Basket, Buick, ...). — [arXiv 2403.11831](https://arxiv.org/pdf/2403.11831); [supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03633-supp.pdf)
- ExBluRF (arXiv 2309.08957) provides blurry-sharp multi-view pairs with **ground-truth blur trajectories**. — [arXiv 2309.08957](https://arxiv.org/html/2309.08957v3)
- BlurRF-Synth (DeepDeblurRF, CVPR 2025): 4,350 blurred-sharp pairs over 150 synthetic scenes, motion and defocus. — [arXiv 2502.14454](https://arxiv.org/html/2502.14454v1)
- Seiskari et al.: 11 short smartphone scenes. — [Zenodo](https://zenodo.org/records/10848124)
- KAIST extreme blur: new RGB-D dataset. — [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Splat-based_3D_Scene_Reconstruction_with_Extreme_Motion-blur_ICCV_2025_paper.pdf)
- BARD-GS: real dynamic blur dataset. — [arXiv 2503.15835](https://arxiv.org/html/2503.15835v1)
- Event-aided drone dataset (arXiv 2602.21101, 2026): synchronised blurred RGB and events with mocap GT poses. Whether it has sharp off-trajectory test views is unknown. — [arXiv 2602.21101](https://arxiv.org/html/2602.21101v1)
- Lv et al. evaluate on Project Aria indoor/outdoor and Quest 3, at room/house scale per the project page (scale not confirmed). — [Project Aria](https://www.projectaria.com/photoreal-reconstruction/)
- TRGS-SLAM evaluates on real fast-motion thermal robot data. — [alphaXiv](https://www.alphaxiv.org/abs/2603.20443)
- RoMo-GS uses handheld, legged-robot and robot-arm inputs (secondary). — [radiancefields](https://radiancefields.com/papers/romo-gs-efficient-online-3d-gaussian-splatting-for-robot-mapping-via-exposure-integrated-motion-blur)
- Oxford Spires (Tao et al., arXiv 2411.10546): six Oxford landmarks. Three synchronised **global-shutter** colour cameras, automotive LiDAR, IMU, and a TLS ground-truth map. The NVS test set includes **sequences in the opposing direction and trajectories far from training**. Finding: SOTA radiance fields (Nerfacto, Splatfacto) **overfit to training poses and generalise poorly to out-of-sequence poses**, and underperform MVS in 3D reconstruction. — [arXiv 2411.10546](https://arxiv.org/html/2411.10546v2)
- A 2026 quadruped multimodal SLAM study found rolling-shutter cameras drop to 33 feature matches vs 188 for global shutter, due to motion blur on a legged platform. This is SLAM, not splatting. — [arXiv 2606.19067](https://arxiv.org/html/2606.19067)

### Inferences
- "Sharpness on off-trajectory views" is essentially unmeasured in the deblurring literature. Deblur benchmarks measure sharp-reference PSNR at training-adjacent poses. Oxford Spires measures off-trajectory NVS but without blur, RS or AE stressors.
- No indoor ground-robot building-scale (~thousands of m2) benchmark with blur, RS or AE and held-out off-trajectory sharp references was found.

### Gaps
- Exact evaluation protocols (test-view selection) of BAD-Gaussians, Seiskari, Lv et al. and MBA-SLAM could not be confirmed from full text.
- ScanNet++ (DSLR vs iPhone, potentially off-trajectory) was not searched, so its fit to this protocol is unverified.

## Q5. Candidate research gaps, with counter-evidence search and closest existing work

### Takeaway
Each component (sensor-driven blur, RS timing, physical exposure/ISP, off-trajectory evaluation) has been done in isolation or on handheld/driving devices. The combination for a LiDAR-inertial ground robot at building scale, with consumer depth cameras under auto-exposure and a surfel/texture representation, appears open. Several sub-gaps are only partly closed.

### Cited Findings
**(a) Joint blur + rolling shutter + exposure, driven by LiDAR-inertial odometry, on a ground robot at building scale. Verdict: OPEN as a combination; PARTLY CLOSED piecewise.**
- Closest: Gaussian Splatting on the Move (VIO velocities, blur + RS, needs known exposure and readout, smartphones, short scenes) — [arXiv 2403.13327](https://arxiv.org/html/2403.13327v3); [repo](https://github.com/spectacularai/3dgs-deblur)
- Lv et al. 2025 (VIBA RS timing, plus exposure/gain and lens shading in GS, on Aria; blur not confirmed) — [arXiv 2506.04444](https://arxiv.org/abs/2506.04444v1)
- TRGS-SLAM (IMU + B-spline, blur + RS, thermal, robot) — [alphaXiv](https://www.alphaxiv.org/abs/2603.20443)
- RoMo-GS (online robot GS with exposure-integrated blur and velocity; camera-only, secondary) — [radiancefields](https://radiancefields.com/papers/romo-gs-efficient-online-3d-gaussian-splatting-for-robot-mapping-via-exposure-integrated-motion-blur)
- SplatAD / NeuRAD (LiDAR + camera RS via timestamps, driving) — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Hess_SplatAD_Real-Time_Lidar_and_Camera_Rendering_with_3D_Gaussian_Splatting_CVPR_2025_paper.html)
- Gaussian-LIC (LIC SLAM with exposure, no blur seen) — [arXiv 2404.06926](https://arxiv.org/pdf/2404.06926)
- Searches for "LiDAR-inertial + motion blur + Gaussian splatting" and "IMU preintegration intra-exposure deblur GS" returned no paper combining them.

**(b) Unknown or variable per-frame exposure duration (auto-exposure) coupled to the blur model. Verdict: PARTLY CLOSED.**
- Casual3DHDR jointly estimates per-frame exposure time, CRF and a continuous trajectory, and couples blur and brightness through exposure time. It is handheld video, learned trajectory, no IMU or LiDAR — [arXiv 2504.17728](https://arxiv.org/html/2504.17728v2)
- MoBGS infers per-frame exposure from latent poses — [arXiv 2504.15122](https://arxiv.org/html/2504.15122v4)
- Deblur4DGS uses exposure-time estimation with regularisation — [arXiv 2412.06424](https://arxiv.org/html/2412.06424v1)
- FAST-LIVO2 estimates exposure online, but for odometry, not rendering — [arXiv 2408.14035](https://arxiv.org/html/2408.14035v2)
- Seiskari et al. require known exposure — [repo](https://github.com/spectacularai/3dgs-deblur)
- Open part: using a sensor trajectory (LIO/IMU) together with metadata-reported but possibly inaccurate or variable exposure, where exposure jointly sets blur extent and brightness, with a PPISP-like per-camera model and a novel-view controller.

**(c) Robot-scale benchmark with off-trajectory ground truth that stresses blur, RS and AE. Verdict: PARTLY CLOSED.**
- Oxford Spires has off-trajectory NVS and TLS GT, but global shutter, outdoor/heritage sites, and no blur or AE stressors — [arXiv 2411.10546](https://arxiv.org/html/2411.10546v2)
- ExBluRF has GT blur trajectories but is small-scale — [arXiv 2309.08957](https://arxiv.org/html/2309.08957v3)
- Deblur-NeRF real uses on-trajectory sharp references — [supp](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Ma_Deblur-NeRF_Neural_Radiance_CVPR_2022_supplemental.pdf)
- The drone event dataset has mocap poses but unclear sharp off-trajectory views — [arXiv 2602.21101](https://arxiv.org/html/2602.21101v1)
- No indoor ground-robot building-scale benchmark with these properties was found.

**(d) Blur, RS and exposure modelling for 2D-surfel / hybrid texture representations (e.g., 2DGS, NeST-style surfels + hash-grid texture), with geometry accuracy measured under blur. Verdict: APPEARS OPEN, weakly verified.**
- A search for "2DGS / surfel + motion blur + mesh" found no direct match. The closest are 3DGS deblurring methods, the historical EWA surface-splatting motion blur work (Heinzle et al. 2010), and the claim that Lv et al.'s image formation model is applicable to GS variants — [search hits: 2DGS repo](https://github.com/hbb1/2d-gaussian-splatting/wiki); [EWA motion blur](https://reality.tf.fau.de/projects/ewamblur/ewamblur.pdf); [arXiv 2506.04444](https://arxiv.org/abs/2506.04444v1)
- Note that RS is analytically awkward for EWA projection, and 3DGUT's sigma-point approach is the known workaround — [arXiv 2412.12507](https://arxiv.org/html/2412.12507v2)

**(e) Replacing learned intra-exposure splines with a LiDAR-inertial continuous-time trajectory as a prior or constraint (vs IMU/VIO). Verdict: OPEN for LiDAR-inertial; PARTLY CLOSED for IMU/VIO.**
- Closest: Seiskari (VIO velocity prior with L2 penalty), TRGS-SLAM (IMU loss on B-spline), Lv (VIBA continuous trajectory) — [arXiv 2403.13327](https://arxiv.org/html/2403.13327v3); [alphaXiv 2603.20443](https://www.alphaxiv.org/abs/2603.20443); [arXiv 2506.04444](https://arxiv.org/abs/2506.04444v1)
- Gaussian-LIC2 has a continuous-time LIC trajectory but no blur or RS image model per the snippets — [arXiv 2507.04004](https://arxiv.org/abs/2507.04004v1)

**(f) Novel-view appearance and exposure control for consistent off-trajectory renders. Verdict: LARGELY CLOSED (method-wise) by PPISP; open for robot multi-camera metadata use.**
- PPISP controller — [arXiv 2601.18336](https://arxiv.org/html/2601.18336)
- IBGS nearest-view exposure — [arXiv 2511.14357](https://arxiv.org/pdf/2511.14357)

**(g) Heterogeneous multi-camera rigs (different ISPs and shutters, e.g. RealSense + ZED) with per-sensor intrinsic photometrics plus per-frame settings. Verdict: PARTLY CLOSED.**
- NeuRAD per-sensor appearance embeddings (secondary) — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.pdf)
- PPISP intrinsic/per-frame split — [arXiv 2601.18336](https://arxiv.org/html/2601.18336)
- Not shown for indoor robots with depth cameras.

### Inferences
- The most defensible novel contribution is not a new deblur loss. It is a **sensor-driven physical image formation model**: GLIM's continuous-time trajectory gives per-row (RS) and per-exposure-sample (blur) poses; per-frame exposure comes from metadata, refined or estimated where missing, and drives both blur length and brightness (PPISP-style per-camera vignetting/CRF); the whole model is implemented for a surfel + hash-texture representation. It should be evaluated with a held-out off-trajectory protocol (Oxford-Spires-style) at building scale.
- The main risk is novelty erosion. RoMo-GS (robot, analytic exposure-integrated blur, velocity), Lv et al. (VIBA + RS + exposure) and Casual3DHDR (exposure-coupled blur and brightness) each cover a slice, and a 2026-2027 paper could combine two of them. These should be monitored.
- Rolling shutter on RealSense D4xx RGB sensors (some D4xx RGB sensors are believed rolling-shutter, and the D455 RGB global-shutter) and ZED's rolling shutter are **unverified here** and need checking against Intel/Stereolabs datasheets before claiming RS matters for this rig.

### Gaps
- RoMo-GS primary text, its LiDAR/IMU use (if any) and its datasets were not verified. It is the single most important paper to check for counter-evidence to gaps (a) and (e).
- LVI-GS, Gaussian-LIC2 and GS-LIVO full texts were not checked for exposure or blur handling. Snippets suggest none model blur or RS.
- Searches were snippet-only. A paper that closes (a) or (d) may exist but be poorly indexed, so the "open" verdicts are provisional.
- Whether RealSense or ZED SDKs expose reliable per-frame exposure duration and RS readout time was not verified.
