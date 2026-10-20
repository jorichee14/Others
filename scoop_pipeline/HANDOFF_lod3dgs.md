# Handoff: LOD-3DGS (LetsGo) digital twin from the SCOOP reference map

Branch `claude/scoop-paper`. Code: `scoop_pipeline/datasets/lod3dgs.py`,
`scoop_pipeline/tests/test_lod3dgs.py`; usage in `scoop_pipeline/README.md`,
section "Gaussian splatting: LOD-3DGS".

## Goal

Train LetsGo's level-of-detail Gaussian splatting on the 20261015 mapping
pass (mobile_2), using the SCOOP reference map and poses instead of
structure from motion.

- Paper: Cui et al., "LetsGo: Large-Scale Garage Modeling and Rendering via
  LiDAR-Assisted Gaussian Primitives", ACM TOG 43(6) 172, 2024, DOI 10.1145/3687762.
- Code: github.com/zhaofuq/LOD-3DGS (training), github.com/zhaofuq/LOD-SIBR-Viewer (viewer).

## What the paper does (read from the PDF)

1. LIV-SLAM poses and a merged LiDAR cloud (Sec. 3.3). In SCOOP this is
   stage 02's BA trajectory and stage 01's TSDF map.
2. Poisson mesh of the cloud, then every face more than 0.1 m from the
   nearest cloud point is deleted (App. B).
3. The mesh is sampled evenly every 4 cm. These points, not the raw LiDAR,
   start the Gaussians (Sec. 4.1). Table 3: mesh points beat raw LiDAR points
   at the same spacing by 2.5–3.4 dB PSNR.
4. Depth loss: L1 between rendered depth and LiDAR depth (Eq. 5).
5. For large scenes, the scene is split into blocks, and the mesh is
   projected into each view to choose the training views per block (Sec. 5.1).
6. Fisheye frames are split into 5 pinhole views. Not needed here: the
   RealSense colour camera is already a pinhole camera.

## What LOD-3DGS's code reads (checked in the code)

- COLMAP text files: `sparse/0/cameras.txt` (PINHOLE only), `sparse/0/images.txt`
  (world to camera, `qw qx qy qz tx ty tz`), and `sparse/0/points3D.ply`
  (x y z nx ny nz red green blue).
- **It ignores cx and cy and assumes the image centre is the principal point**
  (`scene/dataset_readers.py`, readColmapCameras). `lod3dgs.py` therefore
  crops and resamples each image to the largest window centred on the
  principal point.
- `depths/<image name>.png`, opened as 8-bit grey + alpha (`utils/camera_utils.py`):
  - grey = warped depth: d/20 below 10 m, 1 − 5/d above (`utils/image_utils.py`
    warpped_depth);
  - alpha = valid-depth mask.

  Rendered depth is camera z-depth.
- `--use_lod` reads the Octree loader. On the first run it converts
  points3D.ply to LAS (needs `laspy`) and calls
  `PotreeConverter/bin/Release/Converter.exe` with `shell=True`. That only
  works on Windows. **Linux needs a patch:** build the PotreeConverter
  submodule and call its binary without `shell=True`. Not done yet.

## The SCOOP reference map (20261015, `map_stages/pipeline_config_mapping_20261015.json`)

- **Machine:** mobile_2 (colour: `/mobile_2/color/image_raw`; LiDAR:
  `/mobile_2/ouster/points`).
- **Poses:** `odometry/mobile_2/traj_lidar_ba_glim.txt`. This is GLIM after
  stage-02 bundle adjustment, the reference. `traj_lidar_refined.txt` (02b) is
  only a comparison: do not use it.
- **Map:** `01_build_map` builds it in these steps, saving a file after each:
  1. TSDF fusion at 2 cm, trunc 8 cm, min_weight 3 → `merged_<tag>.pcd`
  2. moving-object removal with free-space carving → `static_<tag>.pcd`
  3. denoise → `denoised_<tag>.pcd`
  4. colorize → `colored_<tag>.pcd` = `map_final_<tag>.pcd`

  Not anchored or flattened, so the map is in the BA trajectory's frame.
- **Tag:** `mapping_20261015` (`map_stages/pipeline_common.py:653`, `<pass>_<date>`).

## Done (pushed; this session and another session on the same branch)

`lod3dgs.py` has four subcommands. Run `-h` on each for every flag.
- **`mesh <map> --traj <TUM> [--ceil auto|Z] [--voxel] [--show]`**: writes
  `mapping/mesh_<tag>.ply`, a Poisson mesh of the map with faces more than
  0.1 m from the map deleted. The other session added:
  - mirror ghosts removed (points no direct ray from the path reaches;
    `--keep-ghosts` keeps them);
  - floating pieces, and pieces open to the sky, removed (`--min-piece`);
  - objects meshed separately at a finer cell (`--no-objects`,
    `--object-cell`, ...);
  - colours: the median of the 5 nearest map points (`--colour-k`).

  `--ceil` also writes `*_noceil.ply`. With `auto`, the cut is the first
  strong height peak 2.1 m or more above the floor; the floor comes from the
  robot's height on `--traj`.
- **`view <map> --traj <TUM>`**: a viewing copy of the point cloud with
  walls flattened, stray points removed, and **floor holes filled** (up to
  `--max-hole` m², only where `--roof` of the hole has ceiling over it).
- **`cut <mesh or cloud> --traj <TUM> [--ceil auto|Z]`**: a copy without the
  ceiling, without meshing again.
- **`scene <replica export> --map <map> [--every N]`**: the LOD-3DGS scene:
  - images and depth, recentred on the principal point;
  - depth in LOD-3DGS's format;
  - cameras.txt and images.txt;
  - points3D.ply sampled every 4 cm from the trimmed Poisson mesh,
    coloured from the frames.

  `scene` does not yet use the `mesh` command's ghost, piece and object
  steps. It has its own simpler Poisson step.

`python scoop_pipeline/tests/test_lod3dgs.py` passes: view, ghosts, objects,
mesh with ceiling cut, and scene. LOD-3DGS's own colmap_loader and plyfile
read the `scene` output.

## State of the user's run

The user is running the mesh command on the robot PC:
```
P=~/workspaces/isaac_ros-dev/data/processed/20261015/mapping
python3 scoop_pipeline/datasets/lod3dgs.py mesh $P/mapping/map_final_mapping_20261015.pcd \
    --traj $P/odometry/mobile_2/traj_lidar_ba_glim.txt --ceil auto --voxel 0.04 --show
```
The user reported **many holes in the floor of the map**. The cause is not
yet known.

## Next steps

1. **Look at the mesh** with the user: walls, floor holes, any surfaces in
   empty space. Check the printed height histogram and where the ceiling
   was cut; if a single height doesn't fit, pass `--ceil Z`.
2. **Find which step makes the floor holes** (G=mapping_20261015, M=$P/mapping, T=the BA traj):
   ```
   python3 scoop_pipeline/check.py maps $M/merged_$G.pcd $M/static_$G.pcd --traj $T --save $M/lost_dynamic.ply
   python3 scoop_pipeline/check.py maps $M/static_$G.pcd $M/denoised_$G.pcd --traj $T --save $M/lost_denoise.ply
   ```
   The floor is in the height band just below the path.
   - **Lost from merged to static:** carving at grazing angles
     (`01_build_map.py:85`). Raise `01_build_map.remove_dynamic.carve.grazing_margin`
     from 0.03 to about 0.06. Delete `static_`, `denoised_`, `colored_` and
     `map_final_`, keep `merged_`, and re-run 01.
   - **Lost from static to denoised:** loosen `denoise` (nb 30, std 3.0).
   - **Already missing in merged:** `tsdf.min_weight`, or no LiDAR returns
     (dark or glossy floor, robot blind zone).

   Hole shape helps: round patches where people walked → moving-object
   removal; holes growing with distance from the path → grazing; one stripe
   under the path → LiDAR blind zone.
   For viewing, `lod3dgs.py view` already fills floor holes. For training,
   the fix belongs in 01, so the map itself is complete. `view` is a
   viewing copy only.
3. **Replica export** with the same poses and map:
   ```
   B=~/workspaces/isaac_ros-dev/data/work/20261015/mapping/mobile_2/mirc_dataset_mapping_2_run_20261015_mobile_2_retimed
   python3 scoop_pipeline/datasets/mcap_convert.py $B --format replica \
       --pose-tum $P/odometry/mobile_2/traj_lidar_ba_glim.txt \
       --color /mobile_2/color/image_raw --info /mobile_2/color/camera_info \
       --lidar /mobile_2/ouster/points --depth-source map --map $P/mapping/map_final_mapping_20261015.pcd
   python3 scoop_pipeline/check.py replica $P/datasets/mobile_2/replica_map_mapping_20261015
   ```
   - If it warns about non-zero distortion, the RealSense colour images must
     be undistorted first.
   - `check.py replica` must show depth edges on image edges. If they don't,
     the images and poses are out of sync in time.
4. **Build the scene:**
   ```
   python3 scoop_pipeline/datasets/lod3dgs.py scene $P/datasets/mobile_2/replica_map_mapping_20261015 \
       --map $P/mapping/map_final_mapping_20261015.pcd --every 3
   ```
5. **LOD-3DGS setup and training:**
   - `git clone --recursive https://github.com/zhaofuq/LOD-3DGS`
   - Install its environment and CUDA extensions (simple-knn,
     diff-gaussian-rasterization), plus `laspy` and `plyfile`.
   - Patch the PotreeConverter call for Linux (see above).
   - Train with:
     ```
     python train.py -s <scene> --use_lod --sh_degree 2 --depths depths \
         --densification_interval 10000 --iterations 300000 --scaling_lr 0.0015 \
         --position_lr_init 0.000016 --opacity_reset_interval 300000 \
         --densify_until_iter 200000 --data_device cpu -r 1
     ```
     These are the flags from its README.

## Open choices

- Should `scene` reuse the `mesh` command's ghost, piece and object steps
  for its starting points?

- The TSDF map is already a noise-averaged surface, and the paper's mesh gain
  was measured against raw LiDAR. Starting from `map_final` thinned to 4 cm,
  with its own colours and no Poisson, may be just as good. One training run
  of each would settle it.
