# NeST-Splatting on a LiDAR scene (Phase A)

Surfels on the LiDAR surface, appearance in a hash-grid texture field
(NeST-Splatting, ICCV 2025, Apache 2.0), trained the way LetsGo trains a
LiDAR-initialised model. Three things NeST does not have, as a patch against
its commit `058a3df`:

- `fetchPly` reads the scene's colours and mesh normals instead of random
  colours and zero normals;
- `create_from_pcd` starts every surfel lying on the mesh normal (its
  rotation's third axis) instead of at a random orientation;
- `train.py` adds a one-sided LiDAR depth loss from the scene's `depths/`:
  a surface rendered in front of the LiDAR one is pushed back, never pulled
  forward onto the map's own smoothing error. Weight and start iteration
  come from the yaml (`loss.lambda_depth`, `loss.depth_iter`).

## Set up (once)

```bash
cd ~/workspaces && git clone https://github.com/zhangxin-cg/nest-splatting.git
cd nest-splatting && git checkout 058a3df
git apply ~/workspaces/Others/scoop_pipeline/nest/nest_lidar.patch
conda env create -f environment.yml && conda activate nest_splatting
```

`environment.yml` builds the surfel rasteriser, simple-knn, the grid
encoder and tinycudann from source; that is the step that can fail, and it
fails on the compiler or CUDA version, not on anything here.

## Size the texture grid to the scene

`encoding.hashgrid.range` is the cube the texture covers, in the scene's own
frame. Read the bounds and set it to cover them with a margin:

```bash
SCENE=~/workspaces/isaac_ros-dev/data/processed/20261015/mapping/datasets/mobile_2/lod3dgs_map_mapping_20261015
python3 - <<'PY'
import numpy as np, os
p = os.path.expanduser("$SCENE/sparse/0/points3D.ply")
with open(p, "rb") as f:
    while f.readline().strip() != b"end_header": pass
    dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("nx","<f4"),("ny","<f4"),("nz","<f4"),("r","u1"),("g","u1"),("b","u1")])
    a = np.frombuffer(f.read(), dtype=dt)
lo = [a[k].min() for k in "xyz"]; hi = [a[k].max() for k in "xyz"]
print("bounds", np.round(lo, 1), np.round(hi, 1))
half = max(abs(v) for v in lo + hi) + 2
print(f"range: [-{half:.0f}, {half:.0f}]")
PY
```

Put the printed `range` into `scoop_indoor.yaml`. The grid's finest level is
2^13 cells across that range, so a 100 m range is 1.2 cm texels.

## Train

```bash
conda activate nest_splatting && cd ~/workspaces/nest-splatting
cp ~/workspaces/Others/scoop_pipeline/nest/scoop_indoor.yaml configs/
python train.py -s $SCENE -m ~/workspaces/lod3dgs_runs/nest_20261015 \
  --yaml configs/scoop_indoor.yaml -r 1 --eval \
  --test_iterations 10000 35000 70000 --save_iterations 10000 35000 70000
```

The first line of training output must say `[LiDAR init] 3268773 of 3268773
surfels oriented to the mesh normal`. If it does not, the patch is not in.

## Score it the same two ways as before

`train.py --eval` prints the held-out PSNR at each test iteration (every 8th
image, as 3DGS does). For the like-for-like against the LOD-3DGS runs, render
the same 693 views and score with `splat_psnr.py` -- that needs a NeST
renderer for `splat_walk.py --render-views`, which is the next thing to add
once a model exists.
