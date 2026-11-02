# Object Gaussians from a rosbag: SAM3 → SAM 3D Objects

`bag_to_objects.py` takes RGB + registered depth from a ROS 2 bag, segments objects with
**SAM3** (text prompts), and reconstructs each one as a **complete, metric** set of Gaussians
with **SAM 3D Objects**, using the measured depth as its pointmap. Output is a standard 3DGS
`.ply` per object, posed in the camera optical frame in metres.

```
bag ──extract──► rgb.png, depth.npy, pointmap.npy, K ──segment (SAM3)──► masks/*.png
                                                       ──reconstruct (SAM 3D Objects)──► objects/*_cam.ply
```

The stages exchange files only: SAM3 (Python 3.12, recent torch) and SAM 3D Objects
(torch 2.5.1, pytorch3d, kaolin, ≥32 GB VRAM) do not share an environment.

**Two reconstruction backends:**

| subcommand | model | GPU | how it becomes metric |
|---|---|---|---|
| `reconstruct` | SAM 3D Objects | ≥32 GB | the depth pointmap is a model input; pose comes out of the model |
| `reconstruct-idol` | [Idol](https://github.com/speridlabs/idol-sdk) | ~7.2 GB (fits an RTX 3080 12 GB) | object comes out in a unit cube; scale + pose fitted to the depth in the mask (CPU, ~40 s) |

`reconstruct-idol` fit: ICP at fixed scale on the visible part in both directions (measured →
object, and object points visible from the camera → measured), started from 24 yaws × 3 scales
about the camera's vertical axis. The scale is then searched by golden section. Each candidate
is scored by distance to the measured points plus the fraction of object points that contradict
the image (outside the mask, or in front of a measured surface). It assumes a roughly level camera
(the generated object is upright).

## Setup (once)

```bash
# bag reading: any env, e.g. scoop
pip install rosbags opencv-python-headless numpy
```

```bash
# SAM3: request access at https://huggingface.co/facebook/sam3, then
conda create -n sam3 python=3.12 -y && conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git ~/workspaces/sam3 && pip install -e ~/workspaces/sam3
pip install rosbags opencv-python-headless
hf auth login
```

```bash
# Idol (12 GB GPUs): request access to the gated DINOv3 and FLUX.2-dev repos on Hugging Face, then
git clone https://github.com/speridlabs/idol-sdk.git ~/workspaces/idol-sdk
pip install -e ~/workspaces/idol-sdk plyfile scipy   # in the sam3 env (torch >= 2.5)
```

```bash
# SAM 3D Objects (>= 32 GB GPUs): request access at https://huggingface.co/facebook/sam-3d-objects, then follow
# its doc/setup.md (mamba env sam3d-objects + checkpoints/hf). Also:
git clone https://github.com/facebookresearch/sam-3d-objects.git ~/workspaces/sam-3d-objects
pip install plyfile scipy   # inside the sam3d-objects env
```

## Example run on the 2026-10-15 coop site_1 bags

```bash
cd ~/workspaces/Others/ed305_digital_twin/objects
BAG=../../../isaac_ros-dev/data/work/20261015/coop/site_1/mobile_1
OUT=../../../isaac_ros-dev/data/work/20261015/coop/site_1/objects_mobile_1
```

**1. See the topics.** Defaults are the ZED wrapper's
`/zed/zed_node/left/image_rect_color`, `/zed/zed_node/depth/depth_registered`,
`/zed/zed_node/left/camera_info`. If the bag namespaces them (e.g. `/mobile_1/zed/...`),
pass them with `--image-topic`, `--depth-topic`, `--info-topic` in step 2.

```bash
python bag_to_objects.py topics $BAG
```

**2. Extract frames.** 5 frames, one every 5 s, from 10 s in; or `--times 12.5 40 73` for
moments where the other robot or a piece of furniture is in view.

```bash
python bag_to_objects.py extract $BAG --out $OUT --start 10 --every 5 --max-frames 5
```

**3. Segment.** Check `frame_*/overlay.png`.

```bash
conda activate sam3
python bag_to_objects.py segment --out $OUT --prompts "robot,chair,table"
```

**4. Reconstruct.** `--only robot` restricts it to that prompt.

```bash
conda activate sam3d-objects
python bag_to_objects.py reconstruct --out $OUT --sam3d-repo ~/workspaces/sam-3d-objects --only robot
```

On the RTX 3080 (12 GB), use Idol instead. Stop other GPU jobs first: Idol needs ~7.2 GB.

```bash
conda activate sam3
python bag_to_objects.py reconstruct-idol --out $OUT --only robot
```

Either backend prints, per object, the number of Gaussians, the extent in metres, and the median
distance from the measured depth points inside the mask to the nearest Gaussian (the
visible-side error). Open `objects/<name>_cam.ply` in any 3DGS viewer (SuperSplat,
`gsplat` viewer), together with `pointmap.npy` if you want to check the fit.

## Files per frame

| file | content |
|---|---|
| `rgb.png` | image as recorded |
| `depth.npy` | float32 m, same size as `rgb.png`, NaN = no depth |
| `pointmap.npy` | float32 H×W×3, m, camera optical frame (x right, y down, z forward) |
| `meta.json` | stamp, `frame_id`, K, topics, image-depth time offset, detected objects |
| `masks/<name>.png`, `overlay.png` | SAM3 masks and a preview |
| `objects/<name>_cam.ply` | object in the camera optical frame, metres |
| `objects/<name>_local.ply` | object in the model's normalized frame |
| `objects/<name>_input.png` | (Idol) the crop the model saw |
| `objects/<name>_pose.json` | centre, extent, scale, visible-side error, raw model pose |

## Notes

- **Unseen sides are generated**, not measured. Only the part near measured depth
  (visible-side error) is checked against the sensor.
- SAM 3D Objects expects its pointmap in the PyTorch3D camera convention (x left, y up);
  the script converts to and from the optical frame, so everything on disk is optical.
- To place an object in the room/map frame, compose `_cam.ply` with the camera's pose at
  `meta.json` `stamp` (TF `map → frame_id`).
