# Dense static point cloud from radar detections (DREAM-PCD, adapted)

Builds a **dense, denoised, static point cloud of the environment** from the
three calibrated IWR6843ISK radars, following the signal-processing recipe of
**DREAM-PCD** ([arXiv:2309.15374](https://arxiv.org/abs/2309.15374)) — but
adapted, because their input is **raw ADC data** and ours is the chip's
**CFAR detections** (`x, y, z, snr, doppler` on `/points_all`).

- `densify_core.py` — the math (pure numpy, no ROS): ego-motion from
  Doppler, static/dynamic split, voxel evidence accumulation,
  information-form fusion, PLY export.
- `radar_densify_node.py` — live ROS 2 node (defaults = this rig's
  2026-08-19 radar↔lidar extrinsics).
- `densify_offline.py` — the same pipeline over a recorded rosbag2.

---

## Can their method work on point clouds at all?

Stage by stage — two of the three carry over, one cannot:

| DREAM-PCD stage (raw data) | point-cloud domain (this tool) |
|---|---|
| **Ego-motion compensation** — align frames before integrating | **Yes, fully.** The Doppler field of the static world encodes the sensor's velocity: for a static return with unit direction `u`, `doppler = −(u·v_sensor)`. Fit `v` by RANSAC over the frame (Kellner-style) — that *is* "removing ego speed", and it needs no raw data. With all three radars jointly it upgrades to a full 6-DOF **ego twist** `(v, ω)`: each radar `k` at extrinsic `(R_k, t_k)` contributes rows `u^T R_kᵀ [I  −[t_k]×]`, and the differing lever arms and look directions make the rotation rate observable — one radar alone never can. |
| **Non-Coherent Accumulation (NCA)** — sum power over ego-compensated frames to densify + raise SNR | **Yes, analogously.** Detections replace power bins: every frame's static points are transformed into one world frame and accumulated on a **voxel evidence grid**. A voxel must be hit in ≥ `min_frames` *distinct* frames to be emitted. That threshold is the non-coherent integration gain: real structure is re-detected frame after frame, while CFAR flicker noise and most multipath ghosts appear once and die. Density grows with observation time exactly as NCA's does. |
| **Synthetic Aperture Accumulation (SAA)** — coherently combine echoes across positions for angular resolution | **No — not coherently.** CFAR detection discards the **phase**; without phase there is nothing to combine coherently, and SAA also needs the antenna position known to ~λ/4 (≈1 mm at 60 GHz). Once the chip has output points, that information is unrecoverable. Two substitutes recover part of the benefit (below). |
| **Learned denoiser** (their network, trained on their raw-derived dataset) | Replaced by classical filters: the evidence threshold above plus a 26-neighbour radius outlier filter. (Their network is trained on raw-data-derived inputs, so it does not transfer to this representation anyway.) |

### The two aperture substitutes

1. **Three radars = one physically larger (sparse) aperture.** The
   calibration gave `T_os_lidar_radarN` to ~cm/degree accuracy, so all three
   clouds fuse into a single map. radar1/2 are mounted rolled relative to
   each other, so their *weak axes are perpendicular* — where their fields of
   view overlap, each constrains the axis the other measures poorly. This is
   an aperture increase you get for free and DREAM-PCD (single radar)
   doesn't have.

2. **Viewpoint diversity + anisotropic fusion.** Each detection is given the
   rig's measured noise model — σ_range = 5 cm along the ray,
   `r·σ_az` (3°) and `r·σ_el` (8°) across it — as a 3×3 information matrix,
   and voxels fuse points in information form (`p̂ = (ΣΛ)⁻¹ Ση`). As the rig
   moves, the same surface is seen from different angles; the error
   ellipsoids **cross**, and their intersection is tighter in cross-range
   than any single look — which is what a synthetic aperture buys, obtained
   *statistically* instead of coherently. The gain is √N-ish rather than
   SAA's dramatic beam sharpening, but it is the honest ceiling for
   detection-level data.

**Bottom line:** yes — "remove ego speed, then accumulate frames" is exactly
right and is implemented here; "increase antenna aperture" is the one step
that cannot be ported coherently, and is substituted by multi-radar fusion +
motion-diversity weighted averaging. If you ever need true SAA resolution,
you must record raw ADC (DCA1000 capture card on the IWR6843) — no
processing of `/points_all` can reconstruct it.

---

## Pipeline

Per radar frame:

1. **Gate** range `[min_range, max_range]` and optional `min_snr`.
2. **Ego velocity** from Doppler (RANSAC, `static_doppler_thresh` = 0.15 m/s
   inlier band). A stationary rig is detected and shortcut (`v = 0`).
3. **Static / dynamic split** — inliers of the fit are static; walking
   people, other robots, etc. are rejected *before* mapping (published on
   `~/dynamic_points` for inspection).
4. **Pose** of the base at the cloud stamp (`ego_mode`, below), then
   transform static points radar → base (calibrated extrinsic) → map.
5. **Accumulate** on the voxel evidence grid with per-detection anisotropic
   information (SNR-weighted).

On output (periodic `~/map`, or `~/save` → PLY):

6. Keep voxels seen in ≥ `min_frames` distinct frames (and ≥ `min_hits`
   detections), fuse each voxel's position in information form (sub-voxel,
   *not* snapped to the grid), then drop voxels with < `min_neighbors`
   occupied 26-neighbours.

### Where the pose comes from (`ego_mode`)

| mode | source | when |
|---|---|---|
| `tf` (default) | TF `map_frame ← base_frame` (GLIM / lidar odometry) | whenever the Ouster runs — best accuracy, radar error stays the only error |
| `odom` | a `nav_msgs/Odometry` topic | odometry without TF |
| `doppler` | dead reckoning from the fitted ego twist (6-DOF with ≥2 radars in view of enough static world; translation-only fallback) | radar-only operation. Drift is unbounded — good for short sweeps and for proving the radar-only case |
| `static` | identity | rig on a tripod; you still get densification + denoising from pure integration |

Doppler sign: TI reports range-rate **positive for receding**, so
`doppler_sign` defaults to `-1`. Classification is sign-invariant; the sign
only affects the *direction* of dead-reckoned motion (if your `doppler`
mode map comes out mirrored along the direction of travel, flip it to `+1`).

---

## Run

```bash
pip install -r requirements.txt      # numpy (ROS 2 provides the rest)

python3 radar_densify_node.py --ros-args \
  -p ego_mode:=tf -p map_frame:=odom -p base_frame:=os_lidar \
  -p pc_field_snr:=intensity \
  -p voxel_m:=0.10 -p min_frames:=3

ros2 topic pub -1 /radar_densify/save  std_msgs/msg/Empty "{}"   # → PLY
ros2 topic pub -1 /radar_densify/reset std_msgs/msg/Empty "{}"
```

RViz: add `~/map` (colour by `frames` to see the evidence), `~/static_points`
and `~/dynamic_points` to check the split is right — a person walking through
must show up dynamic (red-flag if not: `static_doppler_thresh` too loose, or
the person moves tangentially, where Doppler is blind — the evidence
threshold still removes them from the map since they don't persist per voxel).

Offline, over a bag:

```bash
python3 densify_offline.py my_bag/ --odom-topic /glim/odom -o dense_map.ply
python3 densify_offline.py my_bag/ --ego-mode doppler -o dense_map.ply   # radar-only
python3 densify_offline.py my_bag/ --ego-mode static \
    --radar radar1:/radar1/radar/points_all -o one_radar_static.ply
```

(For `ego_mode:=tf` there is no offline path — play the bag and run the live
node; TF interpolation belongs to tf2.)

### Parameters

| param | default | meaning |
|---|---|---|
| `radarN_enable` / `radarN_topic` | true / `/radarN/radar/points_all` | any subset of the three radars |
| `radarN_t_xyz`, `radarN_quat_xyzw` | 2026-08-19 solves | `T_base_radar` (base = `os_lidar`) |
| `pc_field_snr` / `pc_field_doppler` | `intensity` / `doppler` | field names; a missing doppler field disables the split (everything treated static) |
| `min_range` / `max_range` / `min_snr` | 0.3 / 25 / 0 | detection gates |
| `ego_mode` | `tf` | `tf` \| `odom` \| `doppler` \| `static` |
| `map_frame` / `base_frame` | `odom` / `os_lidar` | accumulation frame / rig frame |
| `static_doppler_thresh` | 0.15 m/s | ego-fit inlier band = static/dynamic boundary |
| `doppler_sign` | −1 | TI convention (see above) |
| `sigma_range_m` / `sigma_az_deg` / `sigma_el_deg` | 0.05 / 3 / 8 | the per-detection noise model (same numbers as the calibration) |
| `voxel_m` | 0.10 | evidence/fusion voxel |
| `min_frames` | 3 | distinct-frame evidence threshold — **the** denoise knob |
| `min_hits` | 0 | optional extra threshold on raw detection count |
| `min_neighbors` | 2 | 26-neighbourhood outlier filter (0 = off) |
| `output_path` | `radar_dense_map.ply` | `~/save` target |
| `twist_sync_s` | 0.10 s | max stamp skew between radars entering the joint twist |

**Tuning:** `min_frames=3` at 10–20 fps assumes tens of seconds of data; for
a very short sweep (< ~5 s) drop it to 2. Raise it (5–8) in heavy-multipath
rooms. Larger `voxel_m` = denser-looking, smoother, less detail; 0.10 m is a
good match for this rig's noise at indoor ranges.

---

## Validation (synthetic, matching the rig's noise model)

- Ego velocity: recovered to **3 mm/s** with 20 % dynamic outliers in frame;
  static/dynamic split accuracy 100 %.
- Joint 3-radar twist (using this rig's actual extrinsics): linear velocity
  to 15 mm/s, angular rate to ~5°/s (the ~15 cm lever arms bound it — for
  precise rotation use `ego_mode:=tf`).
- Accumulation, 40 true scatterers on a wall, 5 random multipath ghosts per
  frame, σ_el = 8°: after 200 frames the map holds ~1000 fused points
  (≈25× densification), plane error below the single-look cross-range
  noise, **zero** ghosts surviving `min_frames=3` + neighbour filter.

## Alternatives, for the record

- **Raw-data route** (what DREAM-PCD actually runs, incl. real SAA): needs a
  DCA1000EVM per radar to stream ADC; heavy but the only way to beat the
  chip's angular resolution physically.
- **Learned point-cloud upsampling / diffusion enhancement**: recent works
  (e.g. range-image diffusion enhancers) mostly still condition on raw
  spectra or need paired lidar ground truth for training — with the Ouster
  on this rig you *could* self-collect such pairs later; this pipeline then
  becomes the input-side baseline.
- **Radar-inertial odometry** (EKF over the Doppler ego-velocity + IMU) would
  fix `doppler` mode's rotation drift without the lidar; worth it only if
  the Ouster ever leaves the rig.
