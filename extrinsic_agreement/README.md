# Extrinsic-agreement check via a shared static target (the chair)

Quantify **how much the `mirc_dataset_20260706` extrinsics agree with each
other**, using the one thing every sensor saw: the fixed chair.

## The principle

The chair is a single rigid point that is **static in the `map` frame**. If every
extrinsic (and pose) were perfect, then taking each sensor's *own* observation of
the chair and pushing it through that sensor's transform chain into `map` would
land them all on the **same point**. They don't. The **spread of those landing
points, in metres, is the extrinsic disagreement** — and the *pattern* of the
spread says which extrinsic is wrong.

Two complementary tests, both implemented here:

### 1. Rigid-rig test — the clean one (no trajectory, drift-immune)
ZED, LiDAR, radar1, radar2 are bolted to MP#1, so a rig-to-rig extrinsic is all
that separates them — **no pose is needed**. Compare the chair across the pair in
one shared sensor frame:

```
residual = p_chair^zed  −  T_zed_sensor · p_chair^sensor      (in the ZED frame)
```

The residual is *purely* that one extrinsic's error plus target-localization
noise. `/glim`, `/vo_pose`, and all drift cancel. **This is the definitive test
for the LiDAR↔ZED extrinsic you flagged.**

> **The LiDAR↔ZED sheet entry is self-contradictory.** The arrow reads
> `os_lidar → zed` (that is `T_zed_lidar`); the label reads `T_lidar_camera`
> (that is its *inverse*). They differ by an inversion — and using the wrong one
> throws the LiDAR cloud off by **metres** (exactly the "wonkiness" you saw).
> `--lidar-interp auto` runs both and reports which one makes the chair agree, so
> the direction stops being a guess.

### 2. Map-consensus test — the whole chain
Push every sensor's chair observation into `map` through its full chain, take a
robust per-axis-median consensus, and report each sensor's deviation:

```
ZED        P^map = T_map_zed(t) · p^zed                      # t from /glim/camera_pose
LiDAR      P^map = T_map_zed(t) · T_zed_lidar · p^lidar       # rides on the ZED pose
RealSense  P^map = T_map_rs(t)  · p^rs                        # t from /vo_pose (START/MIDDLE only)
Arducam    fixed T_map_arducam ; RGB-only → reproject the consensus into it
```

Each sensor's residual splits into:
- **signed bias** (a consistent offset) → **a real extrinsic error**,
- **scatter** (random spread) → target-localization noise, extrinsic is fine.

Aggregating over many frames is what separates the two. A single frame can't.

### 3. Reprojection cross-check (pixels)
Project the consensus chair point into each camera and compare to where the chair
was actually detected. This is the **only** handle on the fixed **Arducam** (RGB,
no depth): the **vertical (`v`) pixel bias is the direct symptom of the suspected
z error**.

> **Honest limit on the Arducam z.** The chair is a *single static point*, so a
> monocular fixed camera can only recover the **bearing** to it — you see the z
> error as a vertical pixel offset, but range and height trade off along the ray,
> so you can't solve the *metric* z from the chair alone. To pin the metric z, add
> **≥3 non-collinear static landmarks** (fill `extra_points` in the YAML — a table
> corner, a floor marking, etc., with their known `map` coordinates) so it becomes
> a proper PnP, or register the Arducam image against the uploaded aggregated
> point-cloud map. The chair gives you the symptom; the extra points give you the
> number.

## Files

| file | what it is | tested? |
|---|---|---|
| `dataset.py` | every intrinsic/extrinsic from the sheet, one place, explicit frames | via self-test |
| `se3.py` | SE(3) algebra with **one** stated convention (`T_a_b · p_b = p_a`) | via self-test |
| `agreement.py` | the three tests + a printed report; CLI over an observations YAML | via self-test |
| `test_agreement.py` | synthetic ground-truth self-test of the transform math | **run me** |
| `extract_observations.py` | interactive rosbag → observations YAML (needs ROS 2) | env-specific |
| `observations.example.yaml` | the input format (placeholder numbers) | — |

The **analysis math is unit-tested** (`test_agreement.py` builds a known chair,
synthesizes each sensor's view through the inverse transforms, and checks that
perfect extrinsics give exactly zero, an injected 40 mm rig error reads back as
40 mm, the wrong LiDAR direction is off by metres, and a 100 mm Arducam z error
shows up as a vertical pixel bias). The **rosbag extractor is ROS-2 glue** and
runs in your dataset environment, not in CI.

## Usage

```bash
pip install numpy scipy pyyaml            # analysis deps
python3 test_agreement.py                 # prove the math (no ROS needed)
```

```bash
# 1. Pull chair observations out of the bag at a few times where it's visible
#    (start/middle — where /vo_pose is still good and both platforms pass it):
python3 extract_observations.py mirc_dataset_20260706_complete \
    --times 12.5 15.0 21.3 28.0 \
    --lidar-roi  1.5 4.0  -1.0 1.0  -0.6 0.8 \    # os_lidar box around the chair
    --out observations.yaml

# 2. Quantify the agreement:
python3 agreement.py observations.yaml --lidar-interp auto
```

## Reading the report

- **Rigid-rig RMS** is the cleanest single number per rig extrinsic. A large
  *signed bias* is a real mounting error; large *scatter* with ~0 bias is just
  click/centroid noise.
- **Map-consensus** additionally folds in pose accuracy and the map anchors, so a
  sensor that's clean in the rigid-rig test but bad here implicates its *pose*
  (e.g. `/vo_pose` drift) or a *map-anchor* extrinsic, not the rig.
- **Reprojection** `v_bias` on the Arducam is your z symptom; add `extra_points`
  to turn the symptom into a metric correction.

## Practical notes

- **Same physical point.** All sensors must localize the *same* part of the chair.
  A segmented-chair centroid is only as repeatable as the segmentation; a
  distinctive stable feature (a specific leg's floor contact, a seat corner) is
  more consistent across viewpoints. Whatever you pick sets the noise floor — you
  can't claim sub-centimetre extrinsic agreement from a centimetre-level click.
- **Time sync.** For the moving platforms, use the pose at the *image timestamp*
  (`se3.slerp_pose` interpolates between pose samples if you need it).
- **`/vo_pose` drift.** Restrict RealSense frames to the start/middle, which is
  exactly where the chair is seen anyway.
- **Radar.** A chair is a weak, angularly-smeared radar target; radar1/2 rows are
  supported but expect them to be the noisiest, and treat a bad radar number as
  "radar can't see this target well," not necessarily a bad extrinsic.
- **Rotation observability.** One point mostly exercises translation (and rotation
  only through its lever arm). Capture the chair from several platform poses; if
  you fit a chair *orientation* too, `se3.rotation_angle_deg` compares it.
