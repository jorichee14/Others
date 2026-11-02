# Localization in ED305: built map vs the USD digital twin

**Question.** Does the Stretch 4 localize better against the map built
from its own lidars (2Fast-2Lamaa mapping) or against the fixed USD
digital twin of the room?

**Fair comparison.** Same robot, same recorded rosbag, same
localization algorithm and settings; only the map changes. Accuracy is
measured against ground truth from the ED305 camera network.

## Step 1: Map A, the built map

Drive the room once with the Stretch 4, run 2Fast-2Lamaa mapping,
save the point-cloud map.

## Step 2: Map B, the USD twin as a point cloud

2Fast-2Lamaa localizes against a point cloud, so the twin must be
converted to the same format.

1. Read the USD in Python (`pxr`, or Isaac Sim's own Python): every
   `UsdGeom.Mesh`, its vertices and faces, and its world transform.
2. Check units (`metersPerUnit`) and up axis (Isaac Sim is usually
   Z-up, many USD assets are Y-up; centimetres are common).
3. Sample points on the mesh surfaces (Open3D or trimesh) at the same
   density as Map A, then voxel-downsample with the same voxel size.
4. Remove objects that are not in the real room (if any).
5. Save as the same file type as Map A.

## Step 3: One common frame

Register Map A to Map B: a rough manual alignment (3 matching points),
then ICP. The result is the transform from the built map to the twin
frame. All poses below are reported in the twin frame.

The leftover cloud-to-mesh distance after ICP is itself a result:
where the real room differs from the twin (moved chairs, objects
added since the twin was made).

## Step 4: Ground truth from the camera network

1. Put an AprilTag board on top of the Stretch 4.
2. Calibrate the ED305 cameras (intrinsics; extrinsics in the twin
   frame, from tags placed at known positions in the twin).
3. Detect the board in every camera frame; fuse into the robot's pose
   at each time.
4. Sync camera and robot clocks (chrony, or a visible event at the
   start of the recording).

Simpler fallback: tape 8–10 marks on the floor, measure their positions
in the twin, and stop the robot on each; ground truth only at those
points.

## Step 5: Runs

Record 4–5 rosbags with ground truth running:
- a loop around the room;
- approach to the table and the pick area;
- a run after moving some furniture (where the twin is now wrong);
- a run with people walking.

Replay each bag through 2Fast-2Lamaa localization twice: once on Map A,
once on Map B.

## Step 6: Metrics

| Metric | What it shows |
|---|---|
| Absolute trajectory error (`evo_ape`) against ground truth | overall accuracy |
| Relative pose error (`evo_rpe`) | local drift and jumps |
| Error near the table / pick area | accuracy where manipulation starts |
| Error per region against the map–room difference map of step 3 | whether errors come from places where the twin is outdated |
| Localization failures and recovery | robustness |
| Time per scan, CPU | cost on the robot |

## What to expect

- The twin is clean and complete but fixed: it misses clutter and
  anything moved since it was made.
- The built map matches what the lidars actually see but is noisier
  and only covers where the robot drove.
- So the twin may win in an unchanged room and lose where the room has
  changed. The furniture-moved run shows this directly.
