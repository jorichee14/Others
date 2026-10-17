# analysis/odom

Trajectory analysis; reads outputs, writes nothing back.

- `zed_vs_lidar.py <pipeline config> [--at <stamp> ...] [--every 60]` -- how far the
  ZED's own tracking (its map_zed, from the bag's /tf, placed by 05's map -> map_zed)
  drifts from the LiDAR trajectory over a run, as the ZED left optical frame in map.
  `--at` reports chosen stamps too, e.g. an infra camera extrinsic yaml's stamp: its
  ChArUco `pose_in_map` is in map_zed, so it carries that drift.
- `compare_trajs.py <pipeline config> [--traj NAME=FILE ...] [--plot out.png]` -- GLIM's,
  02's and 02b's trajectories of a pass (odometry/<machine>/) against each other: position
  and rotation difference as they are and after a rigid fit, the difference of the motion
  over 1 s / 10 s / 100 s (short-term vs drift), and each one's loop closure on the boards.
