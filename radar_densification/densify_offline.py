#!/usr/bin/env python3
"""
Offline densification: rosbag2 -> dense static point cloud (PLY).

Same pipeline as radar_densify_node.py (ego-speed removal by Doppler RANSAC,
static extraction, voxel evidence accumulation, information-form fusion,
outlier filter) but batch, over a recorded bag — repeatable parameter sweeps
without replaying.

Ego pose comes from --odom-topic (nav_msgs/Odometry of base in map),
--ego-mode doppler (radar-only dead reckoning; joint 6-DOF twist when
several radars are enabled), or --ego-mode static. For a TF-based pose
(GLIM), play the bag and run the live node instead.

Example (this rig, three radars, lidar odometry recorded as /glim/odom):

  python3 densify_offline.py my_bag/ --odom-topic /glim/odom \
      --voxel 0.10 --min-frames 3 -o dense_map.ply
"""

import argparse
import sys

import numpy as np

from densify_core import (VoxelEvidenceMap, detection_information,
                          estimate_base_twist, estimate_sensor_velocity,
                          quat_to_R, radius_outlier_filter,
                          rotate_information, save_ply, so3_exp)
from radar_densify_node import DEFAULT_RADARS


def read_bag(path, topics):
    """Yield (topic, stamp_sec, msg) in time order for the given topics."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        print(f"warning: topics not in bag: {missing}", file=sys.stderr)
    reader.set_filter(rosbag2_py.StorageFilter(
        topics=[t for t in topics if t in types]))
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        msg = deserialize_message(raw, get_message(types[topic]))
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        t = stamp.sec + stamp.nanosec * 1e-9 if stamp else t_ns * 1e-9
        yield topic, t, msg


def cloud_to_arrays(msg, f_snr, f_dop):
    from sensor_msgs_py import point_cloud2 as pc2
    have = {f.name for f in msg.fields}
    opt = [f for f in (f_snr, f_dop) if f in have]
    rows = np.array([tuple(p) for p in pc2.read_points(
        msg, field_names=["x", "y", "z"] + opt, skip_nans=True)], float)
    if rows.size == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    n = len(rows)
    snr = rows[:, 3 + opt.index(f_snr)] if f_snr in opt else np.zeros(n)
    dop = rows[:, 3 + opt.index(f_dop)] if f_dop in opt else np.zeros(n)
    return rows[:, :3], snr, dop


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("bag", help="rosbag2 directory")
    ap.add_argument("-o", "--output", default="radar_dense_map.ply")
    ap.add_argument("--radar", action="append", metavar="NAME:TOPIC",
                    help="override/limit radars, e.g. radar1:/radar1/radar/"
                         "points_all (repeatable; default: all three)")
    ap.add_argument("--ego-mode", choices=["odom", "doppler", "static"],
                    default="odom")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--pc-field-snr", default="intensity")
    ap.add_argument("--pc-field-doppler", default="doppler")
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--min-snr", type=float, default=0.0)
    ap.add_argument("--static-thresh", type=float, default=0.15,
                    help="doppler residual (m/s) below which a return is static")
    ap.add_argument("--doppler-sign", type=float, default=-1.0)
    ap.add_argument("--sigma-range", type=float, default=0.05)
    ap.add_argument("--sigma-az", type=float, default=3.0)
    ap.add_argument("--sigma-el", type=float, default=8.0)
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--min-hits", type=int, default=0)
    ap.add_argument("--min-neighbors", type=int, default=2)
    args = ap.parse_args()

    radars = {}
    if args.radar:
        for spec in args.radar:
            name, topic = spec.split(":", 1)
            d = DEFAULT_RADARS.get(name)
            if d is None:
                ap.error(f"unknown radar '{name}' (no default extrinsic); "
                         f"known: {list(DEFAULT_RADARS)}")
            radars[name] = {"topic": topic, "R": quat_to_R(np.array(d["q"])),
                            "t": np.array(d["t"])}
    else:
        radars = {n: {"topic": d["topic"], "R": quat_to_R(np.array(d["q"])),
                      "t": np.array(d["t"])} for n, d in DEFAULT_RADARS.items()}
    topic_to_radar = {r["topic"]: n for n, r in radars.items()}

    topics = list(topic_to_radar)
    if args.ego_mode == "odom":
        topics.append(args.odom_topic)

    vmap = VoxelEvidenceMap(args.voxel)
    frame_id = 0
    pose = (np.eye(3), np.zeros(3))       # current T_map_base
    have_odom = args.ego_mode != "odom"
    last = {}                             # radar -> freshest frame (doppler)
    last_twist_stamp = None
    n_static = n_dynamic = 0

    for topic, t, msg in read_bag(args.bag, topics):
        if topic == args.odom_topic and args.ego_mode == "odom":
            q = msg.pose.pose.orientation
            p0 = msg.pose.pose.position
            pose = (quat_to_R([q.x, q.y, q.z, q.w]),
                    np.array([p0.x, p0.y, p0.z]))
            have_odom = True
            continue

        name = topic_to_radar[topic]
        r = radars[name]
        pts, snr, dop = cloud_to_arrays(msg, args.pc_field_snr,
                                        args.pc_field_doppler)
        rng = np.linalg.norm(pts, axis=1)
        keep = (rng >= args.min_range) & (rng <= args.max_range)
        if args.min_snr > 0:
            keep &= snr >= args.min_snr
        pts, snr, dop = pts[keep], snr[keep], dop[keep]
        if len(pts) == 0:
            continue

        if args.ego_mode == "doppler":
            last[name] = {"pts": pts, "doppler": dop, "R": r["R"], "t": r["t"],
                          "stamp": t}
            fresh = [f for f in last.values() if t - f["stamp"] <= 0.10]
            v, w, _ = estimate_base_twist(fresh, thresh=args.static_thresh)
            if v is None and fresh:
                v3, _ = estimate_sensor_velocity(fresh[0]["pts"],
                                                 fresh[0]["doppler"],
                                                 thresh=args.static_thresh)
                if v3 is not None:
                    v, w = fresh[0]["R"] @ v3, np.zeros(3)
            if v is not None and last_twist_stamp is not None:
                dt = t - last_twist_stamp
                if 0.0 < dt < 0.5:
                    Rwb, twb = pose
                    twb = twb + Rwb @ (args.doppler_sign * v * dt)
                    Rwb = Rwb @ so3_exp(args.doppler_sign * w * dt)
                    pose = (Rwb, twb)
            last_twist_stamp = t
        if not have_odom:
            continue                      # wait for the first odom message

        _, static = estimate_sensor_velocity(pts, dop,
                                             thresh=args.static_thresh)
        Rwb, twb = pose
        Rwr, twr = Rwb @ r["R"], Rwb @ r["t"] + twb
        pw = pts[static] @ Rwr.T + twr
        if len(pw):
            Lam = rotate_information(
                detection_information(pts[static], args.sigma_range,
                                      args.sigma_az, args.sigma_el,
                                      snr=snr[static]), Rwr)
            vmap.add(pw, Lam, snr[static], frame_id)
        n_static += int(static.sum())
        n_dynamic += int((~static).sum())
        frame_id += 1

    out_pts, attrs = vmap.extract(args.min_frames, args.min_hits)
    out_pts, attrs = radius_outlier_filter(out_pts, attrs, args.voxel,
                                           args.min_neighbors)
    save_ply(args.output, out_pts, attrs)
    print(f"{frame_id} radar frames | {n_static} static / {n_dynamic} dynamic "
          f"detections | {len(vmap)} voxels touched | "
          f"{len(out_pts)} points kept -> {args.output}")


if __name__ == "__main__":
    main()
