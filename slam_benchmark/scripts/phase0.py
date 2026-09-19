#!/usr/bin/env python3
"""Phase 0, in one command, against the bag.

Everything the plan's stage 0 can extract from the recording itself:

  A  both reference trajectories -> TUM, with the expected_start refusal
  B  the recorded baselines -> TUM  (mobile_1: zed odom + zed pose;
     mobile_2: Isaac cuVSLAM odometry + vo_pose)
  C  board dwell-window CANDIDATES (I2) from each reference trajectory
  D  /tf_static dumped in full; the ZED camera<->IMU chain flagged (I13)
  E  gate B0: place overlap between the agents; closest approach
  F  infra_1: visibility fraction per agent + bearing/range residuals of the
     dynamic radar returns against what each reference predicts (I9's check)
  G  radar field layouts: is there a Doppler field, and under what NAME (I12)

Not here: depth_health.py (run separately, per depth topic), the overnight IMU
log (hardware), the Doppler SIGN (needs G's field name first — the check is
printed at the end).

    python3 scripts/phase0.py --config configs/coop2.yaml \
        --bag ~/path/to/mirc_dataset_coop2_20260828_completed \
        --out runs/phase0

Read the report bottom-up: the last section says which phase-0 gates passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "rosbag_to_opv2v"))

from slambench import Trajectory, load_dataset, save_tum, se3           # noqa: E402
from slambench.collab import (place_overlap, predict_observation,       # noqa: E402
                              observation_residual)

DATATYPE = {1: "int8", 2: "uint8", 3: "int16", 4: "uint16",
            5: "int32", 6: "uint32", 7: "float32", 8: "float64"}
DOPPLERISH = ("velocity", "doppler", "v_r", "vr", "radial", "vel")


# ---------------------------------------------------------------- pure helpers
# (kept import-safe and bag-free so the self-test can drive them)

def pose_topic_to_traj(rows, name):
    """[(stamp_s, t(3), q_xyzw(4))] -> Trajectory, sorted, strictly increasing."""
    if not rows:
        raise ValueError(f"{name}: no messages")
    stamps = np.array([r[0] for r in rows], dtype=np.float64)
    order = np.argsort(stamps, kind="stable")
    stamps = stamps[order]
    poses = np.stack([se3.pose_from_quat(np.asarray(rows[i][1], dtype=float),
                                         np.asarray(rows[i][2], dtype=float))
                      for i in order])
    keep = np.concatenate([[True], np.diff(stamps) > 0])
    return Trajectory(stamps[keep], poses[keep], name=name)


def find_dwells(traj: Trajectory, board_xyz, radius_m=2.0, vmax_mps=0.15,
                min_duration_s=2.0):
    """Candidate dwell windows: near the board AND slow, long enough to count.

    CANDIDATES, not answers — I2 is closed by a person looking at each window,
    because 'slow and near the board' also describes driving past it carefully.
    """
    p = traj.positions
    t = traj.stamps
    if len(t) < 3:
        return []
    v = np.zeros(len(t))
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    v[1:] = step / np.maximum(dt, 1e-6)
    v[0] = v[1]
    near = np.linalg.norm(p - np.asarray(board_xyz, dtype=float), axis=1) <= radius_m
    slow = v <= vmax_mps
    mask = near & slow
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if t[j - 1] - t[i] >= min_duration_s:
                out.append({"t_start": float(t[i]), "t_end": float(t[j - 1]),
                            "duration_s": float(t[j - 1] - t[i]),
                            "mean_dist_m": float(np.mean(np.linalg.norm(
                                p[i:j] - np.asarray(board_xyz, float), axis=1)))})
            i = j
        else:
            i += 1
    return out


def fields_of(msg):
    """PointCloud2 field layout as [(name, typename)], plus Doppler candidates."""
    layout = [(str(f.name), DATATYPE.get(int(f.datatype), f"?{f.datatype}"))
              for f in msg.fields]
    doppler = [n for n, _ in layout if any(k in n.lower() for k in DOPPLERISH)]
    return layout, doppler


def xyz_of(msg, max_points=100000):
    """PointCloud2 -> (N,3) float64, non-finite rows dropped.

    Reads through a structured dtype whose itemsize is the cloud's point_step,
    the same construction ros2opv2v/pointclouds.py uses — one view, no loop.
    """
    fields = {str(f.name): (int(f.offset), int(f.datatype)) for f in msg.fields}
    npdt = {7: "f4", 8: "f8"}
    names, formats, offsets = [], [], []
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise ValueError(f"cloud has no '{axis}' field: {sorted(fields)}")
        off, dt = fields[axis]
        if dt not in npdt:
            raise ValueError(f"field '{axis}' is {DATATYPE.get(dt, dt)}, not float")
        names.append(axis); formats.append(npdt[dt]); offsets.append(off)
    step = int(msg.point_step)
    dtype = np.dtype({"names": names, "formats": formats,
                      "offsets": offsets, "itemsize": step})
    buf = bytes(msg.data)
    n = min(len(buf) // step if step else 0, max_points)
    a = np.frombuffer(buf, dtype=dtype, count=n)
    out = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    return out[np.isfinite(out).all(axis=1)]


def gated_residual(det_az, det_rng, pred_az, pred_rng, az_gate_deg=15.0):
    """Detections matched to ONE agent's prediction by an azimuth gate.

    Gating by the prediction biases the residual MAGNITUDE toward zero, so the
    number this supports is the residual's SIGN and trend, not its absolute
    size — which is exactly what the phase-0 question needs: a consistent-sign
    bearing residual means the node's pose or extrinsic is wrong, and a range
    residual growing with distance means the trajectory drifts along the line
    of sight. State the caveat wherever the numbers are quoted.
    """
    take = np.abs(((det_az - pred_az + 180.0) % 360.0) - 180.0) <= az_gate_deg
    if not take.any():
        return None
    return {"n": int(take.sum()),
            "azimuth_deg": float(np.median(((det_az[take] - pred_az + 180.0) % 360.0) - 180.0)),
            "range_m": float(np.median(det_rng[take] - pred_rng))}


# ------------------------------------------------------------------- the run

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", default="runs/phase0")
    ap.add_argument("--dwell-radius-m", type=float, default=2.0)
    ap.add_argument("--dwell-vmax", type=float, default=0.15)
    ap.add_argument("--b0-radius-m", type=float, default=5.0)
    args = ap.parse_args()

    from ros2opv2v.bagreader import BagReader

    cfg = load_dataset(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = BagReader(str(Path(args.bag).expanduser()), stamp_source="header")
    report = {}

    pose_topics = {
        "reference/mobile_1": "/mobile_1/global_pose",
        "reference/mobile_2": "/mobile_2/global_pose",
        "baseline/zed_odom": "/mobile_1/zed/odom",
        "baseline/zed_pose": "/mobile_1/zed/pose",
        "baseline/cuvslam_odom": "/mobile_2/visual_slam/tracking/odometry",
        "baseline/cuvslam_vo_pose": "/mobile_2/visual_slam/tracking/vo_pose",
    }
    rows = {k: [] for k in pose_topics}
    by_topic = {v: k for k, v in pose_topics.items()}
    radar_topics = ["/mobile_1/radar1/radar/points_all",
                    "/mobile_1/radar2/radar/points_all",
                    "/infra_1/radar/points_all"]
    radar_fields = {}
    infra_dyn = []          # (stamp_s, xyz array in the infra radar frame)
    tf_static = []

    wanted = list(pose_topics.values()) + radar_topics + \
        ["/infra_1/radar/points_dynamic", "/tf_static"]
    for topic, stamp_ns, msg in reader.iter_messages(wanted):
        t = stamp_ns * 1e-9
        if topic in by_topic:
            p = getattr(msg, "pose", msg)
            p = getattr(p, "pose", p)
            rows[by_topic[topic]].append(
                (t, (p.position.x, p.position.y, p.position.z),
                 (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)))
        elif topic in radar_topics:
            if topic not in radar_fields:
                radar_fields[topic] = fields_of(msg)
        elif topic == "/infra_1/radar/points_dynamic":
            try:
                infra_dyn.append((t, xyz_of(msg)))
            except ValueError as e:
                infra_dyn.append((t, str(e)))
        elif topic == "/tf_static":
            for tr in msg.transforms:
                key = (str(tr.header.frame_id), str(tr.child_frame_id))
                if any((t["parent"], t["child"]) == key for t in tf_static):
                    continue          # two latched publishers, one transform
                tf_static.append({
                    "parent": str(tr.header.frame_id), "child": str(tr.child_frame_id),
                    "xyz": [tr.transform.translation.x, tr.transform.translation.y,
                            tr.transform.translation.z],
                    "quat_xyzw": [tr.transform.rotation.x, tr.transform.rotation.y,
                                  tr.transform.rotation.z, tr.transform.rotation.w]})

    # ---- A + B: trajectories out, with the start check on the references
    trajs = {}
    print("\n== A/B: trajectories ==")
    for name in pose_topics:
        try:
            trajs[name] = pose_topic_to_traj(rows[name], name)
        except ValueError as e:
            print(f"  {name:24s} MISSING ({e})")
            continue
        tr = trajs[name]
        path = out / (name.replace("/", "_") + ".tum")
        save_tum(tr, path)
        line = f"  {name:24s} {len(tr):5d} poses  {tr.duration:6.1f} s  {tr.path_length():7.2f} m -> {path}"
        agent = name.split("/")[1] if name.startswith("reference/") else None
        if agent:
            exp = cfg.agent_reference(agent).get("expected_start")
            d = float(np.linalg.norm(tr.positions[0] - np.asarray(exp, float)))
            line += f"   start {d * 1e3:6.1f} mm from anchor {'OK' if d < 0.25 else '** FAIL **'}"
            report[f"start_check/{agent}"] = d
        print(line)

    # ---- C: dwell candidates
    print("\n== C: board dwell candidates (I2) — confirm each by eye ==")
    boards = {a["name"]: a["position"] for a in cfg.raw["reference"]["anchors"]}
    dwells = {}
    for agent in ("mobile_1", "mobile_2"):
        tr = trajs.get(f"reference/{agent}")
        if tr is None:
            continue
        for bname, bpos in boards.items():
            w = find_dwells(tr, bpos, args.dwell_radius_m, args.dwell_vmax)
            dwells[f"{agent}/{bname}"] = w
            t0 = tr.stamps[0]
            desc = ", ".join(f"[t+{x['t_start']-t0:.1f}s..t+{x['t_end']-t0:.1f}s"
                             f" {x['duration_s']:.1f}s @{x['mean_dist_m']:.2f}m]" for x in w) or "none"
            print(f"  {agent:9s} @ {bname:10s} {desc}")
    report["dwell_candidates"] = dwells

    # ---- D: tf_static
    print(f"\n== D: /tf_static — {len(tf_static)} transforms ==")
    for tr in tf_static:
        r, p_, y = se3.rot_to_rpy(se3.quat_to_rot(np.asarray(tr["quat_xyzw"])))
        print(f"  {tr['parent']:32s} -> {tr['child']:32s} "
              f"xyz=({tr['xyz'][0]:+.4f},{tr['xyz'][1]:+.4f},{tr['xyz'][2]:+.4f}) "
              f"rpy=({r:+.2f},{p_:+.2f},{y:+.2f})deg")
    def _zed_imu(t):
        # 'imu' alone is not enough: os_imu (the Ouster's, excluded) matched it
        # on the real bag and produced a false YES. Caught by the first run.
        return any("zed" in t[k].lower() and "imu" in t[k].lower()
                   for k in ("parent", "child"))
    imu_chain = [t for t in tf_static if _zed_imu(t)]
    print(f"  ZED IMU chain in tf_static: "
          f"{'YES — I13 recoverable from the bag' if imu_chain else 'NO — use the SDK or SN*.conf (I13)'}")
    report["tf_static"] = tf_static

    # ---- E: gate B0
    print("\n== E: gate B0 — place overlap ==")
    if "reference/mobile_1" in trajs and "reference/mobile_2" in trajs:
        ov = place_overlap(trajs["reference/mobile_1"], trajs["reference/mobile_2"],
                           radius_m=args.b0_radius_m)
        print("  " + ov["verdict"])
        report["b0"] = {k: v for k, v in ov.items() if k != "verdict"}

    # ---- F: infra visibility + residuals
    print("\n== F: infra_1 — visibility and residuals ==")
    st = cfg.stream("infra_1.radar")
    T_map_cam = se3.pose_from_rpy(**st["static_world_pose"])
    T_cam_radar = se3.pose_from_rpy(**st["reference_frame_from_sensor"])
    T_map_radar = T_map_cam @ T_cam_radar
    good_sweeps = [(t, xyz) for t, xyz in infra_dyn if not isinstance(xyz, str) and len(xyz) >= 1]
    print(f"  dynamic sweeps: {len(infra_dyn)} total, {len(good_sweeps)} with points")
    for agent in ("mobile_1", "mobile_2"):
        tr = trajs.get(f"reference/{agent}")
        if tr is None or not good_sweeps:
            continue
        pred = predict_observation(tr, T_map_radar, axes="ros")   # infra1_link: body frame
        vis, res_az, res_rng, res_t = 0, [], [], []
        for t, xyz in good_sweeps:
            i = int(np.argmin(np.abs(pred["stamps"] - t)))
            if abs(pred["stamps"][i] - t) > 0.15:
                continue
            # collab's 'ros' convention: fwd=x, right=-y, az=atan2(-right,fwd)=atan2(y,x)
            det_az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
            det_rng = np.linalg.norm(xyz, axis=1)
            g = gated_residual(det_az, det_rng, pred["azimuth_deg"][i], pred["range_m"][i])
            if g:
                vis += 1
                res_az.append(g["azimuth_deg"]); res_rng.append(g["range_m"]); res_t.append(t)
        frac = vis / max(len(good_sweeps), 1)
        print(f"  {agent}: matched in {vis}/{len(good_sweeps)} sweeps ({frac:.0%})")
        if res_az:
            az = np.array(res_az); rng = np.array(res_rng)
            print(f"    bearing residual  median {np.median(az):+6.2f} deg  "
                  f"p10..p90 [{np.percentile(az,10):+.2f},{np.percentile(az,90):+.2f}]  "
                  f"{'SYSTEMATIC OFFSET — suspect infra pose/extrinsic/clock' if abs(np.median(az)) > 2 else 'centred — ok'}")
            print(f"    range residual    median {np.median(rng):+6.2f} m   "
                  f"p10..p90 [{np.percentile(rng,10):+.2f},{np.percentile(rng,90):+.2f}]")
            report[f"infra/{agent}"] = {"visible_fraction": frac,
                                        "az_median_deg": float(np.median(az)),
                                        "rng_median_m": float(np.median(rng))}

    # ---- G: radar fields
    print("\n== G: radar Doppler fields (I12) ==")
    for topic in radar_topics:
        if topic not in radar_fields:
            print(f"  {topic}: NO MESSAGES READ")
            continue
        layout, dop = radar_fields[topic]
        print(f"  {topic}\n    fields: " + ", ".join(f"{n}:{t}" for n, t in layout))
        print(f"    doppler candidate(s): {dop if dop else 'NONE — tier 4 closes on this recording'}")
        report[f"fields/{topic}"] = {"layout": layout, "doppler": dop}

    (out / "phase0_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out/'phase0_report.json'}")
    print("\nstill manual after this run:")
    print("  * confirm each dwell candidate by eye and write it into configs/coop2.yaml (I2)")
    print("  * depth_health.py on both depth topics (covariates)")
    print("  * the 3 h static IMU log tonight (I14)")
    print("  * Doppler SIGN: decode the field G named on one known-motion segment; "
          "positive-approaching vs positive-receding flips the trajectory direction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
