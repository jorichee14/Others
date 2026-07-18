#!/usr/bin/env python3
"""Find the chair in the rosbag and report every timestamp it's visible.

Runs an object detector (YOLO, COCO class "chair") over the camera streams in the
bag and reports:
  * a CSV of every chair detection (topic, time, confidence, box, centre pixel),
  * the co-visibility windows where >= N sensors see the chair at once (the useful
    moments for the agreement check), and
  * a ready-to-paste ``--times ...`` list for extract_observations.py.

Optionally (``--emit-observations``) it goes end-to-end: at each suggested time it
takes the detected box centre, deprojects the aligned depth to a 3D point, looks
up the platform pose, and writes an observations.yaml directly — no clicking.

Needs ROS 2 (rosbag2_py, sensor_msgs, cv_bridge, sensor_msgs_py) + OpenCV +
ultralytics, all in your dataset environment. The co-visibility/scheduling logic
(the part with judgement in it) is pure and unit-tested in test_find_chair.py.

Examples:
  # just list where the chair is
  python3 find_chair.py mirc_dataset_20260706_complete --every-sec 0.3

  # go straight to a filled observations file
  python3 find_chair.py mirc_dataset_20260706_complete --every-sec 0.3 \\
      --lidar-roi 1.5 4.0 -1.0 1.0 -0.6 0.8 \\
      --emit-observations observations.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import yaml

# Pure helpers reused from the extractor (its heavy imports are deferred, so this
# import is cheap and ROS-free).
from extract_observations import (DEFAULT_TOPICS, read_bag, nearest,
                                   pose_to_dict, lidar_centroid)

# camera streams the detector runs on, and their sensor label / depth / pose
CAMERA_STREAMS = {
    "zed_image": dict(sensor="zed", depth="zed_depth", pose="zed_pose", K="ZED_LEFT"),
    "rs_image": dict(sensor="realsense", depth="rs_depth", pose="rs_pose", K="REALSENSE"),
    "arducam": dict(sensor="arducam", depth=None, pose=None, K="ARDUCAM"),
}


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested, no ROS / no ML)
# --------------------------------------------------------------------------- #
def _present(times_by_sensor, g, half):
    """Set of sensors with a detection within ``half`` of time ``g``."""
    return {s for s, ts in times_by_sensor.items()
            if any(abs(x - g) <= half for x in ts)}


def covisibility_windows(times_by_sensor: dict, bin_s: float = 0.5,
                         min_sensors: int = 2, required=()):
    """Intervals where the chair is co-visible.

    A grid instant qualifies iff it has ``>= min_sensors`` detections within
    ``bin_s/2`` AND every sensor in ``required`` is among them. So passing
    ``required=("zed","realsense","arducam")`` yields only spans where the chair
    is in *all three cameras at once* — the window's sensor set then holds for
    every instant in it, not just somewhere inside.

    Returns ``[(start_s, end_s, frozenset(sensors)), ...]`` sorted by start.
    """
    required = set(required)
    all_t = sorted(t for ts in times_by_sensor.values() for t in ts)
    if not all_t:
        return []
    half = bin_s / 2.0
    step = max(half, 1e-3)
    windows, cur = [], None
    g = all_t[0]
    while g <= all_t[-1] + 1e-9:
        present = _present(times_by_sensor, g, half)
        ok = len(present) >= min_sensors and required.issubset(present)
        if ok:
            if cur is None:
                cur = [g, g, set(present)]
            else:
                # with `required`, intersect so the reported set is what holds
                # everywhere in the window; without it, keep the union.
                cur[1] = g
                cur[2] = (cur[2] & present) if required else (cur[2] | present)
        elif cur is not None:
            windows.append((cur[0], cur[1], frozenset(cur[2])))
            cur = None
        g += step
    if cur is not None:
        windows.append((cur[0], cur[1], frozenset(cur[2])))
    return windows


def _coverage(times_by_sensor, g, half):
    return len(_present(times_by_sensor, g, half))


def suggest_times(times_by_sensor: dict, bin_s: float = 0.5, min_sensors: int = 2,
                  n: int = 6, required=()):
    """Pick up to ``n`` well-spread timestamps, drawn only from the co-visibility
    windows. Every returned instant is guaranteed to satisfy ``required`` (all
    listed sensors present) — not merely to fall inside a qualifying window."""
    required = set(required)
    half = bin_s / 2.0
    wins = covisibility_windows(times_by_sensor, bin_s, min_sensors, required)
    if not wins:
        return []
    lo, hi = wins[0][0], wins[-1][1]
    covered = [(a, b) for a, b, _ in wins]

    def in_window(t):
        return any(a <= t <= b for a, b in covered)

    def satisfies(t):
        return in_window(t) and required.issubset(_present(times_by_sensor, t, half))

    picks = []
    for i in range(n):
        target = lo + (hi - lo) * (i + 0.5) / n
        if not in_window(target):
            target = min((a for a, b in covered if a >= target), default=None) \
                or max(b for a, b in covered)
        cands = [t for t in np.arange(target - bin_s, target + bin_s, half / 2)
                 if satisfies(t)]
        if not cands:
            continue                       # no instant here meets `required`; skip
        best = max(cands, key=lambda t: (_coverage(times_by_sensor, t, half),
                                         -abs(t - target)))
        if all(abs(best - p) > half for p in picks):
            picks.append(round(float(best), 3))
    return sorted(picks)


# --------------------------------------------------------------------------- #
# Detector (YOLO / COCO chair) — env-specific
# --------------------------------------------------------------------------- #
class ChairDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.35):
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            sys.exit(f"find_chair needs 'ultralytics' (pip install ultralytics). {e}")
        self.model = YOLO(weights)
        self.conf = conf
        self.chair_ids = [i for i, n in self.model.names.items() if n == "chair"]

    def detect(self, bgr):
        """Return the highest-confidence chair box as (conf, cx, cy, x1,y1,x2,y2) or None."""
        res = self.model(bgr, verbose=False, conf=self.conf)[0]
        best = None
        for b in res.boxes:
            if int(b.cls) in self.chair_ids:
                c = float(b.conf)
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                if best is None or c > best[0]:
                    best = (c, (x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2)
        return best


def _robust_depth_in_box(depth, box, frac=0.5):
    """Median of valid depths in the central `frac` of the box (metres)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * frac / 2, (y2 - y1) * frac / 2
    xa, xb = int(cx - hw), int(cx + hw)
    ya, yb = int(cy - hh), int(cy + hh)
    patch = depth[max(0, ya):yb, max(0, xa):xb]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else None


# --------------------------------------------------------------------------- #
# Bag pass
# --------------------------------------------------------------------------- #
def scan_bag(bag, topics_map, every_sec, detector):
    """Detect the chair over the camera streams. Returns detections + buffers."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    bridge = CvBridge()
    wanted = {getattr_topic: key for key, getattr_topic in topics_map.items()}
    dets = {c["sensor"]: [] for c in CAMERA_STREAMS.values()}  # sensor -> [(t, det)]
    last_t = {}      # per camera topic, last sampled time
    buffers = {k: [] for k in DEFAULT_TOPICS}   # keep non-camera msgs for emit
    t0 = None

    for topic, t, data in read_bag(bag, wanted.keys()):
        t0 = t if t0 is None else t0
        rel = t - t0
        key = wanted[topic]
        if key in CAMERA_STREAMS:
            if rel - last_t.get(key, -1e9) < every_sec:
                continue
            last_t[key] = rel
            img = bridge.imgmsg_to_cv2(deserialize_message(data, Image),
                                       desired_encoding="bgr8")
            d = detector.detect(img)
            if d:
                dets[CAMERA_STREAMS[key]["sensor"]].append((rel, d))
        else:
            buffers[key].append((rel, deserialize_message(data, _emit_type(key))))
    return dets, buffers, t0


def _emit_type(key):
    from sensor_msgs.msg import Image, PointCloud2
    from geometry_msgs.msg import PoseStamped
    return {"zed_depth": Image, "rs_depth": Image, "lidar": PointCloud2,
            "zed_pose": PoseStamped, "rs_pose": PoseStamped}[key]


def emit_observations(times, dets, buffers, lidar_roi, required=()):
    """Build an observations dict from detections + depth/pose buffers.

    Any frame missing a chair pixel for a sensor in ``required`` is dropped, so
    every emitted frame really does see the chair in all the required cameras.
    """
    import dataset as ds
    from cv_bridge import CvBridge
    bridge = CvBridge()
    Kmap = {"zed": ds.ZED_LEFT, "realsense": ds.REALSENSE, "arducam": ds.ARDUCAM}
    required = set(required)

    frames, dropped = [], []
    for i, tt in enumerate(times):
        sensors = {}
        for key, meta in CAMERA_STREAMS.items():
            sensor = meta["sensor"]
            # nearest detection for this sensor to tt
            cand = [d for d in dets.get(sensor, []) if abs(d[0] - tt) <= 0.5]
            if not cand:
                continue
            _, det = min(cand, key=lambda d: abs(d[0] - tt))
            conf, cx, cy, x1, y1, x2, y2 = det
            entry = {"pixel": [round(cx, 1), round(cy, 1)]}
            if meta["depth"]:
                dep_msg = nearest(buffers[meta["depth"]], tt)
                if dep_msg is not None:
                    depth = bridge.imgmsg_to_cv2(dep_msg, "passthrough").astype(np.float32)
                    if depth.max() > 100:
                        depth /= 1000.0
                    d = _robust_depth_in_box(depth, (x1, y1, x2, y2))
                    if d:
                        cam = Kmap[sensor]
                        entry["point"] = [round(float(x), 4) for x in cam.deproject([cx, cy], d)[0]]
            if meta["pose"]:
                pose = nearest(buffers[meta["pose"]], tt)
                if pose is not None:
                    entry["pose_map"] = pose_to_dict(pose)
            sensors[sensor] = entry

        if lidar_roi:
            pc = nearest(buffers["lidar"], tt)
            if pc is not None:
                c = lidar_centroid(pc, lidar_roi)
                if c:
                    sensors["lidar"] = {"point": [round(v, 4) for v in c]}

        # enforce the "all required cameras see the chair" guarantee at emit time
        missing = [s for s in required if s not in sensors or "pixel" not in sensors[s]]
        if missing:
            dropped.append((round(float(tt), 3), missing))
            continue
        frames.append({"id": len(frames), "t": round(float(tt), 3), "sensors": sensors})

    if dropped:
        print("\n  dropped frames missing a required camera:")
        for tt, miss in dropped:
            print(f"    t={tt:.2f}s  missing {', '.join(miss)}")
    return {"frames": frames}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--every-sec", type=float, default=0.3,
                    help="sample each camera stream at most this often (s)")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--bin", type=float, default=0.5,
                    help="co-visibility time tolerance (s)")
    ap.add_argument("--min-sensors", type=int, default=2)
    ap.add_argument("--require", nargs="*", default=None,
                    help="sensors that MUST all see the chair at each chosen "
                         "instant. Default: all three cameras "
                         "(zed realsense arducam). Relax with e.g. "
                         "'--require zed realsense', or disable with '--require'.")
    ap.add_argument("--n-times", type=int, default=6)
    ap.add_argument("--csv", default="chair_detections.csv")
    ap.add_argument("--emit-observations", default=None,
                    help="also write a filled observations YAML here")
    ap.add_argument("--lidar-roi", type=float, nargs=6, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    for k, v in DEFAULT_TOPICS.items():
        ap.add_argument(f"--{k}-topic", default=v)
    args = ap.parse_args()

    topics_map = {k: getattr(args, f"{k}_topic") for k in DEFAULT_TOPICS}
    detector = ChairDetector(args.weights, args.conf)
    dets, buffers, _ = scan_bag(args.bag, topics_map, args.every_sec, detector)

    # CSV of everything
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "t_s", "conf", "cx", "cy", "x1", "y1", "x2", "y2"])
        for sensor, rows in dets.items():
            for t, d in rows:
                w.writerow([sensor, f"{t:.3f}", f"{d[0]:.3f}"] + [f"{v:.1f}" for v in d[1:]])
    print(f"\nwrote {sum(len(v) for v in dets.values())} detections -> {args.csv}")
    for sensor, rows in dets.items():
        if rows:
            ts = [t for t, _ in rows]
            print(f"  {sensor:10s}: {len(rows):4d} detections, t in [{min(ts):.1f}, {max(ts):.1f}] s")

    times_by_sensor = {s: [t for t, _ in rows] for s, rows in dets.items() if rows}

    # Default: require the chair in all three cameras at the chosen instant.
    all_cameras = [c["sensor"] for c in CAMERA_STREAMS.values()]
    required = all_cameras if args.require is None else args.require
    min_sensors = max(args.min_sensors, len(required))

    never = [s for s in required if not times_by_sensor.get(s)]
    if never:
        print(f"\n  WARNING: chair never detected in: {', '.join(never)} — no timestamp "
              f"can satisfy all required cameras.\n"
              f"  Check the topic/detector, or relax with --require "
              f"{' '.join(s for s in required if s not in never)}")

    req_txt = ", ".join(required) if required else "(none)"
    wins = covisibility_windows(times_by_sensor, args.bin, min_sensors, required)
    print(f"\nco-visibility windows (require: {req_txt}):")
    if not wins:
        print("  none — relax --require / --min-sensors or check topic names")
    for a, b, sset in wins:
        print(f"  [{a:6.1f} .. {b:6.1f}] s  ({b - a:4.1f}s)  all-present: {', '.join(sorted(sset))}")

    times = suggest_times(times_by_sensor, args.bin, min_sensors, args.n_times, required)
    if times:
        print(f"\nsuggested capture times (each sees the chair in: {req_txt}):")
        print("  --times " + " ".join(f"{t:.2f}" for t in times))

    if args.emit_observations and times:
        obs = emit_observations(times, dets, buffers, args.lidar_roi, required)
        with open(args.emit_observations, "w") as f:
            yaml.safe_dump(obs, f, sort_keys=False)
        print(f"\nwrote filled observations -> {args.emit_observations}")
        print("now run:  python3 agreement.py", args.emit_observations, "--lidar-interp auto")


if __name__ == "__main__":
    main()
