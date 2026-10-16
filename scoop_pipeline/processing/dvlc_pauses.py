#!/usr/bin/env python3
"""Where a robot stood still -> one short bag per pause, for Koide's
direct_visual_lidar_calibration (DVLC).

Runs on the MERGED bag: Ouster packets decoded (decode_ouster.py), put on
capture time (retime.py) and merged with the camera, odometry and IMU topics
of the raw recording (merge_bags.py). See "mobile_2: camera-LiDAR
calibration at the pauses" in the README for those steps.

    python3 processing/dvlc_pauses.py <merged bag> <out_dir> [--list]
        [--odom /mobile_2/visual_slam/tracking/odometry] [--imu /mobile_2/imu]
        [--points /mobile_2/ouster/points]
        [--image /mobile_2/color/image_raw] [--camera-info /mobile_2/color/camera_info]
        [--min-pause 4] [--trim 1] [--max-len 10] [--only 0,2,5]
        [--intervals 120.5-130,300-310]

1. Finds the pauses: 0.1 s bins where, over +-0.25 s,
     VSLAM odometry speed  < --max-speed  (m/s, position change over +-0.5 s)
     IMU mean |gyro|       < --max-gyro   (rad/s)
     IMU accel std         < --max-acc-std (m/s^2, the drive's vibration)
   and runs of such bins at least --min-pause s long. Each run loses --trim s
   at both ends (the robot settling) and keeps at most --max-len s from its
   middle. --list prints the table and stops; --intervals (s from the bag's
   start) replaces the detection; --only keeps the pauses with those numbers.

2. Writes <out_dir>/pause_NN/ per pause, a rosbag2 (MCAP) folder DVLC's
   preprocess reads as one data set: <points>, <image> and <camera-info>
   logged inside the pause, copied byte for byte. Plus <out_dir>/pauses.yaml
   (what was cut, and why).

No ROS needed (scoop conda env). Then, with the commands printed at the end:

    dvlc ros2 run direct_visual_lidar_calibration preprocess <out_dir> <prep> -v \\
        --image_topic <image> --points_topic <points> --camera_info_topic <camera-info>
    dvlc ros2 run direct_visual_lidar_calibration initial_guess_manual <prep>
    dvlc ros2 run direct_visual_lidar_calibration calibrate <prep>
    dvlc ros2 run direct_visual_lidar_calibration viewer <prep>
"""
import argparse
import os
import sys
import warnings

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag, rosmsg                                        # noqa: E402
from scoop.bagwrite import BagWriter                                 # noqa: E402

BIN_S = 0.1          # detection grid
HALF_S = 0.25        # IMU window half width, in s
ODOM_HALF_S = 0.5    # odometry speed: position change over +-0.5 s
ODOM_GAP_S = 0.3     # no odometry within this of a bin -> the bin is not still


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def read_odom(reader, topic):
    """(log ns, positions (N,3)) of a nav_msgs/Odometry or PoseStamped topic."""
    ts = rosmsg.typestore()
    kind = reader.topics()[topic].msgtype
    t, p = [], []
    for _, log_ns, payload, _ in reader.iter_raw([topic]):
        m = ts.deserialize_cdr(payload, kind)
        pos = m.pose.pose.position if kind == "nav_msgs/msg/Odometry" else m.pose.position
        t.append(log_ns)
        p.append((pos.x, pos.y, pos.z))
    return np.array(t, np.int64), np.array(p, np.float64).reshape(-1, 3)


def read_imu(reader, topic):
    """(log ns, gyro (N,3), accel (N,3)) of a sensor_msgs/Imu topic."""
    ts = rosmsg.typestore()
    t, g, a = [], [], []
    for _, log_ns, payload, _ in reader.iter_raw([topic]):
        m = ts.deserialize_cdr(payload, "sensor_msgs/msg/Imu")
        w, la = m.angular_velocity, m.linear_acceleration
        t.append(log_ns)
        g.append((w.x, w.y, w.z))
        a.append((la.x, la.y, la.z))
    return (np.array(t, np.int64), np.array(g, np.float64).reshape(-1, 3),
            np.array(a, np.float64).reshape(-1, 3))


def _window_sum(x, half_bins):
    """Sum of x over bins i-half .. i+half (zeros past the ends)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    return c[np.minimum(i + half_bins + 1, len(x))] - c[np.maximum(i - half_bins, 0)]


def motion_grid(t0_ns, t1_ns, odom, imu):
    """Per 0.1 s bin: centre (s from t0), odometry speed, mean |gyro|, accel std."""
    n = int(np.ceil((t1_ns - t0_ns) * 1e-9 / BIN_S))
    tc = (np.arange(n) + 0.5) * BIN_S

    ot, op = odom
    speed = np.full(n, np.inf)
    if len(ot) >= 2:
        os_ = (ot - t0_ns) * 1e-9
        lo, hi = tc - ODOM_HALF_S, tc + ODOM_HALF_S
        pl = np.stack([np.interp(lo, os_, op[:, k]) for k in range(3)], 1)
        ph = np.stack([np.interp(hi, os_, op[:, k]) for k in range(3)], 1)
        speed = np.linalg.norm(ph - pl, axis=1) / (hi - lo)
        j = np.clip(np.searchsorted(os_, tc), 1, len(os_) - 1)
        gap = np.minimum(np.abs(os_[j] - tc), np.abs(os_[j - 1] - tc))
        inside = (lo >= os_[0]) & (hi <= os_[-1])
        speed[(gap > ODOM_GAP_S) | ~inside] = np.inf

    it, ig, ia = imu
    b = np.clip(((it - t0_ns) * 1e-9 / BIN_S).astype(int), 0, n - 1)
    cnt = np.bincount(b, minlength=n).astype(float)
    gsum = np.bincount(b, np.linalg.norm(ig, axis=1), minlength=n)
    asum = np.stack([np.bincount(b, ia[:, k], minlength=n) for k in range(3)], 1)
    asq = np.stack([np.bincount(b, ia[:, k] ** 2, minlength=n) for k in range(3)], 1)
    h = int(round(HALF_S / BIN_S))
    c = _window_sum(cnt, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        gyro = _window_sum(gsum, h) / c
        m = np.stack([_window_sum(asum[:, k], h) for k in range(3)], 1) / c[:, None]
        v = np.stack([_window_sum(asq[:, k], h) for k in range(3)], 1) / c[:, None] - m ** 2
        acc_std = np.sqrt(np.clip(v, 0, None).sum(1))
    few = c < 5
    gyro[few] = np.inf
    acc_std[few] = np.inf
    return tc, speed, gyro, acc_std


def find_runs(still, tc, min_len_s):
    """[(start s, end s)] of runs of True at least min_len_s long."""
    d = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = []
    for a, b in zip(starts, ends):
        s, e = tc[a] - BIN_S / 2, tc[b - 1] + BIN_S / 2
        if e - s >= min_len_s:
            runs.append((float(s), float(e)))
    return runs


def cut(runs, trim_s, max_len_s):
    """Trim both ends, keep at most max_len_s from the middle."""
    out = []
    for s, e in runs:
        s, e = s + trim_s, e - trim_s
        if e - s <= 0:
            continue
        if max_len_s and e - s > max_len_s:
            mid = 0.5 * (s + e)
            s, e = mid - max_len_s / 2, mid + max_len_s / 2
        out.append((s, e))
    return out


def parse_intervals(text):
    out = []
    for part in text.split(","):
        a, b = (float(x) for x in part.split("-", 1))
        if b <= a:
            raise ValueError(f"interval {part!r}: end before start")
        out.append((a, b))
    return out


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def write_pause(reader, out_dir, lo_ns, hi_ns, topics, schemas, qos):
    """The messages of ``topics`` logged in [lo_ns, hi_ns] -> a new bag,
    byte for byte with their QoS. Returns the topic counts."""
    with BagWriter(out_dir, library="scoop.dvlc_pauses") as w:
        for t in topics:
            w.set_qos(t, qos.get(t, ""))
        for topic, log_ns, payload, _ in reader.iter_raw(topics, lo_ns, hi_ns):
            if lo_ns <= log_ns <= hi_ns:
                w.write_raw(topic, schemas[topic], payload, log_ns)
    return dict(w.counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("out_dir")
    ap.add_argument("--list", action="store_true", help="print the pauses, write nothing")
    ap.add_argument("--odom", default="/mobile_2/visual_slam/tracking/odometry")
    ap.add_argument("--imu", default="/mobile_2/imu")
    ap.add_argument("--points", default="/mobile_2/ouster/points")
    ap.add_argument("--image", default="/mobile_2/color/image_raw")
    ap.add_argument("--camera-info", default="/mobile_2/color/camera_info")
    ap.add_argument("--max-speed", type=float, default=0.01, help="m/s")
    ap.add_argument("--max-gyro", type=float, default=0.02, help="rad/s")
    ap.add_argument("--max-acc-std", type=float, default=0.08, help="m/s^2")
    ap.add_argument("--min-pause", type=float, default=4.0, help="s")
    ap.add_argument("--trim", type=float, default=1.0, help="s off both ends")
    ap.add_argument("--max-len", type=float, default=10.0, help="s kept per pause (0: all)")
    ap.add_argument("--only", default="", help="pause numbers to write, e.g. 0,2,5")
    ap.add_argument("--intervals", default="",
                    help="s from the bag start, e.g. 120.5-130,300-310 (skips detection)")
    a = ap.parse_args()

    try:
        reader = bag.open_bag(a.bag)
        topics = reader.topics()
        cut_topics = [a.points, a.image, a.camera_info]
        need = cut_topics + ([] if a.intervals else [a.odom, a.imu])
        missing = [t for t in need if t not in topics]
        if missing:
            sys.exit(f"error: not in the bag: {missing}\n  topics: {sorted(topics)}")
        t0_ns, t1_ns = reader.time_range()

        if a.intervals:
            pauses = parse_intervals(a.intervals)
            stats = [None] * len(pauses)
        else:
            print(f"reading {a.odom} and {a.imu} ...")
            tc, speed, gyro, acc = motion_grid(t0_ns, t1_ns, read_odom(reader, a.odom),
                                               read_imu(reader, a.imu))
            still = (speed < a.max_speed) & (gyro < a.max_gyro) & (acc < a.max_acc_std)
            pauses = cut(find_runs(still, tc, a.min_pause), a.trim, a.max_len)
            stats = []
            for s, e in pauses:
                k = (tc >= s) & (tc <= e)
                stats.append(dict(speed=float(np.max(speed[k])), gyro=float(np.max(gyro[k])),
                                  acc_std=float(np.max(acc[k]))))
            print(f"{still.mean() * 100:.0f} % of {(t1_ns - t0_ns) * 1e-9:.0f} s still, "
                  f"{len(pauses)} pauses of >= {a.min_pause:g} s")

        print(f"\n  #   start s    end s   len s   max speed  max |gyro|  max acc std")
        for i, ((s, e), st) in enumerate(zip(pauses, stats)):
            extra = ("" if st is None else
                     f"   {st['speed']:.4f}     {st['gyro']:.4f}      {st['acc_std']:.4f}")
            print(f"  {i:<3d} {s:8.1f} {e:8.1f} {e - s:7.1f}{extra}")
        if a.list or not pauses:
            return

        keep = sorted(set(int(x) for x in a.only.split(","))) if a.only else range(len(pauses))
        if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
            sys.exit(f"{a.out_dir} is not empty; pick a new output folder")
        os.makedirs(a.out_dir, exist_ok=True)
        schemas = bag.topic_schemas(reader)
        qos = bag.topic_qos(reader)
        record = dict(bag=os.path.abspath(a.bag), bag_start_ns=int(t0_ns),
                      topics=dict(points=a.points, image=a.image, camera_info=a.camera_info),
                      thresholds=dict(max_speed=a.max_speed, max_gyro=a.max_gyro,
                                      max_acc_std=a.max_acc_std, min_pause=a.min_pause,
                                      trim=a.trim, max_len=a.max_len),
                      pauses=[])
        for i in keep:
            s, e = pauses[i]
            d = os.path.join(a.out_dir, f"pause_{i:02d}")
            counts = write_pause(reader, d, t0_ns + int(s * 1e9), t0_ns + int(e * 1e9),
                                 cut_topics, schemas, qos)
            print(f"wrote {d}: {counts}")
            record["pauses"].append(dict(name=f"pause_{i:02d}", start_s=round(s, 3),
                                         end_s=round(e, 3), counts=counts,
                                         **(stats[i] or {})))
        with open(os.path.join(a.out_dir, "pauses.yaml"), "w") as fh:
            yaml.safe_dump(record, fh, sort_keys=False)
    except (ValueError, bag.BagError) as err:
        sys.exit(f"error: {err}")

    prep = a.out_dir.rstrip("/") + "_preprocessed"
    print(f"""
next (DVLC):
  dvlc ros2 run direct_visual_lidar_calibration preprocess {a.out_dir} {prep} -v \\
    --image_topic {a.image} --points_topic {a.points} --camera_info_topic {a.camera_info}
  dvlc ros2 run direct_visual_lidar_calibration initial_guess_manual {prep}
  dvlc ros2 run direct_visual_lidar_calibration calibrate {prep}
  dvlc ros2 run direct_visual_lidar_calibration viewer {prep}""")


if __name__ == "__main__":
    main()
