#!/usr/bin/env python3
"""
STAGE 07 - build a coop run's cloud in the REFERENCE map frame.

Accumulates the run's Ouster scans into one cloud using stage 08's LiDAR-ICP
poses. (Stage 08 must therefore run first; this stage consumes its lidar track.)

The poses come from stage 08's LiDAR-ICP track (traj_<name>_lidar.tum), which is
already registered to the reference cloud. So the output is in that frame by
construction: no GLIM, no stage 03, no second anchoring. It contains whatever
was in the room during the run — the chairs the mapping cloud lacks — and is
what label_static.py should be pointed at.

    python3 07_build_coop_cloud.py \
        --bag  ../../raw/20260828/mirc_dataset_coop2_20260828_completed \
        --traj map_stages_20260828_outputs/reference_coop2_all/traj_mobile_1_lidar.tum \
        --out  map_stages_20260828_outputs/coop2_in_ref.pcd

Or from pipeline_config.json (block "07_build", sample at the bottom):

    python3 07_build_coop_cloud.py pipeline_config.json

Then:
    python3 scripts/label_static.py --pcd map_stages_20260828_outputs/coop2_in_ref.pcd \
        --dataset <root>/test --propose --map-image /tmp/coop2_top.png

Options worth knowing:
    --t-start / --t-end   seconds from bag start; e.g. 0 40 to build only from the
                          stretch before people walked through (cleaner chairs)
    --voxel               output resolution (m); 0.03 is plenty for box fitting
    --range-max           drop returns beyond this (far corridor noise)
    --keep-quality        only use scans whose stage-08 quality row says nobs=6
                          (needs quality_<name>.csv next to the tum)
"""
import argparse, os, sys, csv
import numpy as np

try:
    from rosbags.rosbag2 import Reader
except ImportError:
    sys.exit("pip install rosbags")
try:
    from rosbags.typesys import Stores, get_typestore
    _TS = get_typestore(Stores.ROS2_HUMBLE)
    _DESER = lambda raw, t: _TS.deserialize_cdr(raw, t)
except Exception:
    from rosbags.serde import deserialize_cdr
    _DESER = lambda raw, t: deserialize_cdr(raw, t)


# ------------------------------------------------------------------ SE(3)
def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def slerp(q0, q1, u):
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0: q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); s = np.sin(th)
    return (np.sin((1-u)*th)*q0 + np.sin(u*th)*q1) / s


def read_tum(path):
    A = np.loadtxt(path)
    if A.ndim == 1: A = A[None]
    A = A[np.argsort(A[:, 0])]
    return A[:, 0], A[:, 1:4], A[:, 4:8]


def pose_at(ts, P, Q, t):
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    t0, t1 = ts[i-1], ts[i]
    u = 0.0 if t1 == t0 else float(np.clip((t - t0) / (t1 - t0), 0, 1))
    R = quat_to_R(slerp(Q[i-1], Q[i], u))
    p = (1-u) * P[i-1] + u * P[i]
    gap = min(abs(t - t0), abs(t1 - t))
    return R, p, gap, not (t0 <= t <= t1)


# ------------------------------------------------------------------ cloud
def pc2_xyz(msg):
    off = {f.name: (f.offset, f.datatype) for f in msg.fields}
    raw = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.point_step)
    cols = []
    for n in ("x", "y", "z"):
        o, dt = off[n]
        if dt != 7: raise RuntimeError(f"{n} not float32")
        cols.append(raw[:, o:o+4].copy().view(np.float32).ravel())
    P = np.stack(cols, 1)
    return P[np.isfinite(P).all(1)]


def write_pcd(path, P):
    P = P.astype(np.float32)
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
           "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
           f"WIDTH {len(P)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(P)}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii")); f.write(P.tobytes())


class VoxelAccum:
    """Running per-voxel centroid without holding every raw point."""
    def __init__(self, voxel):
        self.v = voxel; self.keys = []; self.sums = []; self.cnts = []
        self.k_m, self.s_m, self.c_m = None, None, None

    def add(self, P):
        k = np.floor(P / self.v).astype(np.int64)
        key = (k[:, 0] + (1 << 20)) * (1 << 42) + (k[:, 1] + (1 << 20)) * (1 << 21) + (k[:, 2] + (1 << 20))
        uk, inv = np.unique(key, return_inverse=True)
        s = np.zeros((len(uk), 3)); np.add.at(s, inv, P)
        c = np.bincount(inv, minlength=len(uk)).astype(np.int64)
        self.keys.append(uk); self.sums.append(s); self.cnts.append(c)
        if len(self.keys) >= 40: self.merge()

    def merge(self):
        if not self.keys: return
        K = np.concatenate(([self.k_m] if self.k_m is not None else []) + self.keys)
        S = np.concatenate(([self.s_m] if self.s_m is not None else []) + self.sums)
        C = np.concatenate(([self.c_m] if self.c_m is not None else []) + self.cnts)
        uk, inv = np.unique(K, return_inverse=True)
        s = np.zeros((len(uk), 3)); np.add.at(s, inv, S)
        c = np.zeros(len(uk), np.int64); np.add.at(c, inv, C)
        self.k_m, self.s_m, self.c_m = uk, s, c
        self.keys, self.sums, self.cnts = [], [], []

    def points(self, min_count=1):
        self.merge()
        ok = self.c_m >= min_count
        return self.s_m[ok] / self.c_m[ok, None]


# ------------------------------------------------------------------ main
SAMPLE_CONFIG = """
"07_build": {
  "bag":   "/path/to/mirc_dataset_coop2_20260828_completed",
  "traj":  "map_stages_20260828_outputs/reference_coop2_all/traj_mobile_1_lidar.tum",
  "out":   "map_stages_20260828_outputs/coop2_in_ref.pcd",
  "topic": "/mobile_1/ouster/points",
  "voxel": 0.03, "range_min": 0.7, "range_max": 25.0,
  "t_start": 0, "t_end": 42,        <- optional: only the stretch before people walked through
  "min_count": 2, "keep_quality": true
}
"""


def main():
    argv = sys.argv[1:]
    cfg_defaults = {}
    if argv and not argv[0].startswith("-") and argv[0].endswith(".json"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pipeline_common import load_pipeline
        P = load_pipeline(argv[0])
        blk = P.cfg.get("07_build")
        if blk is None:
            sys.exit("add a '07_build' block to %s (sample at the bottom of this file)" % argv[0])
        cfg_defaults = {k: v for k, v in blk.items()}
        # processed layout: this pass's merged bag, stage 08's LiDAR-ICP track,
        # the cloud into mapping/ (cloud_in_ref_<tag>.pcd)
        cfg_defaults["bag"] = os.path.expanduser(cfg_defaults.get("bag") or P.dataset["bag"])
        cfg_defaults["traj"] = (P.outp(cfg_defaults["traj"]) if cfg_defaults.get("traj")
                                else P.lidar_track_traj())
        cfg_defaults["out"] = P.outp(cfg_defaults.get("out") or "cloud_in_ref_{tag}.pcd")
        print(P.describe())
        argv = argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required="bag" not in cfg_defaults)
    ap.add_argument("--traj", required="traj" not in cfg_defaults,
                    help="stage-08 traj_<name>_lidar.tum (T_map_lidar)")
    ap.add_argument("--out", required="out" not in cfg_defaults)
    ap.add_argument("--topic", default="/mobile_1/ouster/points")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--range-min", type=float, default=0.7)
    ap.add_argument("--range-max", type=float, default=25.0)
    ap.add_argument("--t-start", type=float, default=None, help="s from bag start")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth scan")
    ap.add_argument("--max-gap-ms", type=float, default=150.0,
                    help="skip scans further than this from a trajectory sample")
    ap.add_argument("--min-count", type=int, default=2,
                    help="keep a voxel only if this many returns fell in it "
                         "(drops single-hit noise and thin dynamic smears)")
    ap.add_argument("--keep-quality", action="store_true")
    ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg_defaults.items()})
    args = ap.parse_args(argv)

    ts, P, Q = read_tum(args.traj)
    print(f"trajectory: {len(ts)} poses, {ts[-1]-ts[0]:.1f} s")

    good_t = None
    if args.keep_quality:
        qcsv = os.path.join(os.path.dirname(args.traj),
                            "quality_" + os.path.basename(args.traj)[5:-4] + ".csv")
        if os.path.exists(qcsv):
            good_t = []
            with open(qcsv) as f:
                for r in csv.DictReader(f):
                    if int(r["nobs"]) >= 6 and not r["seed_status"].startswith("fail"):
                        good_t.append(float(r["t"]))
            good_t = np.array(sorted(good_t))
            print(f"quality gate: {len(good_t)} clean scans from {qcsv}")
        else:
            print(f"(no {qcsv}; --keep-quality ignored)")

    acc = VoxelAccum(args.voxel)
    n_used = n_skip_gap = n_skip_q = 0
    with Reader(args.bag) as rd:
        t0 = rd.start_time / 1e9
        conns = [c for c in rd.connections if c.topic == args.topic]
        if not conns: sys.exit(f"{args.topic} not in bag")
        i = -1
        for c, t, raw in rd.messages(connections=conns):
            i += 1
            if i % args.stride: continue
            m = _DESER(raw, c.msgtype)
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            rel = st - t0
            if args.t_start is not None and rel < args.t_start: continue
            if args.t_end is not None and rel > args.t_end: break
            if good_t is not None:
                j = np.searchsorted(good_t, st)
                near = min(abs(good_t[max(j-1, 0)] - st), abs(good_t[min(j, len(good_t)-1)] - st))
                if near > 0.06: n_skip_q += 1; continue
            R, p, gap, extrap = pose_at(ts, P, Q, st)
            if extrap or gap * 1e3 > args.max_gap_ms:
                n_skip_gap += 1; continue
            X = pc2_xyz(m)
            r = np.linalg.norm(X, axis=1)
            X = X[(r > args.range_min) & (r < args.range_max)]
            acc.add(X @ R.T + p)
            n_used += 1
            if n_used % 200 == 0:
                print(f"  {n_used} scans  t={rel:.1f}s", flush=True)
    out = acc.points(args.min_count)
    write_pcd(args.out, out)
    print(f"used {n_used} scans (skipped {n_skip_gap} off-trajectory, {n_skip_q} low-quality)")
    print(f"wrote {args.out}: {len(out)} points at {args.voxel} m voxel, "
          f"x {out[:,0].min():.2f}..{out[:,0].max():.2f}  y {out[:,1].min():.2f}..{out[:,1].max():.2f}")
    print("this cloud is in the same frame as the trajectory it was built from — "
          "the reference map frame if that is what stage 08 registered to.")


if __name__ == "__main__":
    main()
