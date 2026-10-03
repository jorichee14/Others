#!/usr/bin/env python3
"""
STAGE 09 - build a coop run's cloud in the REFERENCE map frame.

Accumulates the run's Ouster scans into one cloud using stage 04's LiDAR-ICP
poses. (Stage 04 must therefore run first; this stage consumes its lidar track.)

The poses come from stage 04's LiDAR-ICP track (traj_<name>_lidar.tum), which is
already registered to the reference cloud. So the output is in that frame by
construction: no GLIM, no stage 05, no second anchoring. It contains whatever
was in the room during the run — the chairs the mapping cloud lacks — and is
what label_static.py should be pointed at.

    python3 09_build_coop_cloud.py \
        --bag  ../../raw/20260828/mirc_dataset_coop2_20260828_completed \
        --traj map_stages_20260828_outputs/reference_coop2_all/traj_mobile_1_lidar.tum \
        --out  map_stages_20260828_outputs/coop2_in_ref.pcd

Or from pipeline_config.json (block "09_build", sample at the bottom):

    python3 09_build_coop_cloud.py pipeline_config.json

Then:
    python3 scripts/label_static.py --pcd map_stages_20260828_outputs/coop2_in_ref.pcd \
        --dataset <root>/test --propose --map-image /tmp/coop2_top.png

Options worth knowing:
    --t-start / --t-end   seconds from the trajectory start; e.g. 0 40 to build only from the
                          stretch before people walked through (cleaner chairs)
    --voxel               output resolution (m); 0.03 is plenty for box fitting
    --range-max           drop returns beyond this (far corridor noise)
    --keep-quality        only use scans stage 04 registered (quality csv status ok)
    --no-deskew           one pose per scan (default: every point through the
                          trajectory at its own time, as 01 / 02 / 04 do)
                          (needs quality_<name>.csv next to the tum)
"""
import argparse, os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import iter_scans, load_traj      # noqa: E402


# ------------------------------------------------------------------ cloud
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
"09_build": {
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
        from pipeline_common import load_pipeline
        P = load_pipeline(argv[0])
        blk = P.cfg.get("09_build")
        if blk is None:
            sys.exit("add a '09_build' block to %s (sample at the bottom of this file)" % argv[0])
        cfg_defaults = {k: v for k, v in blk.items()}
        # processed layout: this pass's merged bag, stage 04's LiDAR-ICP track,
        # the cloud into mapping/ (cloud_in_ref_<tag>.pcd)
        cfg_defaults["bag"] = os.path.expanduser(cfg_defaults.get("bag") or P.dataset["bag"])
        cfg_defaults["traj"] = (P.outp(cfg_defaults["traj"]) if cfg_defaults.get("traj")
                                else P.lidar_track_traj())
        cfg_defaults["out"] = P.outp(cfg_defaults.get("out") or "cloud_in_ref_{tag}.pcd")
        s01 = P.cfg.get("01_build_map", {})
        cfg_defaults.setdefault("deskew", bool(s01.get("deskew", True)))
        cfg_defaults.setdefault("deskew_bins", int(s01.get("deskew_bins", 100)))
        print(P.describe())
        argv = argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required="bag" not in cfg_defaults)
    ap.add_argument("--traj", required="traj" not in cfg_defaults,
                    help="stage-04 traj_<name>_lidar.tum (T_map_lidar)")
    ap.add_argument("--out", required="out" not in cfg_defaults)
    ap.add_argument("--topic", default="/mobile_1/ouster/points")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--range-min", type=float, default=0.7)
    ap.add_argument("--range-max", type=float, default=25.0)
    ap.add_argument("--t-start", type=float, default=None, help="s from the trajectory start")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth scan")
    ap.add_argument("--max-gap-ms", type=float, default=150.0,
                    help="skip scans further than this from a trajectory sample")
    ap.add_argument("--min-count", type=int, default=2,
                    help="keep a voxel only if this many returns fell in it "
                         "(drops single-hit noise and thin dynamic smears)")
    ap.add_argument("--keep-quality", action="store_true")
    ap.add_argument("--no-deskew", dest="deskew", action="store_false",
                    help="place each scan with one pose (default: deskew each point "
                         "through the trajectory at its own time, as the map stages)")
    ap.add_argument("--deskew-bins", type=int, default=100)
    ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg_defaults.items()})
    args = ap.parse_args(argv)

    ts, Ts = load_traj(args.traj)
    ts, Ts = np.asarray(ts), np.asarray(Ts)
    print(f"trajectory: {len(ts)} poses, {ts[-1]-ts[0]:.1f} s")

    good = None
    if args.keep_quality:
        qcsv = os.path.join(os.path.dirname(args.traj),
                            "quality_" + os.path.basename(args.traj)[5:-4] + ".csv")
        if os.path.exists(qcsv):
            good_t = []
            with open(qcsv) as f:
                for r in csv.DictReader(f):
                    if r.get("status") == "ok":
                        good_t.append(float(r["t"]))
            good_t = np.array(sorted(good_t))
            # quality rows are per trajectory pose: the poses they mark ok
            k = np.clip(np.searchsorted(good_t, ts), 0, max(len(good_t) - 1, 0))
            near = np.minimum(np.abs(good_t[k] - ts),
                              np.abs(good_t[np.maximum(k - 1, 0)] - ts)) if len(good_t) else \
                np.full(len(ts), np.inf)
            good = near < 1e-3
            print(f"quality gate: {int(good.sum())} of {len(ts)} poses registered ({qcsv})")
        else:
            print(f"(no {qcsv}; --keep-quality ignored)")
    print("deskew: %s" % ("ON, %d bins, through this trajectory" % args.deskew_bins
                          if args.deskew else "off (one pose per scan)"))

    acc = VoxelAccum(args.voxel)
    n_used = n_skip_q = 0
    i = -1
    for j, X in iter_scans(args.bag, args.topic, ts, args.max_gap_ms * 1e-3,
                           args.range_min, args.range_max, Ts, args.deskew,
                           args.deskew_bins, frame="world"):
        i += 1
        if i % args.stride:
            continue
        rel = ts[j] - ts[0]
        if args.t_start is not None and rel < args.t_start:
            continue
        if args.t_end is not None and rel > args.t_end:
            break
        if good is not None and not good[j]:
            n_skip_q += 1
            continue
        acc.add(X)
        n_used += 1
        if n_used % 500 == 0:
            print(f"  {n_used} scans  t={rel:.1f}s", flush=True)
    out = acc.points(args.min_count)
    write_pcd(args.out, out)
    print(f"used {n_used} scans (skipped {n_skip_q} not registered by 04; scans more than "
          f"{args.max_gap_ms:.0f} ms from a trajectory pose are not read)")
    print(f"wrote {args.out}: {len(out)} points at {args.voxel} m voxel, "
          f"x {out[:,0].min():.2f}..{out[:,0].max():.2f}  y {out[:,1].min():.2f}..{out[:,1].max():.2f}")
    print("this cloud is in the same frame as the trajectory it was built from — "
          "the reference map frame if that is what stage 04 registered to.")


if __name__ == "__main__":
    main()
