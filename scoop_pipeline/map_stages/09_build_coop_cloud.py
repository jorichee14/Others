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
    --ref-map PATH        the reference map's points into the same cloud as well, so
                          the run's own objects stand in the building the mapping
                          pass measured. Off unless asked for (09_build.ref_map,
                          or "auto" for the reference pass's anchored map): a
                          merged cloud cannot tell what the run brought into the
                          room from what was always there. The map's points are
                          added AFTER the dynamic filter, so none is carved away.
    --also TOPIC=TRAJ     another LiDAR's scans into the same cloud, placed by its
                          own stage-04 track (T_map_lidar). From a config, every
                          "kind": "lidar" entry of 04_reference.tracks is added by
                          itself: both robots' Ousters build the coop cloud, each
                          seeing what the other drove past, and each robot's rays
                          carve the other robot out.
    --keep-dynamic        keep the moving things. By default the run's movers (the
                          other robot, the operators) are removed by FREE-SPACE
                          CARVING alone: a 0.15 m grid counts, per voxel, the
                          scans whose rays passed THROUGH it to something farther
                          away, and a voxel seen through at least min_free times,
                          at least as often as it was hit, held something that
                          came and went. The point where a ray STOPPED is what
                          gets taken, never a surface behind it. Nothing else
                          removes a point: the span test and the hit-count guard
                          of stage 01 are both off (min_span_s 0, min_hits 1),
                          because a run sees most of the room once and both
                          would call those surfaces transient. --min-count is
                          the noise guard instead.
                          Stage 01's other test, the time SPAN a voxel was seen
                          over, is off here (min_span_s 0): a mapping pass
                          revisits every surface, but a run drives past most of
                          the room once, so the span test would delete the walls
                          it glimpsed along with the movers. Set min_span_s
                          above 0 to bring it back. The removed points go to
                          <out>_dynamic.pcd. Config block "remove_dynamic" under
                          09_build (voxel, min_span_s, min_hits, carve{...}).
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


def read_points(path):
    """Any pcd -> (N, 3) float64, through Open3D (as 01 and 04 read the map)."""
    import open3d as o3d
    pc = o3d.io.read_point_cloud(os.path.expanduser(path))
    if not len(pc.points):
        raise SystemExit("no points in %s" % path)
    return np.asarray(pc.points, dtype=np.float64)


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
  "min_count": 2, "keep_quality": true, "ref_map": null,
  "remove_dynamic": {"enable": true, "voxel": 0.15, "min_span_s": 0.0, "min_hits": 1,
                     "carve": {"enable": true, "ray_stride": 4, "scan_stride": 2, "max_range": 20.0,
                               "grazing_margin": 0.03, "min_free": 5, "free_ratio": 1.0}}
}
"""

# Carving decides here, and nothing else does.
#   min_span_s 0  a run drives past most of the room once; judging those
#                 surfaces transient empties the cloud of the walls the objects
#                 stand against (coop site_1: 191k of 246k voxels "dynamic" by
#                 span alone, against 55k the rays actually saw through).
#   min_hits 1    the same for the noise guard: a 0.15 m voxel a single sweep
#                 filled densely is a far wall seen once in passing, not noise
#                 (another 134k voxels). Stray single points are dropped by the
#                 OUTPUT voxel's min_count instead, which is what it is for.
REMOVE_DYNAMIC = {"enable": True, "voxel": 0.15, "min_span_s": 0.0, "min_hits": 1,
                  "carve": {"enable": True, "ray_stride": 4, "scan_stride": 2, "max_range": 20.0,
                            "endpoint_margin": 0.0, "grazing_margin": 0.03, "min_free": 5,
                            "free_ratio": 1.0}}


def dynamic_filter(rd):
    """(DynStats, FreeSpaceCarver or None) from the remove_dynamic block, with
    stage 01's classes (map_stages/01_build_map.py), or (None, None) when off."""
    rd = dict(REMOVE_DYNAMIC, **(rd or {}))
    if not rd.get("enable", True):
        return None, None, rd
    import importlib
    M = importlib.import_module("01_build_map")
    dyn = M.DynStats(float(rd["voxel"]))
    cv = dict(REMOVE_DYNAMIC["carve"], **(rd.get("carve") or {}))
    carver = None
    if cv.get("enable", True):
        carver = M.FreeSpaceCarver(float(rd["voxel"]), max_range=float(cv["max_range"]),
                                   ray_stride=int(cv["ray_stride"]), scan_stride=int(cv["scan_stride"]),
                                   endpoint_margin=float(cv["endpoint_margin"]),
                                   grazing_margin=float(cv["grazing_margin"]))
    return dyn, carver, rd


def drop_dynamic(P, dyn, carver, rd):
    """P less the points in voxels judged transient; returns (kept, dropped, counts)."""
    import importlib
    M = importlib.import_module("01_build_map")
    cv = dict(REMOVE_DYNAMIC["carve"], **(rd.get("carve") or {}))
    keys, counts = dyn.dynamic_keys(int(rd["min_hits"]), float(rd["min_span_s"]), carver=carver,
                                    min_free=int(cv["min_free"]), free_ratio=float(cv["free_ratio"]))
    keys = np.asarray(M.as_cpu(keys))
    if not keys.size or not len(P):
        return P, P[:0], counts
    packed = M.pack_voxels(np.floor(P / float(rd["voxel"])).astype(np.int64))
    pos = np.clip(np.searchsorted(keys, packed), 0, keys.size - 1)
    gone = keys[pos] == packed
    return P[~gone], P[gone], counts


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
        cfg_defaults["topic"] = P.points_topic
        # the other robots' LiDARs: 04's lidar tracks, each with its own trajectory
        also = []
        for tr in (P.cfg.get("04_reference") or {}).get("tracks") or []:
            if tr.get("kind") == "lidar" and tr.get("enabled", True):
                topic = tr.get("points_topic") or P.sensor_setting("points_topic", tr["machine"])
                traj = os.path.join(P.reference_dir(), "traj_%s.tum" % tr["name"])
                if os.path.exists(traj):
                    also.append("%s=%s" % (topic, traj))
                else:
                    print("(no %s: track %s not built by 04, left out)" % (traj, tr["name"]))
        cfg_defaults["also"] = also
        if cfg_defaults.get("ref_map") == "auto":
            cfg_defaults["ref_map"] = P.ref_file("anchored_map") if P.reference else ""
        elif cfg_defaults.get("ref_map"):
            cfg_defaults["ref_map"] = P.outp(cfg_defaults["ref_map"])
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
    ap.add_argument("--also", action="append", default=[], metavar="TOPIC=TRAJ",
                    help="another LiDAR's scans, placed by its own stage-04 track")
    ap.add_argument("--min-span-s", type=float, default=None,
                    help="bring back stage 01's span test: a voxel seen over less than this "
                         "many seconds is transient (0, the default here, leaves it off)")
    ap.add_argument("--ref-map", default=None,
                    help="the reference map's pcd, added after the dynamic filter (off by "
                         "default: a merged cloud cannot tell what the run brought in)")
    ap.add_argument("--no-ref-map", dest="ref_map", action="store_const", const="",
                    help="leave the reference map out")
    ap.add_argument("--ref-voxel", type=float, default=0.0,
                    help="voxel for the map's points (0: --voxel)")
    ap.add_argument("--keep-dynamic", action="store_true",
                    help="keep the run's movers (default: removed by span and free-space carving)")
    ap.add_argument("--remove-dynamic", default=None, help=argparse.SUPPRESS)   # the config block
    ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg_defaults.items()})
    args = ap.parse_args(argv)
    rd = None if args.keep_dynamic else (args.remove_dynamic if isinstance(args.remove_dynamic, dict) else {})
    if rd is not None and args.min_span_s is not None:
        rd = dict(rd, min_span_s=args.min_span_s)
    dyn, carver, rd = dynamic_filter(rd) if rd is not None else (None, None, None)
    if dyn is not None:
        cv = rd.get("carve") or {}
        print("remove dynamic: ON, grid %.2f m, %s, %s%s" % (
            rd["voxel"],
            "span >= %.1f s" % rd["min_span_s"] if rd["min_span_s"] > 0
            else "no span test (a run sees most surfaces once)",
            "hits >= %d" % rd["min_hits"] if rd["min_hits"] > 1
            else "no hit-count guard (min_count does that)",
            ", carving every %d-th scan, ray stride %d, free >= %d and >= %.1f x hits" % (
                cv.get("scan_stride", 2), cv.get("ray_stride", 4), cv.get("min_free", 5),
                cv.get("free_ratio", 1.0)) if carver is not None else ", no carving"))
    else:
        print("remove dynamic: off (--keep-dynamic)")

    sources = [(args.topic, args.traj)]
    for item in args.also or []:
        if "=" not in item:
            sys.exit("--also takes TOPIC=TRAJ, got %r" % item)
        topic, traj = item.split("=", 1)
        sources.append((topic.strip(), os.path.expanduser(traj.strip())))
    print("deskew: %s" % ("ON, %d bins, through each trajectory" % args.deskew_bins
                          if args.deskew else "off (one pose per scan)"))
    acc = VoxelAccum(args.voxel)
    n_used_all = 0
    for topic, traj in sources:
        ts, Ts = load_traj(traj)
        ts, Ts = np.asarray(ts), np.asarray(Ts)
        print(f"{topic}: trajectory {traj}: {len(ts)} poses, {ts[-1]-ts[0]:.1f} s")
        good = None
        if args.keep_quality:
            qcsv = os.path.join(os.path.dirname(traj), "quality_" + os.path.basename(traj)[5:-4] + ".csv")
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
                print(f"  quality gate: {int(good.sum())} of {len(ts)} poses registered ({qcsv})")
            else:
                print(f"  (no {qcsv}; --keep-quality ignored)")
        n_used = n_skip_q = 0
        i = -1
        for j, X in iter_scans(args.bag, topic, ts, args.max_gap_ms * 1e-3,
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
            if dyn is not None:
                dyn.add(X, float(ts[j]))
                if carver is not None:
                    carver.add(Ts[j][:3, 3].astype(np.float64), X)
            n_used += 1
            if n_used % 500 == 0:
                print(f"  {n_used} scans  t={rel:.1f}s", flush=True)
        print(f"  used {n_used} scans of {topic} (skipped {n_skip_q} not registered by 04; scans more than "
              f"{args.max_gap_ms:.0f} ms from a trajectory pose are not read)")
        n_used_all += n_used
    out = acc.points(args.min_count)
    if dyn is not None:
        out, gone, (n_static, n_dyn, n_carved) = drop_dynamic(out, dyn, carver, rd)
        dyn_path = os.path.splitext(args.out)[0] + "_dynamic.pcd"
        write_pcd(dyn_path, gone)
        other = n_dyn - n_carved
        print(f"remove dynamic: {n_static} static / {n_dyn} dynamic voxels"
              f"{', %d carved by free-space' % n_carved if carver is not None else ''}"
              f"{', %d by the span or hit-count tests' % other if other else ''}"
              f" -> dropped {len(gone)} points, kept {len(out)}; the dropped ones in {dyn_path}")
    n_run = len(out)
    if args.ref_map:
        ref = read_points(args.ref_map)
        print(f"reference map: {args.ref_map}: {len(ref)} points")
        lo, hi = out.min(axis=0) - 2.0, out.max(axis=0) + 2.0
        inside = np.all((ref >= lo) & (ref <= hi), axis=1)
        if inside.sum() < len(ref):
            print(f"  {len(ref) - int(inside.sum())} of them outside the run's extent "
                  f"(plus 2 m), left out")
        merged = VoxelAccum(args.ref_voxel or args.voxel)
        merged.add(out)
        merged.add(ref[inside])
        out = merged.points(1)
        print(f"  merged: {n_run} from the run + {int(inside.sum())} from the map "
              f"-> {len(out)} at {args.ref_voxel or args.voxel} m "
              f"(the overlap falls in the same voxels)")
    write_pcd(args.out, out)
    print(f"used {n_used_all} scans from {len(sources)} LiDAR(s)"
          + (f"; the reference map's points are in too ({args.ref_map})" if args.ref_map else ""))
    print(f"wrote {args.out}: {len(out)} points at {args.voxel} m voxel, "
          f"x {out[:,0].min():.2f}..{out[:,0].max():.2f}  y {out[:,1].min():.2f}..{out[:,1].max():.2f}")
    print("this cloud is in the same frame as the trajectory it was built from — "
          "the reference map frame if that is what stage 04 registered to.")


if __name__ == "__main__":
    main()
