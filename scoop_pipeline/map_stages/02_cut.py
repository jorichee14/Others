#!/usr/bin/env python3
"""
STAGE 02 - cut floor & ceiling from the map by height.

Pipeline use (reads floor/ceil/voxel from config):
    python3 02_cut.py [pipeline_config.json]

Pick-your-bounds use (prints a z-histogram, changes nothing):
    python3 02_cut.py [pipeline_config.json] --analyze
  Read off the floor/ceiling peaks, put them in 02_cut.{floor,ceil}, re-run.

If floor and ceil are both null in the config, the cut can't run: the script
prints the histogram and exits non-zero so the pipeline stops for you to choose.
"""
import sys
import numpy as np
import open3d as o3d

from pipeline_common import load_pipeline


def ascii_hist(centers, counts, floor, ceil, width=60):
    peak = max(int(counts.max()), 1)
    binw = centers[1] - centers[0] if len(centers) > 1 else 1.0
    lines = []
    for z, n in reversed(list(zip(centers, counts))):
        bar = "#" * int(round(width * n / peak))
        mark = ""
        if floor is not None and abs(z - floor) <= binw / 2:
            mark = "  <-- floor?"
        elif ceil is not None and abs(z - ceil) <= binw / 2:
            mark = "  <-- ceiling?"
        lines.append(f"z={z:+6.2f} | {bar:<{width}} {n:>9,d}{mark}")
    return "\n".join(lines)


def suggest(centers, counts):
    n = len(centers)
    lo_end = max(int(n * 0.40), 1); hi_start = min(int(n * 0.60), n - 1)
    fi = int(np.argmax(counts[:lo_end]))
    ci = hi_start + int(np.argmax(counts[hi_start:]))
    return centers[fi], centers[ci]


def main():
    analyze = "--analyze" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg_path = args[0] if args else "pipeline_config.json"
    P = load_pipeline(cfg_path)
    s = P.stage("02_cut")

    ai = {"x": 0, "y": 1, "z": 2}[s.get("axis", "z")]
    pcd = o3d.io.read_point_cloud(s["input"])
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise SystemExit(f"no points in {s['input']}")
    h = pts[:, ai]

    counts, edges = np.histogram(h, bins=s.get("bins", 100))
    centers = 0.5 * (edges[:-1] + edges[1:])
    s_floor, s_ceil = suggest(centers, counts)
    floor, ceil, voxel = s.get("floor"), s.get("ceil"), s.get("voxel", 0.0)

    print(f"{len(h):,} points | {s.get('axis','z')} range [{h.min():.2f}, {h.max():.2f}] "
          f"| median {np.median(h):.2f}\n")
    print(ascii_hist(centers, counts, floor if floor is not None else s_floor,
                     ceil if ceil is not None else s_ceil))
    print(f"\nsuggested:  floor {s_floor:+.2f}   ceil {s_ceil:+.2f}")
    print(f"config:     floor {floor}   ceil {ceil}   voxel {voxel}")

    if analyze:
        print("\n--analyze: nothing written.")
        return

    if floor is None and ceil is None and (not voxel or voxel <= 0):
        raise SystemExit("\nno floor/ceil/voxel set in 02_cut -> nothing to do. "
                         "Run with --analyze, pick bounds, put them in the config.")

    if floor is not None or ceil is not None:
        lo = floor if floor is not None else h.min() - 1.0
        hi = ceil if ceil is not None else h.max() + 1.0
        keep = np.where((h >= lo) & (h <= hi))[0]
        pcd = pcd.select_by_index(keep)
        print(f"\ncut: kept {len(keep):,}/{len(h):,} ({100*len(keep)/len(h):.1f}%) "
              f"with {lo:.2f} <= {s.get('axis','z')} <= {hi:.2f}")

    if voxel and voxel > 0:
        n0 = len(pcd.points); pcd = pcd.voxel_down_sample(voxel)
        print(f"voxel {voxel} m: {n0:,} -> {len(pcd.points):,}")

    o3d.io.write_point_cloud(s["output"], pcd)
    print(f"saved {s['output']}  ({len(pcd.points):,} points)")


if __name__ == "__main__":
    main()
