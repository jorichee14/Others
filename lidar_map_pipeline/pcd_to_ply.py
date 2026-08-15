#!/usr/bin/env python3
"""
Give every existing .pcd a .ply twin, without re-running the pipeline.

Stage 01 now writes both formats itself; this backfills output directories
produced before that, so an hour of detect does not get repeated for the sake
of a file extension. Conversion is lossless for what these clouds carry
(xyz + rgb) -- both files come from the same in-memory cloud.

Skips a .pcd whose .ply already exists and is newer, so re-running after a
partial pass (or after the pipeline wrote some twins itself) converts only
what is missing. One file is held in memory at a time, so a directory of
5 GB clouds needs no more RAM than the largest single one.

    python3 pcd_to_ply.py <dir-or-file> [more...] [--force]
"""

import argparse
import os
import sys
import open3d as o3d


def find_pcds(paths):
    out = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".pcd"):
            out.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files
                           if f.lower().endswith(".pcd"))
        else:
            print(f"! skipping {p}: not a .pcd or directory")
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+",
                    help="output directories and/or individual .pcd files")
    ap.add_argument("--force", action="store_true",
                    help="rewrite .ply twins even when up to date")
    a = ap.parse_args()

    pcds = find_pcds(a.paths)
    if not pcds:
        raise SystemExit("no .pcd files found")
    print(f"{len(pcds)} .pcd file(s)")
    done = skipped = failed = 0
    for src in pcds:
        dst = src[:-4] + ".ply"
        if not a.force and os.path.exists(dst) \
                and os.path.getmtime(dst) >= os.path.getmtime(src):
            skipped += 1
            continue
        try:
            pc = o3d.io.read_point_cloud(src)
            n = len(pc.points)
            if n == 0:
                print(f"  ! {src}: empty, skipped")
                failed += 1
                continue
            o3d.io.write_point_cloud(dst, pc)
            print(f"  {src} -> {os.path.basename(dst)}  ({n} pts, "
                  f"{os.path.getsize(dst) / 2**20:.0f} MiB)", flush=True)
            done += 1
        except Exception as e:                       # keep going: one bad file
            print(f"  ! {src}: {type(e).__name__}: {e}")   # must not stop 40
            failed += 1
    print(f"\n{done} converted, {skipped} already current, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
