#!/usr/bin/env python3
"""Re-aim (and, where the mount allows, move) cameras to remove blind spots.

    python optimize_cameras.py                          # nominal cameras and mounts
    python optimize_cameras.py --max-moved 4            # touch at most 4 cameras
    python optimize_cameras.py --cameras configs/cameras.yaml --mounts configs/mounts.yaml \\
                               --room configs/room.yaml

Each camera stays inside its range of movement (mounts YAML). The search runs
on a coarser grid (--search-xy, --search-z), the before/after report on the
room's own grid.

Writes to --out (default out/optimized_<cameras stem>_k<k>/):
  moves.md          what to change on each camera, before/after coverage
  cameras.yaml      the new poses (load with coverage_map.py --cameras)
  before/, after/   coverage report and plots as coverage_map.py writes them
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

import coverage_map as cm
from ed305.camera import load_cameras
from ed305.coverage import load_room, visibility
from ed305.placement import aim_from_pose, load_mounts, optimise, room_aim_point

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", default=HERE / "configs/cameras_nominal.yaml", type=Path)
    ap.add_argument("--mounts", default=HERE / "configs/mounts_small.yaml", type=Path)
    ap.add_argument("--room", default=HERE / "configs/room_nominal.yaml", type=Path)
    ap.add_argument("--k", type=int)
    ap.add_argument("--max-moved", type=int, help="change at most this many cameras")
    ap.add_argument("--min-gain", type=float, default=0.001,
                    help="a move must add this fraction of the volume (default 0.001 = 0.1%%)")
    ap.add_argument("--step-deg", type=float, default=10.0, help="coarse angle step (refined to 1/4)")
    ap.add_argument("--step-m", type=float, default=0.5, help="coarse step along a mount segment")
    ap.add_argument("--search-xy", type=float, default=0.2, help="cell size while searching (m)")
    ap.add_argument("--search-z", type=float, default=0.5, help="height step while searching (m)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def evaluate(cams, room, k, out, plots):
    xs, ys, pts = room.grid()
    vis, blocked = visibility(cams, room, pts)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(cm.report(cams, room, vis, ~blocked, k, len(room.z)))
    if plots:
        cm.plots(out, cams, room, xs, ys, vis, blocked, k)
    n = vis.sum(0)[~blocked]
    return float(np.mean(n >= k))


def fmt_aim(a):
    if a is None:
        return "-"
    name, p = a
    return f"{name} ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})"


def main():
    a = parse()
    room = load_room(a.room)
    k = a.k or room.k
    cams = load_cameras(a.cameras)
    mounts = load_mounts(a.mounts, cams)

    search = replace(room, xy_step=a.search_xy,
                     z=np.arange(room.z[0], room.z[-1] + 1e-9, a.search_z) if a.search_z > 0 else room.z)
    _, _, pts = search.grid()
    _, blocked = visibility([], search, pts)
    new, settings, moved = optimise(cams, mounts, search, pts, blocked, k, a.step_deg, a.step_m, a.max_moved, a.min_gain)

    out = a.out or HERE / "out" / f"optimized_{a.cameras.stem}_k{k}"
    before = evaluate(cams, room, k, out / "before", not a.no_plots)
    after = evaluate(new, room, k, out / "after", not a.no_plots)

    doc = {"defaults": {}, "cameras": {}}
    for c, st, m in zip(new, settings, mounts):
        h, t, _ = aim_from_pose(c.T_world_cam)
        ap = room_aim_point(c.T_world_cam, room)
        doc["cameras"][c.name] = {
            "width": c.width, "height": c.height, "K": c.K.round(6).tolist(), "dist": c.dist.tolist(),
            "T_world_cam": c.T_world_cam.round(6).tolist(),
            "aim": {"heading": round(float(h), 1), "tilt": round(float(t), 1),
                    "aim_point": None if ap is None else {"surface": ap[0], "xyz": [round(float(v), 3) for v in ap[1]]}},
        }
    with open(out / "cameras.yaml", "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)

    lines = [f"# Camera moves, k = {k}", "",
             f"Target volume covered by >= {k} cameras: **{100 * before:.1f}% -> {100 * after:.1f}%** "
             f"({len(moved)} of {len(cams)} cameras changed).", "",
             "To set a camera: put a marker at the new aim point (where the image centre falls on the floor "
             "or a wall), turn the camera until the marker is at the image centre with the image level, "
             "then re-calibrate that camera.", "",
             "| camera | change (seen from behind the camera) | heading (deg) | tilt (deg) | position (m) | aim point (x, y, z) |",
             "|---|---|---|---|---|---|"]
    for i, (c0, c1) in enumerate(zip(cams, new)):
        if i not in moved:
            continue
        h0, t0, _ = aim_from_pose(c0.T_world_cam)
        h1, t1, _ = aim_from_pose(c1.T_world_cam)
        dp = c1.center - c0.center
        pos = "same" if np.linalg.norm(dp) < 1e-3 else f"moved {np.linalg.norm(dp):.2f} to {np.round(c1.center, 2).tolist()}"
        dh, dt = (h1 - h0 + 180) % 360 - 180, t1 - t0
        turn = [f"turn {abs(dh):.0f} deg {'left' if dh > 0 else 'right'}"] if abs(dh) >= 0.5 else []
        turn += [f"tilt {abs(dt):.0f} deg {'down' if dt > 0 else 'up'}"] if abs(dt) >= 0.5 else []
        lines.append(f"| {c1.name} | {', '.join(turn) or '-'} | {h0:.0f} -> {h1:.0f} | {t0:.0f} -> {t1:.0f} | {pos} | "
                     f"{fmt_aim(room_aim_point(c0.T_world_cam, room))} -> {fmt_aim(room_aim_point(c1.T_world_cam, room))} |")
    if not moved:
        lines.append("| (none: no change within the mount ranges helps) | | | | | |")
    lines += ["", "Unchanged: " + (", ".join(c.name for i, c in enumerate(cams) if i not in moved) or "none") + ".",
              "", "Full reports: before/report.md, after/report.md."]
    (out / "moves.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
