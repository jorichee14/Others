#!/usr/bin/env python3
"""A camera's range of rotation, from a video of it being turned.

Record the camera's own stream while turning it (see README, "Measuring a
mount's range"):

    ffmpeg -rtsp_transport tcp -i rtsp://192.168.1.56:8554/cam_2 -c copy cam_2_turn.mp4

then

    python measure_rotation.py cam_2_turn.mp4 --camera cam_2
    python measure_rotation.py cam_2_turn.mp4 --camera cam_2 --cameras configs/cameras.yaml

The video must start with the camera where it is calibrated (the pose in
--cameras); every angle is measured from there. Turn slowly to each stop:
full left, full right, back, full up, full down, then back to the start.

Writes to --out (default out/rotation_<camera>/):
  rotation.csv   per frame: heading, tilt, roll, pan from start, inliers
  rotation.png   pan and tilt over time, with the extremes marked
  mount.yaml     pan / tilt range for configs/mounts.yaml
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from ed305.camera import load_cameras
from ed305.placement import aim_from_pose, wrap
from ed305.rotation import RotationTracker, angle_deg, video_frames

HERE = Path(__file__).resolve().parent
BLUE, INK, MUTED, GRID = "#2a78d6", "#1a1a19", "#6b6a66", "#e4e3df"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file, or an rtsp:// URL to measure live")
    ap.add_argument("--camera", required=True, help="camera name in --cameras, e.g. cam_2")
    ap.add_argument("--cameras", default=HERE / "configs/cameras_nominal.yaml", type=Path,
                    help="intrinsics and the starting pose (default: nominal, so a guessed K)")
    ap.add_argument("--every", type=int, default=3, help="use every Nth frame (default 3: 10 per s at 30 fps)")
    ap.add_argument("--scale", type=float, default=0.5, help="resize frames by this first (default 0.5: 1080p -> 540p)")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="degrees taken off each end of the range for mounts.yaml (default 2)")
    ap.add_argument("--out", type=Path)
    return ap.parse_args()


def plot(path, t, pan, tilt, name):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ax, y, label in ((axs[0], pan, "pan from start (deg, + = left)"), (axs[1], tilt, "tilt (deg, 90 = down)")):
        ax.plot(t, y, color=BLUE, lw=2)
        for i in (int(np.argmin(y)), int(np.argmax(y))):
            ax.plot(t[i], y[i], "o", ms=8, mfc=BLUE, mec="white", mew=2, zorder=3)
            ax.annotate(f"{y[i]:.1f}", (t[i], y[i]), textcoords="offset points", xytext=(6, 6 if i == np.argmax(y) else -14),
                        fontsize=9, color=INK)
        ax.set_ylabel(label, color=MUTED, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[1].set_xlabel("time (s)", color=MUTED, fontsize=9)
    axs[0].set_title(f"{name}: rotation while turned by hand", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    a = parse()
    cams = {c.name: c for c in load_cameras(a.cameras)}
    if a.camera not in cams:
        raise SystemExit(f"no {a.camera} in {a.cameras}")
    cam = cams[a.camera]
    R_world_first = cam.T_world_cam[:3, :3]
    h0, t0, _ = aim_from_pose(cam.T_world_cam)

    frames, fps = video_frames(a.video)
    tracker = RotationTracker(cam.K, cam.dist, scale=a.scale)
    rows, lost = [], []
    for st in tracker.track(frames, fps=fps, every=a.every, log=lost.append):
        T = np.eye(4)
        T[:3, :3] = R_world_first @ st.R
        h, t, r = aim_from_pose(T)
        rows.append((st.frame, st.t, h, t, r, wrap(h - h0), st.inliers, st.keyframe, angle_deg(st.R)))
    if len(rows) < 2:
        raise SystemExit("could not follow the rotation: is the room lit, and is the turn slow?")
    arr = np.array([r[:6] for r in rows], float)

    out = a.out or HERE / "out" / f"rotation_{a.camera}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rotation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "heading_deg", "tilt_deg", "roll_deg", "pan_from_start_deg", "inliers", "keyframe",
                    "angle_from_start_deg"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}", r[6],
                        int(r[7]), f"{r[8]:.2f}"])
    plot(out / "rotation.png", arr[:, 1], arr[:, 5], arr[:, 3], a.camera)

    pan = (float(arr[:, 5].min()), float(arr[:, 5].max()))
    tilt = (float(arr[:, 3].min()), float(arr[:, 3].max()))
    m = a.margin
    mount = {a.camera: {"pan": [round(min(0.0, pan[0] + m), 1), round(max(0.0, pan[1] - m), 1)],
                        "tilt": [round(min(t0, tilt[0] + m), 1), round(max(t0, tilt[1] - m), 1)]}}
    (out / "mount.yaml").write_text(
        f"# {a.camera}, measured from {Path(str(a.video)).name}; {m:g} deg taken off each end\n"
        + yaml.safe_dump(mount, default_flow_style=None))

    end = rows[-1][8]
    print(f"{a.camera}: start heading {h0:.1f}, tilt {t0:.1f} (from {a.cameras.name})")
    print(f"  pan  {pan[0]:+.1f} .. {pan[1]:+.1f} deg from start (+ = left)")
    print(f"  tilt {tilt[0]:.1f} .. {tilt[1]:.1f} deg (0 level, 90 down)")
    print(f"  roll {arr[:, 4].min():+.1f} .. {arr[:, 4].max():+.1f} deg (should stay near its start)")
    print(f"  {len(rows)} frames placed, {len(lost)} lost, {sum(r[7] for r in rows)} keyframes")
    print(f"  end vs start: {end:.2f} deg" + ("  <- turned back to the start? then this is the drift" if end < 3 else
                                               "  (not back at the start: no drift check)"))
    if "nominal" in a.cameras.name:
        print("  K is the nominal guess: angles scale with the focal length; use the calibrated cameras.yaml")
    print(f"\nfor configs/mounts.yaml:\n  {a.camera}: {mount[a.camera]}\n-> {out}")


if __name__ == "__main__":
    main()
