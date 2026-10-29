#!/usr/bin/env python3
"""The views the capture never took: sideways and backward, from every pose.

A splat is only constrained along the directions it was photographed from.
One forward-facing pass down a corridor photographs every wall at the same
glancing angle, so each Gaussian's shape across that angle is never pinned
down, and from any other angle it is a streak. LetsGo's scanner avoids this
by splitting a fisheye into five pinholes at every pose; the ZED is one
pinhole, facing forward. This makes the other pinholes after the fact: the
same poses, turned left, right and back, as a scene of virtual cameras that
splat_walk --render-views can render and a diffusion fixer can clean into
pseudo-views for retraining.

    python3 scoop_pipeline/datasets/virtual_views.py <scene> <out_scene> \\
        [--every 10] [--yaws=-90,90,180] [--pitches 0]

(--yaws takes the equals sign: a leading minus is otherwise read as a flag.)

Writes <out_scene>/sparse/0/{cameras.txt, images.txt} -- the real camera's
intrinsics, the virtual poses -- and <out_scene>/sources.json naming, for
each virtual view, the real image it was turned from, which is the fixer's
reference. The yaw is about the world's up (GLIM's z), so a 180 looks back
down the corridor the robot came along, not at the floor.
"""
import argparse
import json
import math
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def quaternion_rotation(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation_quaternion(R):
    """w, x, y, z of a rotation matrix (the signs COLMAP expects)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z


def read_images(path):
    """[(id, camera_id, name, c2w)] from a COLMAP images.txt, in file order."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            v = line.split()
            if len(v) >= 10 and v[0].isdigit():
                R = quaternion_rotation(*(float(c) for c in v[1:5]))
                t = np.array([float(c) for c in v[5:8]])
                c2w = np.eye(4)
                c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
                out.append((int(v[0]), int(v[8]), v[9], c2w))
    return out


def turned(c2w, yaw_deg, pitch_deg):
    """The same camera centre, turned about the world's up and then tilted.

    Yaw about world z keeps a level camera level: a 180 looks back along the
    corridor. The tilt is then about the camera's own right axis, so an
    upward pitch looks at the ceiling whichever way the yaw left it facing.
    """
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    R = Rz @ c2w[:3, :3]
    if pitch_deg:
        b = math.radians(pitch_deg)
        right = R[:, 0]
        K = np.array([[0, -right[2], right[1]], [right[2], 0, -right[0]], [-right[1], right[0], 0]])
        R = (np.eye(3) + math.sin(b) * K + (1 - math.cos(b)) * K @ K) @ R
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, c2w[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="the real scene (sparse/0/images.txt)")
    ap.add_argument("out", help="the scene of virtual cameras to write")
    ap.add_argument("--every", type=int, default=10, help="turn every Nth real pose")
    ap.add_argument("--yaws", default="-90,90,180",
                    help="degrees about up; 0 is the real view (write --yaws=-90,... for a leading minus)")
    ap.add_argument("--pitches", default="0", help="degrees of tilt, + looks up; one per yaw or one for all")
    a = ap.parse_args()

    scene, out = os.path.expanduser(a.scene), os.path.expanduser(a.out)
    yaws = [float(v) for v in a.yaws.split(",") if v.strip()]
    pitches = [float(v) for v in a.pitches.split(",") if v.strip()]
    if len(pitches) == 1:
        pitches = pitches * len(yaws)
    if len(pitches) != len(yaws):
        sys.exit("--pitches: one value, or one per yaw")

    images = read_images(os.path.join(scene, "sparse", "0", "images.txt"))
    if not images:
        sys.exit(f"{scene}: no cameras in sparse/0/images.txt")
    os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
    shutil.copyfile(os.path.join(scene, "sparse", "0", "cameras.txt"),
                    os.path.join(out, "sparse", "0", "cameras.txt"))

    rows, sources, next_id = [], {}, 1
    for index in range(0, len(images), max(1, a.every)):
        _, camera_id, name, c2w = images[index]
        stem = re.sub(r"\.\w+$", "", name)
        for yaw, pitch in zip(yaws, pitches):
            label = f"y{int(round(yaw)):+04d}" + (f"p{int(round(pitch)):+03d}" if pitch else "")
            virtual = f"{stem}_{label}.jpg"
            w2c = np.linalg.inv(turned(c2w, yaw, pitch))
            qw, qx, qy, qz = rotation_quaternion(w2c[:3, :3])
            t = w2c[:3, 3]
            rows.append(f"{next_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                        f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {camera_id} {virtual}")
            rows.append("")                        # the (empty) points line COLMAP expects
            sources[virtual] = {"from": name, "yaw": yaw, "pitch": pitch}
            next_id += 1

    with open(os.path.join(out, "sparse", "0", "images.txt"), "w", encoding="utf-8") as handle:
        handle.write("# virtual cameras turned from %s: every %d, yaws %s, pitches %s\n"
                     % (scene, a.every, a.yaws, a.pitches))
        handle.write("\n".join(rows) + "\n")
    with open(os.path.join(out, "sources.json"), "w", encoding="utf-8") as handle:
        json.dump({"scene": scene, "every": a.every, "views": sources}, handle, indent=1)
    print(f"{len(sources)} virtual cameras from {len(range(0, len(images), a.every))} poses "
          f"x {len(yaws)} turn(s) -> {out}")
    print(f"render them:  splat_walk.py <level_5.ply> --render-views {out} <renders>")


if __name__ == "__main__":
    main()
