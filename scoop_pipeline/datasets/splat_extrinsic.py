#!/usr/bin/env python3
"""Find the one correction every camera in a scene is missing.

splat_psnr.py found the training images misaligned with the model by an
amount that depends on depth, the same in every frame and only a little
worse at speed: a camera placed about a centimetre from where the rig says
it is, with a small timestamp offset on top. Neither is a per-frame error,
so neither is anything bundle adjustment with a fixed extrinsic can reach;
both are one number each, and the trained model is the instrument to read
them with. Render a frame from a corrected pose, score it against its own
image, and the correction that scores best across the building is the one
the rig was missing.

Run it in the LOD-3DGS environment (the rasteriser):

    conda activate lod3dgs; export PYTHONNOUSERSITE=1
    python3 scoop_pipeline/datasets/splat_extrinsic.py <level_5.ply> --scene <scene> \\
        [--frames 60] [--scale 0.5] [--time] [--out correction.json] \\
        [--write-images corrected_images.txt]

The correction is a rigid transform in the CAMERA frame (x right, y down,
z forward), applied the same to every camera: c2w' = c2w @ dT. --time also
searches a time offset in frames, each camera slid along its own path by
that fraction of a frame -- the part of the error that grows with speed.
It searches by coordinate descent, coarse to fine, on a few dozen frames
spread over the whole scene at reduced resolution, and reports the PSNR
before and after so the gain is on the record. --write-images writes the
corrected sparse/0/images.txt for retraining.

Reads like splat_psnr: if the search recovers several dB, this was the
fault; if it recovers almost nothing, it was not, whatever the symptom
looked like.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- the maths
def rotation_of(rx, ry, rz):
    """A small rotation from three angles (degrees) about the camera's axes."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform_of(params):
    """(rx, ry, rz in degrees, tx, ty, tz in metres) -> 4 x 4 in the camera frame."""
    T = np.eye(4)
    T[:3, :3] = rotation_of(*params[:3])
    T[:3, 3] = params[3:6]
    return T


def slide(frames, index, delta):
    """The camera-to-world a fraction of a frame before or after this one.

    The rotation is this frame's; the translation moves toward the neighbour
    in the direction of delta. For the tenths of a frame a stamp offset is,
    that is the pose the image was really taken from.
    """
    _, _, here = frames[index]
    if abs(delta) < 1e-9:
        return here.copy()
    step = 1 if delta > 0 else -1
    other = index + step
    if other < 0 or other >= len(frames):
        return here.copy()
    moved = here.copy()
    moved[:3, 3] = here[:3, 3] + abs(delta) * (frames[other][2][:3, 3] - here[:3, 3])
    return moved


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 0 else 10 * math.log10(255.0 ** 2 / mse)


def descend(score, start, steps, floors, rounds=40, log=print):
    """Coordinate descent, coarse to fine, on a score to maximise.

    One parameter at a time, each tried both ways at its current step; a
    step that improved nothing is halved, and the search stops when every
    step is under its floor. Slow and sure: a few hundred renders of a few
    dozen frames, and no gradient through a rasteriser to get wrong.
    """
    best = list(start)
    best_score = score(best)
    steps = list(steps)
    log(f"  start {best_score:.3f} dB")
    for round_ in range(rounds):
        improved = False
        for i in range(len(best)):
            if steps[i] < floors[i]:
                continue
            for sign in (1.0, -1.0):
                trial = list(best)
                trial[i] += sign * steps[i]
                value = score(trial)
                if value > best_score + 1e-4:
                    best, best_score, improved = trial, value, True
                    break
        if not improved:
            steps = [s / 2.0 for s in steps]
        log(f"  round {round_ + 1:2d}  {best_score:.3f} dB  "
            + "  ".join(f"{v:+.4f}" for v in best)
            + ("" if improved else "   (steps halved)"))
        if all(s < f for s, f in zip(steps, floors)):
            break
    return best, best_score


# ------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", help="the trained model's finest level (level_5.ply, or a clean copy)")
    ap.add_argument("--scene", required=True, help="the scene folder (sparse/0 and images/)")
    ap.add_argument("--frames", type=int, default=60, help="frames to score, spread over the scene")
    ap.add_argument("--scale", type=float, default=0.5, help="render and compare at this fraction of full size")
    ap.add_argument("--time", action="store_true", help="also search a time offset (frames)")
    ap.add_argument("--rot-step", type=float, default=0.5, help="first rotation step (degrees)")
    ap.add_argument("--trans-step", type=float, default=0.02, help="first translation step (m)")
    ap.add_argument("--out", default=None, help="write the correction here (json)")
    ap.add_argument("--write-images", default=None, help="write a corrected images.txt here")
    a = ap.parse_args()

    from splat_walk import load_splats, scene_cameras, camera, render
    scene = os.path.expanduser(a.scene)
    cams, frames = scene_cameras(scene)
    print(f"loading {a.ply} ...", flush=True)
    g, deg = load_splats(os.path.expanduser(a.ply))
    print(f"{len(g['xyz'])} splats, SH degree {deg}; {len(frames)} cameras in the scene")

    # Frames spread over the whole scene, never the first or last (a time
    # offset needs a neighbour on both sides).
    picks = np.linspace(1, len(frames) - 2, min(a.frames, len(frames) - 2)).astype(int)
    chosen = []
    for index in picks:
        name, cid, _ = frames[index]
        image = cv2.imread(os.path.join(scene, "images", name))
        if image is None:
            continue
        W, H, fx, fy, cx, cy = cams[cid]
        w, h = int(round(W * a.scale)), int(round(H * a.scale))
        chosen.append((int(index), cid, cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), w, h))
    print(f"scoring {len(chosen)} frames at {a.scale:g} scale", flush=True)

    def score(params):
        rx, ry, rz, tx, ty, tz = params[:6]
        delta = params[6] if len(params) > 6 else 0.0
        dT = transform_of((rx, ry, rz, tx, ty, tz))
        total = 0.0
        for index, cid, image, w, h in chosen:
            W, H, fx, fy, cx, cy = cams[cid]
            fov = 2 * math.degrees(math.atan(H / (2 * fy)))
            c2w = slide(frames, index, delta) @ dT
            eye, fwd, up = c2w[:3, 3], c2w[:3, 2], -c2w[:3, 1]
            bgr = render(g, deg, camera(eye, eye + fwd, up, w, h, fov), w, h)
            total += psnr(bgr, image)
        return total / len(chosen)

    start = [0.0] * 6 + ([0.0] if a.time else [])
    steps = [a.rot_step] * 3 + [a.trans_step] * 3 + ([0.25] if a.time else [])
    floors = [0.02] * 3 + [0.001] * 3 + ([0.02] if a.time else [])
    before = score(start)
    best, after = descend(score, start, steps, floors)

    rx, ry, rz, tx, ty, tz = best[:6]
    delta = best[6] if a.time else 0.0
    print(f"\nPSNR {before:.2f} -> {after:.2f} dB ({after - before:+.2f}) on {len(chosen)} frames")
    print(f"the camera is {1000 * tx:+.1f} mm right, {1000 * ty:+.1f} mm down, {1000 * tz:+.1f} mm "
          f"forward of where the rig put it; turned {rx:+.3f}, {ry:+.3f}, {rz:+.3f} deg about x, y, z"
          + (f"; stamped {delta:+.3f} frames off" if a.time else ""))
    if after - before < 0.5:
        print("that recovers under half a dB: a single correction is not what the images are "
              "missing, whatever the symptom looked like")

    result = {"dT_camera": transform_of(best[:6]).round(8).tolist(),
              "rotation_deg": [rx, ry, rz], "translation_m": [tx, ty, tz],
              "time_offset_frames": delta, "psnr_before": before, "psnr_after": after,
              "frames_scored": len(chosen), "scale": a.scale, "ply": os.path.expanduser(a.ply),
              "scene": scene}
    out = os.path.expanduser(a.out) if a.out else os.path.join(scene, "correction.json")
    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {out}")

    if a.write_images:
        dT = transform_of(best[:6])
        lines = []
        with open(os.path.join(scene, "sparse", "0", "images.txt"), encoding="utf-8") as fh:
            raw = fh.read().splitlines()
        by_name = {name: i for i, (name, _, _) in enumerate(frames)}
        for line in raw:
            v = line.split()
            if len(v) >= 10 and v[0].isdigit() and v[9] in by_name:
                c2w = slide(frames, by_name[v[9]], delta) @ dT
                w2c = np.linalg.inv(c2w)
                R, t = w2c[:3, :3], w2c[:3, 3]
                qw = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
                qx = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
                qy = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
                qz = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
                lines.append(" ".join([v[0], f"{qw:.9f}", f"{qx:.9f}", f"{qy:.9f}", f"{qz:.9f}",
                                       f"{t[0]:.9f}", f"{t[1]:.9f}", f"{t[2]:.9f}", v[8], v[9]]))
            else:
                lines.append(line)
        with open(os.path.expanduser(a.write_images), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {a.write_images}: every camera corrected; the depth maps beside the images "
              "were rendered at the OLD poses -- rebuild the scene or train without the depth loss")


if __name__ == "__main__":
    main()
