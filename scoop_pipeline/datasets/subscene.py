#!/usr/bin/env python3
"""One session's cameras out of a two-session scene, as a scene of their own.

The two-session scene holds October's RealSense and September's ZED in one
frame, told apart by camera id. A model trained on October alone can be
rendered at September's cameras -- trajectories it never drove and a lens it
never saw -- only if those cameras come as a scene the renderer will load:
sparse/0 with just them, the images and depths beside, the surface points
for the loader's sake. Nothing is copied that can be linked.

    python3 scoop_pipeline/datasets/subscene.py <both_scene> <out_scene> --camera 2

Then NeST renders every camera in it as a "train" view (no --eval, so none
are held out) and metrics.py scores train/ -- the real novel-trajectory
number:

    python eval_render.py -s <out_scene> -m <october_model> --yaml ... \\
        --iteration 70000 --skip_test --skip_mesh
    python metrics.py -m <october_model>
"""
import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="the scene holding several sessions (sparse/0, images/, depths/)")
    ap.add_argument("out", help="the scene to write, holding one of them")
    ap.add_argument("--camera", type=int, required=True, help="the camera id to keep (cameras.txt)")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    a = ap.parse_args()

    scene, out = os.path.expanduser(a.scene), os.path.expanduser(a.out)
    sparse_in, sparse_out = os.path.join(scene, "sparse", "0"), os.path.join(out, "sparse", "0")
    os.makedirs(sparse_out, exist_ok=True)
    for sub in ("images", "depths"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    kept_cameras = []
    with open(os.path.join(sparse_in, "cameras.txt"), encoding="utf-8") as src:
        for line in src:
            v = line.split()
            if len(v) >= 8 and v[0].isdigit() and int(v[0]) == a.camera:
                kept_cameras.append(line)
    if not kept_cameras:
        sys.exit(f"{scene}: no camera {a.camera} in cameras.txt")
    with open(os.path.join(sparse_out, "cameras.txt"), "w", encoding="utf-8") as dst:
        dst.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        dst.writelines(kept_cameras)

    names, rows = [], []
    with open(os.path.join(sparse_in, "images.txt"), encoding="utf-8") as src:
        lines = src.read().splitlines()
    def is_pose(line):
        v = line.split()
        return len(v) >= 10 and v[0].isdigit()

    # COLMAP writes each image as a pose line and a points line; keep both.
    for i, line in enumerate(lines):
        if is_pose(line) and int(line.split()[8]) == a.camera:
            rows.append(line)
            following = lines[i + 1] if i + 1 < len(lines) else ""
            rows.append("" if is_pose(following) else following)
            names.append(line.split()[9])
    if not names:
        sys.exit(f"{scene}: no images with camera {a.camera}")
    with open(os.path.join(sparse_out, "images.txt"), "w", encoding="utf-8") as dst:
        dst.write(f"# camera {a.camera} of {scene}: {len(names)} images\n")
        dst.write("\n".join(rows) + "\n")

    for name in names:
        for sub, ext in (("images", None), ("depths", ".png")):
            src = os.path.join(scene, sub, name if ext is None else os.path.splitext(name)[0] + ext)
            dst = os.path.join(out, sub, os.path.basename(src))
            if not os.path.exists(src):
                continue
            if os.path.lexists(dst):
                os.remove(dst)
            if a.copy:
                shutil.copyfile(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)

    points = os.path.join(sparse_in, "points3D.ply")
    target = os.path.join(sparse_out, "points3D.ply")
    if os.path.exists(points):
        if os.path.lexists(target):
            os.remove(target)
        os.symlink(os.path.abspath(points), target)
    print(f"{len(names)} cameras of id {a.camera} -> {out} "
          f"({'copied' if a.copy else 'linked'} images and depths; points3D.ply linked)")


if __name__ == "__main__":
    main()
