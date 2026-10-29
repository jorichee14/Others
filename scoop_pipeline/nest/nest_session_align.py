#!/usr/bin/env python3
"""The rigid transform one session is off by, in the other's frame.

September's cameras were put into October's frame by an ICP between the two
LiDAR maps. A model trained on October alone, rendered at those cameras, is
the right corridor drawn from a camera standing slightly elsewhere: a shift
the same in every view, with depth-dependent disagreement inside each, and
none of it growing with speed -- the signature of one rigid residual shared
by every camera of the session. That residual is one transform, and the
trained model is the instrument to read it with: render a frame from a
moved camera, score it against its own photograph, and the move that scores
best across the session is the error the alignment left.

Run inside the NeST checkout, in its environment:

    cd ~/workspaces/nest-splatting && conda activate nest_splatting
    python ~/workspaces/Others/scoop_pipeline/nest/nest_session_align.py \\
        -s <session_scene> -m <model_dir> --yaml configs/scoop_indoor.yaml \\
        --iteration 70000 [--frames 40] [--write-images] [--also-fix <both_scene> --camera 2]

The correction is in the WORLD frame (c2w' = dT @ c2w), applied the same to
every camera, searched by coordinate descent coarse to fine at half
resolution. --write-images rewrites the session scene's images.txt with the
corrected poses (the original kept as images_before_align.txt); --also-fix
applies the same correction to that camera id's rows in a two-session scene,
so a model trained on both sessions starts from cameras that agree.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())                                   # the NeST checkout
sys.path.insert(0, os.path.join(HERE, "..", "datasets"))
from splat_extrinsic import descend, transform_of, psnr            # noqa: E402
from virtual_views import rotation_quaternion                      # noqa: E402


def w2c_of(cam):
    """The camera's world-to-camera, as NeST stores it (transposed on the GPU)."""
    return cam.world_view_transform.transpose(0, 1).detach().cpu().numpy().astype(np.float64)


def move_camera(cam, w2c, torch):
    """Put a camera at a new world-to-camera, touching only what render() reads."""
    wvt = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
    cam.world_view_transform = wvt
    cam.full_proj_transform = (wvt.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = wvt.inverse()[3, :3]


def rewrite_images(path, correction, camera_id=None):
    """images.txt with every (or one camera id's) pose moved by the world-frame correction."""
    inv = np.linalg.inv(correction)
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    out, moved = [], 0
    for line in lines:
        v = line.split()
        if len(v) >= 10 and v[0].isdigit() and (camera_id is None or int(v[8]) == camera_id):
            w, x, y, z = (float(c) for c in v[1:5])
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = R, [float(c) for c in v[5:8]]
            w2c = w2c @ inv                      # c2w' = dT @ c2w  <=>  w2c' = w2c @ dT^-1
            qw, qx, qy, qz = rotation_quaternion(w2c[:3, :3])
            t = w2c[:3, 3]
            out.append(" ".join([v[0], f"{qw:.9f}", f"{qx:.9f}", f"{qy:.9f}", f"{qz:.9f}",
                                 f"{t[0]:.9f}", f"{t[1]:.9f}", f"{t[2]:.9f}", v[8], v[9]]))
            moved += 1
        else:
            out.append(line)
    backup = os.path.join(os.path.dirname(path), "images_before_align.txt")
    if not os.path.exists(backup):
        os.replace(path, backup)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")
    return moved, backup


def per_frame_search(chosen, args, render_one, torch):
    """Each frame refined alone, six parameters in its own camera frame.

    The whole-session search asks whether one move fixes every view. This
    asks whether ANY move fixes each view: a frame that gains several dB
    from its own small move has a pose error; a frame that gains nothing
    however it is moved is looking at something the model does not hold
    (a thing that moved between sessions, a colour the camera saw
    differently, a surface never captured in October).
    """
    quiet = lambda *a, **k: None
    rows = []
    for k, (cam, w2c, image) in enumerate(chosen):
        def score(params):
            with torch.no_grad():
                move_camera(cam, np.linalg.inv(transform_of(params[:6])) @ w2c, torch)   # c2w' = c2w @ dT
                out = render_one(cam)
            rgb = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return psnr(rgb, image)
        before = score([0.0] * 6)
        best, after = descend(score, [0.0] * 6, [args.rot_step] * 3 + [args.trans_step] * 3,
                              [0.02] * 3 + [0.002] * 3, log=quiet)
        move_camera(cam, w2c, torch)
        shift = 100 * math.sqrt(sum(v * v for v in best[3:6]))
        turn = math.sqrt(sum(v * v for v in best[:3]))
        rows.append((cam.image_name, before, after, after - before, shift, turn))
        print(f"  {k + 1:3d} {cam.image_name:>28s}  {before:6.2f} -> {after:6.2f} dB  ({after - before:+5.2f})"
              f"  moved {shift:5.1f} cm, turned {turn:5.2f} deg", flush=True)
    gains = np.array([r[3] for r in rows])
    shifts = np.array([r[4] for r in rows])
    print(f"\nper-frame refinement on {len(rows)} frames: median gain {np.median(gains):+.2f} dB, "
          f"{int((gains > 1).sum())} frames gain over 1 dB, {int((gains > 3).sum())} over 3 dB; "
          f"median move {np.median(shifts):.1f} cm (max {shifts.max():.1f})")
    print("read it so: gains of several dB with moves of centimetres = per-frame pose error (fix: refine poses);"
          " gains near zero whatever the move = the content differs, not the camera")
    out = args.out or os.path.join(args.source_path, "per_frame_align.json")
    json.dump([{"image": r[0], "psnr_before": r[1], "psnr_after": r[2], "move_cm": r[4], "turn_deg": r[5]}
               for r in rows], open(out, "w"), indent=1)
    print(f"wrote {out}")


def main():
    import torch
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from hash_encoder.modules import INGP
    from hash_encoder.config import Config
    from train import merge_cfg_to_args

    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--rot-step", type=float, default=1.0, help="first rotation step (degrees)")
    parser.add_argument("--trans-step", type=float, default=0.10, help="first translation step (m)")
    parser.add_argument("--out", default="", help="json of the correction (default <scene>/session_align.json)")
    parser.add_argument("--write-images", action="store_true", help="rewrite the scene's images.txt")
    parser.add_argument("--also-fix", default="", help="a two-session scene to apply the same move to")
    parser.add_argument("--camera", type=int, default=-1, help="--also-fix: the camera id of this session there")
    parser.add_argument("--force", action="store_true", help="write poses even when the gain is under half a dB")
    parser.add_argument("--per-frame", action="store_true",
                        help="instead: refine each chosen frame on its own (camera frame), to tell pose from content")
    args = get_combined_args(parser)
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)
    args.resolution = 2                      # half size: four times the frames per second
    args.data_device = "cpu"

    ingp_model = INGP(cfg_model).to("cuda")
    ingp_model.load_model(args.model_path, args.iteration)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    args.cfg = cfg_model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(args.iteration)

    cameras = scene.getTrainCameras() + scene.getTestCameras()
    picks = np.linspace(0, len(cameras) - 1, min(args.frames, len(cameras))).astype(int)
    chosen = [(cameras[i], w2c_of(cameras[i]),
               (cameras[i].original_image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
              for i in picks]
    print(f"{len(chosen)} of {len(cameras)} cameras, at half resolution", flush=True)

    def score(params):
        dT = transform_of(params[:6])
        inv = np.linalg.inv(dT)
        total = 0.0
        with torch.no_grad():
            for cam, w2c, image in chosen:
                move_camera(cam, w2c @ inv, torch)
                out = render(cam, gaussians, pipe, background, ingp=ingp_model, beta=beta,
                             iteration=args.iteration, cfg=cfg_model)["render"]
                rgb = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                total += psnr(rgb, image)
        return total / len(chosen)

    if args.per_frame:
        per_frame_search(chosen, args, render_one=lambda cam: render(
            cam, gaussians, pipe, background, ingp=ingp_model, beta=beta,
            iteration=args.iteration, cfg=cfg_model)["render"], torch=torch)
        return

    start = [0.0] * 6
    before = score(start)
    best, after = descend(score, start, [args.rot_step] * 3 + [args.trans_step] * 3,
                          [0.02] * 3 + [0.002] * 3)
    for cam, w2c, _ in chosen:                 # leave the cameras as they were
        move_camera(cam, w2c, torch)

    rx, ry, rz, tx, ty, tz = best
    print(f"\nPSNR {before:.2f} -> {after:.2f} dB ({after - before:+.2f}) on {len(chosen)} frames")
    print(f"the session sits {100 * tx:+.1f} cm x, {100 * ty:+.1f} cm y, {100 * tz:+.1f} cm z "
          f"from where the alignment put it, turned {rx:+.3f}, {ry:+.3f}, {rz:+.3f} deg about x, y, z "
          f"(world frame)")
    if after - before < 0.5:
        print("that recovers under half a dB: the alignment is not what these views are missing")

    correction = transform_of(best)
    out = args.out or os.path.join(args.source_path, "session_align.json")
    json.dump({"dT_world": correction.round(9).tolist(), "rotation_deg": [rx, ry, rz],
               "translation_m": [tx, ty, tz], "psnr_before": before, "psnr_after": after,
               "frames": len(chosen), "model": args.model_path, "scene": args.source_path,
               "iteration": args.iteration}, open(out, "w"), indent=1)
    print(f"wrote {out}")

    if (args.write_images or args.also_fix) and after - before < 0.5 and not args.force:
        print("poses left untouched (--force to write them anyway)")
        return
    if args.write_images:
        moved, backup = rewrite_images(os.path.join(args.source_path, "sparse", "0", "images.txt"), correction)
        print(f"moved {moved} cameras in {args.source_path}/sparse/0/images.txt (original: {backup})")
    if args.also_fix:
        if args.camera < 0:
            sys.exit("--also-fix needs --camera: which id this session is in that scene")
        moved, backup = rewrite_images(os.path.join(os.path.expanduser(args.also_fix), "sparse", "0", "images.txt"),
                                       correction, camera_id=args.camera)
        print(f"moved {moved} cameras of id {args.camera} in {args.also_fix} (original: {backup})")


if __name__ == "__main__":
    main()
