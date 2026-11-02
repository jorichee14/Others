#!/usr/bin/env python3
"""The flythrough of a NeST model on the lap the other videos use.

Run inside the NeST checkout, in its environment, with the same path json,
anchor frame and eye shift the splat_walk video was made with, so the four
videos (cloud, Sionna, splats, this) line up frame for frame:

    cd ~/workspaces/nest-splatting && conda activate nest_splatting
    python ~/workspaces/Others/scoop_pipeline/nest/nest_fly.py -s <scene> -m <model> \\
        --yaml configs/<yaml> --iteration 140000 \\
        --fly <mesh>_fly_path.json out.mp4 --frame <anchor_frame.json> [--eye-shift DZ] \\
        [--size 1280x720] [--fov 70]

The path's eye and look points (z up, the anchored frame) are moved into the
model's frame with the inverse of the anchor's T_N_world, as splat_walk does;
each frame is a pinhole camera looking from eye to look, x right, y down,
z forward (COLMAP's), rendered with the trained surfels and texture. Novel
views get the session's colour term only (identity for a single session):
a steady exposure along the lap, which is what the per-image terms were
for.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())                                   # the NeST checkout


class FlyCamera:
    """What NeST's render() reads of a camera, for one eye/look pair."""

    def __init__(self, eye, look, up, W, H, fov_deg, torch, getProjectionMatrix, znear=0.01, zfar=100.0):
        eye, look, up = np.asarray(eye, float), np.asarray(look, float), np.asarray(up, float)
        f = look - eye
        f /= np.linalg.norm(f) or 1.0
        r = np.cross(f, up)
        r /= np.linalg.norm(r) or 1.0
        d = np.cross(f, r)
        R = np.vstack([r, d, f])                      # world -> camera rotation
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, -R @ eye
        self.image_width, self.image_height = W, H
        self.FoVy = math.radians(fov_deg)
        self.FoVx = 2 * math.atan(math.tan(self.FoVy / 2) * W / H)
        self.znear, self.zfar = znear, zfar
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_name = "fly"
        self.colmap_id = 1


def video_writer(out, fps, size):
    if shutil.which("ffmpeg"):
        p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                              "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                             stdin=subprocess.PIPE)
        return (lambda bgr: p.stdin.write(np.ascontiguousarray(bgr).tobytes())), (lambda: (p.stdin.close(), p.wait()))
    import cv2
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return vw.write, vw.release


def main():
    import torch
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from gaussian_renderer import render, GaussianModel
    from hash_encoder.modules import INGP
    from hash_encoder.config import Config
    from train import merge_cfg_to_args
    from utils.graphics_utils import getProjectionMatrix

    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--fly", nargs=2, required=True, metavar=("PATH_JSON", "OUT_MP4"))
    parser.add_argument("--frame", default="", help="anchor_frame.json: the path is in the anchored frame")
    parser.add_argument("--eye-shift", type=float, default=0.0, metavar="DZ", help="raise (+) or lower the eye (m)")
    parser.add_argument("--size", default="1280x720", help="WxH")
    parser.add_argument("--fov", type=float, default=70.0, help="vertical field of view (deg)")
    parser.add_argument("--start", type=int, default=0, help="first frame of the path to render")
    parser.add_argument("--count", type=int, default=0, help="frames to render (0: to the end)")
    args = get_combined_args(parser)
    cfg_model = Config(args.yaml)
    merge_cfg_to_args(args, cfg_model)
    args.data_device = "cpu"

    pj = json.load(open(os.path.expanduser(args.fly[0])))
    E, Lk = np.array(pj["eye"], float), np.array(pj["look"], float)
    if args.eye_shift:
        E[:, 2] += args.eye_shift
        Lk[:, 2] += args.eye_shift
    if args.frame:
        af = json.load(open(os.path.expanduser(args.frame)))
        Ti = np.linalg.inv(np.array(af["T_N_world"], float))      # anchored -> the model's frame
        E = E @ Ti[:3, :3].T + Ti[:3, 3]
        Lk = Lk @ Ti[:3, :3].T + Ti[:3, 3]
    fps = int(pj.get("fps", 30))
    W, H = (int(v) for v in args.size.lower().split("x"))
    W, H = W - W % 2, H - H % 2
    stop = len(E) if args.count <= 0 else min(len(E), args.start + args.count)

    ingp_model = INGP(cfg_model).to("cuda")
    ingp_model.load_model(args.model_path, args.iteration)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    args.cfg = cfg_model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    gaussians.base_opacity = cfg_model.surfel.tg_base_alpha
    beta = cfg_model.surfel.tg_beta
    gaussians.XYZ_TYPE = "UV"
    ingp_model.set_active_levels(args.iteration)
    up = np.array([0.0, 0.0, 1.0])
    out = os.path.expanduser(args.fly[1])
    write, release = video_writer(out, fps, (W, H))
    print(f"flying frames {args.start}..{stop - 1} of {len(E)} at {fps} fps, {W}x{H} -> {out}", flush=True)
    with torch.no_grad():
        for i in range(args.start, stop):
            cam = FlyCamera(E[i], Lk[i], up, W, H, args.fov, torch, getProjectionMatrix)
            img = render(cam, gaussians, pipe, background, ingp=ingp_model, beta=beta,
                         iteration=args.iteration, cfg=cfg_model)["render"]
            rgb = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            write(rgb[:, :, ::-1])
            if (i - args.start) % (10 * fps) == 0:
                print(f"  {(i - args.start) // fps} s of {(stop - args.start) // fps}", flush=True)
    release()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
