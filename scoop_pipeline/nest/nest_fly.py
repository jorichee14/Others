#!/usr/bin/env python3
"""The flythrough of a NeST model on the lap the other videos use.

Run inside the NeST checkout, in its environment, with the same path json,
anchor frame and eye shift the splat_walk video was made with, so the four
videos (cloud, Sionna, splats, this) line up frame for frame:

    cd ~/workspaces/nest-splatting && conda activate nest_splatting
    python ~/workspaces/Others/scoop_pipeline/nest/nest_fly.py -s <scene> -m <model> \\
        --yaml configs/<yaml> --iteration 140000 \\
        --fly <mesh>_fly_path.json out.mp4 --frame <anchor_frame.json> [--eye-shift DZ] \\
        [--width 1280] [--unseen-level 3] [--min-views 3]

The camera is the robot's own by default: the training camera's vertical
field of view and aspect, so the frame holds what the camera could have
seen (--fov / --size to override). A wider view shows the ceiling and the
floor under the robot, which no photograph holds.

Surfaces no training photograph saw get only the coarse texture levels
(--unseen-level): a surfel counts as seen when it lies inside a training
camera's frame, in front of it, and not behind the LiDAR depth there, in
at least --min-views cameras. The fine levels of the hash grid are shared
between places by hashing, so on a surface nobody trained they hold another
place's detail: the noisy ceiling. The coarse levels give it a smooth,
plausible colour instead. --unseen-level 24 turns this off.

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


def lidar_depth(path, H, W, torch, device="cuda"):
    """<scene>/depths/<name>.png in metres on the GPU (0 where no return); None if missing."""
    import cv2
    png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    grey = (png[:, :, 0] if png.ndim == 3 else png).astype(np.float32) / 255.0
    metres = np.where(grey < 0.5, 20.0 * grey, 5.0 / np.maximum(1.0 - grey, 1e-6))
    valid = png[:, :, 3] > 0 if png.ndim == 3 and png.shape[2] == 4 else grey > 0
    metres[~valid] = 0.0
    d = torch.from_numpy(metres).to(device)
    if d.shape != (H, W):
        d = torch.nn.functional.interpolate(d[None, None], (H, W), mode="nearest")[0, 0]
    return d


def seen_counts(xyz, cameras, source_path, torch, margin=0.10):
    """Per surfel: how many training cameras see it -- inside the frame, in front,
    and not more than `margin` (plus 5%) behind the LiDAR depth at its pixel."""
    count = torch.zeros(len(xyz), dtype=torch.int32, device=xyz.device)
    ones = torch.ones((len(xyz), 1), device=xyz.device)
    X = torch.cat([xyz, ones], 1)
    print(f"visibility of {len(xyz)} surfels from {len(cameras)} training cameras ...", flush=True)
    for k, cam in enumerate(cameras):
        W, H = cam.image_width, cam.image_height
        d = lidar_depth(os.path.join(source_path, "depths", cam.image_name + ".png"), H, W, torch, xyz.device)
        if d is None:
            continue
        p = X @ cam.world_view_transform                 # row vectors: camera frame (x right, y down, z fwd)
        z = p[:, 2]
        fx = W / (2 * math.tan(cam.FoVx / 2))
        fy = H / (2 * math.tan(cam.FoVy / 2))
        zs = z.clamp(min=1e-3)
        u = (p[:, 0] / zs * fx + W / 2).long()
        v = (p[:, 1] / zs * fy + H / 2).long()
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = ok.nonzero(as_tuple=True)[0]
        dz = d[v[idx], u[idx]]
        vis = (dz > 0) & (z[idx] <= dz * 1.05 + margin)
        count[idx[vis]] += 1
        if k % 500 == 0:
            print(f"  {k} of {len(cameras)}", flush=True)
    return count


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
    parser.add_argument("--size", default="", help="WxH (default: --width at the training camera's aspect)")
    parser.add_argument("--width", type=int, default=1280, help="frame width when --size is not given")
    parser.add_argument("--fov", type=float, default=0.0, help="vertical field of view (deg; default: the training camera's)")
    parser.add_argument("--unseen-level", type=int, default=3,
                        help="texture levels for surfels no training camera saw (24: all, as for seen ones)")
    parser.add_argument("--min-views", type=int, default=3, help="training cameras that must see a surfel to count it seen")
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
    gaussians._appearance_level.data[:] = 24        # every level (a loaded model is otherwise capped at six)

    train = scene.getTrainCameras()
    c0 = train[0]
    fov = args.fov if args.fov > 0 else math.degrees(c0.FoVy)
    if args.size:
        W, H = (int(v) for v in args.size.lower().split("x"))
    else:
        W = args.width
        H = int(round(W * c0.image_height / c0.image_width))
    W, H = W - W % 2, H - H % 2
    print(f"camera: {W}x{H}, vertical field of view {fov:.1f} deg "
          f"(training camera {c0.image_width}x{c0.image_height}, {math.degrees(c0.FoVy):.1f} deg)", flush=True)

    if args.unseen_level < 24:
        seen = seen_counts(gaussians.get_xyz.detach(), train, dataset.source_path, torch)
        unseen = seen < args.min_views
        gaussians._appearance_level.data[unseen] = float(args.unseen_level)
        print(f"surfels seen by fewer than {args.min_views} training cameras: {int(unseen.sum())} of {len(seen)} "
              f"({100.0 * unseen.float().mean().item():.1f}%), drawn with {args.unseen_level} texture levels", flush=True)
    up = np.array([0.0, 0.0, 1.0])
    out = os.path.expanduser(args.fly[1])
    write, release = video_writer(out, fps, (W, H))
    print(f"flying frames {args.start}..{stop - 1} of {len(E)} at {fps} fps, {W}x{H} -> {out}", flush=True)
    with torch.no_grad():
        for i in range(args.start, stop):
            cam = FlyCamera(E[i], Lk[i], up, W, H, fov, torch, getProjectionMatrix)
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
