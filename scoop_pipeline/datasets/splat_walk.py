#!/usr/bin/env python3
"""Walk inside a trained LOD-3DGS model (or any 3DGS point_cloud.ply), or
fly it round the loop `lod3dgs.py fly` flew through the mesh, with
LOD-3DGS's own rasteriser. Run it in the LOD-3DGS environment (torch,
diff_gaussian_rasterization, plyfile; the window is OpenCV's):

    conda activate lod3dgs; export PYTHONNOUSERSITE=1
    python3 scoop_pipeline/datasets/splat_walk.py <run>/point_cloud/iteration_300000/level_5.ply \\
        --scene <scene folder>                       # starts at its first camera
    python3 scoop_pipeline/datasets/splat_walk.py <level_5.ply> --fly <mesh>_fly_path.json out.mp4 \\
        --frame <data>/frames/anchor_frame.json      # the loop was flown in the anchored frame
        [--eye-shift -1.0]                           # the eye down to 0.6 m, nearer the training cameras

The level files beside the one given (level_0.ply the coarsest ...
level_5.ply the finest) are all loaded and each Gaussian is drawn from the
coarsest level whose cell is still small on screen at its distance (--px 2
pixels), as LOD-3DGS's own renderer does; --no-lod draws the one file.
Turn with the arrow keys or by dragging with the left mouse button; move
with W forward, S back, A left, D right, Q up, Z down; R back to the
start; Escape out. The model is in the map's frame (GLIM's world, z up,
the frame the scene was made in); the fly path of the anchored mesh is
moved back with --frame (the inverse of its T_N_world).
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch


def load_splats(path, device="cuda"):
    """A 3DGS .ply -> dict of tensors for the rasteriser (xyz, features
    (N x K x 3, K = (deg + 1)^2), opacity, scales, rotations) and deg."""
    from plyfile import PlyData
    el = PlyData.read(path).elements[0]
    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], 1).astype(np.float32)
    dc = np.stack([np.asarray(el[f"f_dc_{i}"]) for i in range(3)], 1).astype(np.float32)        # N x 3
    rest = sorted([p.name for p in el.properties if p.name.startswith("f_rest_")],
                  key=lambda s: int(s.split("_")[-1]))
    K = len(rest) // 3 + 1
    deg = int(round(math.sqrt(K))) - 1
    fr = np.stack([np.asarray(el[n]) for n in rest], 1).astype(np.float32) if rest else np.zeros((len(xyz), 0), np.float32)
    fr = fr.reshape(len(xyz), 3, K - 1).transpose(0, 2, 1)                                     # N x (K-1) x 3
    feats = np.concatenate([dc[:, None, :], fr], 1)
    op = 1.0 / (1.0 + np.exp(-np.asarray(el["opacity"], np.float32)))[:, None]
    sc = np.exp(np.stack([np.asarray(el[f"scale_{i}"]) for i in range(3)], 1).astype(np.float32))
    rot = np.stack([np.asarray(el[f"rot_{i}"]) for i in range(4)], 1).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)      # noqa: E731
    return {"xyz": t(xyz), "features": t(feats), "opacity": t(op), "scales": t(sc), "rotations": t(rot)}, deg


def load_levels(path, cell=0.04):
    """The level files beside `path` (level_<k>.ply, k = 0 the coarsest; the
    given one for its k, e.g. a _clean copy): [(cell size m, splats)] from
    the coarsest, the finest's cells `cell` m and each coarser level's
    double; or just the given file when it is not a level file."""
    import re
    d, base = os.path.split(path)
    m = re.match(r"level_(\d+)", base)
    if not m:
        g, deg = load_splats(path)
        return [(cell, g)], deg
    ks = sorted({int(re.match(r"level_(\d+)", f).group(1)) for f in os.listdir(d)
                 if re.match(r"level_(\d+)\.ply$", f)} | {int(m.group(1))})
    kmax = ks[-1]
    levels, deg = [], None
    for k in ks:
        f = path if k == int(m.group(1)) else os.path.join(d, f"level_{k}.ply")
        g, deg = load_splats(f)
        levels.append((cell * 2 ** (kmax - k), g))
        print(f"  level {k}: {len(g['xyz'])} splats, cells {cell * 2 ** (kmax - k):.2f} m ({os.path.basename(f)})",
              flush=True)
    return levels, deg


def pick_levels(levels, eye, f_px, px=2.0):
    """The splats to draw this frame, one level per Gaussian by its distance
    from the eye: the coarsest level whose cell projects to px pixels or
    less there (a cell c at distance d covers c * f_px / d pixels), the
    finest nearer than that. One dict of tensors, concatenated."""
    if len(levels) == 1:
        return levels[0][1]
    eye_t = torch.tensor(eye, dtype=torch.float32, device="cuda")
    cells = [c for c, _ in levels]                      # coarsest first
    parts = []
    for i, (c, g) in enumerate(levels):
        d = torch.linalg.norm(g["xyz"] - eye_t, dim=1)
        lo = c * f_px / px                                # from here this level's cells are small enough
        hi = cells[i - 1] * f_px / px if i > 0 else float("inf")   # nearer than the coarser one's
        sel = (d >= lo) & (d < hi) if i < len(levels) - 1 else (d < hi)
        if sel.any():
            parts.append({k: v[sel] for k, v in g.items()})
    return {k: torch.cat([p_[k] for p_ in parts]) for k in parts[0]}


def camera(eye, look, up, W, H, fov_deg, znear=0.05, zfar=1000.0, device="cuda"):
    """The rasteriser's matrices for a camera at eye looking at look: x
    right, y down, z forward (COLMAP's), vertical field of view fov_deg."""
    eye, look, up = np.asarray(eye, float), np.asarray(look, float), np.asarray(up, float)
    f = look - eye
    f /= np.linalg.norm(f) or 1.0
    r = np.cross(f, up)
    r /= np.linalg.norm(r) or 1.0
    d = np.cross(f, r)
    R = np.vstack([r, d, f])                      # world -> camera rotation
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R, -R @ eye
    fovy = math.radians(fov_deg)
    fovx = 2 * math.atan(math.tan(fovy / 2) * W / H)
    tx, ty = math.tan(fovx / 2), math.tan(fovy / 2)
    P = np.zeros((4, 4))
    P[0, 0], P[1, 1] = 1 / tx, 1 / ty
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    view = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
    proj = torch.tensor(P, dtype=torch.float32, device=device).transpose(0, 1)
    return dict(view=view, full=view @ proj, campos=torch.tensor(eye, dtype=torch.float32, device=device),
                tanfovx=tx, tanfovy=ty)


def render(g, deg, cam, W, H, bg=(0.12, 0.12, 0.12), want_depth=False):
    """One frame (H x W x 3 uint8, BGR) of the splats g from the camera;
    want_depth: (frame, depth H x W float32, metres along the view)."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
        bg=torch.tensor(bg, dtype=torch.float32, device="cuda"), scale_modifier=1.0,
        viewmatrix=cam["view"], projmatrix=cam["full"], sh_degree=deg, campos=cam["campos"],
        prefiltered=False, debug=False)
    with torch.no_grad():
        out = GaussianRasterizer(settings)(
            means3D=g["xyz"], means2D=torch.zeros_like(g["xyz"]), shs=g["features"], colors_precomp=None,
            opacities=g["opacity"], scales=g["scales"], rotations=g["rotations"], cov3D_precomp=None)
    img = out[0] if isinstance(out, tuple) else out
    rgb = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if want_depth:
        dep = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        dep = dep.squeeze().cpu().numpy().astype(np.float32) if dep is not None else np.zeros((H, W), np.float32)
        return bgr, dep
    return bgr


def scene_cameras(scene):
    """The scene's cameras (sparse/0): (W, H, fx, fy, cx, cy) and, per
    image, (name, camera-to-world 4 x 4)."""
    cams = {}
    for line in open(os.path.join(scene, "sparse", "0", "cameras.txt")):
        v = line.split()
        if len(v) >= 8 and v[0].isdigit():
            cams[int(v[0])] = (int(v[2]), int(v[3])) + tuple(float(x) for x in v[4:8])
    frames = []
    for line in open(os.path.join(scene, "sparse", "0", "images.txt")):
        v = line.split()
        if len(v) >= 10 and v[0].isdigit():
            qw, qx, qy, qz = map(float, v[1:5])
            t = np.array(list(map(float, v[5:8])))
            R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                          [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                          [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
            frames.append((v[9], int(v[8]), c2w))
    if not cams or not frames:
        raise SystemExit(f"{scene}: no cameras in sparse/0")
    return cams, frames


def render_views(g, deg, scene, out, every=1):
    """The scene's cameras rendered from the splats: out/frameNNNNNN.jpg
    and depthNNNNNN.png (uint16, mm) with out/cameras.json (the camera and
    each frame's camera-to-world), for lod3dgs.py recolour --renders."""
    cams, frames = scene_cameras(scene)
    os.makedirs(out, exist_ok=True)
    meta = {"cameras": {str(k): v for k, v in cams.items()}, "frames": []}
    frames = frames[::max(1, every)]
    print(f"rendering {len(frames)} views of {scene} -> {out}", flush=True)
    for n, (name, cid, c2w) in enumerate(frames):
        W, H, fx, fy, cx, cy = cams[cid]
        fov = 2 * math.degrees(math.atan(H / (2 * fy)))
        eye, fwd = c2w[:3, 3], c2w[:3, 2]
        up_c = -c2w[:3, 1]                                   # the camera's y is down
        bgr, dep = render(g, deg, camera(eye, eye + fwd, up_c, W, H, fov), W, H, want_depth=True)
        cv2.imwrite(os.path.join(out, f"frame{n:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(out, f"depth{n:06d}.png"), np.clip(dep * 1000, 0, 65535).astype(np.uint16))
        meta["frames"].append({"name": name, "camera": cid, "c2w": c2w.round(9).tolist()})
        if n % 200 == 0:
            print(f"  {n} of {len(frames)}", flush=True)
    json.dump(meta, open(os.path.join(out, "cameras.json"), "w"))
    print(f"{len(frames)} views rendered -> {out}", flush=True)


def first_camera(scene):
    """(eye, look) of the first camera of a scene's sparse/0/images.txt
    (world-to-camera qw qx qy qz tx ty tz): its centre, looking its way."""
    for line in open(os.path.join(scene, "sparse", "0", "images.txt")):
        v = line.split()
        if len(v) >= 8 and v[0].isdigit():
            qw, qx, qy, qz = map(float, v[1:5])
            t = np.array(list(map(float, v[5:8])))
            R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                          [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                          [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            eye = -R.T @ t
            fwd = R.T @ np.array([0.0, 0.0, 1.0])
            fwd[2] = 0.0
            fwd /= np.linalg.norm(fwd) or 1.0
            return eye, eye + 2.0 * fwd
    raise SystemExit(f"{scene}: no camera in sparse/0/images.txt")


def video_writer(out, fps, size):
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                              "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                             stdin=subprocess.PIPE)
        return (lambda bgr: p.stdin.write(np.ascontiguousarray(bgr).tobytes())), (lambda: (p.stdin.close(), p.wait()))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return vw.write, vw.release


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", help="a 3DGS .ply, e.g. <run>/point_cloud/iteration_300000/level_5.ply")
    ap.add_argument("--scene", default=None, help="the scene folder: start at its first camera")
    ap.add_argument("--at", default=None, metavar="X,Y[,Z]", help="start here (the model's frame)")
    ap.add_argument("--size", default="1280x720", help="WxH")
    ap.add_argument("--fov", type=float, default=70.0, help="vertical field of view (deg)")
    ap.add_argument("--eye", type=float, default=1.6, help="eye height over the floor (m) when --at has no Z")
    ap.add_argument("--step", type=float, default=0.1, help="metres per key press")
    ap.add_argument("--turn", type=float, default=4.0, help="degrees per arrow key press")
    ap.add_argument("--fly", nargs=2, default=None, metavar=("PATH_JSON", "OUT_MP4"),
                    help="render the video of this fly path (lod3dgs.py fly's <out>_path.json) instead")
    ap.add_argument("--frame", default=None, help="anchor_frame.json: the fly path is in the anchored frame")
    ap.add_argument("--eye-shift", type=float, default=0.0, metavar="DZ",
                    help="with --fly: move the path's eye up or down by DZ m (the mesh fly is at 1.6 m; the robot's "
                         "camera, where the model was trained, at 0.3)")
    ap.add_argument("--render-views", nargs=2, default=None, metavar=("SCENE", "OUT"),
                    help="render the scene's cameras (rgb + depth) to OUT, for lod3dgs.py recolour --renders")
    ap.add_argument("--every", type=int, default=1, help="with --render-views: every Nth camera")
    ap.add_argument("--no-lod", action="store_true", help="the given file alone (default: the level files beside "
                    "it, each Gaussian drawn from the coarsest level whose cell is still small at its distance)")
    ap.add_argument("--cell", type=float, default=0.04, help="the finest level's cell size (m; the scene's spacing)")
    ap.add_argument("--px", type=float, default=2.0, help="a level is used from where its cells cover this many pixels")
    a = ap.parse_args()
    W, H = [int(v) for v in a.size.lower().split("x")]
    print(f"loading {a.ply} ...", flush=True)
    if a.no_lod:
        g0, deg = load_splats(a.ply)
        levels = [(a.cell, g0)]
    else:
        levels, deg = load_levels(a.ply, a.cell)
    xyz = levels[-1][1]["xyz"].cpu().numpy()
    f_px = (H / 2) / math.tan(math.radians(a.fov) / 2)
    print(f"{len(xyz)} splats at the finest level, SH degree {deg}; "
          + (f"{len(levels)} levels drawn by distance (a cell of {a.px:g} px or less)" if len(levels) > 1
             else "one level") , flush=True)
    g_at = lambda eye: pick_levels(levels, eye, f_px, a.px)      # noqa: E731
    up = np.array([0.0, 0.0, 1.0])

    if a.render_views:
        render_views(levels[-1][1], deg, a.render_views[0], a.render_views[1], a.every)   # the finest: for colours
        return
    if a.fly:
        pj = json.load(open(a.fly[0]))
        E, Lk = np.array(pj["eye"], float), np.array(pj["look"], float)
        if a.eye_shift:                 # the path's frame is z up: lower the eye towards the training cameras'
            E[:, 2] += a.eye_shift
            Lk[:, 2] += a.eye_shift
            print(f"eye moved {a.eye_shift:+.2f} m: {E[0, 2]:.2f} m up")
        if a.frame:
            af = json.load(open(a.frame))
            Ti = np.linalg.inv(np.array(af["T_N_world"], float))     # anchored -> the model's (GLIM's world)
            E = E @ Ti[:3, :3].T + Ti[:3, 3]
            Lk = Lk @ Ti[:3, :3].T + Ti[:3, 3]
        fps = int(pj.get("fps", 30))
        write, release = video_writer(a.fly[1], fps, (W, H))
        print(f"flying {len(E)} frames at {fps} fps -> {a.fly[1]}", flush=True)
        for i in range(len(E)):
            write(render(g_at(E[i]), deg, camera(E[i], Lk[i], up, W, H, a.fov), W, H))
            if i % (10 * fps) == 0:
                print(f"  {i // fps} s of {len(E) // fps}", flush=True)
        release()
        return

    if a.at:
        v = [float(x) for x in a.at.split(",")]
        if len(v) == 2:
            near = xyz[np.hypot(xyz[:, 0] - v[0], xyz[:, 1] - v[1]) < 1.0]
            v.append((np.percentile(near[:, 2], 2) if len(near) else np.percentile(xyz[:, 2], 2)) + a.eye)
        eye0 = np.array(v)
        look0 = eye0 + np.array([2.0, 0.0, 0.0])
    elif a.scene:
        eye0, look0 = first_camera(a.scene)
    else:
        c = np.median(xyz, 0)
        eye0 = np.array([c[0], c[1], np.percentile(xyz[:, 2], 2) + a.eye])
        look0 = eye0 + np.array([2.0, 0.0, 0.0])
    print("turn: the arrow keys (left/right, up/down tilt) or drag with the mouse; move: W S A D, Q up, Z down; "
          "R back to the start; Escape out", flush=True)
    st = {"eye": eye0.copy(), "yaw": math.atan2(look0[1] - eye0[1], look0[0] - eye0[0]), "pitch": 0.0, "drag": None}
    win = f"splats {os.path.basename(a.ply)}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["drag"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            st["drag"] = None
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"] is not None:
            dx, dy = x - st["drag"][0], y - st["drag"][1]
            st["drag"] = (x, y)
            st["yaw"] -= dx * 0.004
            st["pitch"] = float(np.clip(st["pitch"] - dy * 0.004, -1.4, 1.4))
    cv2.setMouseCallback(win, on_mouse)
    LEFT, UP, RIGHT, DOWN = 65361, 65362, 65363, 65364            # Linux key codes (waitKeyEx)
    while True:
        fwd = np.array([math.cos(st["pitch"]) * math.cos(st["yaw"]), math.cos(st["pitch"]) * math.sin(st["yaw"]),
                        math.sin(st["pitch"])])
        cv2.imshow(win, render(g_at(st["eye"]), deg, camera(st["eye"], st["eye"] + fwd, up, W, H, a.fov), W, H))
        k = cv2.waitKeyEx(15)
        if k == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        flat = np.array([math.cos(st["yaw"]), math.sin(st["yaw"]), 0.0])
        right = np.array([math.sin(st["yaw"]), -math.cos(st["yaw"]), 0.0])
        moves = {ord("w"): flat, ord("s"): -flat, ord("a"): -right, ord("d"): right, ord("q"): up, ord("z"): -up}
        if k in moves:
            st["eye"] = st["eye"] + a.step * moves[k]
        elif k == LEFT:
            st["yaw"] += math.radians(a.turn)
        elif k == RIGHT:
            st["yaw"] -= math.radians(a.turn)
        elif k == UP:
            st["pitch"] = min(st["pitch"] + math.radians(a.turn), 1.4)
        elif k == DOWN:
            st["pitch"] = max(st["pitch"] - math.radians(a.turn), -1.4)
        elif k == ord("r"):
            st.update(eye=eye0.copy(), yaw=math.atan2(look0[1] - eye0[1], look0[0] - eye0[0]), pitch=0.0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
