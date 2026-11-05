#!/usr/bin/env python3
"""A fly-through video with the object registry on it: every object the
camera sees boxed and named, frame by frame.

    python3 scoop_pipeline/assets/tour_objects.py <fly video .mp4> <registry> [<out .mp4>]
        [--path <video>_path.json] [--mesh <the mesh flown>] [--fov 70]
        [--min-visible 20] [--max-dist 8] [--skip-label wall,floor,ceiling]

The video is lod3dgs.py fly's (its <video>_path.json beside it holds every
frame's eye and look-at point and the mesh); the registry is assets/ after
consolidate (objects.json and each object's points), in the mesh's frame.
Default out: <video>_objects.mp4.

Per frame, the camera as fly had it (x right, y down, z forward; --fov
across the height, the principal point at the centre), and the mesh
ray-cast from it at a quarter of the video's size (Open3D's RaycastingScene:
no display needed) for what is in front. Each object's points projected; the
ones no surface hides (within 5 cm + 2 % of the ray's depth) are what is
seen of it; with --min-visible of them or more, and nearer than --max-dist,
it is boxed: the box round its seen points (2nd..98th percentile), coloured
by label, named by its id (label_NN; "on <id>" when it stands on another), its
distance in the corner. A panel lists what is in view.
Needs open3d, opencv.
"""
import argparse
import colorsys
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


_HUES = {}


def colour_of(label):
    """A bright BGR colour per label: hues a golden-ratio step apart in the
    order the labels first come up, so a handful of labels never look alike."""
    if label not in _HUES:
        _HUES[label] = (0.61803398875 * len(_HUES) + 0.1) % 1.0
    r, g, b = colorsys.hsv_to_rgb(_HUES[label], 0.8, 1.0)
    return int(255 * b), int(255 * g), int(255 * r)


def camera(eye, look, up=(0.0, 0.0, 1.0)):
    """World -> camera rotation and the eye, as lod3dgs.fly_video sets them."""
    eye, look, up = np.asarray(eye, float), np.asarray(look, float), np.asarray(up, float)
    fwd = look - eye
    fwd /= np.linalg.norm(fwd) or 1.0
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) or 1.0
    down = np.cross(fwd, right)
    return np.vstack([right, down, fwd]), eye


def depth_image(scene, R, e, W, H, f, step):
    """Ray-cast depth (along the optical axis) at every step-th pixel."""
    import open3d as o3d
    us, vs = np.arange(0, W, step) + step / 2 - 0.5, np.arange(0, H, step) + step / 2 - 0.5
    U, V = np.meshgrid(us, vs)
    d = np.stack([(U - (W / 2 - 0.5)) / f, (V - (H / 2 - 0.5)) / f, np.ones_like(U)], -1)
    dw = d @ R                                                       # camera -> world directions
    rays = np.concatenate([np.broadcast_to(e, dw.shape), dw], -1).reshape(-1, 6).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(U.shape)
    return t                                                         # t along d (z = 1): the depth


def seen_points(P, R, e, depth, W, H, f, step, max_dist):
    """(u, v, z) of the points of P in view and not hidden."""
    C = (P - e) @ R.T
    z = C[:, 2]
    ok = (z > 0.1) & (z < max_dist)
    if not ok.any():
        return np.zeros((0, 3))
    C, z = C[ok], z[ok]
    u = f * C[:, 0] / z + W / 2 - 0.5
    v = f * C[:, 1] / z + H / 2 - 0.5
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inside], v[inside], z[inside]
    i = np.clip((v / step).astype(int), 0, depth.shape[0] - 1)
    j = np.clip((u / step).astype(int), 0, depth.shape[1] - 1)
    d = depth[i, j]
    vis = ~np.isfinite(d) | (z <= d * 1.02 + 0.05)
    return np.c_[u[vis], v[vis], z[vis]]


def frame_boxes(objs, pts, scene, eye, look, W, H, f, step=4, min_visible=20, max_dist=8.0):
    """The objects seen from (eye, look): [{id, label, on, z, box}]."""
    R, e = camera(eye, look)
    depth = depth_image(scene, R, e, W, H, f, step)
    boxes = []
    for o, P in zip(objs, pts):
        s = seen_points(P, R, e, depth, W, H, f, step, max_dist)
        if len(s) < min_visible:
            continue
        x0, y0 = np.percentile(s[:, :2], 2, axis=0)
        x1, y1 = np.percentile(s[:, :2], 98, axis=0)
        boxes.append({"id": o["id"], "label": o["label"], "on": o.get("on"), "z": float(np.median(s[:, 2])),
                      "box": (int(x0), int(y0), int(x1), int(y1))})
    return boxes


def draw(img, boxes, frame_s):
    """Boxes, names, and the panel of what is in view."""
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in sorted(boxes, key=lambda b: -b["z"]):                  # far first: near ones on top
        x0, y0, x1, y1 = b["box"]
        c = colour_of(b["label"])
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2, cv2.LINE_AA)
        txt = b["id"] + ("  on %s" % b["on"] if b.get("on") else "")
        (tw, th), _ = cv2.getTextSize(txt, font, 0.5, 1)
        ty = y0 - 6 if y0 - th - 8 > 0 else y0 + th + 6
        cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), c, -1)
        cv2.putText(img, txt, (x0 + 3, ty), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        dist = "%.1f m" % b["z"]
        (dw, dh), _ = cv2.getTextSize(dist, font, 0.45, 1)
        cv2.putText(img, dist, (x1 - dw - 4, y1 - 5), font, 0.45, c, 1, cv2.LINE_AA)
    names = {}
    for b in boxes:
        names[b["label"]] = names.get(b["label"], 0) + 1
    lines = ["%d objects in view   %5.1f s" % (len(boxes), frame_s)] + [
        "%-14s %d" % (k, v) for k, v in sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))][:14]
    pw = 220
    ph = 18 * len(lines) + 10
    over = img[8:8 + ph, W - pw - 8:W - 8]
    over[:] = (0.35 * over).astype(np.uint8)
    for k, ln in enumerate(lines):
        col = (255, 255, 255) if k == 0 else colour_of(ln.split()[0])
        cv2.putText(img, ln, (W - pw, 26 + 18 * k), font, 0.45, col, 1, cv2.LINE_AA)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("video")
    ap.add_argument("registry")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--min-visible", type=int, default=20)
    ap.add_argument("--max-dist", type=float, default=8.0)
    ap.add_argument("--max-points", type=int, default=3000, help="per object, for speed")
    ap.add_argument("--skip-label", default="wall,floor,ceiling")
    a = ap.parse_args(argv)
    import open3d as o3d
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datasets"))
    from lod3dgs import video_writer
    from views import read_ply_xyz
    video = os.path.expanduser(a.video)
    stem = os.path.splitext(video)[0]
    pj = json.load(open(os.path.expanduser(a.path or stem + "_path.json")))
    mesh_path = os.path.expanduser(a.mesh or pj["mesh"])
    out = os.path.expanduser(a.out or stem + "_objects.mp4")
    reg = os.path.expanduser(a.registry)
    skip = {s for s in a.skip_label.split(",") if s}
    objs = [o for o in json.load(open(os.path.join(reg, "objects.json")))["objects"] if o["label"] not in skip]
    rng = np.random.default_rng(0)
    pts = []
    for o in objs:
        P = read_ply_xyz(os.path.join(reg, o["points"]))
        if len(P) > a.max_points:
            P = P[rng.choice(len(P), a.max_points, replace=False)]
        pts.append(P)
    m = o3d.io.read_triangle_mesh(mesh_path)
    if not len(m.triangles):
        raise SystemExit("%s: no faces (the mesh fly flew)" % mesh_path)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(pj.get("fps", 30))
    f = (H / 2) / np.tan(np.radians(a.fov) / 2)
    step = 4
    E, Lk = np.asarray(pj["eye"]), np.asarray(pj["look"])
    print("%s: %dx%d at %.0f fps, %d poses; %d objects (%s left out); mesh %s (%d faces) -> %s"
          % (video, W, H, fps, len(E), len(objs), ",".join(sorted(skip)) or "none", mesh_path,
             len(m.triangles), out))
    vw = video_writer(out, fps, (W, H))
    seen_ever, n = {}, 0
    while True:
        ok, img = cap.read()
        if not ok or n >= len(E):
            break
        boxes = frame_boxes(objs, pts, scene, E[n], Lk[n], W, H, f, step, a.min_visible, a.max_dist)
        for b in boxes:
            seen_ever[b["id"]] = b["label"]
        vw.write(draw(img, boxes, n / fps))
        n += 1
        if n % int(10 * fps) == 0:
            print("  %.0f s of %.0f, %d objects seen so far" % (n / fps, len(E) / fps, len(seen_ever)))
    vw.release()
    cap.release()
    print("-> %s: %d frames, %d of %d objects shown at least once" % (out, n, len(seen_ever), len(objs)))


if __name__ == "__main__":
    sys.exit(main())
