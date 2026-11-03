#!/usr/bin/env python3
"""The object registry of a mapped room: every object segmented in the posed
colour frames, lifted to 3D through the map's depth, its sightings merged
across frames. The first stage of asset management; SAM 3D assets and their
check against the measured map build on this registry.

    python3 scoop_pipeline/assets/objects.py <replica export> <registry dir>
        (--prompts "chair,table,monitor" | --tags ram --ram-ckpt <ram_plus.pth>)
        [--segmenter sam3 | --segmenter npz:<dir>] [--every 10] [--min-score 0.5]
        [--voxel 0.02] [--match 0.25] [--min-views 2] [--min-points 150]
        [--max-depth 6.0] [--erode 3] [--max-frames 0]

The input is a Replica-format export of the pass (datasets/mcap_convert.py
--format replica), best with the depth rendered from the map, so a mask finds
measured surface wherever it falls:

    python3 scoop_pipeline/datasets/mcap_convert.py <final bag> <export> --format replica \\
        --pose-is-camera --depth-source map --map <map .pcd> --every 2

Per --every-th frame:
  1. the colour image undistorted (the export keeps the lens's distortion; its
     depth is pinhole, so a mask must be on the pinhole image to fall on it);
  2. segments: SAM 3 with each text prompt ("segment anything with concepts"),
     or masks made elsewhere (--segmenter npz:<dir>, <dir>/<frame>.npz with
     masks (N,H,W) bool, labels (N,) str, scores (N,));
  3. each mask, eroded by --erode px (its edge bleeds onto what is behind),
     back-projected through the depth (0 < d < --max-depth) into the map frame,
     only its largest connected piece kept (5 cm cells): pixels that see past
     the object -- gaps, thin legs -- land on the wall behind, metres away;
  4. matched to an object of the same label that already holds at least
     --match of its 5 cm cells, else a new object.
Then objects of one label that overlap that much are merged (one object seen
in two runs of frames), and those seen in fewer than --min-views frames or
with fewer than --min-points points dropped.

The prompts: --prompts, or --tags ram: the Recognize Anything model (RAM++)
tags every 5th sampled frame and the tags seen in at least 10 % of them, less
the room itself (wall, floor, ceiling, room, ...), become the prompts.

Registry (<registry dir>):
  objects.json        per object: id, label, views (frame, score, mask area,
                      bbox), best_views (the largest, sharpest-scored masks
                      first), centroid, aabb_min/max (map frame, m), points
                      file, n_points, mean score
  points/<id>.ply     the object's measured points (map frame), coloured from
                      the frames, one per --voxel
  masks/<id>_<frame>.png   the masks of its three best views (SAM 3D's input)
  frames.json         the frames used, the prompts, the settings
"""
import argparse
import json
import os
import sys

import numpy as np

ROOM = {"wall", "floor", "ceiling", "room", "office", "classroom", "interior", "building",
        "indoor", "laboratory", "lab", "white", "black", "light", "lamp light", "shadow",
        "ceiling light", "tile", "tile floor", "wood floor", "hardwood", "carpet", "corner",
        "space", "area", "picture frame"}


# ---------------------------------------------------------------- the export
class Export:
    """A Replica-format export: frames, depth, camera-to-world poses, K, D."""

    def __init__(self, root):
        import yaml
        self.root = os.path.expanduser(root)
        cfg = yaml.safe_load(open(os.path.join(self.root, "splatam_data_config.yaml")))
        c = cfg["camera_params"]
        self.K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
        self.W, self.H = int(c["image_width"]), int(c["image_height"])
        self.depth_scale = float(c["png_depth_scale"])
        self.D = np.array(c.get("distortion") or [], float)
        rows = [l.split() for l in open(os.path.join(self.root, "traj.txt")) if l.strip()]
        self.poses = [np.array(r, float).reshape(4, 4) for r in rows]
        self._maps = None

    def __len__(self):
        return len(self.poses)

    def image(self, i):
        """RGB, undistorted onto the depth's pinhole camera."""
        import cv2
        bgr = cv2.imread(os.path.join(self.root, "results", "frame%06d.jpg" % i))
        if bgr is None:
            raise FileNotFoundError("frame%06d.jpg" % i)
        if self.D.size and np.abs(self.D).max() > 0:
            if self._maps is None:
                self._maps = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K,
                                                         (self.W, self.H), cv2.CV_32FC1)
            bgr = cv2.remap(bgr, *self._maps, cv2.INTER_LINEAR)
        return bgr[:, :, ::-1].copy()

    def depth(self, i):
        import cv2
        d = cv2.imread(os.path.join(self.root, "results", "depth%06d.png" % i), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError("depth%06d.png" % i)
        return d.astype(np.float32) / self.depth_scale


# ---------------------------------------------------------------- segmenters
class Sam3Segmenter:
    """SAM 3 with text prompts (github.com/facebookresearch/sam3; the image
    API as its README gives it: build_sam3_image_model, Sam3Processor,
    set_image, set_text_prompt -> masks, boxes, scores). Not run in this
    repository's tests: check it against the installed version."""

    def __init__(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.proc = Sam3Processor(build_sam3_image_model())

    def __call__(self, rgb, prompts, frame=None):
        from PIL import Image
        state = self.proc.set_image(Image.fromarray(rgb))
        out = []
        for p in prompts:
            r = self.proc.set_text_prompt(state=state, prompt=p)
            masks, scores = _np(r["masks"]), _np(r["scores"]).reshape(-1)
            masks = masks.reshape(len(scores), *masks.shape[-2:]) if len(scores) else masks
            for m, s in zip(masks, scores):
                out.append((p, m.astype(bool), float(s)))
        return out


class NpzSegmenter:
    """Masks made elsewhere: <dir>/<frame>.npz (masks (N,H,W), labels, scores)."""

    def __init__(self, d):
        self.d = os.path.expanduser(d)

    def __call__(self, rgb, prompts, frame=None):
        p = os.path.join(self.d, "%06d.npz" % frame)
        if not os.path.isfile(p):
            return []
        z = np.load(p, allow_pickle=True)
        return [(str(l), m.astype(bool), float(s))
                for m, l, s in zip(z["masks"], z["labels"], z["scores"])
                if not prompts or str(l) in prompts]


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy() if x.dtype.is_floating_point else x.detach().cpu().numpy()
    return np.asarray(x)


class RamTagger:
    """Recognize Anything (RAM++, github.com/xinyu1205/recognize-anything):
    ram_plus + inference_ram, as its README gives them. Not run in this
    repository's tests."""

    def __init__(self, ckpt):
        import torch
        from ram import get_transform
        from ram.models import ram_plus
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tf = get_transform(image_size=384)
        self.model = ram_plus(pretrained=os.path.expanduser(ckpt), image_size=384,
                              vit="swin_l").eval().to(self.dev)

    def __call__(self, rgb):
        import torch
        from PIL import Image
        from ram import inference_ram
        with torch.no_grad():
            tags = inference_ram(self.tf(Image.fromarray(rgb)).unsqueeze(0).to(self.dev), self.model)[0]
        return [t.strip().lower() for t in tags.split("|") if t.strip()]


def prompts_from_tags(tag_lists, min_share=0.1):
    """The tags seen in at least min_share of the frames, the room's own left out."""
    from collections import Counter
    n = max(1, len(tag_lists))
    c = Counter(t for tags in tag_lists for t in set(tags))
    return sorted(t for t, k in c.items() if k / n >= min_share and t not in ROOM)


# ---------------------------------------------------------------- lifting
def lift(mask, depth, rgb, K, T_wc, max_depth, erode):
    """A mask's measured points in the map frame, with their colours."""
    import cv2
    m = mask
    if erode > 0:
        m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)) > 0
    v, u = np.nonzero(m & (depth > 0) & (depth < max_depth))
    if not len(u):
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.c_[x, y, z]
    return pc @ T_wc[:3, :3].T + T_wc[:3, 3], rgb[v, u]


def largest_piece(xyz, cell=0.05):
    """The points of the largest 26-connected group of `cell` cells: a mask's
    pixels that see past the object (gaps, thin legs, its bleeding edge) land
    on what is behind it, metres away, and fall off."""
    if len(xyz) == 0:
        return np.zeros(0, bool)
    k = np.floor(xyz / cell).astype(np.int64)
    uniq, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    index = {tuple(u): j for j, u in enumerate(uniq)}
    lab = -np.ones(len(uniq), int)
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]
    sizes = []
    for j in range(len(uniq)):
        if lab[j] >= 0:
            continue
        lab[j] = len(sizes)
        stack, n = [j], 0
        while stack:
            q = stack.pop()
            n += 1
            x, y, z = uniq[q]
            for a, b, c in nb:
                r = index.get((x + a, y + b, z + c))
                if r is not None and lab[r] < 0:
                    lab[r] = lab[j]
                    stack.append(r)
        sizes.append(0)
    counts = np.bincount(lab[inv], minlength=len(sizes))
    return lab[inv] == int(np.argmax(counts))


def cells(xyz, size):
    return set(map(tuple, np.floor(xyz / size).astype(np.int64)))


class Obj:
    def __init__(self, oid, label):
        self.id, self.label = oid, label
        self.keys = set()           # 5 cm cells, for matching
        self.pts = {}               # --voxel cell -> (xyz sum, rgb sum, n)
        self.views = []

    def add(self, xyz, rgb, view, voxel, match_cell):
        self.keys |= cells(xyz, match_cell)
        k = np.floor(xyz / voxel).astype(np.int64)
        for kk, p, c in zip(map(tuple, k), xyz, rgb.astype(np.float64)):
            s = self.pts.get(kk)
            self.pts[kk] = (p, c, 1) if s is None else (s[0] + p, s[1] + c, s[2] + 1)
        self.views.append(view)

    def absorb(self, o):
        self.keys |= o.keys
        for kk, s in o.pts.items():
            t = self.pts.get(kk)
            self.pts[kk] = s if t is None else (t[0] + s[0], t[1] + s[1], t[2] + s[2])
        self.views += o.views

    def points(self):
        v = list(self.pts.values())
        xyz = np.array([s[0] / s[2] for s in v]).reshape(-1, 3)
        rgb = np.array([s[1] / s[2] for s in v]).reshape(-1, 3).round().astype(np.uint8)
        return xyz, rgb


def overlap(a, b):
    """|a & b| / min(|a|, |b|): one object seen partly from one frame matches."""
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return sum(1 for k in small if k in big) / len(small)


def build(export, segment, prompts, every=10, min_score=0.5, voxel=0.02, match=0.25,
          match_cell=0.05, min_views=2, min_points=150, max_depth=6.0, erode=3,
          max_frames=0, log=print):
    """-> (objects, frames used)."""
    objs, frames, nid = [], [], 0
    idx = list(range(0, len(export), max(1, every)))
    if max_frames:
        idx = idx[:max_frames]
    for n, i in enumerate(idx):
        rgb, depth, T = export.image(i), export.depth(i), export.poses[i]
        segs = [s for s in segment(rgb, prompts, frame=i) if s[2] >= min_score]
        frames.append(i)
        for label, mask, score in segs:
            xyz, col = lift(mask, depth, rgb, export.K, T, max_depth, erode)
            keep = largest_piece(xyz, match_cell)
            xyz, col = xyz[keep], col[keep]
            if len(xyz) < 20:
                continue
            ys, xs = np.nonzero(mask)
            view = {"frame": i, "score": round(score, 4), "area": int(mask.sum()),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "_mask": mask}
            keys = cells(xyz, match_cell)
            best, best_ov = None, 0.0
            for o in objs:
                if o.label == label:
                    ov = overlap(keys, o.keys)
                    if ov > best_ov:
                        best, best_ov = o, ov
            if best is None or best_ov < match:
                best = Obj(nid, label)
                nid += 1
                objs.append(best)
            best.add(xyz, col, view, voxel, match_cell)
        if (n + 1) % 50 == 0:
            log("    %d / %d frames, %d objects" % (n + 1, len(idx), len(objs)))
    merged = True                       # one object found in two runs of frames
    while merged:
        merged = False
        for a in objs:
            for b in objs:
                if a is not b and a.label == b.label and overlap(a.keys, b.keys) >= match:
                    a.absorb(b)
                    objs.remove(b)
                    merged = True
                    break
            if merged:
                break
    keep = []
    for o in objs:
        nviews = len({v["frame"] for v in o.views})
        if nviews >= min_views and len(o.pts) >= min_points:
            keep.append(o)
    return keep, frames


# ---------------------------------------------------------------- writing
def write_ply(path, xyz, rgb):
    with open(path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                  % len(xyz)).encode())
        rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        fh.write(rec.tobytes())


def write_registry(out, objs, frames, prompts, settings):
    import cv2
    os.makedirs(os.path.join(out, "points"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    rows = []
    for k, o in enumerate(sorted(objs, key=lambda o: (o.label, -len(o.pts)))):
        oid = "%s_%02d" % (o.label.replace(" ", "_"), sum(1 for p in rows if p["label"] == o.label))
        xyz, rgb = o.points()
        write_ply(os.path.join(out, "points", oid + ".ply"), xyz, rgb)
        views = sorted(o.views, key=lambda v: -v["area"] * v["score"])
        best = []
        for v in views:
            if v["frame"] not in [b["frame"] for b in best]:
                best.append(v)
            if len(best) == 3:
                break
        for v in best:
            cv2.imwrite(os.path.join(out, "masks", "%s_%06d.png" % (oid, v["frame"])),
                        v["_mask"].astype(np.uint8) * 255)
        rows.append({"id": oid, "label": o.label,
                     "n_views": len({v["frame"] for v in o.views}),
                     "mean_score": round(float(np.mean([v["score"] for v in o.views])), 4),
                     "n_points": int(len(xyz)),
                     "centroid": np.round(xyz.mean(0), 4).tolist(),
                     "aabb_min": np.round(xyz.min(0), 4).tolist(),
                     "aabb_max": np.round(xyz.max(0), 4).tolist(),
                     "points": "points/%s.ply" % oid,
                     "best_views": [{k_: v[k_] for k_ in ("frame", "score", "area", "bbox")}
                                    for v in best],
                     "views": [{k_: v[k_] for k_ in ("frame", "score", "area", "bbox")}
                               for v in sorted(o.views, key=lambda v: v["frame"])]})
    json.dump({"frame": "map", "objects": rows}, open(os.path.join(out, "objects.json"), "w"), indent=1)
    json.dump({"frames": frames, "prompts": prompts, "settings": settings},
              open(os.path.join(out, "frames.json"), "w"), indent=1)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("export")
    ap.add_argument("out")
    ap.add_argument("--prompts", default=None, help="comma-separated object names")
    ap.add_argument("--tags", choices=["ram"], default=None)
    ap.add_argument("--ram-ckpt", default=None)
    ap.add_argument("--segmenter", default="sam3", help="sam3, or npz:<dir>")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--match", type=float, default=0.25)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--min-points", type=int, default=150)
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--erode", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)
    ex = Export(a.export)
    out = os.path.expanduser(a.out)
    if os.path.exists(os.path.join(out, "objects.json")):
        raise SystemExit("%s has a registry already; name another folder" % out)
    print("%s: %d frames, %dx%d, distortion %s" % (a.export, len(ex), ex.W, ex.H,
                                                   "undone" if ex.D.size else "none"))
    if a.prompts:
        prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]
    elif a.tags == "ram":
        if not a.ram_ckpt:
            raise SystemExit("--tags ram needs --ram-ckpt <ram_plus_swin_large_14m.pth>")
        tagger = RamTagger(a.ram_ckpt)
        sample = list(range(0, len(ex), max(1, a.every) * 5))
        prompts = prompts_from_tags([tagger(ex.image(i)) for i in sample])
        print("RAM++ over %d frames -> %d prompts: %s" % (len(sample), len(prompts), ", ".join(prompts)))
    else:
        raise SystemExit("give --prompts or --tags ram")
    if not prompts:
        raise SystemExit("no prompts")
    seg = NpzSegmenter(a.segmenter[4:]) if a.segmenter.startswith("npz:") else Sam3Segmenter()
    settings = {k: v for k, v in vars(a).items() if k not in ("export", "out")}
    objs, frames = build(ex, seg, prompts, a.every, a.min_score, a.voxel, a.match,
                         min_views=a.min_views, min_points=a.min_points,
                         max_depth=a.max_depth, erode=a.erode, max_frames=a.max_frames)
    rows = write_registry(out, objs, frames, prompts, settings)
    print("%d frames -> %d objects -> %s/objects.json" % (len(frames), len(rows), out))
    for r in rows:
        print("  %-24s %4d views  %6d pts  centre %s" % (r["id"], r["n_views"], r["n_points"],
                                                         r["centroid"]))


if __name__ == "__main__":
    sys.exit(main())
