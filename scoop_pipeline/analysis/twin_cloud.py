#!/usr/bin/env python3
"""A USD digital twin's FIXED geometry as a labelled point cloud: the ground
truth a map is measured against (twin_compare.py).

    python3 scoop_pipeline/analysis/twin_cloud.py <room.usd> <out .ply>
        [--spacing 0.01] [--movable sofa,chair,rchair] [--root /Room]

    e.g. the Intelligent Home Care Living Lab twin (NYCU room 305, HCIS Lab):
    python3 scoop_pipeline/analysis/twin_cloud.py \\
        Intelligent-Home-Care-Living-Lab-main/room.usd twin_ed305.ply

Every mesh under --root (payloads loaded, instances followed), placed by its
world transform, sampled evenly on its surface, one point per --spacing^2 of
area. The children of --root whose name starts with a --movable word are left
out: furniture that is not where the twin put it on a given day (sofas,
chairs) is no ground truth.

Each point carries a label (uchar), named in <out>.json:
  floor / wall / ceiling   the room shell's meshes, by their box: top at most
                           5 cm over the shell's lowest point = floor; bottom at
                           least 2.4 m up = ceiling (lights hung there too);
                           2 m tall or more and at most 0.4 m deep = wall
  builtin                  the rest of the shell: countertops, cabinets, ...
  <name>                   every other fixed child of --root (big_fridge,
                           washing_machine, big_work_table, ...)
The z axis is the stage's up axis (asserted Z), in metres.
Needs usd-core (pip install usd-core).
"""
import argparse
import json
import os
import re
import sys

import numpy as np


def mesh_triangles(prim, xf_cache):
    """(V world (n,3), F (m,3)) of a UsdGeom.Mesh, its polygons fanned."""
    from pxr import UsdGeom
    m = UsdGeom.Mesh(prim)
    pts = m.GetPointsAttr().Get()
    counts = m.GetFaceVertexCountsAttr().Get()
    idx = m.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not idx:
        return None
    V = np.asarray(pts, np.float64)
    M = np.asarray(xf_cache.GetLocalToWorldTransform(prim), np.float64)   # row-vector convention
    V = V @ M[:3, :3] + M[3, :3]
    counts, idx = np.asarray(counts), np.asarray(idx)
    F, start = [], 0
    for c in counts:
        if c >= 3:
            f = idx[start:start + c]
            F.extend([(f[0], f[k], f[k + 1]) for k in range(1, c - 1)])
        start += c
    return V, np.asarray(F, np.int64).reshape(-1, 3)


def sample(V, F, spacing, rng):
    """Points evenly on the triangles' surface, about one per spacing^2."""
    if not len(F):
        return np.zeros((0, 3))
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    lam = area / spacing ** 2
    n = rng.poisson(lam)
    tri = np.repeat(np.arange(len(F)), n)
    if not len(tri):
        return np.zeros((0, 3))
    r1, r2 = rng.random(len(tri)), rng.random(len(tri))
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[tri] + (s * (1 - r2))[:, None] * b[tri] + (s * r2)[:, None] * c[tri]


def shell_class(bmin, bmax, floor_z):
    h = bmax[2] - bmin[2]
    depth = min(bmax[0] - bmin[0], bmax[1] - bmin[1])
    if bmax[2] <= floor_z + 0.05:
        return "floor"
    if bmin[2] >= floor_z + 2.4:
        return "ceiling"
    if h >= 2.0 and depth <= 0.4:
        return "wall"
    return "builtin"


def twin_points(usd, spacing=0.01, movable=("sofa", "chair", "rchair"), root="/Room",
                seed=0, log=print):
    """-> (xyz (n,3), label (n,), names [label -> name], per-part summary)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(os.path.expanduser(usd), Usd.Stage.LoadAll)
    if UsdGeom.GetStageUpAxis(st) != UsdGeom.Tokens.z:
        raise SystemExit("%s: up axis %s, not Z" % (usd, UsdGeom.GetStageUpAxis(st)))
    mpu = UsdGeom.GetStageMetersPerUnit(st)
    top = st.GetPrimAtPath(root)
    if not top:
        raise SystemExit("no %s in %s" % (root, usd))
    xf = UsdGeom.XformCache()
    rng = np.random.default_rng(seed)
    meshes = {}                                  # child -> [(V, F)]
    for c in top.GetChildren():
        ms = []
        for p in Usd.PrimRange(c, Usd.TraverseInstanceProxies()):
            if p.IsA(UsdGeom.Mesh):
                img = UsdGeom.Imageable(p)
                if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
                    continue
                t = mesh_triangles(p, xf)
                if t is not None and len(t[1]):
                    ms.append((t[0] * mpu, t[1]))
        if ms:
            meshes[c.GetName()] = ms
    # the shell: the child with the most meshes (the room's assembly)
    shell = max(meshes, key=lambda k: len(meshes[k]))
    floor_z = min(V[:, 2].min() for V, _ in meshes[shell])
    names, P, L, summary = [], [], [], []

    def lab(name):
        if name not in names:
            names.append(name)
        return names.index(name)

    for name, ms in meshes.items():
        word = re.sub(r"[_\d]+$", "", name).lower()
        if any(word.startswith(m) for m in movable):
            summary.append({"part": name, "kept": False, "why": "movable"})
            continue
        n0 = sum(len(P_) for P_ in P)
        for V, F in ms:
            if name == shell:
                k = lab(shell_class(V.min(0), V.max(0), floor_z))
            else:
                k = lab(name)
            pts = sample(V, F, spacing, rng)
            P.append(pts)
            L.append(np.full(len(pts), k, np.uint8))
        summary.append({"part": name, "kept": True, "meshes": len(ms),
                        "points": int(sum(len(P_) for P_ in P) - n0)})
    xyz = np.vstack(P) if P else np.zeros((0, 3))
    label = np.concatenate(L) if L else np.zeros(0, np.uint8)
    log("%s: %d parts (%d left out as movable), shell '%s' (%d meshes, floor z %.3f) -> "
        "%d points at %.3f m" % (usd, len(meshes), sum(1 for s in summary if not s["kept"]),
                                 shell, len(meshes[shell]), floor_z, len(xyz), spacing))
    return xyz, label, names, summary


def write_labelled_ply(path, xyz, label):
    with open(path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar label\nend_header\n" % len(xyz)).encode())
        rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "u1")])
        rec["x"], rec["y"], rec["z"], rec["label"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], label
        fh.write(rec.tobytes())


def read_labelled_ply(path):
    """(xyz, label) of a ply written by write_labelled_ply."""
    with open(os.path.expanduser(path), "rb") as fh:
        n = 0
        while True:
            line = fh.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line == "end_header":
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "u1")])
        rec = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
    return np.c_[rec["x"], rec["y"], rec["z"]].astype(np.float64), rec["label"].copy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("usd")
    ap.add_argument("out")
    ap.add_argument("--spacing", type=float, default=0.01)
    ap.add_argument("--movable", default="sofa,chair,rchair")
    ap.add_argument("--root", default="/Room")
    a = ap.parse_args(argv)
    os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")
    xyz, label, names, summary = twin_points(a.usd, a.spacing,
                                             tuple(w.strip() for w in a.movable.split(",") if w.strip()),
                                             a.root)
    out = os.path.expanduser(a.out)
    write_labelled_ply(out, xyz, label)
    counts = np.bincount(label, minlength=len(names))
    meta = {"source": os.path.abspath(os.path.expanduser(a.usd)), "spacing": a.spacing,
            "labels": names, "points_per_label": counts.tolist(), "parts": summary,
            "aabb_min": xyz.min(0).round(4).tolist(), "aabb_max": xyz.max(0).round(4).tolist()}
    json.dump(meta, open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
    for k, nm in enumerate(names):
        print("  %-18s %9d points" % (nm, counts[k]))
    print("-> %s (+ .json)" % out)


if __name__ == "__main__":
    sys.exit(main())
