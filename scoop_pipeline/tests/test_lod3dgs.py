#!/usr/bin/env python
"""Self-test for datasets/lod3dgs.py on a synthetic room.

A 6 x 4 x 3 m room with no ceiling: a dark floor and four walls of four
colours. Twelve cameras inside it at 1.5 m, turned all round and tilted a
little, 160 x 120, f 100, principal point off the centre (70.3, 64.6).
Their images and depth are rendered exactly (ray against plane) and written
as a replica export; the map is the five surfaces sampled every 2 cm.

It must: recentre the camera (the largest window centred on the principal
point) so that the written images, depth and poses agree with the room
rendered by the centred camera at the written pose; store depth the way
LOD-3DGS reads it (warped grey, alpha where there is depth); remove the
surface Poisson puts over the open top (no starting point above the walls,
every point near the map); give the points on each surface its colour.

    python scoop_pipeline/tests/test_lod3dgs.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "datasets"))

import cv2                                                          # noqa: E402
from scipy.spatial import cKDTree                                   # noqa: E402
from scipy.spatial.transform import Rotation                        # noqa: E402
import lod3dgs as L                                                 # noqa: E402

ROOM = np.array([6.0, 4.0, 3.0])
# (axis, value, BGR colour)
PLANES = [(2, 0.0, (40, 40, 40)), (0, 0.0, (30, 30, 220)), (0, 6.0, (30, 200, 30)),
          (1, 0.0, (220, 40, 30)), (1, 4.0, (30, 210, 220))]
W, H, F, CX, CY = 160, 120, 100.0, 70.3, 64.6
SCALE = 6553.5


def render(T_wc, W, H, fx, fy, cx, cy):
    """Exact BGR image and z-depth of the room (0 / black where nothing is hit)."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1) @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    best = np.full((H, W), np.inf)
    img = np.zeros((H, W, 3), np.uint8)
    for a, c, col in PLANES:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (c - o[a]) / d[..., a]
        p = o + s[..., None] * d
        ok = (s > 0) & np.all((p >= -1e-9) & (p <= ROOM + 1e-9), -1) & (s < best)
        best[ok] = s[ok]
        img[ok] = col
    best[~np.isfinite(best)] = 0
    return img, best


def camera(x, y, yaw, pitch):
    f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    T = np.eye(4)
    T[:3, :3] = np.stack([r, np.cross(f, r), f], 1)        # x right, y down, z forward
    T[:3, 3] = [x, y, 1.5]
    return T


def surfaces(step=0.02):
    g = lambda n: np.arange(step / 2, n, step)            # noqa: E731
    pts = []
    for a, c, _ in PLANES:
        o = [i for i in range(3) if i != a]
        A, B = np.meshgrid(g(ROOM[o[0]]), g(ROOM[o[1]]))
        p = np.zeros((A.size, 3))
        p[:, o[0]], p[:, o[1]], p[:, a] = A.ravel(), B.ravel(), c
        pts.append(p)
    return np.concatenate(pts)


def write_pcd(path, P):
    with open(path, "wb") as f:
        f.write(("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                 f"WIDTH {len(P)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(P)}\n"
                 "DATA binary\n").encode())
        P.astype("<f4").tofile(f)


def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        n = int(head.split(b"element vertex ")[1].split()[0])
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                       ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        a = np.fromfile(f, dt, n)
    return np.stack([a["x"], a["y"], a["z"]], 1), np.stack([a["red"], a["green"], a["blue"]], 1)


def main():
    tmp = tempfile.mkdtemp(prefix="lod3dgs_")
    try:
        rep = os.path.join(tmp, "replica_map_mapping_A_20260101")
        os.makedirs(os.path.join(rep, "results"))
        poses = [camera(x, y, yaw, pitch) for (x, y) in ((2.0, 1.5), (4.0, 2.5))
                 for yaw, pitch in zip(np.linspace(0, 2 * np.pi, 6, endpoint=False),
                                       (0.0, -0.2, 0.1, -0.3, 0.0, -0.15))]
        for i, T in enumerate(poses):
            img, D = render(T, W, H, F, F, CX, CY)
            cv2.imwrite(os.path.join(rep, "results", f"frame{i:06d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(os.path.join(rep, "results", f"depth{i:06d}.png"),
                        np.rint(D * SCALE).astype(np.uint16))
        np.savetxt(os.path.join(rep, "traj.txt"), np.array([T.reshape(-1) for T in poses]))
        open(os.path.join(rep, "splatam_data_config.yaml"), "w").write(
            f"dataset_name: \"replica\"\ncamera_params:\n  image_height: {H}\n"
            f"  image_width: {W}\n  fx: {F}\n  fy: {F}\n  cx: {CX}\n  cy: {CY}\n"
            f"  png_depth_scale: {SCALE}\n  crop_edge: 0\n")
        M = surfaces()
        write_pcd(os.path.join(tmp, "map.pcd"), M)

        r = L.convert(rep, os.path.join(tmp, "map.pcd"))
        out = r["out"]
        assert os.path.basename(out) == "lod3dgs_map_mapping_A_20260101", out

        # camera: the largest window centred on (CX, CY)
        cam = open(os.path.join(out, "sparse/0/cameras.txt")).read().splitlines()[-1].split()
        Wc, Hc = int(cam[2]), int(cam[3])
        assert cam[1] == "PINHOLE" and (Wc, Hc) == (141, 109), cam
        assert abs(float(cam[6]) - (Wc - 1) / 2) < 1e-6 and abs(float(cam[7]) - (Hc - 1) / 2) < 1e-6

        # images, depth and poses agree with the room seen by the centred camera
        rows = [l.split() for l in open(os.path.join(out, "sparse/0/images.txt"))
                if l.strip() and not l.startswith("#")]
        assert len(rows) == len(poses)
        for row in rows:
            qw, qx, qy, qz, tx, ty, tz = map(float, row[1:8])
            T_cw = np.eye(4)
            T_cw[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T_cw[:3, 3] = [tx, ty, tz]
            img, D = render(np.linalg.inv(T_cw), Wc, Hc, F, F, (Wc - 1) / 2, (Hc - 1) / 2)
            name = os.path.splitext(row[9])[0]
            got = cv2.imread(os.path.join(out, "images", row[9]))
            la = cv2.imread(os.path.join(out, "depths", name + ".png"), cv2.IMREAD_UNCHANGED)
            assert got.shape == (Hc, Wc, 3) and la.shape == (Hc, Wc, 4), (got.shape, la.shape)
            # away from colour edges (bilinear resampling blends across them)
            flat = cv2.erode((cv2.Laplacian(img.astype(np.float32), cv2.CV_32F) == 0)
                             .all(-1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            err = np.abs(got.astype(int) - img.astype(int)).max(-1)[flat]
            assert np.percentile(err, 99) < 12, (name, np.percentile(err, 99))
            assert (la[..., 0] == la[..., 1]).all() and (la[..., 0] == la[..., 2]).all()
            valid = la[..., 3] == 255
            assert np.mean(valid == (D > 0)) > 0.98, (name, np.mean(valid == (D > 0)))
            # 8-bit steps of 7.8 cm below 10 m (half of it), and the nearest pixel up
            # to half a pixel away (the depth's own slope across it)
            both = valid & (D > 0) & flat
            slope = np.max(np.abs(np.gradient(D)), 0)[both]
            derr = np.abs(L.unwarp_la(la[..., 0])[both] - D[both])
            assert np.mean(derr <= 0.04 + 0.75 * slope) > 0.99, (name, derr.max())

        # starting points: on the walls and floor, nothing over the open top
        P, rgb = read_ply(os.path.join(out, "sparse/0/points3D.ply"))
        assert P[:, 2].max() < ROOM[2] + 0.1, P[:, 2].max()
        dist, _ = cKDTree(M).query(P)
        assert dist.max() < 0.15 and np.percentile(dist, 95) < 0.03, (dist.max(),)
        assert r["faces_kept"] < r["faces"]
        nn = np.sort(cKDTree(P).query(P, k=2)[0][:, 1])
        assert 0.02 < np.median(nn) < 0.06, np.median(nn)
        assert r["coloured"] > 0.6, r["coloured"]
        for a, c, col in PLANES:
            o = [i for i in range(3) if i != a]
            on = ((np.abs(P[:, a] - c) < 0.01) & np.all(P[:, o] > 0.1, 1)
                  & np.all(P[:, o] < ROOM[o] - 0.1, 1) & (rgb != 128).any(1))
            assert on.sum() > 200, (a, c, on.sum())
            med = np.median(rgb[on].astype(int), 0)
            assert np.abs(med - np.array(col)[::-1]).max() < 15, (a, c, med, col)
        print(f"ok: {r['frames']} frames, camera {Wc}x{Hc}, {r['points']} points "
              f"({100 * r['coloured']:.0f}% coloured), mesh {r['faces_kept']}/{r['faces']} faces")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
