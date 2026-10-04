#!/usr/bin/env python3
"""Does a Replica-format sequence (datasets/mcap_convert.py --format replica)
fit itself? Run it on a dataset BEFORE giving it to SplaTAM: SplaTAM tracks the
camera against the depth, so depth that does not fit the colour image, or
poses that do not fit the depth, fail every frame whatever the config says.

    python3 checks/replica_check.py <dataset dir> [--frames 0,500,1000] [--step 5]

1. config vs files: png_depth_scale, resolution and intrinsics of
   splatam_data_config.yaml against the PNGs / JPEGs; traj.txt line count.
2. depth statistics per frame: valid fraction, median / p95, fraction at the
   clip (depth_max), and whether the PNG range matches png_depth_scale.
3. depth <-> image fit: the image gradient at the depth edges relative to
   the image gradient everywhere (1 = unrelated), against the same score for
   the depth mirrored left-right (chance level). Depth that belongs to this
   image scores well above its mirror; depth rendered from the wrong pose or
   the wrong time scores like its mirror.
4. depth <-> pose fit: frame i's depth back-projected, moved to frame i+step
   through traj.txt and compared with frame i+step's depth: the fraction of
   pixels off by more than 10%. A good sequence gives a few percent (edges,
   noise); a time offset or a wrong extrinsic gives tens. (Depth rendered
   from a map at the same poses passes this by construction; step 3 is the
   test for those.) A constant time offset hides while the motion is uniform,
   so look at frames where the robot turns.
5. writes <dataset>/check/frameNNNNNN.png: colour | depth | depth edges on
   the colour image, to look at.

Exit code 1 when a hard check fails (config mismatch, depth scoring at
chance, pose fit over 20%).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml


def load_cfg(d):
    p = os.path.join(d, "splatam_data_config.yaml")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return yaml.safe_load(f)["camera_params"]


def depth_edges(D, lo_rel=0.03):
    """Mask of depth discontinuities: a jump of lo_rel of the depth to a
    4-neighbour, both valid."""
    v = D > 0
    e = np.zeros_like(v)
    for ax in (0, 1):
        a, b = np.roll(D, 1, ax), D
        va = np.roll(v, 1, ax) & v
        jump = np.abs(a - b) > lo_rel * np.maximum(np.minimum(a, b), 1e-3)
        e |= va & jump
    return e


def edge_fit(img, D):
    """Image gradient at the depth edges relative to the image gradient
    everywhere (1 = unrelated), and the same for the depth mirrored
    left-right (chance level)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    G = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    G = cv2.GaussianBlur(G, (5, 5), 0)
    de = depth_edges(D)
    de[:, :3] = de[:, -3:] = de[:3, :] = de[-3:, :] = False
    n = de.sum()
    if n < 50 or G.mean() < 1e-3:
        return np.nan, np.nan, int(n)
    return float(G[de].mean() / G.mean()), float(G[de[:, ::-1]].mean() / G.mean()), int(n)


def pose_fit(D0, D1, T0, T1, K):
    """Fraction of frame 0's depth pixels that, moved to frame 1 through the
    poses (camera-to-world), disagree with frame 1's depth by more than 10%,
    over pixels valid in both."""
    fx, fy, cx, cy = K
    H, W = D0.shape
    v, u = np.nonzero(D0 > 0)
    if len(u) < 100:
        return np.nan, 0
    z = D0[v, u]
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z, np.ones_like(z)], 1)
    Q = (np.linalg.inv(T1) @ T0 @ P.T).T
    zq = Q[:, 2]
    ok = zq > 0.05
    uq = np.round(fx * Q[ok, 0] / zq[ok] + cx).astype(int)
    vq = np.round(fy * Q[ok, 1] / zq[ok] + cy).astype(int)
    zq = zq[ok]
    ok = (uq >= 0) & (uq < W) & (vq >= 0) & (vq < H)
    d1 = D1[vq[ok], uq[ok]]
    m = d1 > 0
    if m.sum() < 100:
        return np.nan, int(m.sum())
    return float((np.abs(zq[ok][m] - d1[m]) > 0.1 * d1[m]).mean()), int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--frames", default=None,
                    help="frame indices to check, comma separated (default: 8 spread out)")
    ap.add_argument("--step", type=int, default=10, help="pose fit: frame i vs i+step")
    ap.add_argument("--depth-max", type=float, default=10.0, help="the converter's clip")
    ap.add_argument("--no-images", action="store_true", help="do not write check/*.png")
    a = ap.parse_args()

    d = a.dataset
    jpgs = sorted(glob.glob(os.path.join(d, "results", "frame*.jpg")))
    pngs = sorted(glob.glob(os.path.join(d, "results", "depth*.png")))
    traj = np.loadtxt(os.path.join(d, "traj.txt")).reshape(-1, 4, 4)
    cfg = load_cfg(d)
    bad = []
    print(f"{d}: {len(jpgs)} colour, {len(pngs)} depth, {len(traj)} poses")
    if not (len(jpgs) == len(pngs) == len(traj)):
        bad.append("file counts differ")
    if not jpgs:
        sys.exit("no frames")

    img0 = cv2.imread(jpgs[0])
    png0 = cv2.imread(pngs[0], cv2.IMREAD_UNCHANGED)
    H, W = img0.shape[:2]
    if png0.dtype != np.uint16:
        bad.append(f"depth PNG is {png0.dtype}, not uint16")
    if png0.shape[:2] != (H, W):
        bad.append(f"depth {png0.shape[:2]} and colour {(H, W)} differ in size")
    if cfg is None:
        print("no splatam_data_config.yaml; assuming Replica's 6553.5 units/m")
        scale, K = 6553.5, None
    else:
        scale = float(cfg["png_depth_scale"])
        K = (cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"])
        print(f"config: {cfg['image_width']}x{cfg['image_height']} fx={K[0]:.1f} "
              f"fy={K[1]:.1f} cx={K[2]:.1f} cy={K[3]:.1f} png_depth_scale={scale}")
        if (cfg["image_width"], cfg["image_height"]) != (W, H):
            bad.append(f"config says {cfg['image_width']}x{cfg['image_height']}, "
                       f"images are {W}x{H}")
        if abs(K[2] - W / 2) > 0.15 * W or abs(K[3] - H / 2) > 0.15 * H:
            bad.append(f"principal point ({K[2]:.0f}, {K[3]:.0f}) far from the image "
                       f"centre of {W}x{H}: intrinsics of another resolution?")
    if K is None:
        K = (0.5 * W, 0.5 * W, W / 2, H / 2)
        print("  (no intrinsics: pose fit uses a guess, treat it as rough)")

    # poses
    R = traj[:, :3, :3]
    orth = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
    step_m = np.linalg.norm(np.diff(traj[:, :3, 3], axis=0), axis=1)
    ang = [np.degrees(np.arccos(np.clip((np.trace(R[i].T @ R[i + 1]) - 1) / 2, -1, 1)))
           for i in range(len(R) - 1)]
    print(f"poses: rotations orthonormal to {orth:.1e}; per-frame motion median "
          f"{np.median(step_m)*100:.1f} cm / {np.median(ang):.2f} deg, max "
          f"{step_m.max()*100:.1f} cm / {np.max(ang):.1f} deg; path {step_m.sum():.1f} m")
    if orth > 1e-3:
        bad.append("traj.txt rotations are not orthonormal")
    if step_m.max() > 1.0:
        bad.append(f"a {step_m.max():.1f} m jump between consecutive frames")

    n = len(jpgs)
    if a.frames:
        idx = [int(x) for x in a.frames.split(",")]
    else:
        idx = sorted({int(x) for x in np.linspace(0, n - 1 - a.step, 8)})
    out = os.path.join(d, "check")
    if not a.no_images:
        os.makedirs(out, exist_ok=True)

    print(f"\n{'frame':>6} {'valid%':>6} {'median':>7} {'p95':>6} {'clip%':>6} "
          f"{'edge fit':>9} {'mirror':>7} {'pose fit':>9}")
    fits, mirrors, pfits, clips = [], [], [], []
    for i in idx:
        img = cv2.imread(jpgs[i])
        D = cv2.imread(pngs[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
        v = D > 0
        if not v.any():
            print(f"{i:6d}   no valid depth at all")
            continue
        med, p95 = np.median(D[v]), np.percentile(D[v], 95)
        clip = float((D[v] >= 0.995 * a.depth_max).mean())
        fit, mir, ne = edge_fit(img, D)
        j = i + a.step
        pf, npx = (np.nan, 0)
        if j < n:
            D1 = cv2.imread(pngs[j], cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
            pf, npx = pose_fit(D, D1, traj[i], traj[j], K)
        print(f"{i:6d} {v.mean()*100:6.1f} {med:7.2f} {p95:6.2f} {clip*100:6.1f} "
              f"{fit:9.2f} {mir:7.2f} {pf*100:8.1f}%")
        fits.append(fit), mirrors.append(mir), pfits.append(pf), clips.append(clip)
        if not a.no_images:
            dn = np.clip(D / max(p95, 1e-3), 0, 1)
            dc = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            dc[~v] = 0
            ov = img.copy()
            ov[depth_edges(D)] = (0, 0, 255)
            cv2.imwrite(os.path.join(out, f"frame{i:06d}.png"),
                        np.concatenate([img, dc, ov], 1))

    if not fits:
        sys.exit("nothing checked")
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        f, m = np.nanmean(fits), np.nanmean(mirrors)
        p = np.nanmedian(pfits)
    print()
    if np.isfinite(f) and np.isfinite(m):
        print(f"depth <-> image: image gradient at depth edges {f:.2f}x the "
              f"average, {m:.2f}x for the mirrored depth (chance)")
        if f < 1.25 * m:
            bad.append("depth does not fit the colour image (scores like its mirror): "
                       "wrong pose / time / extrinsic when the depth was rendered")
    if np.isfinite(p):
        print(f"depth <-> poses: {p*100:.1f}% of pixels off by >10% over {a.step} frames")
        if p > 0.25:
            bad.append("poses do not fit the depth: a time offset between images and "
                       "trajectory, or the wrong camera extrinsic")
    if np.mean(clips) > 0.3:
        print(f"note: {np.mean(clips)*100:.0f}% of valid depth sits at the {a.depth_max} m "
              "clip -- the scene is bigger than --depth-max; use --depth-scale 1000 "
              "--depth-max 60 (SplaTAM's png_depth_scale then 1000)")
    if not a.no_images:
        print(f"images in {out}/ (colour | depth | depth edges in red on the colour)")
    if bad:
        print("\nFAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
