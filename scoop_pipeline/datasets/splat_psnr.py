#!/usr/bin/env python3
"""How well a trained splat model reproduces the scene's own images: PSNR
and L1 of splat_walk.py --render-views' renders against the images they
were rendered for, per camera (each export's: the October frames, the
September ones), so two models are compared on the same views.

    python3 scoop_pipeline/datasets/splat_psnr.py <renders folder> --scene <scene folder>
        [--out psnr.csv]

The renders folder is splat_walk.py --render-views' (frameNNNNNN.jpg and
cameras.json naming the scene image of each); the scene's images are
<scene>/images/<name>. Prints per camera the number of views, the mean and
median PSNR (dB) and the mean L1 (0..1), and all views together; writes a
row per view. Rendering a model on another scene's cameras (the combined
scene's for the October-only model) is how the two are compared on the
same October views: the camera that model never saw scores low, by design.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


def diagnose(render, image):
    """PSNR after the best per-channel gain and bias, and that gain.

    Separates the two ways a model fails on views it was TRAINED on. If the
    geometry and poses are right and only the brightness disagrees -- the
    camera's auto-exposure having moved between a dark corridor and a bright
    window -- then one scalar per channel removes most of the error, and the
    corrected PSNR jumps. If the error survives the correction it is spatial:
    poses, blur, or geometry, none of which a gain can fix.

    A model cannot be blamed for failing to reproduce two different exposures
    of one wall; it can be blamed for the rest.
    """
    a = render.astype(np.float64).reshape(-1, render.shape[-1])
    b = image.astype(np.float64).reshape(-1, image.shape[-1])
    fixed = np.empty_like(a)
    gains = []
    for c in range(a.shape[1]):
        x, y = a[:, c], b[:, c]
        var = float(((x - x.mean()) ** 2).mean())
        gain = float(((x - x.mean()) * (y - y.mean())).mean() / var) if var > 1e-9 else 1.0
        fixed[:, c] = gain * x + (y.mean() - gain * x.mean())
        gains.append(gain)
    fixed = np.clip(fixed, 0, 255)
    mse = float(np.mean((fixed - b) ** 2))
    return (99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)), float(np.mean(gains))


def compare(renders, scene, log=print):
    """[{frame, name, camera, psnr, psnr_gain, gain, l1}] and a summary per camera."""
    import cv2
    meta = json.load(open(os.path.join(renders, "cameras.json")))
    rows = []
    for n, fr in enumerate(meta["frames"]):
        rp = os.path.join(renders, f"frame{n:06d}.jpg")
        ip = os.path.join(scene, "images", fr["name"])
        r, im = cv2.imread(rp), cv2.imread(ip)
        if r is None or im is None:
            continue
        if r.shape != im.shape:
            r = cv2.resize(r, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
        corrected, gain = diagnose(r, im)
        rows.append({"frame": n, "name": fr["name"], "camera": int(fr.get("camera", 1)), "psnr": psnr(r, im),
                     "psnr_gain": corrected, "gain": gain,
                     "l1": float(np.mean(np.abs(r.astype(np.float32) - im.astype(np.float32))) / 255.0)})
    summ = {}
    for cam in sorted({r["camera"] for r in rows}) + ["all"]:
        sel = [r for r in rows if cam == "all" or r["camera"] == cam]
        p = np.array([r["psnr"] for r in sel])
        g = np.array([r["psnr_gain"] for r in sel])
        gains = np.array([r["gain"] for r in sel])
        summ[cam] = {"n": len(sel), "psnr_mean": float(p.mean()), "psnr_median": float(np.median(p)),
                     "psnr_gain_mean": float(g.mean()), "psnr_gain_median": float(np.median(g)),
                     "gain_spread": float(np.percentile(gains, 90) - np.percentile(gains, 10)),
                     "l1": float(np.mean([r["l1"] for r in sel]))}
        log("  camera %-4s %5d views  PSNR mean %.2f dB, median %.2f  L1 %.4f"
            % (cam, len(sel), p.mean(), np.median(p), summ[cam]["l1"]))
        log("               after a per-view gain: mean %.2f dB, median %.2f  (+%.2f); "
            "gain 10-90%% spread %.2f"
            % (g.mean(), np.median(g), g.mean() - p.mean(), summ[cam]["gain_spread"]))
    return rows, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("renders")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default=None, help="default <renders>/psnr.csv")
    a = ap.parse_args()
    renders, scene = os.path.expanduser(a.renders), os.path.expanduser(a.scene)
    print("%s against %s/images" % (renders, scene))
    rows, summ = compare(renders, scene)
    if not rows:
        sys.exit("no render matched an image (cameras.json names against <scene>/images)")
    out = a.out or os.path.join(renders, "psnr.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "name", "camera", "psnr",
                                           "psnr_gain", "gain", "l1"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out)
    lift = summ["all"]["psnr_gain_mean"] - summ["all"]["psnr_mean"]
    if lift >= 2.0:
        print("\n  a per-view gain recovers %.1f dB: the brightness disagrees between "
              "views, not\n  the geometry -- auto-exposure, not poses. Train with "
              "per-image appearance\n  compensation (Gaussian-LIC models exactly this) "
              "and lock the camera's exposure\n  on the next capture." % lift)
    else:
        print("\n  a per-view gain recovers only %.1f dB: the error is spatial, not "
              "photometric.\n  Poses, motion blur or geometry -- bundle-adjust the "
              "images (or optimise the\n  cameras during training) before anything "
              "else." % lift)


if __name__ == "__main__":
    main()
