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


def compare(renders, scene, log=print):
    """[{frame, name, camera, psnr, l1}] and {camera: {n, psnr_mean, psnr_median, l1}} ('all' too)."""
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
        rows.append({"frame": n, "name": fr["name"], "camera": int(fr.get("camera", 1)), "psnr": psnr(r, im),
                     "l1": float(np.mean(np.abs(r.astype(np.float32) - im.astype(np.float32))) / 255.0)})
    summ = {}
    for cam in sorted({r["camera"] for r in rows}) + ["all"]:
        sel = [r for r in rows if cam == "all" or r["camera"] == cam]
        p = np.array([r["psnr"] for r in sel])
        summ[cam] = {"n": len(sel), "psnr_mean": float(p.mean()), "psnr_median": float(np.median(p)),
                     "l1": float(np.mean([r["l1"] for r in sel]))}
        log("  camera %-4s %5d views  PSNR mean %.2f dB, median %.2f  L1 %.4f"
            % (cam, len(sel), p.mean(), np.median(p), summ[cam]["l1"]))
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
        w = csv.DictWriter(fh, fieldnames=["frame", "name", "camera", "psnr", "l1"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
