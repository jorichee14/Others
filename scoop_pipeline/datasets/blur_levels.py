#!/usr/bin/env python3
"""How blurred each training photograph is, as a Gaussian sigma in pixels.

    python3 scoop_pipeline/datasets/blur_levels.py <scene> [--window 10] [--width 424]

For every image in <scene>/images: its sharpness (variance of the Laplacian)
and, as the reference, the sharpest frame within --window frames of it, the
same part of the building seen a moment earlier or later. The reference is
blurred with increasing sigma until its sharpness falls to the frame's own;
that sigma is the frame's blur level, in pixels of the full image. Matching
against a neighbour of the same content keeps a bare wall from counting as
blur: a wall is as flat in the reference as in the frame.

Writes <scene>/blur_sigma.csv (name, sharpness, reference, sigma_px), read by
the blur-matched loss in NeST (nest_blur_loss.patch): a frame's render is
blurred to its sigma before the photometric loss, so a smeared photograph
teaches the model its geometry and colour but not its smear.
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np


def small_grey(path, width):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read {path}")
    h = max(1, int(round(g.shape[0] * width / g.shape[1])))
    return cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA), g.shape[1]


def lapvar(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def sigma_to_match(ref, target, lo=0.0, hi=12.0, steps=14):
    """The Gaussian sigma (small-image pixels) that takes ref's sharpness down to target."""
    if lapvar(ref) <= target:
        return 0.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        v = lapvar(cv2.GaussianBlur(ref, (0, 0), mid))
        lo, hi = (mid, hi) if v > target else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--window", type=int, default=10, help="frames either side to take the reference from")
    ap.add_argument("--width", type=int, default=424, help="working width (px) for the measure")
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="a frame counts as blurred under this fraction of its neighbourhood's sharp level")
    a = ap.parse_args()
    scene = os.path.expanduser(a.scene)
    found = sorted(glob.glob(os.path.join(scene, "images", "*.jpg")) + glob.glob(os.path.join(scene, "images", "*.png")))
    by_name = {}                                   # one file per frame name; the .jpg is the training image
    for p in found:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in by_name or p.lower().endswith(".jpg"):
            by_name[stem] = p
    paths = [by_name[n] for n in sorted(by_name)]
    if not paths:
        raise SystemExit(f"{scene}/images: no images")
    if len(found) != len(paths):
        print(f"{len(found)} files, {len(paths)} frames (a .jpg and a .png of the same name count once)")
    names = sorted(by_name)
    # frames of one session are consecutive by name; the reference is searched within the session only
    session = [n.split("_")[0] if "_" in n else "" for n in names]
    greys, widths = zip(*(small_grey(p, a.width) for p in paths))
    sharp = np.array([lapvar(g) for g in greys])
    print(f"{len(paths)} images; sharpness median {np.median(sharp):.1f}, "
          f"10% {np.percentile(sharp, 10):.1f}, 90% {np.percentile(sharp, 90):.1f}", flush=True)
    rows, sigmas = [], []
    for i in range(len(paths)):
        lo, hi = max(0, i - a.window), min(len(paths), i + a.window + 1)
        cand = [j for j in range(lo, hi) if session[j] == session[i]]
        r = max(cand, key=lambda j: sharp[j])
        # blurred only when well under the neighbourhood's sharp level (its upper quartile):
        # the sharpest of 21 frames is above all the others by construction, and a frame a
        # little under it is a different view of the same place, not a smeared one
        level = np.percentile(sharp[cand], 75)
        s_small = sigma_to_match(greys[r], sharp[i]) if (r != i and sharp[i] < a.ratio * level) else 0.0
        s_full = s_small * widths[i] / a.width
        sigmas.append(s_full)
        rows.append((names[i], f"{sharp[i]:.2f}", names[r], f"{s_full:.3f}"))
    sigmas = np.array(sigmas)
    out = os.path.join(scene, "blur_sigma.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sharpness", "reference", "sigma_px"])
        w.writerows(rows)
    flagged = sigmas > 0
    print(f"blurred frames (under {a.ratio:g} of their neighbourhood's sharp level): {int(flagged.sum())} of {len(sigmas)} "
          f"({100 * flagged.mean():.1f}%); their sigma (full-res px): median {np.median(sigmas[flagged]) if flagged.any() else 0:.2f}, "
          f"90% {np.percentile(sigmas[flagged], 90) if flagged.any() else 0:.2f}, max {sigmas.max():.2f}; "
          f"over 2 px: {100 * np.mean(sigmas > 2):.1f}% of all frames")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
