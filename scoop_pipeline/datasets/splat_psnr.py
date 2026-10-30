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
import re
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


def offset(render, image):
    """The shift that best lines a render up with its image, and what it buys.

    A spatial error has three causes that look identical in a PSNR and call
    for completely different work. The shift tells them apart:

      a consistent shift across views  -- the camera is not where the rig says
        it is: an extrinsic or a principal point, one constant to re-measure
      shifts scattered view to view    -- the poses themselves are noisy, and
        only bundle adjustment fixes that
      no shift, and the gain unmoved   -- nothing is misplaced; the image is
        blurred or the geometry is wrong

    Phase correlation finds it to a fraction of a pixel, and the PSNR after
    shifting says how much of the error was only ever misalignment.
    """
    import cv2
    a = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a * window, b * window)
    moved = cv2.warpAffine(render, np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
                           (render.shape[1], render.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # The same question asked of nine tiles instead of the whole frame. A
    # camera in the wrong PLACE moves near surfaces further across the image
    # than far ones, so the tiles disagree and no single offset can fix the
    # frame; a camera merely pointed slightly wrong moves everything together
    # and the tiles agree. One distinguishes a wrong extrinsic translation or
    # wrong geometry from a wrong rotation, and they are not the same repair.
    h, w = a.shape
    th, tw = h // 3, w // 3
    tile_window = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    tiles = []
    for i in range(3):
        for j in range(3):
            pa = a[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            pb = b[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if pa.shape != (th, tw):
                continue
            (tx, ty), tr = cv2.phaseCorrelate(np.ascontiguousarray(pa) * tile_window,
                                              np.ascontiguousarray(pb) * tile_window)
            tiles.append((tx, ty, tr))
    strong = [t for t in tiles if t[2] >= 0.05]
    if len(strong) >= 4:
        spread = float(np.hypot(np.std([t[0] for t in strong]),
                                np.std([t[1] for t in strong])))
    else:
        spread = float("nan")
    return (float(dx), float(dy), float(response), psnr(moved, image), spread,
            float(np.median([t[2] for t in tiles])) if tiles else 0.0)


def sharpness(image):
    """How much high-frequency detail an image carries (variance of Laplacian).

    Compared between a render and its image it says which of the two is the
    blurred one -- the model smearing Gaussians to cover inconsistent poses,
    or the camera smearing the scene because the robot was moving.
    """
    import cv2
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def steps_between_frames(scene):
    """How far the camera moved into each frame, by image name: {name: metres}.

    The frames are numbered in the order they were taken, so the distance
    between a frame's centre and its neighbours' is the robot's speed at that
    moment, up to the constant frame interval. It is the one thing a static
    extrinsic error does not depend on and a timestamp offset does: a camera
    whose image is stamped a little before or after the pose it is given
    sits further from where it really was the faster the robot went.
    """
    centres = {}
    with open(os.path.join(scene, "sparse", "0", "images.txt"), encoding="utf-8") as handle:
        for line in handle:
            v = line.split()
            if len(v) >= 10 and v[0].isdigit():
                w, x, y, z = (float(c) for c in v[1:5])
                t = np.array([float(c) for c in v[5:8]])
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                centres[v[9]] = -R.T @ t
    by_index = {}
    for name, centre in centres.items():
        m = re.search(r"(\d+)\.\w+$", name)
        if m:
            by_index.setdefault(name.split("frame")[0] if "frame" in name else "",
                                {})[int(m.group(1))] = (name, centre)
    steps = {}
    for series in by_index.values():
        keys = sorted(series)
        for k, index in enumerate(keys):
            before = series[keys[k - 1]][1] if k > 0 else None
            after = series[keys[k + 1]][1] if k + 1 < len(keys) else None
            name, here = series[index]
            gaps = [np.linalg.norm(here - n) for n in (before, after) if n is not None]
            steps[name] = float(np.mean(gaps)) if gaps else float("nan")
    return steps


def pairs_of(renders, scene, gt_dir=None, only="", exclude=None):
    """(n, name, camera, render path, image path) for every view to score.

    Two layouts: splat_walk's (frameNNNNNN.jpg beside a cameras.json naming
    the scene image each one is of), and NeST's eval_render (renders/ and gt/
    holding the same file names), given as --gt-dir.
    """
    if gt_dir:
        names = sorted(f for f in os.listdir(renders)
                       if f.lower().endswith((".png", ".jpg")) and f.startswith(only)
                       and not (exclude and f.startswith(exclude)))
        return [(n, os.path.splitext(f)[0] + ".jpg", 1, os.path.join(renders, f),
                 os.path.join(gt_dir, f)) for n, f in enumerate(names)]
    meta = json.load(open(os.path.join(renders, "cameras.json")))
    return [(n, fr["name"], int(fr.get("camera", 1)), os.path.join(renders, f"frame{n:06d}.jpg"),
             os.path.join(scene, "images", fr["name"]))
            for n, fr in enumerate(meta["frames"]) if fr["name"].startswith(only)
            and not (exclude and fr["name"].startswith(exclude))]


def compare(renders, scene, log=print, gt_dir=None, only="", exclude=None):
    """[{frame, name, camera, psnr, psnr_gain, gain, l1}] and a summary per camera."""
    import cv2
    try:
        steps = steps_between_frames(scene) if scene else {}
    except OSError:
        steps = {}
    rows = []
    for n, name, camera_id, rp, ip in pairs_of(renders, scene, gt_dir, only, exclude):
        fr = {"name": name, "camera": camera_id}
        r, im = cv2.imread(rp), cv2.imread(ip)
        if r is None or im is None:
            continue
        if r.shape != im.shape:
            r = cv2.resize(r, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
        corrected, gain = diagnose(r, im)
        dx, dy, response, shifted, within, tile_response = offset(r, im)
        rows.append({"frame": n, "name": fr["name"], "camera": int(fr.get("camera", 1)), "psnr": psnr(r, im),
                     "psnr_gain": corrected, "gain": gain,
                     "dx": dx, "dy": dy, "shift": float(np.hypot(dx, dy)),
                     "psnr_shift": shifted, "response": response,
                     "within_view": within, "tile_response": tile_response,
                     "sharp_render": sharpness(r), "sharp_image": sharpness(im),
                     "step": steps.get(fr["name"], float("nan")),
                     "l1": float(np.mean(np.abs(r.astype(np.float32) - im.astype(np.float32))) / 255.0)})
    summ = {}
    if not rows:
        return rows, summ
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
        dx = np.array([r["dx"] for r in sel])
        dy = np.array([r["dy"] for r in sel])
        sh = np.array([r["psnr_shift"] for r in sel])
        sr = np.array([r["sharp_render"] for r in sel])
        si = np.array([r["sharp_image"] for r in sel])
        summ[cam].update({"dx_median": float(np.median(dx)), "dy_median": float(np.median(dy)),
                          "shift_spread": float(np.hypot(dx.std(), dy.std())),
                          "psnr_shift_mean": float(sh.mean()),
                          "sharp_render": float(np.median(sr)),
                          "sharp_image": float(np.median(si))})
        within = np.array([r["within_view"] for r in sel], dtype=float)
        within = within[np.isfinite(within)]
        summ[cam]["within_view"] = float(np.median(within)) if len(within) else float("nan")
        summ[cam]["response"] = float(np.median([r["response"] for r in sel]))
        log("               shift median (%.2f, %.2f) px, scatter %.2f px; after it: "
            "%.2f dB (%+.2f)   [peak %.3f]"
            % (np.median(dx), np.median(dy), summ[cam]["shift_spread"],
               sh.mean(), sh.mean() - p.mean(), summ[cam]["response"]))
        log("               within one frame, tiles disagree by %.2f px"
            % summ[cam]["within_view"])
        moving = np.array([r["step"] for r in sel], dtype=float)
        err = np.array([r["within_view"] for r in sel], dtype=float)
        ok = np.isfinite(moving) & np.isfinite(err)
        if ok.sum() >= 12 and moving[ok].std() > 1e-6:
            order = np.argsort(moving[ok])
            third = max(1, ok.sum() // 3)
            slow, fast = err[ok][order[:third]], err[ok][order[-third:]]
            # An error that never varies has no correlation to compute, and
            # that is the static answer, not a missing one.
            summ[cam]["speed_corr"] = (float(np.corrcoef(moving[ok], err[ok])[0, 1])
                                       if err[ok].std() > 1e-9 else 0.0)
            summ[cam]["slow_err"], summ[cam]["fast_err"] = float(slow.mean()), float(fast.mean())
            log("               by speed: slowest third %.2f px, fastest third %.2f px "
                "(r = %+.2f)" % (slow.mean(), fast.mean(), summ[cam]["speed_corr"]))
        else:
            summ[cam]["speed_corr"] = float("nan")
        log("               detail: render %.0f, image %.0f (%s is the blurred one)"
            % (np.median(sr), np.median(si),
               "the render" if np.median(sr) < np.median(si) else "the image"))
    return rows, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("renders")
    ap.add_argument("--scene", default=None, help="the scene (images/, sparse/0) the renders are of")
    ap.add_argument("--gt-dir", default=None,
                    help="NeST's layout: a gt/ folder with the same file names as the renders; "
                         "--scene is then optional (its images.txt gives the speed test)")
    ap.add_argument("--only", default="", help="score only views whose name starts with this (c2_ for one session)")
    ap.add_argument("--exclude", default=None, help="leave out views whose name starts with this")
    ap.add_argument("--out", default=None, help="default <renders>/psnr.csv")
    a = ap.parse_args()
    renders = os.path.expanduser(a.renders)
    scene = os.path.expanduser(a.scene) if a.scene else None
    gt_dir = os.path.expanduser(a.gt_dir) if a.gt_dir else None
    if not scene and not gt_dir:
        sys.exit("give --scene (splat_walk's layout) or --gt-dir (NeST's)")
    print("%s against %s" % (renders, gt_dir or (scene + "/images")))
    rows, summ = compare(renders, scene, gt_dir=gt_dir, only=a.only, exclude=a.exclude)
    if not rows:
        sys.exit("no view matched (names against <scene>/images, or none starts with --only; "
                 "ls the renders dir to see the prefixes)")
    out = a.out or os.path.join(renders, "psnr.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "name", "camera", "psnr",
                                           "psnr_gain", "gain", "dx", "dy", "shift",
                                           "psnr_shift", "response", "within_view",
                                           "tile_response", "sharp_render",
                                           "sharp_image", "step", "l1"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out)
    a = summ["all"]
    systematic = float(np.hypot(a["dx_median"], a["dy_median"]))
    helps = a["psnr_shift_mean"] - a["psnr_mean"]
    within = a["within_view"]
    print("")
    if within == within and within >= 1.0:
        # Asked first, because a camera in the wrong place also responds a
        # little to a global shift and would otherwise read as a rotation.
        print("  tiles within one frame disagree by %.1f px: the misalignment depends "
              "on depth,\n  which is what a camera in the wrong PLACE looks like -- not "
              "the rotations, and not\n  something bundle adjustment fixes with the "
              "extrinsic held fixed." % within)
        corr = a.get("speed_corr", float("nan"))
        if corr == corr and a["fast_err"] >= 1.5 * a["slow_err"] and corr >= 0.3:
            print("  It grows with speed (%.1f px slow, %.1f px fast): the image is "
                  "stamped at a\n  different time than the pose it was given. Find the "
                  "camera-to-pose time offset\n  and re-export; a static extrinsic "
                  "would not care how fast the robot went."
                  % (a["slow_err"], a["fast_err"]))
        elif corr == corr:
            print("  It does not grow with speed (%.1f px slow, %.1f px fast): a static "
                  "offset. The\n  camera-to-LiDAR translation is wrong by about a "
                  "centimetre; re-measure it, or\n  let training optimise each camera's "
                  "pose, which absorbs it." % (a["slow_err"], a["fast_err"]))
    elif helps < 0.5:
        # A shift that buys nothing found no real misalignment, whatever number
        # it reported: a peak in noise is still a peak.
        if a["sharp_render"] < 0.85 * a["sharp_image"]:
            print("  no shift helps (%+.2f dB) and nothing moves within a frame "
                  "(%.1f px), but the\n  render carries %.0f%% of the image's detail: "
                  "the model is smearing. Capacity,\n  densification, or Gaussians "
                  "covering observations that disagree."
                  % (helps, within, 100 * a["sharp_render"] / max(1e-9, a["sharp_image"])))
        else:
            print("  no shift helps (%+.2f dB) and the render is as sharp as the "
                  "image: the\n  images themselves are the limit -- motion blur, noise "
                  "or rolling shutter." % helps)
    elif systematic >= 1.0 and systematic > a["shift_spread"]:
        print("  every view is off by about the same (%.1f, %.1f) px, more than they\n"
              "  differ from each other (%.1f px): one constant is wrong, not the poses.\n"
              "  Re-measure the camera-to-LiDAR extrinsic and the principal point before\n"
              "  touching the training."
              % (a["dx_median"], a["dy_median"], a["shift_spread"]))
    elif a["shift_spread"] >= 1.0:
        print("  the views are off by %.1f px in scattered directions: the poses are\n"
              "  noisy. Bundle-adjust the images with the LiDAR poses as priors, or\n"
              "  optimise the cameras during training, then retrain."
              % a["shift_spread"])
    elif a["sharp_render"] < 0.6 * a["sharp_image"]:
        print("  nothing is misplaced (%.1f px) and the render carries %.0f%% of the\n"
              "  image's detail: the model is smearing. Capacity, densification or an\n"
              "  over-strong regulariser -- not the capture."
              % (a["shift_spread"], 100 * a["sharp_render"] / max(1e-9, a["sharp_image"])))
    else:
        print("  nothing is misplaced (%.1f px) and the render is as sharp as the image:\n"
              "  the images themselves are the limit -- motion blur, noise or rolling\n"
              "  shutter on a moving base." % a["shift_spread"])
    lift = a["psnr_gain_mean"] - a["psnr_mean"]
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
