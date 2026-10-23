#!/usr/bin/env python3
"""Why was a board seen, or not: every sampled frame of one camera stream
scanned for one board, with what the detector saw in it.

    python3 scoop_pipeline/map_stages/board_scan.py <pipeline config> --board board_4
        [--camera NAME]            a 07_build_cameras entry (its bag, topics,
                                   rectified); default the dataset's own camera
        [--image-topic T --camera-info T [--rectified] [--bag BAG]]   any stream
        [--from S] [--to S]        seconds into the stream (default all of it)
        [--period 0.2]             seconds between sampled frames (0: every frame)
        [--out DIR]                default processed/<pass>/frames/board_scan_<board>_<camera>/
        [--save found|all|none]    annotated frames written (default found: the
                                   frames with any marker of the board's dictionary)

Per sampled frame: the markers of the board's dictionary found (how many are
the board's own ids, their size in pixels), the chessboard corners
interpolated from them, and the pose attempt's verdict (ok, or why not:
no_corners, few_corners, pnp_fail, ambiguous, reproj). Frames with nothing of
the dictionary count as absent. Written: a line per frame that had a marker,
the totals, a contact sheet of the whole stream (green border: pose found,
yellow: markers but no pose, none: absent) so what the camera looked at is
seen at a glance, and the annotated frames (--save).

What it tells apart: the board never in view (absent everywhere, the sheet
shows what was in view instead), in view but too small or blurred (markers
found but no or few corners: marker size in px), in view and read but the pose
rejected (reproj, ambiguous), or read fine and only too briefly for stage 05's
section gate (ok in fewer than min_section_samples frames).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2                                                             # noqa: E402
from pipeline_boards import (bank_from_config, iter_images, pick_intrinsics,  # noqa: E402
                             read_tf_and_info, _marker_centers_sizes)

VERDICTS = ("ok", "no_corners", "few_corners", "pnp_fail", "ambiguous", "reproj")


def scan_frame(bank, b, gray, K, D):
    """What the detector sees of board `b` in one image: dict with n_markers
    (of the board's dictionary), n_mine (the board's own ids), px (median
    marker side, px), n_corners, verdict, det (Detection or None), corners,
    ids."""
    adet = bank.by_dict[b.dict_name][0]
    corners, ids, _ = adet.detectMarkers(gray)
    r = {"n_markers": 0, "n_mine": 0, "px": 0.0, "n_corners": 0, "verdict": "absent",
         "det": None, "corners": [], "ids": np.zeros(0, int), "range_m": float("nan")}
    if ids is None or len(ids) == 0:
        return r
    ids = np.asarray(ids).flatten()
    mine = np.array([int(i) in b.id_set for i in ids])
    r.update(n_markers=int(len(ids)), n_mine=int(mine.sum()), corners=list(corners), ids=ids)
    if mine.any():
        _, S = _marker_centers_sizes([c for c, m in zip(corners, mine) if m])
        r["px"] = float(np.median(S))
    if not mine.any():
        return r
    cc, cid = b.corners_full(gray)
    r["n_corners"] = 0 if cid is None else int(len(cid))
    before = dict(b.reject)
    d = b.pose_from_corners(cc, cid, K, D)
    if d is not None:
        r["verdict"], r["det"] = "ok", d
        r["range_m"] = float(np.linalg.norm(d.T[:3, 3]))
    else:
        r["verdict"] = next((k for k in VERDICTS if b.reject.get(k, 0) > before.get(k, 0)), "rejected")
    r["cc"], r["cid"] = cc, cid
    return r


def annotate(gray, r, b):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if r["corners"]:
        cv2.aruco.drawDetectedMarkers(img, r["corners"], r["ids"].reshape(-1, 1))
    if r.get("cid") is not None and r["n_corners"]:
        for p in np.asarray(r["cc"], float).reshape(-1, 2):
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), 1)
    return img


def contact_sheet(thumbs, cols=8, w=200):
    """thumbs: [(gray, label, colour)] -> one image."""
    if not thumbs:
        return None
    h = int(w * thumbs[0][0].shape[0] / thumbs[0][0].shape[1])
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.full((rows * (h + 4), cols * (w + 4), 3), 255, np.uint8)
    for k, (g, label, col) in enumerate(thumbs):
        t = cv2.cvtColor(cv2.resize(g, (w, h)), cv2.COLOR_GRAY2BGR)
        if col is not None:
            cv2.rectangle(t, (0, 0), (w - 1, h - 1), col, 4)
        cv2.putText(t, label, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        y, x = (k // cols) * (h + 4), (k % cols) * (w + 4)
        sheet[y:y + h, x:x + w] = t
    return sheet


def scan(frames, bank, b, K, D, t_from=0.0, t_to=None, out=None, save="found", sheet_n=240,
         log=print):
    """frames: iterable of (stamp, gray). -> (records, summary); writes the
    annotated frames and the contact sheet under out."""
    recs, thumbs, t0 = [], [], None
    n_abs = 0
    for st, gray in frames:
        if t0 is None:
            t0 = st
        rel = st - t0
        if rel < t_from or (t_to is not None and rel > t_to):
            continue
        r = scan_frame(bank, b, gray, K, D)
        r["t"], r["rel"] = st, rel
        recs.append({k: v for k, v in r.items() if k not in ("corners", "ids", "cc", "cid", "det")}
                    | {"reproj": r["det"].reproj if r["det"] else float("nan")})
        col = (0, 200, 0) if r["verdict"] == "ok" else ((0, 200, 255) if r["n_mine"] else None)
        thumbs.append((gray, "%.1fs" % rel, col))
        if r["n_mine"]:
            log("  %8.1f s  markers %2d (%2d of %s, %4.0f px)  corners %2d  %-11s"
                % (rel, r["n_markers"], r["n_mine"], b.name, r["px"], r["n_corners"], r["verdict"])
                + (("  range %.1f m  reproj %.2f px" % (r["range_m"], r["det"].reproj)) if r["det"] else ""))
        else:
            n_abs += 1
        if out and (save == "all" or (save == "found" and r["n_markers"])):
            cv2.imwrite(os.path.join(out, "%s_%07.1fs_%s.jpg" % (b.name, rel, r["verdict"])),
                        annotate(gray, r, b), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if sheet_n and len(thumbs) > sheet_n:          # evenly over the stream
        thumbs = [thumbs[i] for i in np.linspace(0, len(thumbs) - 1, sheet_n).astype(int)]
    n = len(recs)
    counts = {v: sum(1 for r in recs if r["verdict"] == v) for v in ("absent",) + VERDICTS}
    seen = [r["rel"] for r in recs if r["n_mine"]]
    summ = {"frames": n, "absent": n_abs, "with_dict_markers": sum(1 for r in recs if r["n_markers"]),
            "with_board_markers": len(seen), "verdicts": counts,
            "seen_from_s": min(seen) if seen else None, "seen_to_s": max(seen) if seen else None,
            "px_median": float(np.median([r["px"] for r in recs if r["n_mine"]])) if seen else None}
    if out:
        sh = contact_sheet(thumbs)
        if sh is not None:
            cv2.imwrite(os.path.join(out, "contact_sheet.jpg"), sh, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return recs, summ


def camera_of(P, name):
    """(bag, image_topic, camera_info_topic, rectified, K, D fallbacks) of a
    07_build_cameras entry, or the dataset's camera (name None)."""
    if name is None:
        S = P.sensor
        return P.dataset["bag"], S.image_topic, S.camera_info_topic, P.rectified, S.K, S.dist, "dataset"
    for cam in (P.stage("07_build_cameras").get("cameras") or []):
        if cam.get("name") == name:
            return (cam.get("bag") or P.dataset["bag"], cam["image_topic"], cam.get("camera_info_topic"),
                    bool(cam.get("rectified", False)), cam.get("K"), cam.get("dist"), name)
    raise SystemExit("no camera %r in 07_build_cameras (have: %s)"
                     % (name, ", ".join(c.get("name", "?") for c in P.stage("07_build_cameras").get("cameras") or [])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--board", required=True)
    ap.add_argument("--camera", default=None, help="a 07_build_cameras entry's name")
    ap.add_argument("--image-topic", default=None)
    ap.add_argument("--camera-info", default=None)
    ap.add_argument("--rectified", action="store_true")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--period", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", choices=("found", "all", "none"), default="found")
    a = ap.parse_args()
    from pipeline_common import load_pipeline
    P = load_pipeline(a.config)
    bag, itopic, ctopic, rect, Kf, Df, cname = camera_of(P, a.camera)
    if a.image_topic:
        itopic, ctopic, rect, cname = a.image_topic, a.camera_info or ctopic, a.rectified, "topic"
    bag = os.path.expanduser(a.bag or bag)
    bank = bank_from_config(P.cfg, [a.board])
    b = bank.designs[0]
    print(b.describe())
    infos, _, available = read_tf_and_info(bag, [ctopic] if ctopic else [], want_tf=False)
    if itopic not in available:
        raise SystemExit("no %s in %s\n  topics: %s" % (itopic, bag, "\n  ".join(available)))
    K, D, from_bag = pick_intrinsics(infos, ctopic, rect, Kf, Df)
    print("camera %s: %s (%s, fx=%.1f, %s)" % (cname, itopic, "bag CameraInfo" if from_bag else "config",
                                             K[0, 0], "rectified" if rect else "distorted, D from the info"))
    out = os.path.expanduser(a.out) if a.out else P.outp("board_scan_%s_%s" % (a.board, cname))
    os.makedirs(out, exist_ok=True)                     # frames/board_scan_<board>_<camera>/
    t_wall = time.time()
    print("scanning %s every %.2f s%s -> %s" % (bag, a.period,
                                                 (" from %.0f s" % a.t_from) if a.t_from else "", out))
    _, summ = scan(iter_images(bag, itopic, a.period), bank, b, K, D, a.t_from, a.t_to, out, a.save)
    v = summ["verdicts"]
    print("\n%d frames in %.0f s: %d with no marker of %s at all, %d with another board's markers only, "
          "%d with %s's markers" % (summ["frames"], time.time() - t_wall, summ["absent"], b.dict_name,
                                     summ["with_dict_markers"] - summ["with_board_markers"],
                                     summ["with_board_markers"], b.name))
    if summ["with_board_markers"]:
        print("  seen %.1f .. %.1f s into the stream, marker side median %.0f px; verdicts: %s"
              % (summ["seen_from_s"], summ["seen_to_s"], summ["px_median"],
                 ", ".join("%s %d" % (k, v[k]) for k in VERDICTS if v[k])))
        if v["ok"]:
            print("  pose found in %d frames (stage 05 keeps a section of min_section_samples, 4, "
                  "within section_gap, 3 s)" % v["ok"])
        if v["few_corners"] or v["no_corners"]:
            print("  markers found but the chessboard corners not (min_corners %d): too small, blurred, "
                  "or partly out of frame -- look at the saved frames" % b.min_corners)
    else:
        print("  never in this stream: the contact sheet shows what the camera looked at")
    print("wrote %s/contact_sheet.jpg%s" % (out, "" if a.save == "none" else " and the annotated frames"))


if __name__ == "__main__":
    main()
