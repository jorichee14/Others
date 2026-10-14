#!/usr/bin/env python3
"""Re-evaluate an existing calibration WITHOUT refitting: how well do the stored intrinsics and
extrinsics explain a set of recorded frames?

  python evaluate.py --data data                 # the original test views/events (split from the CSVs)
  python evaluate.py --data data_check --all     # a fresh recording, e.g. a month later: everything is test

Use it after the calibration day to detect a bumped or drifted camera: record a short new
session with capture.py (intr and/or extr mode, any person name) into a new folder and run this
on it. Writes results/evaluation/<stamp>/cameras.csv (per camera) and details.csv (per view,
event and tag), and prints the per-camera table with the stored calibrate errors beside the new ones.
"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np

from calib.common import FloorTags, load_boards, load_intrinsics, pose_row, read_csv, read_json, write_csv
from calib.extr import evaluate_events, evaluate_tags, load_observations, rms_of
from calib.intr import evaluate as eval_views, load_views, view_features


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--all", action="store_true", help="treat every view and event as test (new recording)")
    ap.add_argument("--cams", nargs="*", type=int, default=list(range(1, 17)))
    args = ap.parse_args()
    boards, tags = load_boards(), FloorTags()
    intr = load_intrinsics(args.results)
    if not intr:
        raise SystemExit("no intrinsics in %s/intrinsics" % args.results)
    out_dir = os.path.join(args.results, "evaluation", time.strftime("%Y%m%d_%H%M%S"))
    details, summary = [], {}
    stored_i = {int(r["camera"]): r for r in read_csv(os.path.join(args.results, "intrinsics", "summary.csv"))}
    stored_e = {int(r["camera"]): r for r in read_csv(os.path.join(args.results, "extrinsics", "cameras.csv"))}
    known_views = {r["file"]: r["split"] for r in read_csv(os.path.join(args.results, "intrinsics", "views.csv"))}
    known_events = {r["event"]: r["split"] for r in read_csv(os.path.join(args.results, "extrinsics", "events.csv"))}

    # --- intrinsics: reprojection of the test views with the stored K, dist
    for cam in args.cams:
        if cam not in intr:
            continue
        views = load_views(args.data, cam)
        if not args.all:
            views = [v for v in views if known_views.get(v["path"], v["split"]) == "test"]
        if not views:
            continue
        K, dist = intr[cam]
        errs = eval_views(views, boards, K, dist)
        for v, e in zip(views, errs):
            f = view_features(boards[v["board"]], v["corners"], v["ids"], K, dist) or {}
            details.append({"type": "intr_view", "camera": cam, "name": v["path"], "rms_px": round(e, 3),
                            **{k: round(f[k], 3) for k in f}})
        s = summary.setdefault(cam, {"camera": cam})
        s["intr_views"], s["intr_rms_px"] = len(views), round(rms_of(errs), 3)
        s["intr_rms_stored_px"] = stored_i.get(cam, {}).get("rms_test_px", "")

    # --- extrinsics: board refit per event with cameras fixed, floor tags
    ext_path = os.path.join(args.results, "extrinsics", "cameras.json")
    if os.path.exists(ext_path):
        ext = read_json(ext_path)["cameras"]
        T_cw = {}
        for c, v in ext.items():
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = np.array(v["R_cw"]), np.array(v["t_cw"])
            T_cw[int(c)] = T
        events, tag_obs = load_observations(args.data, boards, tags, intr)
        names = sorted(events) if args.all else [e for e in sorted(events) if known_events.get(e, "test") == "test"]
        evals = evaluate_events(events, names, T_cw, intr)
        per = defaultdict(list)
        for e, (T, p) in evals.items():
            details.append({"type": "extr_event", "camera": "", "name": e, "rms_px": round(rms_of(p.values()), 3),
                            **pose_row(T)})
            for c, v in p.items():
                per[c].append(v)
        if not args.all:
            tag_obs = [t for t in tag_obs if t[1] and known_events.get(t[1], "test") == "test"]
        floor = defaultdict(list)
        for r in evaluate_tags(tag_obs, T_cw, intr):
            floor[r["camera"]].append(r["floor_err_m"])
            details.append({"type": "floor_tag", "camera": r["camera"], "name": "%s tag %d" % (r["event"], r["tag"]),
                            "rms_px": round(r["rms_px"], 3), "floor_err_cm": round(100 * r["floor_err_m"], 1)})
        for cam in sorted(set(per) | set(floor)):
            s = summary.setdefault(cam, {"camera": cam})
            s["extr_events"], s["extr_rms_px"] = len(per[cam]), round(rms_of(per[cam]), 3) if per[cam] else ""
            s["extr_rms_stored_px"] = stored_e.get(cam, {}).get("rms_test_px", "")
            s["floor_err_cm"] = round(100 * rms_of(floor[cam]), 1) if floor[cam] else ""

    for cam, s in summary.items():
        bad = []
        if s.get("intr_rms_px", 0) and float(s["intr_rms_px"]) > 1.0:
            bad.append("intrinsics")
        if s.get("extr_rms_px", "") != "" and float(s["extr_rms_px"]) > 1.5:
            bad.append("extrinsics")
        if s.get("floor_err_cm", "") != "" and float(s["floor_err_cm"]) > 5:
            bad.append("floor>5cm")
        s["status"] = "OK" if not bad else "CHECK " + " ".join(bad)
    fields = ["camera", "intr_views", "intr_rms_px", "intr_rms_stored_px", "extr_events", "extr_rms_px",
              "extr_rms_stored_px", "floor_err_cm", "status"]
    rows = [summary[c] for c in sorted(summary)]
    write_csv(os.path.join(out_dir, "cameras.csv"), rows, fields)
    write_csv(os.path.join(out_dir, "details.csv"), details,
              ["type", "camera", "name", "rms_px", "floor_err_cm", "x", "y", "z", "roll", "pitch", "yaw"])
    print("%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-9s %s" % (
        "cam", "intr n", "intr rms", "stored", "extr n", "extr rms", "stored", "floor cm", "status"))
    for r in rows:
        print("%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-9s %s" % tuple(str(r.get(k, "")) for k in fields))
    print("written", out_dir)


if __name__ == "__main__":
    main()
