#!/usr/bin/env python3
"""Extrinsics of all cameras in the ROOM FRAME, from the frame sets saved by capture.py --mode extr
plus the floor tags seen in any frame (intr or extr).

  python calibrate_extrinsics.py --data data --results results

Needs results/intrinsics/cam_N.json for every camera involved (calibrate_intrinsics.py).

Method
  1. Board pose per camera per event from solvePnP (initial values).
  2. Cameras that see 2+ surveyed floor tags get an initial room pose from PnP on the tags;
     the rest are chained through events they share with already-placed cameras.
  3. One bundle adjustment over all cameras and all CALIBRATE events: reprojection of every
     ChArUco corner and every floor-tag corner (Huber loss). Floor tags pin the room frame.
  4. TEST: every 5th event is held out. With the cameras fixed, the board pose of each test
     event is refit and the reprojection error per camera is the test error. Floor tags seen in
     test events give the floor-point error in metres.
Writes
  results/extrinsics/cameras.json   per camera: K, dist, R_cw, t_cw, position, roll pitch yaw
  results/extrinsics/cameras.csv    per camera: x y z roll pitch yaw, rms calib/test, floor error
  results/extrinsics/events.csv     per event: split, cameras, rms, board pose in the room
  results/extrinsics/tags.csv       per floor-tag observation: split, px error, floor error
  results/extrinsics/report.md
"""
import argparse
import os
from collections import defaultdict

import numpy as np

from calib.common import FloorTags, euler_zyx_deg, inv_T, load_boards, load_intrinsics, pose_row, read_csv, write_csv, write_json
from calib.extr import (bundle, evaluate_events, evaluate_tags, initial_poses, load_observations, per_camera,
                        rms_of, split_events)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--keep-split", action="store_true",
                    help="keep the calib/test split recorded in an earlier results/extrinsics/events.csv")
    args = ap.parse_args()
    out_dir = os.path.join(args.results, "extrinsics")
    boards, tags = load_boards(), FloorTags()
    intr = load_intrinsics(args.results)
    if not intr:
        raise SystemExit("no intrinsics in %s/intrinsics; run calibrate_intrinsics.py first" % args.results)
    events, tag_obs = load_observations(args.data, boards, tags, intr)
    if not events:
        raise SystemExit("no extrinsic events: run capture.py --mode extr and hold the board still")
    known = {r["event"]: r["split"] for r in read_csv(os.path.join(out_dir, "events.csv"))} if args.keep_split else None
    split = split_events(events, known)
    calib_ev = {e: v for e, v in events.items() if split[e] == "calib"}
    test_ev = sorted(e for e in events if split[e] == "test")
    calib_tags = [t for t in tag_obs if t[1] is None or split.get(t[1]) == "calib"]
    print("%d events with 2+ cameras (%d calib, %d test), %d floor-tag observations, %d cameras with intrinsics" % (
        len(events), len(calib_ev), len(test_ev), len(tag_obs), len(intr)))

    T_cw, T_wb = initial_poses(calib_ev, calib_tags, intr)
    missing = sorted(set(intr) - set(T_cw))
    if missing:
        print("WARNING: no path to cameras %s (they share no still board view with the others)" % missing)
    T_cw, T_wb, rms0, rms1, per_cam, n_ev = bundle(calib_ev, calib_tags, intr, T_cw, T_wb)
    print("bundle adjustment: %d cameras, %d events, rms %.2f -> %.2f px" % (len(T_cw), n_ev, rms0, rms1))

    # evaluation: calib events (as fitted) and test events (board refit, cameras fixed)
    evals = {e: (T_wb[e], {}) for e in T_wb}
    evals.update(evaluate_events(events, test_ev, T_cw, intr))
    test_per = per_camera({e: v for e, v in evals.items() if split[e] == "test"}, split)
    tag_rows = evaluate_tags(tag_obs, T_cw, intr)
    for r in tag_rows:
        r["split"] = split.get(r["event"], "calib") if r["event"] else "calib"
    floor = defaultdict(list)
    for r in tag_rows:
        if r["split"] == "test":
            floor[r["camera"]].append(r["floor_err_m"])
    n_cal = defaultdict(int)
    for e in T_wb:
        for c in events[e]:
            n_cal[c] += 1

    cams_out, cam_rows = {}, []
    for c in sorted(T_cw):
        K, dist = intr[c]
        T_wc = inv_T(T_cw[c])
        p = pose_row(T_wc)
        rt = rms_of(test_per[c]["test"])
        fe = rms_of(floor[c]) if floor[c] else float("nan")
        cams_out[str(c)] = {"K": K, "dist": dist, "R_cw": T_cw[c][:3, :3], "t_cw": T_cw[c][:3, 3],
                            "position_world": T_wc[:3, 3], "roll_pitch_yaw_deg": euler_zyx_deg(T_wc[:3, :3]),
                            "rms_px": per_cam.get(c), "rms_test_px": rt, "floor_err_test_m": fe}
        cam_rows.append({"camera": c, **p, "rms_calib_px": round(per_cam.get(c, float("nan")), 3),
                         "rms_test_px": round(rt, 3), "n_calib_events": n_cal[c], "n_test_events": len(test_per[c]["test"]),
                         "floor_err_test_cm": round(100 * fe, 1) if np.isfinite(fe) else "",
                         "status": "OK" if (per_cam.get(c, 9) < 1.0 and (not np.isfinite(rt) or rt < 1.5)) else "CHECK"})
    ev_rows = []
    for e in sorted(events):
        if e not in evals:
            continue
        T, per = evals[e]
        ev_rows.append({"event": e, "split": split[e], "n_cams": len(events[e]),
                        "cams": " ".join(str(c) for c in sorted(events[e])),
                        "rms_px": round(rms_of(per.values()), 3) if per else "", **pose_row(T)})
    for r in tag_rows:
        r["rms_px"], r["floor_err_cm"] = round(r["rms_px"], 3), round(100 * r.pop("floor_err_m"), 1)

    write_json(os.path.join(out_dir, "cameras.json"),
               {"frame": "room (floor tags), x y along walls, z up, metres; roll pitch yaw: R_wc = Rz(yaw) Ry(pitch) Rx(roll), degrees",
                "cameras": cams_out})
    write_csv(os.path.join(out_dir, "cameras.csv"), cam_rows)
    write_csv(os.path.join(out_dir, "events.csv"), ev_rows)
    write_csv(os.path.join(out_dir, "tags.csv"), tag_rows,
              ["camera", "event", "split", "tag", "rms_px", "floor_err_cm"])

    t_all = rms_of([v for c in test_per for v in test_per[c]["test"]])
    f_all = rms_of([v for c in floor for v in floor[c]])
    lines = ["# ED305 camera extrinsics", "",
             "Room frame from floor tags: %d tag observations. Bundle rms %.2f px over %d calibrate events; "
             "test rms %.2f px over %d held-out events; floor-point error on held-out tags %.1f cm." % (
                 len(tag_obs), rms1, n_ev, t_all, len(test_ev), 100 * f_all if np.isfinite(f_all) else float("nan")),
             "", "| cam | x | y | z | roll | pitch | yaw | rms calib | rms test | floor err cm | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cam_rows:
        lines.append("| %(camera)d | %(x).3f | %(y).3f | %(z).3f | %(roll).1f | %(pitch).1f | %(yaw).1f | "
                     "%(rms_calib_px).2f | %(rms_test_px).2f | %(floor_err_test_cm)s | %(status)s |" % r)
    lines += ["", "Cameras hang 2.5 to 2.7 m above the floor: z should be in that range.",
              "Re-run after any camera is moved. Re-check later with evaluate.py; verify live with verify.py."]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
