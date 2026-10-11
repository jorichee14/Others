#!/usr/bin/env python3
"""Check the calibration on the floor plane.

Put a floor tag (any id listed in floor_markers.yaml, or a spare id) at a spot, measure its
real position, run:
  python verify.py --tag 11 --x 2.00 --y 4.00
The tag is triangulated from every camera that sees it (live from the RTSP streams) and the
error against the measured spot is printed. Do this at 10 spots across the room; target < 5 cm.
Without --x/--y it just prints the triangulated position (handy to survey the tags themselves).
"""
import argparse
import time

import cv2
import numpy as np

from calib.common import FloorTags, camera_urls, read_json
from calib.streams import open_streams


def triangulate(rays):
    """Least-squares point closest to all rays; rays = [(origin, unit dir)]."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", type=int, required=True)
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--seconds", type=float, default=4.0, help="how long to look")
    args = ap.parse_args()
    calib = read_json("%s/extrinsics/cameras.json" % args.results)["cameras"]
    tags = FloorTags()
    streams = open_streams(camera_urls([int(c) for c in calib]))
    time.sleep(2)
    seen = {}
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        for cam, s in streams.items():
            frame, _ = s.latest()
            if frame is None:
                continue
            det = tags.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if args.tag in det:
                seen[cam] = det[args.tag].mean(0)
        time.sleep(0.2)
    for s in streams.values():
        s.stop()
    if len(seen) < 2:
        raise SystemExit("tag %d seen by %d camera(s); need 2+" % (args.tag, len(seen)))
    rays = []
    for cam, centre in seen.items():
        c = calib[str(cam)]
        K, dist, R, t = np.array(c["K"]), np.array(c["dist"]), np.array(c["R_cw"]), np.array(c["t_cw"])
        pn = cv2.undistortPoints(centre.reshape(1, 1, 2), K, dist).reshape(2)
        d_cam = np.array([pn[0], pn[1], 1.0])
        d_w = R.T @ d_cam
        rays.append((-R.T @ t, d_w / np.linalg.norm(d_w)))
    p = triangulate(rays)
    print("tag %d seen by cameras %s" % (args.tag, sorted(seen)))
    print("triangulated: x %.3f  y %.3f  z %.3f  (z should be ~0 on the floor)" % tuple(p))
    if args.x is not None and args.y is not None:
        err = np.hypot(p[0] - args.x, p[1] - args.y)
        print("floor-plane error vs measured (%.2f, %.2f): %.1f cm  %s" % (
            args.x, args.y, err * 100, "OK" if err < 0.05 else "CHECK"))


if __name__ == "__main__":
    main()
