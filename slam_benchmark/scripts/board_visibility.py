#!/usr/bin/env python3
"""When was each surveyed board actually IN THE CAMERA, per agent.

phase0's dwell finder used position and speed only, which cannot tell "parked
facing the board" from "drove slowly past it". The reference trajectory carries
full 6-DoF poses, so the question is geometric and answerable without a human:
project each board's surveyed position into the camera and ask whether it lands
inside the image.

What this settles automatically:
  * in front of the camera vs behind it          (sign of z in the optical frame)
  * inside the field of view vs outside          (projection against width/height)
  * how far away, and where in frame             (range, pixel position)
  * the board's FACING, when --anchor-frame supplies its orientation — the
    incidence angle between the camera's line of sight and the board normal
  * whether the ChArUco is BIG ENOUGH TO DETECT at that range. The coop2 boards
    are small (18 x 14 cm, 15 mm markers; rs_anchor 22.5 x 17.5 cm, 18 mm), so
    range decides detectability far more sharply than field of view does: at
    4 m a 15 mm marker is 2 px across and no detector will find it.

What it CANNOT settle: occlusion, motion blur, exposure. Only running the
detector settles those — and that run also yields the pose measurement tier 2
actually needs.

    python3 scripts/board_visibility.py --config configs/coop2.yaml --bag <bag> \
        --agent mobile_1 --reference runs/phase0/reference_mobile_1.tum
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "rosbag_to_opv2v"))

from slambench import load_dataset, load_tum, se3          # noqa: E402

CAMERA_INFO = {"mobile_1": "/mobile_1/zed/left/camera_info",
               "mobile_2": "/mobile_2/color/camera_info"}


def project(traj, board_xyz, K, w, h, margin_px=0):
    """Board position -> (visible mask, range, pixel u, v) per pose.

    The trajectory's poses ARE the camera optical frame on both agents (see
    coop2.yaml reference.agents[*].frame), so `inv(pose)` takes a map point
    straight into the camera with no extrinsic in between.
    """
    p = np.asarray(board_xyz, dtype=np.float64)
    n = len(traj)
    uv = np.full((n, 2), np.nan)
    rng = np.full(n, np.nan)
    vis = np.zeros(n, dtype=bool)
    for i in range(n):
        T = se3.invert(traj.poses[i])
        q = T[:3, :3] @ p + T[:3, 3]           # board in optical frame
        rng[i] = float(np.linalg.norm(q))
        if q[2] <= 1e-6:                        # behind the camera
            continue
        u = K[0, 0] * q[0] / q[2] + K[0, 2]
        v = K[1, 1] * q[1] / q[2] + K[1, 2]
        uv[i] = (u, v)
        vis[i] = (margin_px <= u <= w - margin_px) and (margin_px <= v <= h - margin_px)
    return vis, rng, uv


def windows(stamps, mask, min_duration_s):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if stamps[j - 1] - stamps[i] >= min_duration_s:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--agent", required=True, choices=["mobile_1", "mobile_2"])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--anchor-frame", default=None,
                    help="anchor_frame_coop2.json — supplies each board's "
                         "ORIENTATION, which turns 'in frame' into 'facing us'")
    ap.add_argument("--min-bit-px", type=float, default=1.6,
                    help="minimum pixels per ArUco bit cell. The pipeline "
                         "detected `anchor` at 0.72 m (~1.9 px/bit) with 112 "
                         "views, so 1.6 is a permissive floor, not a guess.")
    ap.add_argument("--max-incidence-deg", type=float, default=60.0,
                    help="with --anchor-frame: reject views more oblique than "
                         "this to the board normal")
    ap.add_argument("--max-range-m", type=float, default=6.0)
    ap.add_argument("--margin-px", type=float, default=40.0,
                    help="keep the board this far from the image edge")
    ap.add_argument("--min-duration-s", type=float, default=1.0)
    ap.add_argument("--max-speed", type=float, default=0.25,
                    help="m/s; a board streaking past is visible but not a dwell")
    args = ap.parse_args()

    from ros2opv2v.bagreader import BagReader

    cfg = load_dataset(args.config)
    traj = load_tum(args.reference)
    reader = BagReader(str(Path(args.bag).expanduser()), stamp_source="header")

    K = w = h = None
    for _, _, msg in reader.iter_messages([CAMERA_INFO[args.agent]]):
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        w, h = int(msg.width), int(msg.height)
        break
    if K is None:
        print(f"error: no {CAMERA_INFO[args.agent]} in the bag", file=sys.stderr)
        return 1
    hfov = 2 * np.degrees(np.arctan(w / (2 * K[0, 0])))
    vfov = 2 * np.degrees(np.arctan(h / (2 * K[1, 1])))
    print(f"{args.agent}: {w}x{h}, fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"-> FoV {hfov:.1f} x {vfov:.1f} deg")
    print(f"frame: {cfg.agent_reference(args.agent).get('frame')}  "
          f"({len(traj)} poses, {traj.duration:.1f} s)\n")

    # board designs, from the mapping pipeline's config
    DESIGNS = {"anchor":    dict(sq=0.020, marker=0.015, bits=4),
               "anchor_b":  dict(sq=0.020, marker=0.015, bits=4),
               "rs_anchor": dict(sq=0.025, marker=0.018, bits=5)}
    normals = {}
    if args.anchor_frame:
        import json
        af = json.load(open(args.anchor_frame))
        for bname, b in af.get("boards", {}).items():
            q = np.asarray(b["qxyzw"], dtype=np.float64)
            R = se3.quat_to_rot(q / np.linalg.norm(q))
            normals[bname] = R[:, 2]     # board_axes "ros": +z out of the board
        print(f"board orientations loaded for {sorted(normals)}\n")

    v = np.zeros(len(traj))
    if len(traj) > 1:
        step = np.linalg.norm(np.diff(traj.positions, axis=0), axis=1)
        v[1:] = step / np.maximum(np.diff(traj.stamps), 1e-6)
        v[0] = v[1]
    t0 = traj.stamps[0]
    yaml_rows = []

    for a in cfg.raw["reference"]["anchors"]:
        vis, rng, uv = project(traj, a["position"], K, w, h, args.margin_px)
        d = DESIGNS.get(a["name"], DESIGNS["anchor"])
        bit_px = d["marker"] * K[0, 0] / np.maximum(rng, 1e-6) / (d["bits"] + 2)
        detectable = bit_px >= args.min_bit_px
        ok = vis & (rng <= args.max_range_m) & (v <= args.max_speed) & detectable
        if a["name"] in normals:
            los = np.asarray(a["position"], dtype=np.float64) - traj.positions
            los /= np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-9)
            # incidence between the board normal and the direction TO the camera
            cos_i = np.clip(-(los @ normals[a["name"]]), -1.0, 1.0)
            inc = np.degrees(np.arccos(np.abs(cos_i)))
            facing = (cos_i > 0) & (inc <= args.max_incidence_deg)
            ok &= facing
        wins = windows(traj.stamps, ok, args.min_duration_s)
        print(f"{a['name']} (survey sigma {a.get('uncertainty_m')} m, "
              f"{d['sq']*9*100:.0f}x{d['sq']*7*100:.0f} cm, "
              f"{d['marker']*1000:.0f} mm DICT_{d['bits']}X{d['bits']})")
        print(f"  in frame: {np.mean(vis):6.1%}   detectable (>={args.min_bit_px} "
              f"px/bit, i.e. <={d['marker']*K[0,0]/(args.min_bit_px*(d['bits']+2)):.2f} m): "
              f"{np.mean(vis & detectable):6.1%}"
              + (f"   facing: {np.mean(vis & facing):6.1%}" if a["name"] in normals else "")
              + f"   ALL: {np.mean(ok):6.1%}")
        if not wins:
            far = vis & (~detectable)
            print(f"  NO USABLE WINDOW for this agent." +
                  (f" The board IS in frame {np.mean(far):.1%} of the time but "
                   f"too far to detect (closest {rng[vis].min():.2f} m)."
                   if far.any() else "") + "\n")
            continue
        # rank by pose count among the detectable windows -- NOT by duration,
        # which would prefer a long far-away window over a short close one.
        best = max(wins, key=lambda ij: ij[1] - ij[0])
        for (i, j) in wins:
            mark = " <- best" if (i, j) == best else ""
            print(f"  t+{traj.stamps[i]-t0:7.2f} .. t+{traj.stamps[j-1]-t0:7.2f} s "
                  f"({traj.stamps[j-1]-traj.stamps[i]:5.2f} s, {j-i:3d} poses)  "
                  f"range {rng[i:j].mean():4.2f} m  "
                  f"{bit_px[i:j].mean():4.2f} px/bit  "
                  f"pixel ({np.nanmean(uv[i:j,0]):5.0f},{np.nanmean(uv[i:j,1]):5.0f})"
                  f"{mark}")
        i, j = best
        yaml_rows.append((a["name"], traj.stamps[i], traj.stamps[j-1],
                          rng[i:j].mean(), j - i))
        print()

    if yaml_rows:
        print("paste into configs/coop2.yaml reference.anchors[*].window "
              "(ABSOLUTE stamps, best window each):")
        for name, s, e, r, n in yaml_rows:
            print(f"  # {name}: {n} poses, mean range {r:.2f} m, "
                  f"t+{s-t0:.1f}..t+{e-t0:.1f} s into the run")
            print(f"  window: [{s:.3f}, {e:.3f}]")

    print("\nCANDIDATES, not proof. Geometry cannot see the board's FACING "
          "(coop2 surveys\nposition, not normal), nor occlusion or blur. To "
          "settle those definitively,\nrun the ChArUco detector over "
          f"{CAMERA_INFO[args.agent].replace('camera_info','image_rect_color' if args.agent=='mobile_1' else 'image_raw')} "
          "on these windows —\n../radar_camera_calibration/general_charuco.py "
          "already does the detection and\nPnP; a detected board is proof of "
          "visibility AND a direct pose measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
