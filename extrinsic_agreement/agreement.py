#!/usr/bin/env python3
"""Quantify how well the mirc_dataset extrinsics agree, using a shared target.

Feed it per-frame observations of one static target (the chair) localized in each
sensor's own frame. It runs two families of test:

  1. RIGID-RIG test  (MP#1: zed / lidar / radar1 / radar2)
     Compares the target across rigidly-mounted sensors *in a shared sensor
     frame*, using ONLY the rig extrinsics. No trajectory, immune to pose drift.
     Isolates a single extrinsic. For the LiDAR<->ZED extrinsic it tries both
     directional interpretations of the (contradictory) calibration sheet.

  2. MAP-CONSENSUS test  (cross-platform)
     Pushes each sensor's target observation into the map frame through its full
     chain (pose x extrinsic), takes a robust consensus, and reports each
     sensor's signed deviation (systematic bias = extrinsic error) and scatter
     (localization noise). Reprojects the consensus into each camera for a
     pixel-space cross-check (this is where the fixed Arducam is exercised).

See observations.example.yaml for the input format.  Run the self-test with
``python3 test_agreement.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import yaml

import dataset as ds
from se3 import se3, inv, apply, chain


# --------------------------------------------------------------------------- #
# Input model
# --------------------------------------------------------------------------- #
@dataclass
class SensorObs:
    point: np.ndarray | None = None      # target in this sensor's frame (3,)
    pixel: np.ndarray | None = None      # detected target pixel (2,), for cameras
    pose_map: np.ndarray | None = None   # T_map_sensor at this frame's time (4x4)
    extra_points: dict | None = None     # name -> point (for PnP-style checks)


@dataclass
class Frame:
    fid: str
    t: float | None
    sensors: dict  # sensor_name -> SensorObs


def _mat_from_pose(node) -> np.ndarray:
    return se3(node["trans"], node["quat"])


def load_frames(path: str) -> list[Frame]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    frames = []
    for i, fr in enumerate(doc.get("frames", [])):
        sensors = {}
        for name, s in (fr.get("sensors") or {}).items():
            sensors[name] = SensorObs(
                point=np.asarray(s["point"], float) if s.get("point") is not None else None,
                pixel=np.asarray(s["pixel"], float) if s.get("pixel") is not None else None,
                pose_map=_mat_from_pose(s["pose_map"]) if s.get("pose_map") else None,
                extra_points={k: np.asarray(v, float) for k, v in (s.get("extra_points") or {}).items()},
            )
        frames.append(Frame(fid=str(fr.get("id", i)), t=fr.get("t"), sensors=sensors))
    return frames


# --------------------------------------------------------------------------- #
# 1. Rigid-rig test
# --------------------------------------------------------------------------- #
# All MP#1 extrinsics expressed as T_zed_<sensor> (map a sensor point into ZED).
def _rig_transforms(lidar_interp: str):
    return {
        "lidar": ds.T_zed_lidar(lidar_interp),
        "radar1": ds.T_zed_radar1,
        "radar2": ds.T_zed_radar2,
    }


def rigid_rig(frames: list[Frame], lidar_interp: str = "arrow") -> dict:
    """Compare each MP#1 sensor's target against the ZED, in the ZED frame.

    Residual = p_zed - T_zed_sensor @ p_sensor. Pure extrinsic error + noise.
    For 'lidar', if lidar_interp == 'auto', both sheet interpretations are run
    and the one with the smaller RMS is reported (with the loser for contrast).
    """
    out = {}
    interps = ["arrow", "label"] if lidar_interp == "auto" else [lidar_interp]

    for sensor in ("lidar", "radar1", "radar2"):
        best = None
        for interp in (interps if sensor == "lidar" else ["arrow"]):
            T = _rig_transforms(interp)[sensor]
            res = []
            for fr in frames:
                z = fr.sensors.get("zed")
                o = fr.sensors.get(sensor)
                if z is None or o is None or z.point is None or o.point is None:
                    continue
                res.append(z.point - apply(T, o.point))
            if not res:
                continue
            res = np.array(res)
            rec = _summarize_vec(res)
            rec["interp"] = interp if sensor == "lidar" else None
            rec["n"] = len(res)
            if best is None or rec["rms"] < best["rms"]:
                if best is not None:
                    rec["worse_alt"] = {"interp": best["interp"], "rms": best["rms"]}
                best = rec
            elif sensor == "lidar":
                best.setdefault("worse_alt", {"interp": interp, "rms": rec["rms"]})
        if best is not None:
            out[sensor] = best
    return out


# --------------------------------------------------------------------------- #
# 2. Map-consensus test
# --------------------------------------------------------------------------- #
def target_in_map(frame: Frame, sensor: str, lidar_interp: str = "arrow") -> np.ndarray | None:
    """Lift a sensor's target observation into the map frame, or None."""
    s = frame.sensors.get(sensor)
    if s is None:
        return None

    if sensor == "zed":
        if s.point is None or s.pose_map is None:
            return None
        return apply(s.pose_map, s.point)               # T_map_zed @ p_zed

    if sensor == "lidar":
        z = frame.sensors.get("zed")
        if s.point is None or z is None or z.pose_map is None:
            return None
        T = chain(z.pose_map, ds.T_zed_lidar(lidar_interp))   # T_map_zed @ T_zed_lidar
        return apply(T, s.point)

    if sensor in ("radar1", "radar2"):
        z = frame.sensors.get("zed")
        if s.point is None or z is None or z.pose_map is None:
            return None
        T = chain(z.pose_map, getattr(ds, f"T_zed_{sensor}"))
        return apply(T, s.point)

    if sensor == "realsense":
        if s.point is None or s.pose_map is None:
            return None
        return apply(s.pose_map, s.point)               # T_map_rs @ p_rs

    return None


DEPTH_SENSORS = ("zed", "lidar", "realsense", "radar1", "radar2")


def map_consensus(frames: list[Frame], consensus_sensors=("zed", "lidar", "realsense"),
                  lidar_interp: str = "arrow") -> dict:
    """Per-sensor deviation from a robust per-frame consensus in the map frame."""
    per_sensor_res = {s: [] for s in DEPTH_SENSORS}
    per_frame = []

    for fr in frames:
        pts = {s: target_in_map(fr, s, lidar_interp) for s in DEPTH_SENSORS}
        pts = {s: p for s, p in pts.items() if p is not None}
        if not pts:
            continue
        used = [pts[s] for s in consensus_sensors if s in pts]
        if len(used) < 1:
            continue
        cons = np.median(np.array(used), axis=0)        # robust per-axis
        row = {"fid": fr.fid, "consensus": cons, "range_m": {}}
        for s, p in pts.items():
            per_sensor_res[s].append(p - cons)
            row["range_m"][s] = float(np.linalg.norm(p))
        per_frame.append(row)

    summary = {}
    for s, res in per_sensor_res.items():
        if res:
            summary[s] = _summarize_vec(np.array(res))
            summary[s]["n"] = len(res)
    return {"per_sensor": summary, "per_frame": per_frame,
            "consensus_sensors": list(consensus_sensors)}


# --------------------------------------------------------------------------- #
# Reprojection cross-check (cameras). Uses each frame's own consensus point.
# --------------------------------------------------------------------------- #
def reprojection(frames: list[Frame], consensus_sensors=("zed", "lidar", "realsense"),
                 lidar_interp: str = "arrow") -> dict:
    cam_of = {"zed": ds.ZED_LEFT, "realsense": ds.REALSENSE, "arducam": ds.ARDUCAM}
    errs = {c: [] for c in cam_of}

    for fr in frames:
        used = []
        for s in consensus_sensors:
            p = target_in_map(fr, s, lidar_interp)
            if p is not None:
                used.append(p)
        if not used:
            continue
        cons_map = np.median(np.array(used), axis=0)

        for cam_name, cam in cam_of.items():
            s = fr.sensors.get(cam_name)
            if s is None or s.pixel is None:
                continue
            # pose of this camera in map:
            if cam_name == "arducam":
                T_map_cam = ds.T_map_arducam
            elif cam_name == "zed":
                T_map_cam = s.pose_map
            elif cam_name == "realsense":
                T_map_cam = s.pose_map
            if T_map_cam is None:
                continue
            p_cam = apply(inv(T_map_cam), cons_map)
            uv = cam.project(p_cam)[0]
            if np.any(np.isnan(uv)):
                continue
            errs[cam_name].append(uv - s.pixel)

    out = {}
    for c, e in errs.items():
        if e:
            e = np.array(e)
            out[c] = {"n": len(e),
                      "mean_px": e.mean(axis=0).tolist(),
                      "rms_px": float(np.sqrt((e ** 2).sum(axis=1).mean())),
                      "u_bias": float(e[:, 0].mean()),
                      "v_bias": float(e[:, 1].mean())}
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _summarize_vec(res: np.ndarray) -> dict:
    """res: (N,3) residual vectors. Returns bias/scatter/rms breakdown (mm)."""
    res_mm = res * 1000.0
    norms = np.linalg.norm(res_mm, axis=1)
    return {
        "bias_mm": res_mm.mean(axis=0).tolist(),         # systematic -> extrinsic error
        "scatter_mm": res_mm.std(axis=0).tolist(),        # random -> localization noise
        "rms": float(np.sqrt((norms ** 2).mean())),
        "median_norm_mm": float(np.median(norms)),
        "max_norm_mm": float(norms.max()),
    }


def _fmt3(v):
    return "[" + ", ".join(f"{x:+7.1f}" for x in v) + "]"


def print_report(frames, lidar_interp):
    print(f"\nLoaded {len(frames)} frame(s).\n")

    print("=" * 70)
    print("1. RIGID-RIG TEST  (MP#1, pose-free, isolates one extrinsic)")
    print("=" * 70)
    rr = rigid_rig(frames, lidar_interp="auto" if lidar_interp == "arrow" else lidar_interp)
    if not rr:
        print("  (no MP#1 sensor pairs with the ZED found)")
    for sensor, r in rr.items():
        tag = f" [interp={r['interp']}]" if r.get("interp") else ""
        print(f"\n  ZED <-> {sensor}{tag}   n={r['n']}")
        print(f"    RMS            : {r['rms']:7.1f} mm   (target in the ZED frame)")
        print(f"    signed bias    : {_fmt3(r['bias_mm'])} mm   <- extrinsic error")
        print(f"    scatter (1sigma): {_fmt3(r['scatter_mm'])} mm   <- localization noise")
        print(f"    median | max   : {r['median_norm_mm']:6.1f} | {r['max_norm_mm']:6.1f} mm")
        if r.get("worse_alt"):
            a = r["worse_alt"]
            print(f"    other interp   : '{a['interp']}' gives RMS {a['rms']:.1f} mm "
                  f"({a['rms'] / max(r['rms'], 1e-9):.1f}x worse) -> the sheet direction is settled")

    # map-consensus / reprojection need a concrete LiDAR direction; if the user
    # asked for 'auto', adopt whichever direction the rigid-rig test preferred.
    resolved = lidar_interp
    if lidar_interp == "auto":
        resolved = rr.get("lidar", {}).get("interp") or "arrow"
        print(f"\n  (map-frame stages use LiDAR interp='{resolved}', from the rigid-rig test)")

    print("\n" + "=" * 70)
    print("2. MAP-CONSENSUS TEST  (cross-platform, full chain)")
    print("=" * 70)
    mc = map_consensus(frames, lidar_interp=resolved)
    print(f"  consensus from: {', '.join(mc['consensus_sensors'])}  (per-axis median)\n")
    for s, r in mc["per_sensor"].items():
        print(f"  {s:10s} n={r['n']:2d}  RMS {r['rms']:7.1f} mm   "
              f"bias {_fmt3(r['bias_mm'])}  scatter {_fmt3(r['scatter_mm'])} mm")
    if not mc["per_sensor"]:
        print("  (need per-frame pose_map for the moving platforms)")

    print("\n" + "=" * 70)
    print("3. REPROJECTION CROSS-CHECK  (consensus point -> each camera)")
    print("=" * 70)
    rp = reprojection(frames, lidar_interp=resolved)
    if not rp:
        print("  (no camera pixels provided)")
    for c, r in rp.items():
        note = ""
        if c == "arducam":
            note = "   <- vertical (v) bias is the symptom of the suspected z error"
        print(f"  {c:10s} n={r['n']:2d}  RMS {r['rms_px']:6.1f} px   "
              f"u_bias {r['u_bias']:+6.1f}  v_bias {r['v_bias']:+6.1f} px{note}")

    print("\nReading the numbers:")
    print("  * a large SIGNED bias  = a real extrinsic error (a rotation/translation offset)")
    print("  * large SCATTER, ~0 bias = target-localization noise, extrinsic is fine")
    print("  * rigid-rig RMS is the cleanest single number per rig extrinsic")
    print("  * map-consensus folds in pose accuracy + the map anchors too")
    print("  * Arducam is RGB-only + sees one static point: reprojection gives its")
    print("    ANGULAR error only. To pin the metric z, add >=3 non-collinear static")
    print("    landmarks (extra_points) or align it to the aggregated cloud (see README).\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("observations", help="YAML file of per-frame target observations")
    ap.add_argument("--lidar-interp", choices=["arrow", "label", "auto"], default="arrow",
                    help="how to read the contradictory LiDAR<->ZED sheet entry "
                         "(default: arrow; rigid-rig always tries both)")
    args = ap.parse_args(argv)

    frames = load_frames(args.observations)
    if not frames:
        print("No frames found in", args.observations, file=sys.stderr)
        return 1
    print_report(frames, args.lidar_interp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
