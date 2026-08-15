#!/usr/bin/env python3
"""
Validate a pipeline_config.json for stage 01 BEFORE spending an hour on a bag.

Stage 01 reads its config lazily: `flatten` is not touched until after the
merge, and `detect.weights` not until after colorize. So a missing key does not
fail at startup -- it fails forty minutes in, having already done the expensive
work, and on a resume it fails again at the same place. This walks every key
the stage will eventually read and reports all of the problems at once.

Checks, in rough order of how much time each one saves:

  * required keys present, with the right types
  * dataset paths exist (bag, trajectory, calibration)
  * image_width/height MATCH the resolution the intrinsics were calibrated at
    -- stage 01 resizes images to (W, H) but does NOT rescale fx/fy/cx/cy, so a
    mismatch silently projects everything to the wrong pixels
  * voxel sizes are sane and mutually consistent
  * detect weights exist, ultralytics importable, class names resolvable
  * synthesize rules reference classes the asset library can actually serve

    python3 check_config.py [pipeline_config.json]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ERR, WARN = [], []


def err(m):
    ERR.append(m)


def warn(m):
    WARN.append(m)


# key -> type; nested blocks are dicts of the same shape
REQUIRED = {
    "lidar_min": float, "lidar_max": float, "time_tol": float,
    "scan_voxel": float, "final_voxel": float, "flush_every": int,
    "image_width": int, "image_height": int,
    "anchor_camera_start": bool, "output": str,
    "denoise": {"enable": bool, "nb": int, "std": float},
    "colorize": {"enable": bool, "img_stride": int, "max_range": float,
                 "voxel": float, "drop_gray": bool},
    "flatten": {"enable": bool, "min": int, "dist": float,
                "max_planes": int},
}


def typename(t):
    return {float: "number", int: "integer", bool: "true/false",
            str: "string"}[t]


def check_type(path, val, want):
    if want is float and isinstance(val, (int, float)) and not isinstance(val, bool):
        return True
    if want is int and isinstance(val, int) and not isinstance(val, bool):
        return True
    if want is bool and isinstance(val, bool):
        return True
    if want is str and isinstance(val, str):
        return True
    err(f"{path}: expected {typename(want)}, got {type(val).__name__} ({val!r})")
    return False


def walk(spec, cfg, prefix):
    for k, want in spec.items():
        path = f"{prefix}.{k}"
        if k not in cfg:
            err(f"{path}: MISSING (stage 01 reads this with cfg[\"{k}\"], "
                f"so it raises KeyError rather than defaulting)")
            continue
        if isinstance(want, dict):
            if not isinstance(cfg[k], dict):
                err(f"{path}: expected an object")
                continue
            walk(want, cfg[k], path)
        else:
            check_type(path, cfg[k], want)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "pipeline_config.json"
    if not os.path.exists(path):
        raise SystemExit(f"no such config: {path}")
    try:
        cfg = json.load(open(path))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}\n"
                         f"  (JSON has no comments -- a stray // or trailing "
                         f"comma will land here)")
    print(f"checking {path}")

    ds = cfg.get("dataset")
    if not isinstance(ds, dict):
        raise SystemExit("config has no 'dataset' block; nothing else can be "
                         "checked without it")
    for k in ("bag", "traj", "calib_json", "out_dir"):
        if k not in ds:
            err(f"dataset.{k}: MISSING")
        elif k != "out_dir" and not os.path.exists(ds[k]):
            err(f"dataset.{k}: does not exist -> {ds[k]}")

    if "01_build_map" not in cfg:
        raise SystemExit("config has no '01_build_map' block")
    s = cfg["01_build_map"]
    walk(REQUIRED, s, "01_build_map")

    # ---- geometry sanity --------------------------------------------------
    sv, fv = s.get("scan_voxel", 0), s.get("final_voxel", 0)
    if isinstance(sv, (int, float)) and isinstance(fv, (int, float)):
        if sv <= 0 and fv <= 0:
            err("01_build_map: scan_voxel and final_voxel are both <= 0. The "
                "merge accumulator is voxel-based and cannot be memory-bounded "
                "without one of them; stage 01 exits on this.")
        elif min(v for v in (sv, fv) if v > 0) < 0.005:
            warn(f"01_build_map: finest voxel is {min(v for v in (sv, fv) if v > 0)} m. "
                 "Below ~5 mm the accumulator holds a voxel per sensor noise "
                 "sample and memory stops tracking surface area.")
    if isinstance(s.get("lidar_min"), (int, float)) and \
            isinstance(s.get("lidar_max"), (int, float)) and \
            s["lidar_min"] >= s["lidar_max"]:
        err(f"01_build_map: lidar_min ({s['lidar_min']}) >= lidar_max "
            f"({s['lidar_max']}); every return is filtered out and the merge "
            f"produces an empty cloud")

    # ---- the intrinsics/resolution trap -----------------------------------
    calib = None
    if isinstance(ds.get("calib_json"), str) and os.path.exists(ds["calib_json"]):
        try:
            calib = json.load(open(ds["calib_json"]))
        except json.JSONDecodeError as e:
            err(f"dataset.calib_json is not valid JSON: {e}")
    if calib is not None:
        cam = calib.get("camera", {})
        intr = cam.get("intrinsics")
        if not (isinstance(intr, list) and len(intr) == 4):
            err("calibration.camera.intrinsics must be [fx, fy, cx, cy]")
        else:
            fx, fy, cx, cy = intr
            W, H = s.get("image_width"), s.get("image_height")
            if isinstance(W, int) and isinstance(H, int):
                # the principal point sits near the image centre for any sane
                # camera, so 2*cx is a good estimate of the width the
                # intrinsics belong to
                if abs(2 * cx - W) > 0.15 * W or abs(2 * cy - H) > 0.15 * H:
                    err(f"image_width/height ({W}x{H}) do not match the "
                        f"resolution these intrinsics were calibrated at "
                        f"(cx={cx:.1f}, cy={cy:.1f} imply about "
                        f"{2 * cx:.0f}x{2 * cy:.0f}).\n"
                        f"      Stage 01 resizes images to (W, H) but does NOT "
                        f"rescale fx/fy/cx/cy, so every projection lands on the "
                        f"wrong pixel. Either set W/H to the calibration "
                        f"resolution, or rescale the intrinsics by "
                        f"{W / max(2 * cx, 1):.4f}.")
        if not isinstance(calib.get("results", {}).get("T_lidar_camera"), list):
            err("calibration.results.T_lidar_camera must be "
                "[x, y, z, qx, qy, qz, qw]")

    # ---- detect ------------------------------------------------------------
    d = s.get("detect", {})
    if d.get("enable"):
        w = d.get("weights")
        if not w:
            err("detect.enable is true but detect.weights is not set")
        else:
            wp = w if os.path.isabs(w) else os.path.join(
                os.path.dirname(os.path.abspath(path)), w)
            if not os.path.exists(wp):
                err(f"detect.weights not found -> {wp}\n"
                    f"      (paths here resolve against the CONFIG directory, "
                    f"not out_dir; see assets/README.md to fetch weights)")
            elif "seg" not in os.path.basename(wp):
                warn(f"detect.weights is {os.path.basename(wp)} -- detect needs "
                     f"a SEGMENTATION checkpoint (-seg). A detection-only model "
                     f"returns no masks and every frame will be skipped.")
        try:
            import ultralytics                                  # noqa: F401
        except ImportError:
            err("detect.enable is true but ultralytics is not installed "
                "(pip install ultralytics)")
        vote = d.get("vote", {})
        if vote.get("min_frames", 3) > 1 and d.get("min_baseline", 0) > 0.5:
            warn(f"detect.min_baseline={d['min_baseline']} m with "
                 f"vote.min_frames={vote.get('min_frames', 3)}: the pose gate "
                 f"may not admit enough views for a small object to clear "
                 f"min_frames, and it will be dropped silently.")
        if d.get("structure") is not None:
            warn("detect.structure is no longer read -- stage 01 detects "
                 "objects only. Structure classification now lives under "
                 "synthesize.structure, its sole consumer.")
        if d.get("max_range", 8.0) > s.get("lidar_max", 50.0):
            warn("detect.max_range exceeds lidar_max; points that far away "
                 "were never merged into the map.")

    # ---- synthesize --------------------------------------------------------
    y = s.get("synthesize", {})
    if y.get("enable"):
        if not d.get("enable"):
            err("synthesize.enable is true but detect.enable is false; "
                "synthesize consumes detect's instances and never runs")
        root = y.get("assets", "assets")
        root = root if os.path.isabs(root) else os.path.join(
            os.path.dirname(os.path.abspath(path)), root)
        man = os.path.join(root, "manifest.json")
        if not os.path.exists(man):
            err(f"synthesize.assets has no manifest.json -> {man}")
        else:
            have = set(json.load(open(man)).get("assets", {}))
            for cls, action in (y.get("rules") or {}).items():
                acts = ([action] if isinstance(action, str)
                        else list(action.values()))
                if "replace" in acts and cls not in have:
                    warn(f"rules[{cls!r}] says replace but the asset library "
                         f"has no {cls!r}; those instances fall back to keep.")

    for m in ERR:
        print(f"  ERROR  {m}")
    for m in WARN:
        print(f"  warn   {m}")
    if ERR:
        print(f"\n{len(ERR)} error(s), {len(WARN)} warning(s) -- fix the errors "
              f"before running stage 01")
        return 1
    print(f"\nconfig OK ({len(WARN)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
