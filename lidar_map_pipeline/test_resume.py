#!/usr/bin/env python3
"""
Resume-path test for stage 01: prove the expensive stages are actually skipped.

Colorize reads every image in the bag and projects the whole map per frame. It
is the most expensive thing in stage 01 to repeat, and until this test existed
there was nothing stopping a refactor from quietly re-running it -- the symptom
is not a wrong answer, just an hour of wasted wall-clock, which no assertion
about the output cloud would ever catch.

So the test asserts the SKIP, not the result: main() is run with a bag path
that does not exist. Any code path that opens the bag dies immediately, which
means a passing run is proof that merge, dynamic, denoise and colorize were all
bypassed. Everything else here (calibration, trajectory, config) is real.

    python3 test_resume.py
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load_stage01():
    spec = importlib.util.spec_from_file_location(
        "stage01_resume", os.path.join(HERE, "01_build_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp, cloud_name, n=5000, colored=True):
    """A dataset with everything EXCEPT a bag."""
    out = os.path.join(tmp, "out")
    os.makedirs(out, exist_ok=True)
    json.dump({"camera": {"intrinsics": [350.0, 350.0, 320.0, 180.0],
                          "distortion_coeffs": [0, 0, 0, 0, 0]},
               "results": {"T_lidar_camera": [0.0, 0.0, 0.0, 0, 0, 0, 1]},
               "meta": {}},
              open(os.path.join(tmp, "calibration.json"), "w"))
    with open(os.path.join(tmp, "traj.txt"), "w") as f:
        for i in range(10):
            f.write(f"{100.0 + i * 0.1:.3f} {i * 0.1:.3f} 0 0 0 0 0 1\n")

    rng = np.random.default_rng(5)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(rng.uniform(-3, 3, (n, 3)))
    if colored:
        pc.colors = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    o3d.io.write_point_cloud(os.path.join(out, cloud_name), pc)

    cfg = {
        "dataset": {
            # deliberately absent: touching it is the failure this test detects
            "bag": os.path.join(tmp, "NO_SUCH_BAG"),
            "traj": os.path.join(tmp, "traj.txt"),
            "calib_json": os.path.join(tmp, "calibration.json"),
            "out_dir": out,
        },
        "01_build_map": {
            "gpu": False,
            "lidar_min": 0.3, "lidar_max": 15.0, "time_tol": 0.04,
            "scan_voxel": 0.01, "final_voxel": 0.01, "flush_every": 200,
            "image_width": 640, "image_height": 360,
            "denoise": {"enable": True, "nb": 30, "std": 3.0},
            "colorize": {"enable": True, "img_stride": 1, "max_range": 10.0,
                         "voxel": 0.0, "drop_gray": False},
            "flatten": {"enable": False, "dist": 0.04, "min": 8000,
                        "max_planes": 40},
            "detect": {"enable": False},
            "anchor_camera_start": False,
            "output": "map_final.pcd",
        },
    }
    p = os.path.join(tmp, "cfg.json")
    json.dump(cfg, open(p, "w"), indent=2)
    return p, out, n


def run(mod, cfg_path):
    argv = sys.argv
    sys.argv = ["01_build_map.py", cfg_path]
    try:
        mod.main()
        return None
    except BaseException as e:                 # noqa: BLE001 -- reported, not swallowed
        return e
    finally:
        sys.argv = argv


def main():
    mod = load_stage01()
    print("stage 01 resume paths (bag deliberately missing)\n")

    # ---- resume from colored.pcd ------------------------------------------
    tmp = tempfile.mkdtemp(prefix="resume_colored_")
    try:
        cfg, out, n = fixture(tmp, "colored.pcd")
        print("[case] colored.pcd present, colorize enabled")
        e = run(mod, cfg)
        check("main() completed without opening the bag", e is None,
              "" if e is None else f"{type(e).__name__}: {e}")
        fp = os.path.join(out, "map_final.pcd")
        check("map_final.pcd written", os.path.exists(fp))
        if os.path.exists(fp):
            got = o3d.io.read_point_cloud(fp)
            check("point count preserved end to end", len(got.points) == n,
                  f"{len(got.points)} vs {n}")
            check("colours carried through", got.has_colors())
        # every .pcd this stage writes must have a .ply twin
        plyp = os.path.join(out, "map_final.ply")
        check("map_final.ply twin written", os.path.exists(plyp))
        if os.path.exists(plyp):
            tw = o3d.io.read_point_cloud(plyp)
            check("ply twin matches the pcd", len(tw.points) == n
                  and tw.has_colors(), f"{len(tw.points)} pts")
        # colored.pcd was written by the FIXTURE, not by this run -- the
        # backfill sweep must twin it anyway, since a resume never rewrites it
        check("pre-existing colored.pcd got a .ply twin by backfill",
              os.path.exists(os.path.join(out, "colored.ply")))
        # the tell-tale: colorize would have written these
        check("merge/denoise did not re-run",
              not os.path.exists(os.path.join(out, "merged.pcd"))
              and not os.path.exists(os.path.join(out, "denoised.pcd")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- resume from denoised.pcd with colorize OFF ------------------------
    tmp = tempfile.mkdtemp(prefix="resume_denoised_")
    try:
        cfg, out, n = fixture(tmp, "denoised.pcd", colored=False)
        c = json.load(open(cfg))
        c["01_build_map"]["colorize"]["enable"] = False
        json.dump(c, open(cfg, "w"))
        print("\n[case] denoised.pcd present, colorize disabled")
        e = run(mod, cfg)
        check("main() completed without opening the bag", e is None,
              "" if e is None else f"{type(e).__name__}: {e}")
        check("map_final.pcd written",
              os.path.exists(os.path.join(out, "map_final.pcd")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- a stale colored.pcd must NOT be used when colorize is off ---------
    tmp = tempfile.mkdtemp(prefix="resume_offbutpresent_")
    try:
        cfg, out, n = fixture(tmp, "colored.pcd")
        c = json.load(open(cfg))
        c["01_build_map"]["colorize"]["enable"] = False
        json.dump(c, open(cfg, "w"))
        print("\n[case] colored.pcd present but colorize disabled")
        e = run(mod, cfg)
        # nothing upstream exists, so it must fall through to merge and die on
        # the missing bag -- silently reusing a colorized cloud when colorize
        # was turned off would hand back a different cloud than asked for
        check("falls through to merge instead of reusing colored.pcd",
              e is not None,
              "reused it" if e is None else f"{type(e).__name__}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
