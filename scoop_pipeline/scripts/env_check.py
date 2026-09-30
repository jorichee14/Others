#!/usr/bin/env python3
"""Is this environment ready for the SCooP pipeline?

    conda activate scoop
    python scripts/env_check.py

Imports every package the pipeline uses and prints its version, checks the
pieces that fail late rather than at import (cv2.aruco, open3d's native
libraries, the Ouster SDK module layout, a working GPU), and warns when ROS
has leaked into this Python. Exits non-zero if a required package is missing.
"""
import importlib
import os
import platform
import sys
import warnings

warnings.simplefilter("ignore")

REQUIRED = ["numpy", "scipy", "yaml", "open3d", "cv2", "mcap", "mcap_ros2",
            "rosbags", "ouster.sdk"]
OPTIONAL = ["cupy", "pandas", "matplotlib", "jupyterlab"]


def version(name):
    mod = importlib.import_module(name)
    v = getattr(mod, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version as dist_version
            dist = {"yaml": "pyyaml", "cv2": "opencv-contrib-python-headless",
                    "mcap_ros2": "mcap-ros2-support", "ouster.sdk": "ouster-sdk",
                    "rosbags": "rosbags", "mcap": "mcap"}.get(name, name)
            v = dist_version(dist)
        except Exception:
            v = "?"
    return mod, v


def main():
    print(f"python {platform.python_version()}  ({sys.executable})")
    bad = 0
    mods = {}
    for name in REQUIRED + OPTIONAL:
        try:
            mods[name], v = version(name)
            print(f"  ok       {name:12s} {v}")
        except Exception as e:
            req = name in REQUIRED
            bad += req
            print(f"  {'MISSING' if req else 'absent '}  {name:12s} "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:70]}")

    print("\nchecks")
    cv2 = mods.get("cv2")
    if cv2 is not None:
        ok = hasattr(cv2, "aruco")
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  cv2.aruco (ChArUco boards)"
              + ("" if ok else ": install opencv-CONTRIB-python-headless, and only one cv2"))
    o3d = mods.get("open3d")
    if o3d is not None:
        import numpy as np
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.rand(1000, 3)))
        n = len(pc.voxel_down_sample(0.2).points)
        print(f"  ok    open3d voxel_down_sample ({n} voxels)")
        try:
            cuda = o3d.core.cuda.is_available()
        except Exception:
            cuda = False
        print(f"  {'ok  ' if cuda else 'info'}  open3d CUDA: {cuda} (CPU denoise otherwise)")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scoop import ouster
        print(f"  ok    ouster SDK layout: {ouster.sdk().name}")
    except Exception as e:
        bad += 1
        print(f"  FAIL  ouster SDK layout: {e}")
    cp = mods.get("cupy")
    if cp is not None:
        try:
            n = cp.cuda.runtime.getDeviceCount()
            x = cp.arange(12, dtype=cp.float32).reshape(4, 3)
            float((x * 2).sum() + cp.argsort(x[:, 0]).sum())      # compiles a kernel
            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            name = name.decode() if isinstance(name, bytes) else name
            print(f"  ok    GPU: {n} device(s), {name}, kernels compile")
        except Exception as e:
            print(f"  info  GPU not usable ({type(e).__name__}: {str(e)[:60]}); "
                  f"stage 01 will run on the CPU")

    ros = [p for p in sys.path if "/opt/ros" in p]
    if ros or os.environ.get("ROS_DISTRO"):
        print(f"\n  WARN  ROS is on this Python (ROS_DISTRO={os.environ.get('ROS_DISTRO')}, "
              f"{len(ros)} sys.path entries). Use a shell without `source /opt/ros/...`.")

    print("\n" + ("ready" if not bad else f"{bad} problem(s) above"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
