#!/usr/bin/env python3
"""
Check that one environment can run the whole pipeline, and say precisely what
is wrong when it cannot.

Run it after setup_env.sh, and again any time a pip install has touched the
env -- an ABI break shows up here as a clean report instead of as a traceback
four minutes into a merge.

    python verify_env.py
"""
import sys
import importlib

REQUIRED = [
    ("numpy", "arrays", True),
    ("scipy", "KD-trees for colour propagation and meshing", True),
    ("open3d", "point clouds, voxel downsample, RANSAC, Poisson", True),
    ("cv2", "image decode, masks, PNG cache", True),
    ("rosbags", "reading the ROS2 bag", True),
]
OPTIONAL = [
    ("cupy", "GPU backend for stage 01 (falls back to CPU)"),
    ("ultralytics", "YOLO object detection, stage 01 [6]"),
    ("torch", "backend for ultralytics"),
    ("yaml", "inventory output (a built-in writer is used otherwise)"),
]


def probe(name):
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "?"), None
    except Exception as e:                 # ImportError, but ABI breaks raise
        return False, None, f"{type(e).__name__}: {e}"    # AttributeError too


def main():
    print(f"python {sys.version.split()[0]}\n{sys.executable}\n")
    bad = []

    print("required:")
    for name, why, _ in REQUIRED:
        ok, ver, err = probe(name)
        print(f"  {'ok ' if ok else 'FAIL'}  {name:<12} {ver or ''}"
              f"{'' if ok else '  <- ' + err}")
        if not ok:
            bad.append((name, why, err))

    print("\noptional:")
    for name, why in OPTIONAL:
        ok, ver, err = probe(name)
        print(f"  {'ok ' if ok else '-- '}  {name:<12} {ver or why}")

    # The ABI trap: numpy 2 with modules compiled against numpy 1 does not fail
    # at install time, only on import, and the message points at scipy rather
    # than at the numpy upgrade that actually caused it.
    print()
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        o3d_ok, o3d_ver, _ = probe("open3d")
        if not o3d_ok:
            print(f"numpy/open3d ABI: cannot check, open3d did not import "
                  f"(numpy is {np.__version__})")
        elif major >= 2 and o3d_ver and o3d_ver < "0.19":
            print(f"WARNING numpy {np.__version__} with open3d {o3d_ver}: this "
                  f"wheel is built against the numpy 1.x ABI and will crash.\n"
                  f"        pip install 'numpy<2' -c constraints.txt")
            bad.append(("numpy/open3d ABI", "", ""))
        else:
            print(f"numpy/open3d ABI: consistent "
                  f"(numpy {np.__version__}, open3d {o3d_ver})")
    except Exception:
        pass

    ok, _, _ = probe("cupy")
    if ok:
        try:
            import cupy as cp
            n = cp.cuda.runtime.getDeviceCount()
            props = cp.cuda.runtime.getDeviceProperties(0)
            nm = props["name"]
            free, total = cp.cuda.runtime.memGetInfo()
            float(cp.zeros(4).sum())       # actually execute a kernel
            print(f"GPU: {nm.decode() if isinstance(nm, bytes) else nm}, "
                  f"{n} device(s), {free / 2**30:.1f}/{total / 2**30:.1f} GiB "
                  f"free, kernel launch ok")
        except Exception as e:
            print(f"GPU: cupy imports but cannot run ({type(e).__name__}: {e})"
                  f"\n     stage 01 will fall back to the CPU")
    else:
        print("GPU: cupy absent -> stage 01 runs on the CPU")

    ok, _, _ = probe("torch")
    if ok:
        try:
            import torch
            print(f"torch CUDA: {'available' if torch.cuda.is_available() else 'NOT available (YOLO on CPU)'}")
        except Exception as e:
            print(f"torch CUDA: check failed ({type(e).__name__})")

    # open3d's PyPI wheel is CPU-only; the CUDA tensor API needs a source build
    o3d_ok, _, _ = probe("open3d")
    if o3d_ok:
        cpu_only = "no (PyPI wheel is CPU-only; GPU denoise falls back)"
        try:
            import open3d as o3d
            has = o3d.core.cuda.is_available()
            print(f"open3d CUDA: {'yes' if has else cpu_only}")
        except Exception:
            print(f"open3d CUDA: {cpu_only}")

    print()
    if bad:
        print(f"NOT READY: {len(bad)} problem(s) above")
        for n, why, _ in bad:
            if why:
                print(f"  {n} is needed for {why}")
        return 1
    print("READY: this environment can run the whole pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
