#!/usr/bin/env python
"""Self-test for datasets/splat_psnr.py: a scene of two cameras' images and
renders that are the images with noise of two known levels: the PSNR per
camera matches the noise, the rows and the csv are written.

    python scoop_pipeline/tests/test_splat_psnr.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
import splat_psnr as SP                                     # noqa: E402


def main():
    import cv2
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="splat_psnr_")
    try:
        scene, renders = os.path.join(tmp, "scene"), os.path.join(tmp, "renders")
        os.makedirs(os.path.join(scene, "images"))
        os.makedirs(renders)
        meta = {"frames": []}
        for n in range(8):
            cam = 1 if n < 5 else 2
            name = ("frame%06d.jpg" if cam == 1 else "c2_frame%06d.jpg") % n
            im = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(scene, "images", name), im, [cv2.IMWRITE_JPEG_QUALITY, 100])
            im = cv2.imread(os.path.join(scene, "images", name))
            sigma = 5.0 if cam == 1 else 20.0
            r = np.clip(im.astype(np.float32) + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(renders, f"frame{n:06d}.jpg"), r, [cv2.IMWRITE_JPEG_QUALITY, 100])
            meta["frames"].append({"name": name, "camera": cam, "c2w": np.eye(4).tolist()})
        json.dump(meta, open(os.path.join(renders, "cameras.json"), "w"))
        rows, summ = SP.compare(renders, scene, log=lambda *_: None)
        assert len(rows) == 8 and summ[1]["n"] == 5 and summ[2]["n"] == 3, summ
        assert 32 < summ[1]["psnr_mean"] < 36 and 21 < summ[2]["psnr_mean"] < 25, summ   # sigma 5 -> ~34 dB, 20 -> ~22
        assert summ["all"]["n"] == 8 and summ[1]["l1"] < summ[2]["l1"], summ
        r = subprocess.run([sys.executable, os.path.join(ROOT, "datasets", "splat_psnr.py"), renders, "--scene", scene],
                           capture_output=True, text=True)
        assert r.returncode == 0 and os.path.exists(os.path.join(renders, "psnr.csv")), r.stdout + r.stderr
        print("ok splat_psnr: camera 1 %.1f dB, camera 2 %.1f dB over %d views; csv written"
              % (summ[1]["psnr_mean"], summ[2]["psnr_mean"], summ["all"]["n"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
