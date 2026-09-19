#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data: no bag, no container, no network.

Fabricates a reference trajectory, a room-shaped reference cloud, and two fake
"methods" — one good, one with a deliberate 1.5% scale error — then runs the
real eval_run.py and aggregate.py over them. It proves the plumbing before the
first real run, and it is the thing to re-run after touching the evaluator.

    python3 scripts/smoke_e2e.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench import Trajectory, extrinsic_matrix, save_ply, save_tum, se3  # noqa: E402


def room_cloud(rng, n=120_000):
    """Walls, floor and a few boxes over an 8.5 x 14.3 m room."""
    x0, x1, y0, y1, z0, z1 = -0.5, 8.5, -14.3, 0.5, 0.0, 2.6
    pts = [
        np.column_stack([rng.uniform(x0, x1, n // 3), rng.uniform(y0, y1, n // 3),
                         np.zeros(n // 3)]),                               # floor
        np.column_stack([np.full(n // 6, x0), rng.uniform(y0, y1, n // 6),
                         rng.uniform(z0, z1, n // 6)]),                    # wall
        np.column_stack([np.full(n // 6, x1), rng.uniform(y0, y1, n // 6),
                         rng.uniform(z0, z1, n // 6)]),
        np.column_stack([rng.uniform(x0, x1, n // 6), np.full(n // 6, y0),
                         rng.uniform(z0, z1, n // 6)]),
        np.column_stack([rng.uniform(x0, x1, n // 6), np.full(n // 6, y1),
                         rng.uniform(z0, z1, n // 6)]),
    ]
    return np.concatenate(pts)


def lap(n=1560, rate=10.0, t0=1000.0):
    """A 156 s loop around the room, heading tangent — a pushcart route."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy, rx, ry = 4.0, -7.0, 3.0, 5.5
    poses = np.zeros((n, 4, 4))
    for i, a in enumerate(th):
        x, y = cx + rx * np.cos(a), cy + ry * np.sin(a)
        dx, dy = -rx * np.sin(a), ry * np.cos(a)
        poses[i] = se3.pose_from_rpy(x, y, 0.19, 0.0, 0.0,
                                     np.degrees(np.arctan2(dy, dx)))
    return Trajectory(t0 + np.arange(n) / rate, poses, "reference",
                      frame="zed_left_camera_optical_frame", world="map")


def main() -> int:
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp(prefix="slambench-smoke-"))
    runs = tmp / "runs" / "smoke"
    ref = lap()
    save_tum(ref, runs / "reference" / "mobile_1.tum")

    ref_cloud = room_cloud(rng)
    save_ply(ref_cloud, tmp / "map.ply")

    # The extrinsic a real method's output carries: the Ouster's offset from the
    # ZED optical frame the reference describes.
    E = extrinsic_matrix({"x": 0.067064, "y": -0.092192, "z": -0.074147,
                          "roll": 2.2959, "pitch": 89.5267, "yaw": -87.6614})

    # method A: good. 8 mm of noise, in its own arbitrary world frame.
    origin = se3.pose_from_rpy(-2.0, 3.0, 0.0, 0, 0, 25.0)
    good = ref.transform_right(E).transform_left(se3.invert(origin))
    good.poses[:, :3, 3] += rng.normal(scale=0.008, size=(len(good), 3))
    save_tum(good, runs / "method_good" / "mobile_1.ouster" / "trajectory.tum")
    save_ply(ref_cloud[::4] + rng.normal(scale=0.01, size=(len(ref_cloud[::4]), 3)),
             runs / "method_good" / "mobile_1.ouster" / "map.ply")

    # method B: a 1.5% scale error, which se3-aligned ATE mostly launders and
    # RPE does not. This is the row that shows why both are reported.
    bad = ref.transform_right(E).transform_left(se3.invert(origin))
    bad.poses[:, :3, 3] *= 1.015
    save_tum(bad, runs / "method_scaled" / "mobile_1.ouster" / "trajectory.tum")

    # A dataset config with the map tier switched on, and `raw` map frame so the
    # smoke run does not depend on a trajectory alignment being sane.
    cfg = yaml.safe_load((ROOT / "configs" / "coop2.yaml").read_text())
    cfg["dataset"]["name"] = "smoke"
    cfg["reference"]["map_cloud"] = str(tmp / "map.ply")
    cfg["eval"]["volume"] = {"min": [-1.0, -15.0, -0.2], "max": [9.0, 1.0, 3.0]}
    cfg["eval"]["resolution_m"] = 0.05
    cfg["eval"]["map_frame"] = "raw"
    # Tier 2 needs the board OBSERVED, not merely approached: a window alone
    # measures the standoff (see absolute_check). Synthesise the detections the
    # ChArUco+PnP step would produce -- the board's position in the camera frame
    # at each pose -- so the smoke test exercises the path the real runs use.
    board = ref.positions[10]
    cfg["reference"]["anchors"][0]["window"] = [float(ref.stamps[0]), float(ref.stamps[20])]
    cfg["reference"]["anchors"][0]["position"] = board.tolist()
    cfg["reference"]["anchors"][0]["observations"] = [
        {"stamp": float(ref.stamps[k]),
         "p_board_cam": (se3.invert(ref.poses[k])[:3, :3] @ board
                         + se3.invert(ref.poses[k])[:3, 3]).tolist()}
        for k in range(21)]
    (tmp / "smoke.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    for name in ("method_good", "method_scaled"):
        m = yaml.safe_load((ROOT / "configs" / "methods" / "kiss_icp.yaml").read_text())
        m["name"] = name
        m["outputs"] = ["trajectory", "map"] if name == "method_good" else ["trajectory"]
        (tmp / f"{name}.yaml").write_text(yaml.safe_dump(m, sort_keys=False))

    rc = 0
    for name in ("method_good", "method_scaled"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_run.py"),
                            "--config", str(tmp / "smoke.yaml"),
                            "--method", str(tmp / f"{name}.yaml"),
                            "--stream", "mobile_1.ouster",
                            "--run", str(runs / name / "mobile_1.ouster"),
                            "--reference", str(runs / "reference" / "mobile_1.tum")])
        rc |= r.returncode
    rc |= subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate.py"),
                          "--runs", str(runs), "--config", str(tmp / "smoke.yaml"),
                          "--out", str(tmp / "report.md")]).returncode
    print("\n" + (tmp / "report.md").read_text())

    # The assertions the smoke test exists for.
    import json
    g = json.loads((runs / "method_good" / "mobile_1.ouster" / "metrics.json").read_text())
    b = json.loads((runs / "method_scaled" / "mobile_1.ouster" / "metrics.json").read_text())
    checks = [
        ("good ATE is small", g["ate"]["ate_trans_m"]["rmse"] < 0.02),
        ("good map scores", g.get("map_status") == "ok"),
        # RPE at 1 m is dominated by per-pose noise, so it does NOT separate a
        # small scale error from a noisy-but-unbiased one. At 5 m the scale error
        # has grown linearly while the noise has not. Both assertions are the
        # lesson: pick the RPE delta against the error you are trying to see.
        ("RPE at 1 m does not separate scale error from noise",
         b["rpe"][0]["rpe_trans_m"]["rmse"] < 2 * g["rpe"][0]["rpe_trans_m"]["rmse"]),
        ("RPE at 5 m catches the scale error",
         b["rpe"][1]["rpe_trans_m"]["rmse"] > 3 * g["rpe"][1]["rpe_trans_m"]["rmse"]),
        ("se3 reports the scale it did not apply",
         abs(b["ate"]["alignment"]["scale_observed"] - 1 / 1.015) < 0.01),
        ("anchor tier ran", isinstance(g.get("absolute_check"), list)
         and g["absolute_check"][0]["residual_m"] is not None),
    ]
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        rc |= 0 if ok else 1
    print(f"\nartifacts: {tmp}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
