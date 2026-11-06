import json

import numpy as np

from make_room_cameras import (find_intrinsics, floor_from_points, floor_plane, levelling, quat_to_R,
                               read_intrinsics)


def test_floor_is_the_lowest_large_layer():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 10, (20000, 2)), -0.52 + rng.normal(0, 0.01, 20000)]
    ceiling = np.c_[rng.uniform(0, 10, (30000, 2)), 2.6 + rng.normal(0, 0.01, 20000 + 10000)]
    clutter = np.c_[rng.uniform(0, 10, (300, 2)), rng.uniform(-3, -1, 300)]      # a few points below the floor
    z, share = floor_from_points(np.vstack([floor, ceiling, clutter]))
    assert abs(z + 0.52) < 0.005 and share > 0.3


def test_quaternion_and_intrinsics(tmp_path):
    R = quat_to_R([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])                  # 90 deg about z
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "cam_3.json").write_text(json.dumps({"camera_matrix": [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                                                           "dist_coeffs": [0.1, 0.2], "img_w": 640, "img_h": 480}))
    (tmp_path / "a" / "cam_3_2.json").write_text(json.dumps({"fx": 5, "fy": 5, "cx": 2, "cy": 3, "dist": [0.3],
                                                             "image_size": [1920, 1080]}))
    (tmp_path / "extrinsic_x__cam_3.yaml").write_text("R: [1]\n")
    p = find_intrinsics(tmp_path, 3)
    assert p.name == "cam_3_2.json"
    K, dist, w, h = read_intrinsics(tmp_path / "a" / "cam_3.json")
    assert K[0, 2] == 2 and dist.tolist() == [0.1, 0.2, 0, 0, 0] and (w, h) == (640, 480)
    K, dist, w, h = read_intrinsics(p)
    assert K[0, 0] == 5 and K[1, 2] == 3 and (w, h) == (1920, 1080)


def test_tilted_floor_is_levelled():
    rng = np.random.default_rng(1)
    s = np.tan(np.radians(1.0))                                   # the floor falls 1.7 cm per m along x
    x, y = rng.uniform(0, 11, 40000), rng.uniform(0, 6, 40000)
    floor = np.c_[x, y, -0.6 - s * x + rng.normal(0, 0.008, 40000)]
    ceiling = np.c_[x[:30000], y[:30000], 2.1 - s * x[:30000]]
    n, d, rms, share = floor_plane(np.vstack([floor, ceiling]))
    assert abs(np.degrees(np.arccos(n[2])) - 1.0) < 0.05 and rms < 0.012
    T = levelling(n, d)
    z = floor @ T[:3, :3].T[:, 2] + T[2, 3]
    assert abs(z[x < 1].mean()) < 0.005 and abs(z[x > 10].mean()) < 0.005
