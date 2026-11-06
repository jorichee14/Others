import json

import numpy as np

from make_room_cameras import find_intrinsics, floor_from_points, quat_to_R, read_intrinsics


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
    (tmp_path / "a" / "cam_3_2.json").write_text(json.dumps({"K": [[5, 0, 2], [0, 5, 3], [0, 0, 1]], "dist": [0.3]}))
    (tmp_path / "extrinsic_x__cam_3.yaml").write_text("R: [1]\n")
    p = find_intrinsics(tmp_path, 3)
    assert p.name == "cam_3_2.json"
    K, dist, w, h = read_intrinsics(tmp_path / "a" / "cam_3.json")
    assert K[0, 2] == 2 and dist.tolist() == [0.1, 0.2, 0, 0, 0] and (w, h) == (640, 480)
