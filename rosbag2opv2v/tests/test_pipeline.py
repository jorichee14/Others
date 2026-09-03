"""End-to-end tests: synthetic bag -> OPV2V export -> OpenCOOD-style reading.

Run with:  python -m unittest discover -s tests   (from the package root)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from rosbag2opv2v import transforms as tf                      # noqa: E402
from rosbag2opv2v.config import Config                          # noqa: E402
from rosbag2opv2v.convert import Converter                      # noqa: E402
from rosbag2opv2v.pcd_io import read_pcd, write_pcd             # noqa: E402
from rosbag2opv2v.verify import verify, load_points             # noqa: E402


def opencood_x_to_world(pose):
    """Verbatim copy of ``opencood.utils.transformation_utils.x_to_world``."""
    x, y, z, roll, yaw, pitch = pose[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def opencood_x1_to_x2(x1, x2):
    return np.linalg.inv(opencood_x_to_world(x2)) @ opencood_x_to_world(x1)


class TestPoseEncoding(unittest.TestCase):
    def test_round_trip_matches_opencood(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            matrix = tf.make_matrix(rng.normal(size=3) * 4,
                                    tf.quat_to_matrix(*quat))
            pose = tf.matrix_to_opv2v_pose(matrix)
            np.testing.assert_allclose(opencood_x_to_world(pose), matrix,
                                       atol=1e-9)

    def test_gimbal_lock_is_stable(self):
        matrix = tf.make_matrix([1.0, 2.0, 3.0],
                                tf.rpy_deg_to_matrix(0.0, 90.0, 30.0))
        pose = tf.matrix_to_opv2v_pose(matrix)
        np.testing.assert_allclose(opencood_x_to_world(pose), matrix, atol=1e-6)


class TestPcdIO(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(3)
        xyz = (rng.normal(size=(200, 3)) * 3).astype(np.float32)
        intensity = rng.random(200).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for binary in (True, False):
                path = os.path.join(tmp, "cloud_%s.pcd" % binary)
                write_pcd(path, xyz, intensity, binary=binary)
                back = read_pcd(path)
                self.assertEqual(back.shape, (200, 4))
                np.testing.assert_allclose(back[:, :3], xyz, atol=1e-4)
                # intensity survives as 8-bit, exactly like real OPV2V data
                np.testing.assert_allclose(back[:, 3], intensity, atol=2e-3)

    def test_open3d_reads_what_we_write(self):
        try:
            import open3d as o3d
        except ImportError:
            self.skipTest("open3d is not installed")
        rng = np.random.default_rng(4)
        xyz = (rng.normal(size=(50, 3))).astype(np.float32)
        intensity = rng.random(50).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cloud.pcd")
            write_pcd(path, xyz, intensity)
            pcd = o3d.io.read_point_cloud(path)
            np.testing.assert_allclose(np.asarray(pcd.points), xyz, atol=1e-6)
            np.testing.assert_allclose(np.asarray(pcd.colors)[:, 0], intensity,
                                       atol=2e-3)


class TestEndToEnd(unittest.TestCase):
    """Convert a synthetic bag whose true geometry we know, then check it."""

    seconds = 6.0

    @classmethod
    def setUpClass(cls):
        import make_synthetic_bag as gen

        cls.gen = gen
        cls.tmp = tempfile.mkdtemp(prefix="rosbag2opv2v-test-")
        cls.bag = os.path.join(cls.tmp, "synthetic.mcap")
        gen.main.__globals__["sys"].argv = [
            "make_synthetic_bag", "--out", cls.bag,
            "--seconds", str(cls.seconds)]
        gen.main()

        cfg = Config.load(os.path.join(ROOT, "configs", "mirc_coop2.yaml"))
        cfg.scenario_seconds = None
        cfg.min_frames_per_scenario = 5
        cfg.splits = None
        cfg.split = "train"
        cls.out = os.path.join(cls.tmp, "opv2v")
        Converter(cfg, cls.bag, cls.out, verbose=False).convert()
        cls.scenario = os.path.join(cls.out, "train", "mirc_coop2_000")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load_yaml(self, cav, stamp):
        import yaml
        with open(os.path.join(self.scenario, cav, stamp + ".yaml")) as handle:
            return yaml.safe_load(handle)

    def _stamps(self, cav="1"):
        return sorted(f[:-5] for f in os.listdir(os.path.join(self.scenario, cav))
                      if f.endswith(".yaml"))

    def test_structure_matches_opencood_expectations(self):
        cavs = sorted(d for d in os.listdir(self.scenario)
                      if os.path.isdir(os.path.join(self.scenario, d)))
        self.assertEqual(cavs, ["-1", "1", "2"])
        for cav in cavs:
            int(cav)                     # OpenCOOD calls int() on the folder
        reference = self._stamps("1")
        self.assertGreater(len(reference), 20)
        for cav in cavs:
            self.assertEqual(self._stamps(cav), reference)
            for stamp in reference:
                for suffix in (".pcd", ".yaml"):
                    self.assertTrue(os.path.isfile(os.path.join(
                        self.scenario, cav, stamp + suffix)))

    def test_verifier_passes(self):
        result = verify(self.out, sample=10, verbose=False)
        self.assertTrue(result["ok"], result["errors"])
        self.assertGreater(result["mean_cross_agent_voxel_overlap"], 0.4)
        self.assertGreater(result["gt_boxes_with_ego_points"], 0.9)

    def test_ground_truth_boxes_match_the_simulated_truth(self):
        """The box OpenCOOD reconstructs must sit where the robot really was."""
        gen = self.gen
        static = {child: tf.make_matrix(xyz, tf.rpy_deg_to_matrix(*rpy))
                  for _, child, xyz, rpy in gen.STATIC_TF}
        for stamp in self._stamps()[::7]:
            params = self._load_yaml("1", stamp)
            t = params["mirc"]["stamp"]
            content = params["vehicles"][2]
            object_pose = [content["location"][i] + content["center"][i]
                           for i in range(3)] + list(content["angle"])
            object_to_lidar = opencood_x1_to_x2(object_pose,
                                                params["lidar_pose"])

            truth_world = gen.robot_b_pose(t)
            truth_centre = truth_world[:3, 3] + np.array([0.0, 0.0, 0.30])
            lidar_world = gen.robot_a_pose(t) @ static["os_sensor"]
            expected = (truth_centre - lidar_world[:3, 3]) @ lidar_world[:3, :3]

            np.testing.assert_allclose(object_to_lidar[:3, 3], expected,
                                       atol=2e-3)
            np.testing.assert_allclose(
                content["extent"], [0.30, 0.25, 0.30], atol=1e-9)

    def test_depth_agent_cloud_lands_in_the_room(self):
        """Robot B is RGBD-only: its cloud is back-projected depth, so this
        exercises optical->body rotation, the TF extrinsic and the pose."""
        stamps = self._stamps("2")[::5]
        inside_fraction = []
        for stamp in stamps:
            params = self._load_yaml("2", stamp)
            points = load_points(os.path.join(self.scenario, "2",
                                              stamp + ".pcd"))
            world = (points[:, :3] @
                     opencood_x_to_world(params["lidar_pose"])[:3, :3].T +
                     opencood_x_to_world(params["lidar_pose"])[:3, 3])
            inside = ((np.abs(world[:, 0]) <= gen_room_x() + 0.15) &
                      (np.abs(world[:, 1]) <= gen_room_y() + 0.15) &
                      (world[:, 2] >= -0.15) & (world[:, 2] <= 2.75))
            inside_fraction.append(inside.mean())
        self.assertGreater(float(np.mean(inside_fraction)), 0.98)

    def test_ego_cloud_reprojects_onto_the_room_walls(self):
        stamp = self._stamps()[3]
        params = self._load_yaml("1", stamp)
        points = load_points(os.path.join(self.scenario, "1", stamp + ".pcd"))
        matrix = opencood_x_to_world(params["lidar_pose"])
        world = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
        inside = ((np.abs(world[:, 0]) <= gen_room_x() + 0.1) &
                  (np.abs(world[:, 1]) <= gen_room_y() + 0.1) &
                  (world[:, 2] >= -0.1) & (world[:, 2] <= 2.7))
        self.assertGreater(inside.mean(), 0.999)
        # the rest of the sweep is the room shell; ~14% of the synthetic sweep
        # is the other robot's body, which is not on a wall
        on_wall = (np.abs(np.abs(world[:, 0]) - gen_room_x()) < 0.1) | \
                  (np.abs(np.abs(world[:, 1]) - gen_room_y()) < 0.1) | \
                  (np.abs(world[:, 2]) < 0.1)
        self.assertGreater(on_wall.mean(), 0.8)

    def test_telemetry_is_time_joined(self):
        params = self._load_yaml("1", self._stamps()[5])
        wifi = params["mirc"]["wifi"]
        self.assertIn("rssi_dbm", wifi)
        self.assertLessEqual(abs(wifi["_dt"]), 1.0)

    def test_emitted_opencood_config_is_loadable(self):
        import yaml

        from rosbag2opv2v import opencood_hypes

        text = opencood_hypes.build(self.out, fusion="intermediate")
        hypes = yaml.safe_load(text)
        low = hypes["preprocess"]["cav_lidar_range"]
        voxel = hypes["preprocess"]["args"]["voxel_size"]
        for i in range(2):
            span = (low[i + 3] - low[i]) / voxel[i]
            self.assertAlmostEqual(span, round(span), places=6)
            self.assertEqual(round(span) % 8, 0)
        self.assertAlmostEqual((low[5] - low[2]) / voxel[2], 1.0, places=6)
        self.assertEqual(hypes["fusion"]["core_method"],
                         "IntermediateFusionDataset")


def gen_room_x() -> float:
    import make_synthetic_bag as gen
    return gen.ROOM_X


def gen_room_y() -> float:
    import make_synthetic_bag as gen
    return gen.ROOM_Y


if __name__ == "__main__":
    unittest.main()
