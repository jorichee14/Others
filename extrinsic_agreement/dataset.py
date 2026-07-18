"""mirc_dataset_20260706 — sensor registry.

Every intrinsic and extrinsic from the dataset sheet, transcribed once, here, so
the analysis never hard-codes a number inline. Frame names match the sheet.

map = NAV (x-fwd, y-left, z-up). All "map -> X" entries express X's pose in map,
i.e. ``T_map_X`` under the convention in se3.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from se3 import se3, inv


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple  # (k1, k2, p1, p2, k3), plumb_bob

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1.0]])

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project (N,3) points in this camera's OPTICAL frame to (N,2) pixels.

        Optical frame = x-right, y-down, z-forward. Points with z<=0 project to
        NaN. Distortion is applied (plumb_bob) so this is honest for the raw
        RealSense / Arducam feeds; the ZED feed is rectified (zero distortion).
        """
        p = np.atleast_2d(np.asarray(points_cam, float))
        z = p[:, 2]
        xn = p[:, 0] / z
        yn = p[:, 1] / z
        k1, k2, p1, p2, k3 = self.distortion
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        uv = np.column_stack([u, v])
        uv[z <= 0] = np.nan
        return uv

    def deproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Back-project pixel(s) + metric depth to 3D in the optical frame.

        Uses the pinhole model without undistortion (adequate for the small
        distortions here and for a coarse chair centroid; undistort upstream if
        you need better). ``depth`` is z along the optical axis, in metres.
        """
        uv = np.atleast_2d(np.asarray(uv, float))
        d = np.atleast_1d(np.asarray(depth, float))
        x = (uv[:, 0] - self.cx) / self.fx * d
        y = (uv[:, 1] - self.cy) / self.fy * d
        return np.column_stack([x, y, d])


ZED_LEFT = Camera("zed_left_camera_optical_frame", 640, 360,
                  268.160, 268.160, 324.606, 177.114, (0, 0, 0, 0, 0))
REALSENSE = Camera("camera_color_optical_frame", 640, 480,
                   387.2485, 386.7093, 319.2134, 239.7600,
                   (-0.056839, 0.064692, -0.000467, 0.000595, -0.020723))
ARDUCAM = Camera("arducam_optical_frame", 960, 600,
                 603.840, 602.190, 469.250, 307.930,
                 (-0.089745, -0.031476, -0.000773, -0.000888, 0.017394))


# --------------------------------------------------------------------------- #
# Fixed / anchor extrinsics  (T_map_child)
# --------------------------------------------------------------------------- #
# map -> board (origin; NAV->OPT). Not on the chair chain, kept for completeness.
T_map_board = se3([0.0, 0.0, 0.0],
                  [0.499282, 0.491698, 0.508301, 0.500580])

# Where MP#1 (ZED) starts, in map. A *reference* start pose only — for the moving
# platform use the per-timestamp /glim/camera_pose, not this.
T_map_zed_start = se3([0.385131, 0.126192, 0.180677],
                      [-0.030750, 0.007227, -0.999048, 0.030073])

# Where MP#2 (RealSense) starts, in map. Anchor for /vo_pose if that pose stream
# is expressed in a VSLAM-local start frame rather than in map.
T_map_rs_start = se3([-16.735830, -4.696244, 0.513120],
                     [-0.501359, 0.476104, -0.513143, 0.508575])

# IN#1 Arducam — a FIXED node. This full transform is the thing under test.
# NOTE: the sheet flags the translation z (1.926039) as suspect.
T_map_arducam = se3([-4.145592, -6.339344, 1.926039],
                    [-0.772587, -0.329235, 0.193157, 0.507351])


# --------------------------------------------------------------------------- #
# MP#1 rigid-rig extrinsics
# --------------------------------------------------------------------------- #
# The sheet is internally contradictory here: the arrow "os_lidar -> zed" reads
# as T_zed_lidar, while the label "T_lidar_camera" reads as its inverse. We store
# the raw numbers once and expose BOTH interpretations; the rigid-rig test picks
# whichever makes the chair agree. This ambiguity alone can cause the observed
# LiDAR-cloud "wonkiness".
_LIDAR_ZED_RAW = se3([-0.074928, -0.066971, -0.091627],
                     [-0.497829, -0.498035, 0.501789, 0.502329])

# radar_link expressed in the ZED optical frame -> T_zed_radar*.
T_zed_radar1 = se3([0.213, -0.019, -0.021],
                   [-0.5359, 0.5906, -0.4177, -0.4354])
T_zed_radar2 = se3([-0.0999, -0.0124, -0.0011],
                   [0.7882, -0.0406, 0.6121, 0.0499])

# RealSense internal depth->color (only needed if you localize the chair in the
# depth optical frame instead of the color-aligned depth image).
T_rscolor_rsdepth = se3([-0.059189, -0.000142, 0.000505],
                        [-0.002966, 0.000832, 0.001305, 0.999994])


def T_zed_lidar(interpretation: str = "arrow") -> np.ndarray:
    """Return ``T_zed_lidar`` (maps a LiDAR point into the ZED optical frame).

    ``interpretation``:
      - ``"arrow"``  : trust the sheet arrow "os_lidar -> zed", i.e. the raw
                       numbers already are T_zed_lidar. Returns the raw matrix.
      - ``"label"``  : trust the label "T_lidar_camera" (= T_lidar_zed), i.e. the
                       raw numbers are T_lidar_zed. Returns its inverse.
    """
    if interpretation == "arrow":
        return _LIDAR_ZED_RAW.copy()
    if interpretation == "label":
        return inv(_LIDAR_ZED_RAW)
    raise ValueError(f"interpretation must be 'arrow' or 'label', got {interpretation!r}")


CAMERAS = {c.name: c for c in (ZED_LEFT, REALSENSE, ARDUCAM)}
