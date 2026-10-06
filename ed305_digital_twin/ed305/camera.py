"""Pinhole cameras placed in the ED305 room frame.

Room frame (all configs): origin on the floor at the red dot of the wiki's
camera map (under the middle of cams 10 and 11), z up, x toward the right
wall, y toward the top (door) wall of the floor plan.

Camera frame: OpenCV, x right, y down, z forward (optical axis).

A camera entry in the cameras YAML gives intrinsics and ONE of:
  T_world_cam: 4x4            camera -> room (p_room = T @ p_cam)
  rvec, tvec:  OpenCV         room -> camera, as from cv2.solvePnP / calibrateCamera
  position, look_at: [x,y,z]  nominal mount, optical axis through look_at
"""
from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class Camera:
    name: str
    K: np.ndarray            # 3x3
    width: int
    height: int
    T_world_cam: np.ndarray  # 4x4, camera -> room
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))  # k1 k2 p1 p2 k3

    @property
    def center(self):
        return self.T_world_cam[:3, 3]

    @property
    def axis(self):
        return self.T_world_cam[:3, 2]

    def to_cam(self, pts):
        """(N,3) room points -> (N,3) camera points."""
        R, t = self.T_world_cam[:3, :3], self.T_world_cam[:3, 3]
        return (pts - t) @ R

    def project(self, pts):
        """(N,3) room points -> (N,2) pixels (with distortion), (N,) depth."""
        pc = self.to_cam(pts)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn, yn = pc[:, 0] / z, pc[:, 1] / z
        k1, k2, p1, p2, k3 = self.dist
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        u = self.K[0, 0] * xd + self.K[0, 1] * yd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        return np.stack([u, v], 1), z, r2

    def in_view(self, pts, near=0.1, margin_px=0):
        """Points in front of the camera that land inside the image.

        The distortion polynomial folds back far outside the image, so a point
        also has to be within the undistorted radius of the image corners
        (plus 20%): a wide-angle point cannot wrap into the frame.
        """
        uv, z, r2 = self.project(pts)
        corners = np.array([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]], float)
        xn = (corners[:, 0] - self.K[0, 2]) / self.K[0, 0]
        yn = (corners[:, 1] - self.K[1, 2]) / self.K[1, 1]
        r2_max = 1.44 * np.max(xn * xn + yn * yn)
        m = margin_px
        return ((z > near) & (r2 <= r2_max)
                & (uv[:, 0] >= m) & (uv[:, 0] < self.width - m)
                & (uv[:, 1] >= m) & (uv[:, 1] < self.height - m)), z

    def fov_deg(self):
        h = 2 * np.degrees(np.arctan2(self.width / 2, self.K[0, 0]))
        v = 2 * np.degrees(np.arctan2(self.height / 2, self.K[1, 1]))
        return h, v


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    """Camera -> room transform for a camera at `position` aimed at `target`."""
    p, t = np.asarray(position, float), np.asarray(target, float)
    z = t - p
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    if abs(z @ up) > 0.999:          # straight down/up: image top toward +y
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = p
    return T


def _rodrigues(rvec):
    r = np.asarray(rvec, float).reshape(3)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def _pose(name, c):
    if "T_world_cam" in c:
        T = np.asarray(c["T_world_cam"], float)
        if T.shape != (4, 4):
            raise ValueError(f"{name}: T_world_cam must be 4x4")
        return T
    if "rvec" in c and "tvec" in c:   # OpenCV room -> camera
        R = _rodrigues(c["rvec"])
        t = np.asarray(c["tvec"], float).reshape(3)
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T
    if "position" in c and "look_at" in c:
        return look_at(c["position"], c["look_at"])
    raise ValueError(f"{name}: give T_world_cam, rvec+tvec, or position+look_at")


def _intrinsics(name, c, defaults):
    c = {**defaults, **c}
    if "K" in c:
        K = np.asarray(c["K"], float).reshape(3, 3)
    elif "fx" in c:
        K = np.array([[c["fx"], 0, c.get("cx", c["width"] / 2)],
                      [0, c.get("fy", c["fx"]), c.get("cy", c["height"] / 2)],
                      [0, 0, 1]], float)
    else:
        raise ValueError(f"{name}: give K or fx")
    dist = np.zeros(5)
    d = np.asarray(c.get("dist", []), float).ravel()
    dist[: min(5, d.size)] = d[:5]
    return K, int(c["width"]), int(c["height"]), dist


def load_cameras(path):
    """Cameras YAML -> list of Camera, in file order.

    `defaults:` (optional) fills any intrinsics key a camera leaves out.
    A camera with `enabled: false` is skipped.
    """
    with open(path) as f:
        doc = yaml.safe_load(f)
    defaults = doc.get("defaults", {}) or {}
    cams = []
    for name, c in doc["cameras"].items():
        if c.get("enabled", True) is False:
            continue
        K, w, h, dist = _intrinsics(name, c, defaults)
        cams.append(Camera(str(name), K, w, h, _pose(name, c), dist))
    return cams
