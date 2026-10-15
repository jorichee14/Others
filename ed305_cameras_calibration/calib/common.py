"""Shared helpers: config loading, boards and detectors, pose maths, file layout."""
import json
import os
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(HERE, "config")

_DICTS = {k: getattr(cv2.aruco, k) for k in dir(cv2.aruco) if k.startswith("DICT_")}


def load_yaml(name):
    if name == "boards.yaml" and os.environ.get("ED305_BOARDS"):   # tests use their own boards
        name = os.environ["ED305_BOARDS"]
    path = name if os.path.isabs(name) or os.path.exists(name) else os.path.join(CONFIG, name)
    with open(path) as f:
        return yaml.safe_load(f)


def camera_urls(cams=None):
    cfg = load_yaml("cameras.yaml")["cameras"]
    cams = cams or sorted(cfg)
    return {int(c): cfg[int(c)] for c in cams}


# ---------------------------------------------------------------- boards and detectors
class Board:
    """One ChArUco board and its detector."""

    def __init__(self, name, spec):
        self.name = name
        self.dict_name = spec["dictionary"]
        self.dictionary = cv2.aruco.getPredefinedDictionary(_DICTS[self.dict_name])
        self.board = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]),
                                            float(spec["square_m"]), float(spec["marker_m"]),
                                            self.dictionary)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(self.board, cv2.aruco.CharucoParameters(), params)
        self.n_corners = (spec["squares_x"] - 1) * (spec["squares_y"] - 1)
        self.object_points = self.board.getChessboardCorners().reshape(-1, 3)

    def detect(self, gray):
        """Returns (corners Nx2 float32, ids N int) or (None, None)."""
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) < 4:
            return None, None
        return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(int)


def load_boards(names=None):
    spec = load_yaml("boards.yaml")["boards"]
    names = names or sorted(spec)
    return {n: Board(n, spec[n]) for n in names}


class FloorTags:
    """Single AprilTags on the floor; corners in tag frame: (-s/2,s/2) (s/2,s/2) (s/2,-s/2) (-s/2,-s/2)."""

    def __init__(self):
        spec = load_yaml("boards.yaml")["floor_tags"]
        self.size = float(spec["size_m"])
        self.dictionary = cv2.aruco.getPredefinedDictionary(_DICTS[spec["dictionary"]])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    def detect(self, gray):
        """Returns {tag_id: 4x2 corners} in OpenCV marker corner order (top-left clockwise)."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2).astype(np.float32) for c, i in zip(corners, ids.reshape(-1))}

    def world_corners(self):
        """{tag_id: 4x3 world corners} from floor_markers.yaml (z = 0)."""
        out = {}
        s = self.size / 2
        local = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)  # marker corner order
        for m in load_yaml("floor_markers.yaml")["markers"]:
            yaw = np.deg2rad(float(m.get("yaw_deg", 0)))
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            out[int(m["id"])] = (local @ R.T) + np.array([m["x"], m["y"], float(m.get("z", 0))])
        return out


# ---------------------------------------------------------------- poses
def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    T[:3, 3] = np.asarray(tvec, float).reshape(3)
    return T


def T_to_rt(T):
    rvec, _ = cv2.Rodrigues(np.asarray(T, float)[:3, :3])
    return rvec.reshape(3), np.asarray(T, float)[:3, 3].copy()


def inv_T(T):
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def project(K, dist, rvec, tvec, pts3):
    img, _ = cv2.projectPoints(np.asarray(pts3, float).reshape(-1, 1, 3), np.asarray(rvec, float),
                               np.asarray(tvec, float), K, dist)
    return img.reshape(-1, 2)


# ---------------------------------------------------------------- files
def now_tag():
    return time.strftime("%Y%m%d_%H%M%S") + "_%03d" % int((time.time() % 1) * 1000)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_np)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(str(type(o)))


def read_intrinsics_file(path):
    """(K, dist, image_size or None) from a file made elsewhere: this repo's cam_N.json (K, dist),
    the ED305 wiki results (camera_matrix, dist_coeffs), a ROS camera_info YAML from
    camera_calibration (camera_matrix.data, distortion_coefficients.data) or from
    `ros2 topic echo --once <camera_info>` (k, d, width, height)."""
    j = load_yaml(path) if path.endswith((".yaml", ".yml")) else read_json(path)

    def first(*keys):
        for k in keys:
            if k in j and j[k] is not None:
                v = j[k]
                return v["data"] if isinstance(v, dict) and "data" in v else v
        return None

    K = first("K", "camera_matrix", "k")
    dist = first("dist", "dist_coeffs", "distortion_coefficients", "d", "D")
    if K is None:
        raise ValueError("%s: no K / camera_matrix / k" % path)
    size = j.get("image_size")
    if size is None and j.get("width") and j.get("height"):
        size = [j["width"], j["height"]]
    if size is None and j.get("image_width") and j.get("image_height"):
        size = [j["image_width"], j["image_height"]]
    dist = np.array(dist if dist is not None else [], float).reshape(-1)
    return np.array(K, float).reshape(3, 3), (dist if dist.size else np.zeros(5)), \
        (tuple(int(v) for v in size) if size else None)


def load_intrinsics(results_dir):
    """{cam: (K, dist)} from results/intrinsics/cam_N.json."""
    out = {}
    d = os.path.join(results_dir, "intrinsics")
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.startswith("cam_") and fn.endswith(".json"):
            j = read_json(os.path.join(d, fn))
            out[int(j["camera"])] = (np.array(j["K"], float), np.array(j["dist"], float).reshape(-1))
    return out


def euler_zyx_deg(R):
    """roll, pitch, yaw (deg) of a rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    R = np.asarray(R, float)
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def pose_row(T_wx):
    """x, y, z (m) and roll, pitch, yaw (deg) of a frame given by its pose in the world (T world<-x)."""
    T = np.asarray(T_wx, float)
    r, p, y = euler_zyx_deg(T[:3, :3])
    return {"x": round(T[0, 3], 4), "y": round(T[1, 3], 4), "z": round(T[2, 3], 4),
            "roll": round(r, 3), "pitch": round(p, 3), "yaw": round(y, 3)}


def write_csv(path, rows, fields=None, append=False):
    import csv
    if not rows and not fields:
        return
    fields = fields or list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not (append and os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def read_csv(path):
    import csv
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
