"""Intrinsics: board-pose features of a view (x, y, z, roll, pitch, yaw), the calibrate/test split,
the fit on the calibrate set and the evaluation on the test set."""
import glob
import os

import cv2
import numpy as np

from calib.common import cam_name, read_json

IMAGE_SIZE = (1920, 1080)
TEST_EVERY = 5          # every 5th saved view of a camera is a TEST view (never used for fitting)
FIELDS = ["x", "y", "z", "roll", "pitch", "yaw"]
# span the saved views must cover for a bar to be full (like the X Y Size Skew bars of the ROS tool)
TARGET = {"x": 0.6, "y": 0.6, "z": float(np.log(2.5)), "roll": 60.0, "pitch": 50.0, "yaw": 50.0}
MIN_VIEWS = 20          # calibrate views wanted before "READY"

# Lens models. "ed305" is how the ED305 cameras were calibrated in March 2026 (cal_params.py):
# OpenCV pinhole, radial k1 k2 only (p1 = p2 = k3 = 0), principal point at the image centre.
MODELS = {
    "ed305": "standard lens: k1 k2, principal point at the image centre (as the March 2026 ED305 calibration)",
    "full": "standard lens: k1 k2 p1 p2 k3, principal point estimated",
    "fisheye": "fisheye lens (over ~120 deg): equidistant model, k1 k2 k3 k4",
}
DEFAULT_MODEL = "full"


def _fisheye_flag(name):
    """cv2.fisheye.CALIB_<name> (OpenCV 4) or cv2.CALIB_<name> (OpenCV 5)."""
    for mod, key in ((cv2.fisheye, "CALIB_" + name), (cv2, "fisheye_CALIB_" + name), (cv2, "CALIB_" + name)):
        if hasattr(mod, key):
            return getattr(mod, key)
    raise AttributeError("OpenCV has no fisheye flag " + name)


def project_model(obj, rvec, tvec, K, dist, model="full"):
    """Project 3D points (board frame) into the image with the given lens model. Returns Nx2."""
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    r, t = np.asarray(rvec, np.float64).reshape(3, 1), np.asarray(tvec, np.float64).reshape(3, 1)
    if model == "fisheye":
        p, _ = cv2.fisheye.projectPoints(obj, r, t, np.asarray(K, np.float64), np.asarray(dist, np.float64).reshape(4, 1))
    else:
        p, _ = cv2.projectPoints(obj, r, t, np.asarray(K, np.float64), None if dist is None else np.asarray(dist, np.float64))
    return p.reshape(-1, 2)


def split_for(index):
    """Split of the index-th saved view of a camera (0-based)."""
    return "test" if index % TEST_EVERY == TEST_EVERY - 1 else "calib"


def nominal_K(image_size=IMAGE_SIZE):
    """Rough K used for the coverage bars until the camera has a real calibration."""
    w, h = image_size
    return np.array([[1.0 * w, 0, w / 2], [0, 1.0 * w, h / 2], [0, 0, 1]], float)


def board_pose(board, corners, ids, K, dist=None, model="full"):
    """(rvec, tvec) of the board in the camera frame from one view, or None."""
    o, i = board.board.matchImagePoints(corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
    if o is None or len(o) < 6:
        return None
    if model == "fisheye":   # undistort to normalised coordinates, then an ideal pinhole
        i = cv2.fisheye.undistortPoints(i.reshape(-1, 1, 2).astype(np.float64), np.asarray(K, np.float64),
                                        np.asarray(dist, np.float64).reshape(4, 1))
        K, dist = np.eye(3), np.zeros(5)
    dist = np.zeros(5) if dist is None else dist
    ok, r, t = cv2.solvePnP(o, i, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    ok, r, t = cv2.solvePnP(o, i, K, dist, r, t, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return (r.reshape(3), t.reshape(3)) if ok else None


def view_features(board, corners, ids, K, dist=None, image_size=IMAGE_SIZE, model="full"):
    """Where the board was in this view.
    x, y: board centre in the image, 0..1 of width/height
    z: distance of the board centre along the optical axis (m)
    roll: in-plane rotation of the board (deg); pitch, yaw: tilt about the image x / y axis (deg)
    Returns None when the pose cannot be solved."""
    w, h = image_size
    p = board_pose(board, corners, ids, K, dist, model)
    if p is None:
        return None
    r, t = p
    R, _ = cv2.Rodrigues(r)
    centre = R @ board.object_points.mean(0) + t
    n = R[:, 2]
    if n[2] < 0:
        n = -n
    c = corners.mean(0)
    return {"x": float(c[0] / w), "y": float(c[1] / h), "z": float(centre[2]),
            "roll": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            "pitch": float(np.degrees(np.arctan2(n[1], n[2]))),
            "yaw": float(np.degrees(np.arctan2(n[0], n[2])))}


def coverage(feats):
    """{field: 0..1} how much of the wanted span the saved views cover (z on a log scale)."""
    out = {}
    for k in FIELDS:
        v = np.array([f[k] for f in feats if f and f.get(k) is not None], float)
        if len(v) < 2:
            out[k] = 0.0
            continue
        if k == "z":
            v = np.log(np.maximum(v, 0.05))
        out[k] = float(min(1.0, np.ptp(v) / TARGET[k]))
    return out


def ready(feats):
    return len(feats) >= MIN_VIEWS and all(v >= 1.0 for v in coverage(feats).values())


# ---------------------------------------------------------------- saved views
def load_views(data_dir, cam, session=1):
    """All saved intr views of a camera (all people merged), each a dict with board, corners, ids,
    path, person, split. Views saved without a split get one by order (every 5th = test)."""
    views = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "intr", cam_name(cam, session), "*.json"))):
        j = read_json(path)
        if j.get("charuco") and len(j["charuco"]["ids"]) >= 6:
            views.append({"board": j["board"], "corners": np.array(j["charuco"]["corners"], np.float32),
                          "ids": np.array(j["charuco"]["ids"], np.int32), "path": path,
                          "person": path.split(os.sep)[-4], "split": j.get("split"),
                          "image_size": tuple(j.get("image_size") or IMAGE_SIZE)})
    for k, v in enumerate(views):
        if v["split"] not in ("calib", "test"):
            v["split"] = split_for(k)
    return views


def _points(views, boards):
    obj, img = [], []
    for v in views:
        o, i = boards[v["board"]].board.matchImagePoints(v["corners"].reshape(-1, 1, 2), v["ids"].reshape(-1, 1))
        obj.append(o.reshape(-1, 3).astype(np.float32))
        img.append(i.reshape(-1, 2).astype(np.float32))
    return obj, img


def calibrate(views, boards, flags=0, image_size=IMAGE_SIZE, model="full"):
    """Fit K and distortion on these views with the given lens model: rms, K, dist, per-view rms."""
    obj, img = _points(views, boards)
    if model == "fisheye":
        # cv2.fisheye.calibrate needs 1xN point arrays and a sensible starting focal length (without one it
        # converges to nonsense): start from lenses of 90 to 210 deg across the image width, keep the best fit
        O = [o.reshape(1, -1, 3).astype(np.float64) for o in obj]
        I = [i.reshape(1, -1, 2).astype(np.float64) for i in img]
        w, h = image_size
        best, err = None, None
        for half_deg in (45, 60, 75, 90, 105):
            f0 = (w / 2.0) / np.radians(half_deg)
            K0 = np.array([[f0, 0, w / 2.0], [0, f0, h / 2.0], [0, 0, 1]])
            try:
                out = cv2.fisheye.calibrate(
                    O, I, (int(w), int(h)), K0, np.zeros((4, 1)),
                    flags=_fisheye_flag("USE_INTRINSIC_GUESS") | _fisheye_flag("RECOMPUTE_EXTRINSIC") | _fisheye_flag("FIX_SKEW"),
                    criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9))
            except cv2.error as e:
                err = e
                continue
            if best is None or out[0] < best[0]:
                best = out
        if best is None:
            raise RuntimeError("fisheye fit failed (%s)" % str(err).split("error:")[-1].strip())
        rms, K, dist, rvecs, tvecs = best
    else:
        K0 = None
        if model == "ed305":   # principal point exactly at (w/2, h/2), as cal_params.py did
            K0 = cv2.initCameraMatrix2D(obj, img, tuple(image_size))
            K0[0, 2], K0[1, 2] = image_size[0] / 2.0, image_size[1] / 2.0
            flags |= (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
                      | cv2.CALIB_FIX_K3)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, tuple(image_size), K0, None if K0 is None else np.zeros(5),
                                                         flags=flags)
    dist = dist.reshape(-1)
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        p = project_model(o, r, t, K, dist, model)
        per_view.append(float(np.sqrt(np.mean(np.sum((p - i) ** 2, axis=1)))))
    return float(rms), K, dist, per_view


def evaluate(views, boards, K, dist, model="full"):
    """Per-view reprojection rms (px) of views NOT used in the fit: pose from solvePnP with the given
    K and dist, then reprojection. The honest number for how good the calibration is."""
    out = []
    for v in views:
        p = board_pose(boards[v["board"]], v["corners"], v["ids"], K, dist, model)
        if p is None:
            out.append(float("nan"))
            continue
        o, i = _points([v], boards)
        pr = project_model(o[0], p[0], p[1], K, dist, model)
        out.append(float(np.sqrt(np.mean(np.sum((pr - i[0]) ** 2, axis=1)))))
    return out


def fit_camera(views, boards, flags=0, image_size=IMAGE_SIZE, model=DEFAULT_MODEL):
    """Fit on the calibrate views (dropping clear outliers once), evaluate on the test views.
    Returns None when there are fewer than 10 calibrate views, else a dict:
    K, dist, rms_calib, rms_test, n_calib, n_test, dropped, views (each with split, used, rms_px, features)."""
    calib = [v for v in views if v["split"] == "calib"]
    test = [v for v in views if v["split"] == "test"]
    if len(calib) < 10:
        return None
    rms, K, dist, per = calibrate(calib, boards, flags, image_size, model)
    med = np.median(per)
    keep = [v for v, e in zip(calib, per) if e <= max(2 * med, 0.5)]
    dropped = [v for v, e in zip(calib, per) if e > max(2 * med, 0.5)]
    if dropped and len(keep) >= 10:
        rms, K, dist, per = calibrate(keep, boards, flags, image_size, model)
    else:
        keep, dropped = calib, []
    for v, e in zip(keep, per):
        v["used"], v["rms_px"] = True, e
    for v, e in zip(dropped, evaluate(dropped, boards, K, dist, model)):
        v["used"], v["rms_px"] = False, e
    for v, e in zip(test, evaluate(test, boards, K, dist, model)):
        v["used"], v["rms_px"] = False, e
    for v in views:
        v["features"] = view_features(boards[v["board"]], v["corners"], v["ids"], K, dist, image_size, model)
    t = [v["rms_px"] for v in test if np.isfinite(v["rms_px"])]
    return {"K": K, "dist": dist, "model": model, "rms_calib": rms, "rms_test": float(np.sqrt(np.mean(np.square(t)))) if t else float("nan"),
            "n_calib": len(keep), "n_test": len(test), "dropped": len(dropped), "views": views,
            "coverage": coverage([v["features"] for v in views if v["split"] == "calib"])}


def view_rows(cam, res):
    """CSV rows (one per view) for a fit_camera result."""
    rows = []
    for v in res["views"]:
        f = v.get("features") or {}
        rows.append({"camera": cam, "person": v["person"], "board": v["board"], "file": v["path"],
                     "split": v["split"], "used": int(v["used"]), "corners": len(v["ids"]),
                     "rms_px": round(v["rms_px"], 3) if np.isfinite(v["rms_px"]) else "",
                     **{k: (round(f[k], 3) if k in f else "") for k in FIELDS}})
    return rows


def summary_row(cam, res):
    K, d = res["K"], res["dist"]
    cov = res["coverage"]
    row = {"camera": cam, "n_calib": res["n_calib"], "n_test": res["n_test"], "dropped": res["dropped"],
           "rms_calib_px": round(res["rms_calib"], 3), "rms_test_px": round(res["rms_test"], 3),
           "fx": round(K[0, 0], 2), "fy": round(K[1, 1], 2), "cx": round(K[0, 2], 2), "cy": round(K[1, 2], 2)}
    names = ["k1", "k2", "k3", "k4"] if res.get("model") == "fisheye" else ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    row["model"] = res.get("model", "")
    for k, name in enumerate(names[:len(d)]):
        row[name] = float("%.6g" % d[k])
    for k in FIELDS:
        row["cov_" + k] = round(cov[k], 2)
    row["status"] = status(res)
    return row


def status(res):
    if not np.isfinite(res["rms_test"]):
        return "NO TEST VIEWS"
    if res["rms_test"] > 1.0 or res["rms_test"] > 1.6 * res["rms_calib"] + 0.2:
        return "CHECK: test error high (overfit or bad views)"
    if res["rms_calib"] > 0.6:
        return "CHECK: rms > 0.6 px (board not flat/sharp?)"
    low = [k for k, v in res["coverage"].items() if v < 1.0]
    if low:
        return "CHECK: more views varying " + " ".join(low)
    return "OK"
