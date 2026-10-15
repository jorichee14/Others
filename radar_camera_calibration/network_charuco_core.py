"""Pure logic of network_charuco.py (no ROS): the board pose in one image, "is the board still",
and per-camera pooling / convergence / lock of an extrinsic, the same way general_charuco.py does
it for one pair (average of the samples, stable for a while -> LOCKED)."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def average_se3(Xs):
    """Mean of 4x4 rigid transforms (translations averaged, quaternions averaged in one hemisphere)."""
    ts = np.array([X[:3, 3] for X in Xs])
    qs = np.array([Rot.from_matrix(X[:3, :3]).as_quat() for X in Xs])
    qs = qs * np.sign(qs @ qs[0])[:, None]
    q = qs.mean(0)
    q /= np.linalg.norm(q)
    X = np.eye(4)
    X[:3, :3] = Rot.from_quat(q).as_matrix()
    X[:3, 3] = ts.mean(0)
    return X


def rt_error(X, Y):
    """(translation difference in m, rotation difference in deg) between two 4x4 transforms."""
    d = np.linalg.inv(X) @ Y
    return float(np.linalg.norm(X[:3, 3] - Y[:3, 3])), float(np.degrees(np.linalg.norm(Rot.from_matrix(d[:3, :3]).as_rotvec())))


def pose_dict(T):
    """translation, quaternion (x y z w) and roll-pitch-yaw (deg, xyz) of a 4x4 transform, JSON-ready."""
    R = Rot.from_matrix(T[:3, :3])
    return {"translation": [float(v) for v in T[:3, 3]],
            "quaternion_xyzw": [float(v) for v in R.as_quat()],
            "rpy_deg": [float(v) for v in R.as_euler("xyz", degrees=True)],
            "T": [[float(v) for v in row] for row in T]}


class BoardPose:
    """T_cam_board (4x4) of one ChArUco board in one grey image, with the corner count and the
    reprojection rms (px). Standard OpenCV lens model (K, 5 distortion coefficients)."""

    def __init__(self, board, min_corners=8):
        self.det = cv2.aruco.CharucoDetector(board)
        self.obj = board.getChessboardCorners().reshape(-1, 3)
        self.min_corners = min_corners

    def __call__(self, gray, K, D):
        cc, cid, _, _ = self.det.detectBoard(gray)
        n = 0 if cid is None else len(cid)
        if n < self.min_corners:
            return None, None, n
        objp = self.obj[cid.ravel()].astype(np.float64)
        img = cc.reshape(-1, 2).astype(np.float64)
        ok, r, t = cv2.solvePnP(objp, img, K, D, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None, None, n
        ok, r, t = cv2.solvePnP(objp, img, K, D, r, t, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, n
        proj, _ = cv2.projectPoints(objp, r, t, K, D)
        reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(r)[0]
        T[:3, 3] = t.ravel()
        return T, reproj, n


class Still:
    """Is the board still in one camera? Still = over the last `still_s` seconds the board pose moved
    less than `still_m` and `still_deg`. `hold` counts the still periods (a new hold each time the
    board becomes still again), so samples from different board placements can be told apart."""

    def __init__(self, still_s=0.6, still_m=0.003, still_deg=0.3):
        self.still_s, self.still_m, self.still_deg = still_s, still_m, still_deg
        self.hist, self.still, self.hold = [], False, 0

    def add(self, t, T):
        if T is None:
            self.hist, self.still = [], False
            return False
        self.hist = [(ti, Ti) for ti, Ti in self.hist if t - ti <= 2 * self.still_s] + [(t, T)]
        span = [(ti, Ti) for ti, Ti in self.hist if t - ti <= self.still_s]
        covered = self.hist[0][0] <= t - self.still_s
        still = covered and all(
            dt < self.still_m and dr < self.still_deg for dt, dr in (rt_error(T, Ti) for _, Ti in span))
        if still and not self.still:
            self.hold += 1
        self.still = still
        return still


class Pair:
    """One camera B: its pose samples (T_world_B, or T_A_B without a world frame) pooled and averaged
    until stable, then LOCKED. Locks when, over the last `conv_window` averages, the average moved
    less than `conv_t_std` (m) / `conv_rot_std` (deg) for `conv_hold` samples in a row, with at least
    `conv_pool` samples from at least `min_holds` different board placements."""

    def __init__(self, name, pool_cap=400, conv_window=15, conv_hold=10, conv_t_std=0.003, conv_rot_std=0.2,
                 conv_pool=30, min_holds=3):
        self.name = name
        self.pool_cap, self.conv_window, self.conv_hold = pool_cap, conv_window, conv_hold
        self.conv_t_std, self.conv_rot_std, self.conv_pool, self.min_holds = conv_t_std, conv_rot_std, conv_pool, min_holds
        self.pool, self.holds, self.est = [], set(), []
        self.stable, self.X, self.locked = 0, None, None
        self.t_std = self.rot_std = float("inf")
        self.reproj = []

    def add(self, X, hold, reproj=None):
        """One sample (4x4) from board placement `hold`. Returns True when this sample locked it."""
        if self.locked is not None:
            return False
        self.pool = (self.pool + [X])[-self.pool_cap:]
        self.holds.add(hold)
        if reproj is not None:
            self.reproj = (self.reproj + [reproj])[-self.pool_cap:]
        self.X = average_se3(self.pool)
        self.est = (self.est + [(self.X[:3, 3].copy(), Rot.from_matrix(self.X[:3, :3]).as_rotvec())])[-self.conv_window:]
        if len(self.est) >= self.conv_window:
            ts = np.array([e[0] for e in self.est])
            rs = np.array([e[1] for e in self.est])
            self.t_std = float(np.linalg.norm(ts.std(0)))
            self.rot_std = float(np.degrees(np.linalg.norm(rs.std(0))))
        ok = (self.t_std < self.conv_t_std and self.rot_std < self.conv_rot_std
              and len(self.pool) >= self.conv_pool and len(self.holds) >= self.min_holds)
        self.stable = self.stable + 1 if ok else 0
        if self.stable >= self.conv_hold:
            self.locked = self.X.copy()
            return True
        return False

    def spread(self):
        """Spread of the samples around the average: (rms m, rms deg). Large = a bad board pose or
        intrinsics, or a camera / robot that moved during the collection."""
        if self.X is None:
            return float("nan"), float("nan")
        e = np.array([rt_error(self.X, P) for P in self.pool])
        return float(np.sqrt(np.mean(e[:, 0] ** 2))), float(np.sqrt(np.mean(e[:, 1] ** 2)))

    def status(self):
        st, sr = self.spread()
        return {"samples": len(self.pool), "holds": len(self.holds), "t_std_mm": self.t_std * 1000,
                "rot_std_deg": self.rot_std, "spread_mm": st * 1000, "spread_deg": sr,
                "reproj_b_px": float(np.mean(self.reproj)) if self.reproj else None,
                "stable": self.stable, "locked": self.locked is not None}
