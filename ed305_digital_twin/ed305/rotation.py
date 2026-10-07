"""How a camera turned, from its own video.

When a camera turns about a point close to its lens, two of its frames
differ by a pure rotation: a scene point's viewing direction b (unit
vector, camera frame) moves as b_t = R b_0, whatever the scene's depth.
So R comes from matched features alone: undistort, turn pixels into
directions with K, fit the rotation (Kabsch) to the RANSAC inliers.

Frames are matched to a keyframe, not to the previous frame, so the error
does not grow with the frame rate; a new keyframe is taken when the overlap
with the old one runs low. Drift still adds up over keyframes, which is why
the video should end back where it started: the end-to-start angle is the
check.
"""
from dataclasses import dataclass

import cv2
import numpy as np


def kabsch(a, b):
    """Rotation R minimising |b - R a| over rows of unit vectors a, b."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


@dataclass
class Step:
    frame: int
    t: float
    R: np.ndarray          # camera at this frame -> camera at the first frame
    inliers: int
    keyframe: bool


class RotationTracker:
    def __init__(self, K, dist=None, scale=1.0, max_features=4000, min_inliers=60, rekey_inliers=250,
                 rekey_angle=12.0, ransac_deg=0.15):
        """scale: frames are resized by this before feature detection (K follows)."""
        self.scale = scale
        self.K = np.asarray(K, float).copy()
        self.K[:2] *= scale
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_inliers = min_inliers
        self.rekey_inliers = rekey_inliers
        self.rekey_angle = rekey_angle
        # RANSAC threshold on normalized coordinates: about ransac_deg of angle
        self.thresh = np.tan(np.radians(ransac_deg))
        self.key = None            # (keypoint bearings, descriptors, R_first_from_key)

    def _features(self, img):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)   # dim room: lift contrast for the detector
        kps, des = self.sift.detectAndCompute(g, None)
        if des is None or len(kps) < 8:
            return None, None
        px = np.array([k.pt for k in kps], np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(px, self.K, self.dist).reshape(-1, 2)
        b = np.column_stack([n, np.ones(len(n))])
        return b / np.linalg.norm(b, axis=1, keepdims=True), des

    def _match(self, b0, d0, b1, d1):
        """Rotation taking key directions b0 to frame directions b1, and its inlier count."""
        pairs = self.matcher.knnMatch(d0, d1, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < self.min_inliers:
            return None, len(good)
        a = b0[[m.queryIdx for m in good]]
        c = b1[[m.trainIdx for m in good]]
        # RANSAC on a homography of normalized points (pure rotation = homography), then Kabsch on inliers
        _, mask = cv2.findHomography(a[:, :2] / a[:, 2:], c[:, :2] / c[:, 2:], cv2.RANSAC, self.thresh,
                                     maxIters=4000, confidence=0.999)
        if mask is None:
            return None, 0
        inl = mask.ravel().astype(bool)
        if inl.sum() < self.min_inliers:
            return None, int(inl.sum())
        R = kabsch(a[inl], c[inl])
        # re-select on the rotation itself and refit
        err = np.linalg.norm(c - a @ R.T, axis=1)
        inl = err < 2 * self.thresh
        if inl.sum() < self.min_inliers:
            return None, int(inl.sum())
        return kabsch(a[inl], c[inl]), int(inl.sum())

    def track(self, frames, fps=30.0, every=1, log=None):
        """frames: iterable of images. Yields Step per used frame; skips frames it cannot place."""
        for i, img in enumerate(frames):
            if i % every:
                continue
            b, d = self._features(img)
            if b is None:
                if log:
                    log(f"frame {i}: no features")
                continue
            if self.key is None:
                self.key = (b, d, np.eye(3))
                yield Step(i, i / fps, np.eye(3), len(b), True)
                continue
            kb, kd, R_first_key = self.key
            R_frame_key, n = self._match(kb, kd, b, d)
            if R_frame_key is None:
                if log:
                    log(f"frame {i}: lost ({n} matches); turn slower or add texture to the view")
                continue
            R_first_frame = R_first_key @ R_frame_key.T
            rekey = n < self.rekey_inliers or angle_deg(R_frame_key) > self.rekey_angle
            if rekey:
                self.key = (b, d, R_first_frame)
            yield Step(i, i / fps, R_first_frame, n, rekey)


def video_frames(src):
    """Frames from a file or an rtsp:// URL, and the frame rate."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                yield img
        finally:
            cap.release()
    return gen(), fps
