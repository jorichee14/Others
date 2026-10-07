import cv2
import numpy as np

from ed305.camera import Camera
from ed305.placement import aim_from_pose, pose_from_aim, wrap
from ed305.rotation import RotationTracker

W, H, F = 640, 360, 500.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
BW, BH, BF = 2400, 1600, 693.0       # wide virtual view the frames are cut from
KB = np.array([[BF, 0, BW / 2], [0, BF, BH / 2], [0, 0, 1]])


def texture(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((BH, BW), np.float32)
    for s in (3, 9, 27):
        n = rng.random((BH // s + 1, BW // s + 1)).astype(np.float32)
        img += cv2.resize(n, (BW, BH), interpolation=cv2.INTER_CUBIC) / s ** 0.3
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for _ in range(600):
        c = (int(rng.integers(0, BW)), int(rng.integers(0, BH)))
        cv2.circle(img, c, int(rng.integers(3, 25)), int(rng.integers(0, 256)), -1)
    return img


def turn_sequence():
    pans = list(np.linspace(0, 25, 26)) + list(np.linspace(25, -20, 46)) + list(np.linspace(-20, 0, 21))
    seq = [(p, 30.0) for p in pans]
    tilts = list(np.linspace(30, 42, 13)) + list(np.linspace(42, 18, 25)) + list(np.linspace(18, 30, 13))
    return seq + [(0.0, t) for t in tilts]


def test_recovers_pan_and_tilt_range():
    base = texture()
    h0 = 20.0
    R_world_first = pose_from_aim([0, 0, 2.6], h0, 30)[:3, :3]
    frames = []
    for pan, tilt in turn_sequence():
        R_world_frame = pose_from_aim([0, 0, 2.6], h0 + pan, tilt)[:3, :3]
        R_first_frame = R_world_first.T @ R_world_frame
        Hfb = KB @ R_first_frame @ np.linalg.inv(K)       # frame pixel -> base pixel
        frames.append(cv2.warpPerspective(base, Hfb, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))
    steps = list(RotationTracker(K).track(frames))
    assert len(steps) == len(frames)
    pans, tilts = [], []
    for st in steps:
        T = np.eye(4)
        T[:3, :3] = R_world_first @ st.R
        h, t, r = aim_from_pose(T)
        pans.append(wrap(h - h0))
        tilts.append(t)
        assert abs(r) < 0.5
    assert abs(min(pans) + 20) < 0.5 and abs(max(pans) - 25) < 0.5
    assert abs(min(tilts) - 18) < 0.5 and abs(max(tilts) - 42) < 0.5
    assert abs(pans[-1]) < 0.5 and abs(tilts[-1] - 30) < 0.5      # back at the start
