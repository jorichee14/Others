"""Synthetic recordings for testing motion_sync: one world, several overlapping views, known start offsets."""
import cv2
import numpy as np


def make_scene(folder, offsets, seconds=40.0, fps=30.0, n_walkers=6, size=(320, 180), seed=0, views=None):
    """Writes cam_<i>.avi per offset (seconds this camera started later than the world clock).
    Returns {camera name: offset}."""
    rng = np.random.default_rng(seed)
    W, H = 640, 360
    bg = cv2.GaussianBlur(rng.integers(60, 200, (H, W, 3)).astype(np.uint8), (0, 0), 3)
    n = int((seconds + max(offsets) + 1) * fps)
    pos = rng.uniform([50, 50], [W - 50, H - 50], (n_walkers, 2))
    vel = rng.normal(0, 40, (n_walkers, 2))
    cols = rng.integers(0, 255, (n_walkers, 3))
    world = []
    for i in range(n):
        f = bg.copy()
        pause = (i // int(fps * 3)) % 3 == 0                       # people stop now and then: bursts of motion
        for k in range(n_walkers):
            if not (pause and k % 2):
                vel[k] += rng.normal(0, 30, 2) / fps
                pos[k] += vel[k] / fps
            for d in (0, 1):
                lim = (W, H)[d]
                if pos[k, d] < 30 or pos[k, d] > lim - 30:
                    vel[k, d] *= -1
                    pos[k, d] = np.clip(pos[k, d], 30, lim - 30)
            x, y = pos[k].astype(int)
            cv2.ellipse(f, (x, y), (14, 30), 0, 0, 360, tuple(int(c) for c in cols[k]), -1)
        world.append(f)
    views = views or [(0, 0), (160, 0), (320, 0), (0, 180), (160, 120), (320, 180)]
    out = {}
    for c, off in enumerate(offsets):
        x0, y0 = views[c % len(views)]
        name = f"cam_{c + 1}"
        vw = cv2.VideoWriter(str(folder / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        start = int(round(off * fps))
        for i in range(start, start + int(seconds * fps)):
            crop = world[i][y0:y0 + size[1], x0:x0 + size[0]].astype(np.int16)
            crop = crop + rng.normal(0, 4, crop.shape).astype(np.int16)
            vw.write(np.clip(crop, 0, 255).astype(np.uint8))
        vw.release()
        out[name] = off
    return out
