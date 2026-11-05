"""Frames, labels and splits for fine-tuning YOLO on the room cameras' recordings.

No torch here: sampling a video, YOLO label lines, matching a model's classes
to ours by name, and the train / val split. The command-line tools are
extract_frames.py, prelabel.py, make_dataset.py and train_yolo.py.

A dataset folder (``root``) looks like
    images/<video stem>_t<ms>.jpg   sampled frames
    labels/<same stem>.txt          YOLO labels: cls cx cy w h, normalised
    frames.csv                      image, camera, video, t (s)
    classes.txt                     one class name per line, index = line
"""
import csv
import re
from pathlib import Path

import cv2
import numpy as np

FRAMES_CSV = "frames.csv"
FIELDS = ["image", "camera", "video", "t"]


def camera_of(path):
    """'cam_7' from a file name like cam_7_2026-11-03.mp4 or cam7.mp4; else the stem."""
    m = re.search(r"cam[_-]?(\d+)", Path(path).name, re.I)
    return f"cam_{int(m.group(1))}" if m else Path(path).stem


def thumb(frame, size=(160, 90)):
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(g, size, interpolation=cv2.INTER_AREA), (3, 3), 0)


def change_fraction(a, b, thresh=15):
    """Fraction of thumbnail pixels that changed by more than thresh grey levels."""
    return float(np.mean(cv2.absdiff(a, b) > thresh))


def sample_video(path, every=1.0, min_change=0.005, max_frames=None):
    """Yield (t seconds, BGR frame): one frame per `every` seconds, skipping
    frames where under `min_change` of the image changed since the last kept
    one (a still room gives the same picture again and again)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 30.0
    step = max(1, round(every * fps))
    last, kept, i = None, 0, -1
    try:
        while cap.grab():
            i += 1
            if i % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            th = thumb(frame)
            if last is not None and change_fraction(th, last) < min_change:
                continue
            last = th
            yield i / fps, frame
            kept += 1
            if max_frames and kept >= max_frames:
                break
    finally:
        cap.release()


def read_frames_csv(root):
    p = Path(root) / FRAMES_CSV
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return [dict(r, t=float(r["t"])) for r in csv.DictReader(f)]


def append_frames_csv(root, rows):
    p = Path(root) / FRAMES_CSV
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def yolo_line(cls, xyxy, w, h):
    """One YOLO label line from a pixel box, clipped to the image."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    x0, x1 = np.clip([x0, x1], 0, w)
    y0, y1 = np.clip([y0, y1], 0, h)
    return f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def read_labels(path):
    """[(cls, cx, cy, w, h)] from a YOLO label file; [] if it is missing or empty."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().split("\n"):
        v = line.split()
        if len(v) >= 5:
            out.append((int(v[0]), *map(float, v[1:5])))
    return out


def class_map(model_names, classes):
    """{model class index: our class index} for classes with the same name.

    model_names is a model's .names ({0: 'person', ...} or a list). Our classes
    a model does not have are not in the map: they are labelled by hand (or by
    a model fine-tuned on them, whose names then match)."""
    if not isinstance(model_names, dict):
        model_names = dict(enumerate(model_names))
    ours = {c.strip().lower(): i for i, c in enumerate(classes)}
    return {int(k): ours[v.lower()] for k, v in model_names.items() if v.lower() in ours}


def split_rows(rows, val_cameras=(), val_frac=0.2, gap=10.0):
    """'train', 'val' or 'skip' per row.

    With val_cameras, those cameras are val (a view the model never saw) and
    the rest train. Otherwise each video's last val_frac of its sampled time
    is val, and train frames within `gap` seconds before that are skipped:
    neighbouring frames are near copies, and a random split would put copies
    of the val frames in train.
    """
    val_cameras = set(val_cameras)
    if val_cameras:
        return ["val" if r["camera"] in val_cameras else "train" for r in rows]
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r["t"])
    cut = {}
    for v, ts in by_video.items():
        ts = sorted(ts)
        n_val = int(round(len(ts) * val_frac))
        cut[v] = ts[len(ts) - n_val] if n_val > 0 else float("inf")
    out = []
    for r in rows:
        c = cut[r["video"]]
        out.append("val" if r["t"] >= c else "skip" if r["t"] > c - gap else "train")
    return out
