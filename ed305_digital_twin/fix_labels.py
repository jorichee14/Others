#!/usr/bin/env python3
"""Correct the boxes by hand, one frame after the other, in a simple window.

    python fix_labels.py data/overhead                 # the frames in val.txt
    python fix_labels.py data/overhead --list train.txt
    python fix_labels.py data/overhead --camera cam_3  # every frame of one camera

Mouse
  left drag            draw a new box
  right click in a box delete it (the smallest box under the pointer)
Keys
  d or space   save, next frame          a   save, previous frame
  z            undo on this frame        c   delete every box on this frame
  q or Esc     quit (start again later: it opens the first frame not done yet)

What to fix: every person gets a box, even if only partly visible (box the
visible part); nothing else gets one (chairs, bags, screens, reflections); two
people under one box: delete it, draw two.

Saves straight into labels/<frame>.txt and notes the frame in reviewed.txt.
Needs OpenCV with windows (opencv-python, not opencv-python-headless).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from ed305.yolo_data import read_frames_csv, read_labels, yolo_line

BAR = 44
GREEN, YELLOW, WHITE = (60, 200, 60), (0, 220, 255), (255, 255, 255)


class Editor:
    def __init__(self, root, images, max_w=1600, max_h=860, cls=0):
        self.root, self.images, self.cls = Path(root), images, cls
        self.max_w, self.max_h = max_w, max_h
        self.reviewed_file = self.root / "reviewed.txt"
        self.reviewed = set(self.reviewed_file.read_text().split()) if self.reviewed_file.exists() else set()
        self.i = next((k for k, p in enumerate(images) if p.stem not in self.reviewed), 0)
        self.drag = None
        self.load()

    # ------------------------------------------------------------- frames
    def label_path(self, p):
        return self.root / "labels" / f"{p.stem}.txt"

    def load(self):
        p = self.images[self.i]
        self.img = cv2.imread(str(p))
        if self.img is None:
            raise SystemExit(f"cannot read {p}")
        h, w = self.img.shape[:2]
        self.w, self.h = w, h
        self.s = min(self.max_w / w, self.max_h / h, 1.0)
        self.boxes = []                                   # [cls, x0, y0, x1, y1] in image pixels
        for c, cx, cy, bw, bh in read_labels(self.label_path(p)):
            self.boxes.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        self.undo = []
        self.drag = None

    def save(self, done=True):
        """Write the boxes; done=True also marks the frame checked (leaving it with d / a)."""
        p = self.images[self.i]
        self.label_path(p).parent.mkdir(exist_ok=True)
        self.label_path(p).write_text("".join(yolo_line(int(b[0]), b[1:], self.w, self.h) + "\n" for b in self.boxes))
        if done and p.stem not in self.reviewed:
            self.reviewed.add(p.stem)
            with open(self.reviewed_file, "a") as f:
                f.write(p.stem + "\n")

    def go(self, step):
        self.save()
        self.i = int(np.clip(self.i + step, 0, len(self.images) - 1))
        self.load()

    # ------------------------------------------------------------- editing
    def to_img(self, x, y):
        return x / self.s, (y - BAR) / self.s

    def change(self):
        self.undo.append([b[:] for b in self.boxes])

    def on_mouse(self, event, x, y, flags=0, param=None):
        ix, iy = self.to_img(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (ix, iy, ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag = (self.drag[0], self.drag[1], ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            x0, y0 = self.drag[:2]
            self.drag = None
            b = [min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy)]
            b = [float(np.clip(b[0], 0, self.w)), float(np.clip(b[1], 0, self.h)),
                 float(np.clip(b[2], 0, self.w)), float(np.clip(b[3], 0, self.h))]
            if (b[2] - b[0]) * self.s >= 5 and (b[3] - b[1]) * self.s >= 5:
                self.change()
                self.boxes.append([self.cls] + b)
        elif event == cv2.EVENT_RBUTTONDOWN:
            hits = [k for k, b in enumerate(self.boxes) if b[1] <= ix <= b[3] and b[2] <= iy <= b[4]]
            if hits:
                k = min(hits, key=lambda k: (self.boxes[k][3] - self.boxes[k][1]) * (self.boxes[k][4] - self.boxes[k][2]))
                self.change()
                del self.boxes[k]

    def key(self, k):
        """Returns False to quit."""
        if k in (ord("d"), ord(" ")):
            self.go(+1)
        elif k == ord("a"):
            self.go(-1)
        elif k == ord("z") and self.undo:
            self.boxes = self.undo.pop()
        elif k == ord("c") and self.boxes:
            self.change()
            self.boxes = []
        elif k in (ord("q"), 27):
            if self.undo:                      # changed: keep the changes, but this frame is not finished
                self.save(done=False)
            return False
        return True

    # ------------------------------------------------------------- drawing
    def render(self):
        img = cv2.resize(self.img, (round(self.w * self.s), round(self.h * self.s)), interpolation=cv2.INTER_AREA)
        lw = 2
        for b in self.boxes:
            p0 = (round(b[1] * self.s), round(b[2] * self.s))
            p1 = (round(b[3] * self.s), round(b[4] * self.s))
            cv2.rectangle(img, p0, p1, GREEN, lw)
        if self.drag:
            x0, y0, x1, y1 = self.drag
            cv2.rectangle(img, (round(x0 * self.s), round(y0 * self.s)), (round(x1 * self.s), round(y1 * self.s)), YELLOW, lw)
        bar = np.zeros((BAR, img.shape[1], 3), np.uint8)
        p = self.images[self.i]
        done = len(self.reviewed & {q.stem for q in self.images})
        cv2.putText(bar, f"{self.i + 1}/{len(self.images)}  {p.name}  {len(self.boxes)} boxes"
                         f"{'  (done before)' if p.stem in self.reviewed else ''}   done {done}/{len(self.images)}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.putText(bar, "drag: new box   right click: delete   d/space: next   a: back   z: undo   q: quit",
                    (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return np.vstack([bar, img])


def frame_list(root, list_file=None, camera=None):
    root = Path(root)
    if camera:
        names = [r["image"] for r in read_frames_csv(root) if r["camera"] == camera]
    else:
        lines = (root / (list_file or "val.txt")).read_text().split()
        names = [l.replace("\\", "/").split("/")[-1] for l in lines]   # works with paths written on another machine
    images = [root / "images" / n for n in names]
    missing = [p.name for p in images if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} frames not in {root / 'images'}, e.g. {missing[:3]}")
    if not images:
        raise SystemExit("no frames to show")
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder")
    ap.add_argument("--list", help="file with the frames to fix (default val.txt)")
    ap.add_argument("--camera", help="instead: every frame of this camera")
    ap.add_argument("--max-width", type=int, default=1600, help="window size limit in pixels (default 1600x860)")
    ap.add_argument("--max-height", type=int, default=860)
    a = ap.parse_args()
    ed = Editor(a.root, frame_list(a.root, a.list, a.camera), a.max_width, a.max_height)
    win = "fix_labels"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        raise SystemExit("this OpenCV has no windows: pip uninstall -y opencv-python-headless && "
                         "pip install --force-reinstall opencv-python")
    cv2.setMouseCallback(win, ed.on_mouse)
    while True:
        cv2.imshow(win, ed.render())
        k = cv2.waitKey(20) & 0xFF
        if k != 255 and not ed.key(k):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:      # window closed with the X
            ed.key(ord("q"))
            break
    cv2.destroyAllWindows()
    print(f"done {len(ed.reviewed & {p.stem for p in ed.images})} of {len(ed.images)} frames; "
          f"labels in {a.root / 'labels'}")


if __name__ == "__main__":
    main()
