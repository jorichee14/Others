#!/usr/bin/env python3
"""Look at the labels: frames with their boxes drawn, a sheet per camera.

    python show_labels.py data/overhead                  # 6 random frames per camera
    python show_labels.py data/overhead --per-camera 12
    python show_labels.py data/overhead --unsure 24      # the frames review.csv puts first

Writes <root>/check/<camera>.jpg (and unsure.jpg). Look for what goes wrong
again and again: people missed in one camera, boxes on chairs, bags or
screens, one box over two people. Those are worth fixing before training.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from ed305.yolo_data import read_frames_csv, read_labels

TILE = (640, 360)


def draw(root, image, classes):
    img = cv2.imread(str(root / "images" / image))
    if img is None:
        return np.zeros((TILE[1], TILE[0], 3), np.uint8)
    h, w = img.shape[:2]
    boxes = read_labels(root / "labels" / f"{Path(image).stem}.txt")
    for c, cx, cy, bw, bh in boxes:
        p0 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
        p1 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
        cv2.rectangle(img, p0, p1, (60, 200, 60), 3)
        if len(classes) > 1:
            cv2.putText(img, classes[c], (p0[0], p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 200, 60), 2)
    img = cv2.resize(img, TILE, interpolation=cv2.INTER_AREA)
    label = f"{image}  {len(boxes)} boxes" if (root / "labels" / f"{Path(image).stem}.txt").exists() else f"{image}  NO LABEL FILE"
    cv2.rectangle(img, (0, 0), (TILE[0], 24), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def sheet(path, tiles, cols=3):
    while len(tiles) % cols:
        tiles.append(np.full((TILE[1], TILE[0], 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--per-camera", type=int, default=6)
    ap.add_argument("--unsure", type=int, default=12, help="also a sheet of the first N frames in review.csv (0: none)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = a.root
    cls_file = root / "classes.txt"
    classes = cls_file.read_text().splitlines() if cls_file.exists() else ["person"]
    out = root / "check"
    out.mkdir(exist_ok=True)
    rng = random.Random(a.seed)

    by_cam = {}
    for r in read_frames_csv(root):
        by_cam.setdefault(r["camera"], []).append(r["image"])
    for cam, images in sorted(by_cam.items()):
        pick = sorted(rng.sample(images, min(a.per_camera, len(images))))
        sheet(out / f"{cam}.jpg", [draw(root, i, classes) for i in pick])
    print(f"{len(by_cam)} camera sheets in {out}")

    review = root / "review.csv"
    if a.unsure and review.exists():
        with open(review, newline="") as f:
            first = [r["image"] for r in csv.DictReader(f)][:a.unsure]
        if first:
            sheet(out / "unsure.jpg", [draw(root, i, classes) for i in first])
            print(f"{out / 'unsure.jpg'}: the {len(first)} frames with the most unsure boxes")


if __name__ == "__main__":
    main()
