#!/usr/bin/env python3
"""Draft labels from a pretrained YOLO, to correct by hand instead of drawing every box.

    python prelabel.py data/overhead                              # person, COCO yolo26x
    python prelabel.py data/overhead --classes person,robot
    python prelabel.py data/overhead --model runs/detect/ed305/weights/best.pt --classes person,robot

The model's classes are matched to --classes by name. A COCO model knows
"person" but not "robot": robots are drawn by hand in the first round, and
from the second round on a model fine-tuned here pre-labels them too.

Never overwrites a label file that exists (it may be reviewed already):
only frames without labels/<stem>.txt are labelled. --overwrite to redo all.

Writes to the dataset folder:
  labels/<stem>.txt   YOLO boxes (an empty file = nothing in the frame)
  classes.txt         --classes, one per line
  review.csv          per new frame: boxes, unsure boxes (conf < --unsure); review those first
"""
import argparse
import csv
from pathlib import Path

from ed305.yolo_data import class_map, yolo_line


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder from extract_frames.py")
    ap.add_argument("--model", default="yolo26x.pt", help="weights (default yolo26x.pt: COCO, largest)")
    ap.add_argument("--classes", default="person", help="comma-separated, in index order (default person)")
    ap.add_argument("--conf", type=float, default=0.25, help="keep boxes above this (default 0.25)")
    ap.add_argument("--unsure", type=float, default=0.5, help="boxes below this are flagged in review.csv")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size (default 1280: far people are small)")
    ap.add_argument("--device", default=None, help="e.g. 0, cpu (default: auto)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="also redo frames that have labels")
    return ap.parse_args()


def main():
    a = parse()
    from ultralytics import YOLO

    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    lbl_dir = a.root / "labels"
    lbl_dir.mkdir(exist_ok=True)
    cls_file = a.root / "classes.txt"
    if cls_file.exists() and cls_file.read_text().splitlines() != classes:
        raise SystemExit(f"{cls_file} has {cls_file.read_text().splitlines()}, not {classes}: "
                         "class indices would mix up. Use the same --classes, or start a new folder.")
    cls_file.write_text("\n".join(classes) + "\n")

    model = YOLO(a.model)
    cmap = class_map(model.names, classes)
    missing = [c for i, c in enumerate(classes) if i not in cmap.values()]
    print(f"{a.model}: pre-labels {[classes[i] for i in sorted(set(cmap.values()))]}"
          + (f"; label by hand: {missing}" if missing else ""))
    if not cmap:
        raise SystemExit("the model has none of these classes")

    images = sorted((a.root / "images").glob("*.jpg"))
    todo = [p for p in images if a.overwrite or not (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"{len(todo)} of {len(images)} frames to label")
    if not todo:
        return

    review = []
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        results = model.predict([str(p) for p in chunk], conf=a.conf, imgsz=a.imgsz, device=a.device,
                                classes=sorted(cmap), verbose=False)
        for p, r in zip(chunk, results):
            h, w = r.orig_shape
            lines, unsure = [], 0
            for c, xyxy, conf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                lines.append(yolo_line(cmap[int(c)], xyxy, w, h))
                unsure += conf < a.unsure
            (lbl_dir / f"{p.stem}.txt").write_text("".join(l + "\n" for l in lines))
            review.append({"image": p.name, "boxes": len(lines), "unsure": unsure})
        print(f"  {min(i + a.batch, len(todo))}/{len(todo)}", end="\r")
    print()

    n_box = sum(r["boxes"] for r in review)
    n_unsure = sum(r["unsure"] for r in review)
    rp = a.root / "review.csv"
    if rp.exists():                      # keep earlier rounds' rows for frames not redone here
        done = {r["image"] for r in review}
        with open(rp, newline="") as f:
            review += [dict(r, boxes=int(r["boxes"]), unsure=int(r["unsure"]))
                       for r in csv.DictReader(f) if r["image"] not in done]
    review.sort(key=lambda r: (-r["unsure"], -r["boxes"]))
    with open(rp, "w", newline="") as f:
        w = csv.DictWriter(f, ["image", "boxes", "unsure"])
        w.writeheader()
        w.writerows(review)
    print(f"{n_box} boxes, {n_unsure} below conf {a.unsure}; review list: {rp}")


if __name__ == "__main__":
    main()
