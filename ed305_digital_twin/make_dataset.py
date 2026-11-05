#!/usr/bin/env python3
"""Train / val split and data.yaml for the labelled frames.

    python make_dataset.py data/overhead                          # last 20% of each video is val
    python make_dataset.py data/overhead --val-cameras cam_3,cam_12   # those cameras are val

Split by time or by camera, never frame by frame: frames a second apart are
near copies, so a random split puts the val frames in train too and the val
score says nothing. By time, train frames within --gap seconds before the
val part are left out. By camera, val measures a view the model never saw
(what a new or re-aimed camera will get).

Only frames with a labels/<stem>.txt are used. Val labels must be reviewed:
scored against raw pre-labels, the score only says how much the model agrees
with the pre-labelling model.

Writes to the dataset folder: train.txt, val.txt, data.yaml (for train_yolo.py).
"""
import argparse
from collections import Counter
from pathlib import Path

import yaml

from ed305.yolo_data import read_frames_csv, read_labels, split_rows


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--val-cameras", default="", help="comma-separated cameras held out as val")
    ap.add_argument("--val-frac", type=float, default=0.2, help="by time: share of each video for val (default 0.2)")
    ap.add_argument("--gap", type=float, default=10.0, help="by time: seconds left out before val (default 10)")
    return ap.parse_args()


def main():
    a = parse()
    root = a.root.resolve()
    classes = [c for c in (root / "classes.txt").read_text().splitlines() if c.strip()]
    rows = read_frames_csv(root)
    labelled = [r for r in rows if (root / "labels" / f"{Path(r['image']).stem}.txt").exists()]
    if len(labelled) < len(rows):
        print(f"{len(rows) - len(labelled)} frames have no labels yet and are left out")
    val_cams = [c.strip() for c in a.val_cameras.split(",") if c.strip()]
    unknown = set(val_cams) - {r["camera"] for r in labelled}
    if unknown:
        raise SystemExit(f"no labelled frames from {sorted(unknown)}")
    split = split_rows(labelled, val_cams, a.val_frac, a.gap)

    lists = {"train": [], "val": []}
    cams = {"train": Counter(), "val": Counter()}
    boxes = {"train": Counter(), "val": Counter()}
    for r, s in zip(labelled, split):
        if s == "skip":
            continue
        lists[s].append(str(root / "images" / r["image"]))
        cams[s][r["camera"]] += 1
        for b in read_labels(root / "labels" / f"{Path(r['image']).stem}.txt"):
            boxes[s][b[0]] += 1
    if not lists["train"] or not lists["val"]:
        raise SystemExit(f"train {len(lists['train'])}, val {len(lists['val'])} frames: need both")

    for s, paths in lists.items():
        (root / f"{s}.txt").write_text("\n".join(paths) + "\n")
    data = {"path": str(root), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(classes))}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    for s in ("train", "val"):
        per_cls = ", ".join(f"{classes[c]} {n}" for c, n in sorted(boxes[s].items())) or "no boxes"
        per_cam = ", ".join(f"{c} {n}" for c, n in sorted(cams[s].items()))
        print(f"{s}: {len(lists[s])} frames ({per_cls})\n  {per_cam}")
    print(f"skipped (gap): {split.count('skip')}\n{root / 'data.yaml'}")


if __name__ == "__main__":
    main()
