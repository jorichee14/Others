#!/usr/bin/env python3
"""Fine-tune YOLO on the room cameras' frames, and compare with the model it started from.

    python train_yolo.py data/overhead/data.yaml
    python train_yolo.py data/overhead/data.yaml --model yolo26m.pt --epochs 150 --name ed305_m
    python train_yolo.py data/overhead/data.yaml --set freeze=10 --set lr0=0.002

1. Scores the pretrained --model on val, for the classes it has by name
   (COCO: person), so "did fine-tuning help" has an answer.
2. Fine-tunes it (ultralytics), every class in data.yaml.
3. Scores the best checkpoint on the same val, writes compare.md.

Augmentation defaults for wall/ceiling cameras: left-right flips yes,
upside-down flips no (the cameras look down at an angle, so an upside-down
person never happens), little rotation. --set key=value passes anything
else to ultralytics' train().

Output: runs/detect/<name>/ (weights/best.pt, ultralytics' plots), compare.md there.
"""
import argparse
from pathlib import Path

import yaml

from ed305.yolo_data import class_map

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="data.yaml from make_dataset.py")
    ap.add_argument("--model", default="yolo26s.pt", help="starting weights (default yolo26s.pt, COCO)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="default 1280: people far from the camera are small")
    ap.add_argument("--batch", type=float, default=-1, help="-1: as many as fit in GPU memory (default)")
    ap.add_argument("--device", default=None, help="e.g. 0 or 0,1 or cpu (default: auto)")
    ap.add_argument("--name", default="ed305")
    ap.add_argument("--project", type=Path, default=HERE / "runs/detect")
    ap.add_argument("--no-baseline", action="store_true", help="skip scoring the pretrained model")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra ultralytics train() argument")
    return ap.parse_args()


def scores(metrics, names):
    """{class name: (precision, recall, AP50, AP50-95)} for the classes val had boxes of."""
    out = {}
    for j, c in enumerate(metrics.ap_class_index):
        out[names[int(c)]] = tuple(float(v) for v in metrics.box.class_result(j))
    return out


def validate(model, data, imgsz, device, classes=None):
    names = yaml.safe_load(Path(data).read_text())["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    m = model.val(data=str(data), imgsz=imgsz, device=device, classes=classes, plots=False, verbose=False)
    return scores(m, {int(k): v for k, v in names.items()})


def value(s):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def main():
    a = parse()
    if not a.data.exists():
        raise SystemExit(f"{a.data} does not exist: run make_dataset.py on the dataset folder first")
    from ultralytics import YOLO

    data = yaml.safe_load(a.data.read_text())
    names = data["names"] if isinstance(data["names"], dict) else dict(enumerate(data["names"]))
    names = {int(k): v for k, v in names.items()}

    before = {}
    if not a.no_baseline:
        base = YOLO(a.model)
        # The pretrained model's predictions count only where its class index
        # and name both match ours (COCO person = 0); other classes are not scored.
        same = sorted(k for k, v in class_map(base.names, [names[i] for i in sorted(names)]).items() if k == v)
        if same:
            before = {k: v for k, v in validate(base, a.data, a.imgsz, a.device, same).items()
                      if k in {names[i] for i in same}}
        else:
            print(f"{a.model} has no class at the same index and name as data.yaml: no baseline")

    extra = {k: value(v) for k, v in (s.split("=", 1) for s in a.set)}
    args = dict(data=str(a.data.resolve()), epochs=a.epochs, imgsz=a.imgsz,
                batch=int(a.batch) if a.batch == int(a.batch) else a.batch, device=a.device,
                project=str(a.project.resolve()), name=a.name, exist_ok=False, patience=30,
                fliplr=0.5, flipud=0.0, degrees=5.0, plots=True)
    args.update(extra)
    model = YOLO(a.model)
    model.train(**args)
    run = Path(model.trainer.save_dir)
    best = run / "weights/best.pt"
    after = validate(YOLO(best), a.data, a.imgsz, a.device)

    lines = [f"# {a.name}: {a.model} fine-tuned", "",
             f"data `{a.data}`, imgsz {a.imgsz}, epochs {a.epochs}, best `{best}`", "",
             "| class | | precision | recall | AP50 | AP50-95 |", "|---|---|---|---|---|---|"]
    for c in [names[i] for i in sorted(names)]:
        for tag, d in (("pretrained", before), ("fine-tuned", after)):
            if c in d:
                p, r, ap50, ap = d[c]
                lines.append(f"| {c} | {tag} | {p:.3f} | {r:.3f} | {ap50:.3f} | {ap:.3f} |")
            elif tag == "fine-tuned" or before:
                lines.append(f"| {c} | {tag} | - | - | - | - |")
    text = "\n".join(lines) + "\n"
    (run / "compare.md").write_text(text)
    print("\n" + text + f"\n{run / 'compare.md'}")


if __name__ == "__main__":
    main()
