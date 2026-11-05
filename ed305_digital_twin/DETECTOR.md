# Fine-tuning YOLO on the room cameras

A detector for people (and robots) as the room cameras see them: from the
wall/ceiling, at a steep angle, often cut by the image edge, sitting behind
desks and each other. A COCO model already finds most of the people in these
views; where it fails is crowds (several people on the sofa merge or get
doubled), people half out of the frame or behind furniture, and anything not
in COCO (robots). Fine-tuning fixes those on our views.

```
recordings ─► extract_frames.py ─► prelabel.py ─► review by hand ─► make_dataset.py ─► train_yolo.py
                                       ▲  (CVAT / Label Studio)                            │
                                       └──────── next round: --model best.pt ◄─────────────┘
```

```bash
pip install -r requirements.txt -r requirements-yolo.txt    # torch for your CUDA first
```

## 1. Frames

```bash
python extract_frames.py rec/cam_*.mp4 --out data/overhead
```

One frame per second, only when enough of the picture changed (a still room
is the same image over and over). The camera name comes from the file name
(`cam_7_....mp4` -> `cam_7`). Aim for frames from every camera, empty room
included (empty frames teach it not to fire on chairs and bags), with the
lights as they are used. 1000-3000 frames is a good first round; more
cameras beats more minutes from one camera.

## 2. Pre-labels

```bash
python prelabel.py data/overhead --classes person,robot      # yolo26x, COCO
```

Writes `labels/<stem>.txt` for frames that have none (never overwrites one,
it may be reviewed). COCO has no robot: draw those by hand in round 1.
`review.csv` lists frames with the most unsure boxes first.

## 3. Review (the part that matters)

Training on raw pre-labels teaches the model the pre-labelling model's
mistakes. Import `images/` + `labels/` + `classes.txt` into Label Studio
(YOLO import) or CVAT (Ultralytics YOLO format), fix, export YOLO back into
`labels/`. Rules, so the boxes are the same across people:

- Every visible person, however little shows; box the visible part only.
- People behind a desk or monitor: box what is visible, don't guess the legs.
- Reflections and people on screens: no box.
- A frame with nobody: keep it, with an empty label file.

At least the val frames must be reviewed (see 4).

## 4. Split

```bash
python make_dataset.py data/overhead                           # by time: last 20% of each video
python make_dataset.py data/overhead --val-cameras cam_3,cam_12    # by camera
```

Never a random split: frames a second apart are near copies, so val would be
in train too. By camera answers "does it work on a camera it never saw"; by
time answers "does it work later on the same cameras".

## 5. Train

```bash
python train_yolo.py data/overhead/data.yaml                     # yolo26s, imgsz 1280
python train_yolo.py data/overhead/data.yaml --model yolo26m.pt --name ed305_m
```

Scores the pretrained model on val first, fine-tunes, scores again, and
writes `runs/detect/<name>/compare.md` (precision, recall, AP50, AP50-95 per
class, before and after). imgsz 1280 because people at the far wall are
small in 1920x1080. Upside-down flips are off (these cameras never see an
upside-down person). `--set key=value` passes anything else to ultralytics
(`--set freeze=10` with few frames, `--set batch=8` if memory runs out).

## 6. Next rounds

Pre-label new frames with the fine-tuned model (its class names match, so
it pre-labels robots too), review, add, retrain:

```bash
python extract_frames.py rec/new/*.mp4 --out data/overhead
python prelabel.py data/overhead --classes person,robot --model runs/detect/ed305/weights/best.pt
```

Add frames where the current model is wrong: the crowded sofa corner,
people at the image edge, robots among people.

## Tested

The pipeline end to end (extract, pre-label, both splits, a 2-epoch
CPU run, compare.md) on synthetic videos made from one ED305 frame; the
frame-sampling and split logic in `tests/test_yolo_data.py`. Not yet trained
on real recordings, so no numbers yet.
