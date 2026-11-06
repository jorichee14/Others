# Fine-tuning YOLO on the room cameras

A detector for people (and robots) as the room cameras see them: from the
wall/ceiling, at a steep angle, often cut by the image edge, sitting behind
desks and each other. A COCO model already finds most of the people in these
views; where it fails is crowds (several people on the sofa merge or get
doubled), people half out of the frame or behind furniture, and anything not
in COCO (robots). Fine-tuning fixes those on our views.

```
calibrated + synced:  16 recordings ─► multiview_label.py ─────────────────────┐
                                                                               ├─► make_dataset.py ─► train_yolo.py
by hand:              recordings ─► extract_frames.py ─► prelabel.py ─► review ┘
```

```bash
pip install -r requirements.txt -r requirements-yolo.txt    # torch for your CUDA first
```

## Person labels without hand labelling: all 16 cameras at once

With the cameras calibrated (room frame, floor at z = 0) and the recordings on
one clock, the cameras check each other instead of a person checking the
boxes. `multiview_label.py` replaces steps 1-3 below for people:

```bash
python multiview_label.py rec/ --calib results/extrinsics/cameras.json --out data/overhead --preview 20
python make_dataset.py data/overhead --val-cameras cam_3,cam_12
python train_yolo.py data/overhead/data.yaml
```

`rec/` is one recording per camera made at the same time (`cam_N.mkv`), plus
the `cam_N_times.csv` that the calibration repo's `flash_sync.py` writes
(switch the lights off and on once while recording). `cameras.json` is the
calibration's `results/extrinsics/cameras.json`.

How it decides, per time step (every `--every` s, default 1):

1. The pretrained detector runs on every camera's frame with a low threshold (0.05).
2. A person at floor point (x, y) has a known box in every camera (an upright
   cylinder: sitting, mid or standing size, computed once per floor cell).
   A floor point that boxes from 2+ cameras agree with becomes a person, and
   those boxes are taken; repeat. Taking the boxes out is what keeps two
   people's crossing view lines from making a third, "ghost" person.
3. Per camera:

| box | label |
|---|---|
| agrees with a person 2+ cameras see (any score) | kept: this is where it gains over one camera, unsure boxes of real people are confirmed |
| no other camera agrees, score >= 0.6 | kept (a spot only this camera sees), unless 3+ other cameras see that spot clearly and found nobody: dropped (a person on a screen, a coat) |
| no other camera agrees, score < 0.6 | dropped |
| a person other cameras agree on, clearly in view here (whole box in the image, not behind someone nearer), no box | **missed**: that frame gets no label file, so it is not trained on with a missing person (`--add-missed` writes the expected box instead) |

Outputs next to the usual dataset files: `multiview.csv` (per frame: kept,
dropped, missed, labelled or left out), `people.csv` (every person's floor
position per time step: floor-plane ground truth for the camera-only
baselines in PLAN phase 5) and, with `--preview N`, `preview/step_*.jpg`:
all cameras with green = kept, red = dropped, yellow = missed, and the floor
view. **Look at the previews first**: people on the floor view where they
are in the room, and green boxes on the people, mean calibration and sync are
good enough. Boxes consistently beside people, or many misses, mean a camera's
calibration or its time offset is off.

What it cannot fix: people no camera's detector finds (it only confirms and
fills in), and a person hidden behind furniture in one camera counts as a miss
there (furniture is not modelled), which only leaves that frame out. Lower
`--single-conf` keeps more boxes seen by one camera only; `--min-views 3` is
stricter about what counts as a person.

The robot was standing still in these recordings, so it is not labelled:
these labels are person only (classes.txt = person).

## Labels by hand (one camera, or no calibration)

### 1. Frames

```bash
python extract_frames.py rec/cam_*.mp4 --out data/overhead
```

One frame per second, only when enough of the picture changed (a still room
is the same image over and over). The camera name comes from the file name
(`cam_7_....mp4` -> `cam_7`). Aim for frames from every camera, empty room
included (empty frames teach it not to fire on chairs and bags), with the
lights as they are used. 1000-3000 frames is a good first round; more
cameras beats more minutes from one camera.

### 2. Pre-labels

```bash
python prelabel.py data/overhead --classes person,robot      # yolo26x, COCO
```

Writes `labels/<stem>.txt` for frames that have none (never overwrites one,
it may be reviewed). COCO has no robot: draw those by hand in round 1.
`review.csv` lists frames with the most unsure boxes first.

Look at them before going on (5 minutes):

```bash
python show_labels.py data/overhead          # check/<camera>.jpg, check/unsure.jpg
```

A camera where people are always missed, boxes on chairs or screens, one box
over two people: fix those first (lower `--conf`, or correct more frames).

### 3. Review (the part that matters)

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
python make_dataset.py data/overhead --val-cameras cam_3,cam_12 --val-frames 300   # only 300 val frames
```

`--val-frames` keeps that many val frames, evenly spread over time: those
are the ones to correct by hand (listed in `val.txt`); the rest of the val
cameras' frames are not used, so the val cameras stay unseen in training.

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

Multi-view: `tests/test_multiview.py` on the nominal 16-camera layout (people
found within 0.3 m, low-score boxes of real people confirmed, a lone box
dropped, a confident box on an empty floor refuted, no ghosts among 7 people),
and `multiview_label.py` end to end on 16 rendered recordings with people cut
from a real ED305 frame and each camera starting at a different time (with
`_times.csv`): the three people found by 13-14 cameras, within 5 cm of where
they were placed. Not run on real recordings yet.

Single camera: the pipeline end to end (extract, pre-label, both splits, a 2-epoch
CPU run, compare.md) on synthetic videos made from one ED305 frame; the
frame-sampling and split logic in `tests/test_yolo_data.py`. Not yet trained
on real recordings, so no numbers yet.
