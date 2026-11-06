# Room tracker: one identity per person, from any sensors

`room_track.py`, `ed305/tracking/`. Identity is a person's path through the
room, in the room frame and on one clock, estimated from every sensor at once.
It is solved as a data-association problem, not frame by frame in each camera.

## The problem

All measurements from all sensors are grouped into paths (one per person) or
explained as clutter. The best grouping maximises

    prod over paths [ p_start * prod p_motion * p_appearance * p_detected * p_end ] * prod p_clutter

subject to: each measurement used once, one per sensor per scan per path,
paths start and end in entry zones (the door) unless at the recording's edges.
In -log form it is a minimum-cost partition. Every piece of evidence is one more
cost term, and every sensor enters through the same terms.

## How it is solved

| stage | what | how |
|---|---|---|
| 1. tracklets (online) | short paths without doubt | Kalman filter per person on the floor (constant velocity), each scan matched to the tracks by an optimal assignment on log-likelihood ratios; a report that fits two tracks almost equally is not used; tracks end when their evidence drops |
| 2. linking (global) | tracklets -> identities | one assignment over all tracklets, end -> next start (gap closing as min-cost flow, solved exactly): motion over the gap, appearance, identity, "could they have stayed unseen that long", against ending here and starting there |

The costs, per sensor `s` and track at `x`:

    update   log P_D,s(x) + log N(z; Hx, S) - log clutter_s   [+ appearance] [+ identity]
    miss     log(1 - P_D,s(x))
    new      -log(birth_density / clutter_s)

so a miss only counts where that sensor should have seen the person, and a noisy
sensor needs more reports before it can start a track.

## The sensor contract

A sensor gives `Measurement`s and has a `SensorModel` (`ed305/tracking/model.py`):

| | |
|---|---|
| `Measurement` | `t` (common clock), `sensor`, `xy` floor position in the room frame, `cov` its 2x2 covariance; optional `feat` (appearance), `ident` (known identity) |
| `SensorModel` | `p_detect`, `p_detect_static` + `min_speed` (Doppler radar loses people who stand still; averaged over how sure the speed is), `coverage` (where it sees), `clutter` (false reports per m^2 per scan), `ident` (a self-reporting sensor only sees its own target) |

Adding a sensor = writing its measurements (`ed305/tracking/adapters.py`: a
function, or the CSV format `t,sensor,x,y,sxx,sxy,syy[,ident][,f0,...]`) and
its entry in a sensors YAML (`configs/sensors_example.yaml`). The tracker does
not change. In ED305:

| sensor | adapter | notes |
|---|---|---|
| 16 room cameras | `people_to_measurements()` from `multiview.py` (+ ReID features) | position noise shrinks with the number of agreeing cameras; coverage from the calibration |
| mmWave radar | `polar_measurement()` per cluster | `p_detect_static` low, coarse, many false points, no appearance |
| robots (pose) | `pose_measurement()` | identity known: the robot's path is fixed, nobody else can take it |
| robot ZED / LiDAR detections | `Measurement` in the room frame via the robot pose | add the robot's pose uncertainty to `cov` |

Appearance: the model (how similar features of the same / different people
are) is fitted from the data itself (`fit_appearance()`): features within a
tracklet are the same person, tracklets at the same time far apart are different
people. No labels needed.

## Simulated room (known identities)

`python room_track.py --sim dense|sparse|fused --out out/sim` (`ed305/tracking/sim.py`):
8 people coming in through the door, walking, sitting 3-20 s, some leaving;
2 of them in near-identical clothes; a robot reporting its pose. Mean over 6 seeds:

| scenario | tracklets only (stage 1) | room tracker | switches | tracks (truth 9) |
|---|---|---|---|---|
| dense cameras (10 Hz, 7 cm, misses, merged detections) | IDF1 0.74 | **0.97** | 14 | 9.7 |
| cameras + radar (half the room, 20 cm, loses static people, 1 false point per scan) | 0.62 | **0.98** | 17 | 9.5 |
| sparse cameras (2 m blind band through the room) | 0.41 | **0.77** | 28 | 9.0 |

What the simulations taught (each was a wrong turn first; `tests/test_room_tracker.py` keeps them):
- A sensor that reports one identity must not count as a miss for everyone else (it ended every track).
- Cutting tracks where people come close made identities worse in every case; holding back unclear reports is enough.
- Starting a new person right next to an existing track must be expensive for a sensor with much clutter (radar).
- A long gap is cheap if there is a blind spot within reach to have stayed in, not along a straight line.
- Adding the radar does not cost identity accuracy (0.98 with, 0.97 without on the same runs).

The sparse case is the hard one: people who sit in the blind band for tens of
seconds come back with only appearance to tell them apart; the near-identical
pair is mixed up there.

## On ED305 data

Needs: the cameras calibrated in the room frame (`cameras.json`, floor z = 0)
and the recordings on one clock (`cam_N_times.csv` next to each video). Two ways
to get the times files:

| | how | when |
|---|---|---|
| `motion_sync.py rec/` | from the motion the cameras see together: people moving change several cameras at the same instant; cross-correlation per camera pair, one least-squares fit over all pairs (`ed305/sync.py`) | recordings you already have; needs people moving, offsets up to `--max-lag` (default 10 s; raise it if the recordings were started by hand minutes apart) |
| `flash_sync.py` (calibration repo) | lights off and on once while recording | new recordings |

```bash
python motion_sync.py ~/Downloads/yolo_20261105_10 --check-drift
```

It prints each camera's offset and how well the camera pairs agree (residual in
ms; a frame is 33 ms). A camera that shares no motion with the others is
reported NOT SYNCED and gets no times file. Tested on a synthetic 6-camera
scene with known start offsets: every offset within about 1 ms. One recording
per camera in the folder (move the `.mp4` copies out if the `.mkv` are there too).

```bash
pip install -r requirements.txt -r requirements-yolo.txt
python room_track.py --rec rec/ --calib results/extrinsics/cameras.json \
    --model runs/detect/ed305_v1/weights/best.pt --reid yolo26s-reid.onnx \
    --sensors configs/sensors_example.yaml --video cam_3 --out out/run1
```

Outputs `tracks.csv` (paths in metres), `boxes.csv` (every camera box with its
room identity), `cam_3_ids.mp4`, `summary.md`. Other sensors join with
`--measurements radar.csv robot_1.csv` and their models in `--sensors`.

Tested end to end on 16 rendered recordings (people cut from a real ED305 frame,
each camera starting at a different time, real detector and ReID): every person
one unbroken track. Not yet run on real recordings.

## Not yet

- Live: stage 1 is online; stage 2 runs on the whole recording. Live use
  re-runs stage 2 over a sliding window (seconds of delay, can still revise).
- Measured on real data: label identities for a 2 min clip (click each person's
  id) and score with `ed305.tracking.id_metrics` (IDF1, switches). Every change
  is then judged by that number.
- ReID fine-tuned on these cameras: the room tracker's confident tracks give the
  same-person crops for it.
