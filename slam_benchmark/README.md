# slam_benchmark

A bench for comparing SLAM systems on the MIRC **coop2** recording, on two axes
at once: **trajectory accuracy** (ATE/RPE) and **map quality** (accuracy,
completeness, Chamfer, F-score) — across LiDAR, RGB-D, and, where it exists,
inertial input.

Adding a method is a config plus a Dockerfile. The evaluation never changes.

```bash
pip install -r requirements.txt
python3 scripts/test_slambench.py     # 32 self-tests, no bag, no ROS, no GPU
python3 scripts/smoke_e2e.py          # the whole pipeline on synthetic data

# once, per agent
python3 scripts/export_reference.py --config configs/coop2.yaml \
    --agent mobile_1 --out runs/coop2_20260828/reference/mobile_1.tum

# per method
python3 scripts/run_method.py  --config configs/coop2.yaml \
    --method configs/methods/kiss_icp.yaml --stream mobile_1.ouster --execute
python3 scripts/eval_run.py    --config configs/coop2.yaml \
    --method configs/methods/kiss_icp.yaml --stream mobile_1.ouster \
    --run runs/coop2_20260828/kiss_icp/mobile_1.ouster \
    --reference runs/coop2_20260828/reference/mobile_1.tum
python3 scripts/aggregate.py   --runs runs/coop2_20260828 \
    --config configs/coop2.yaml --out results/coop2.md
```

| | |
|---|---|
| `slambench/` | the evaluator: SE(3), trajectories, alignment, ATE/RPE, point clouds, map metrics, tables. numpy + pyyaml, nothing else |
| `configs/coop2.yaml` | the sequence: streams, extrinsics, what the reference is and how far it can be trusted |
| `configs/methods/` | one file per method — modality, stream, indoor re-parameterisation, and what the result will and will not support |
| `docker/` | one image per method; the interface is three files in `/out` |
| `scripts/` | export the reference, run a method, score a run, build the table |

## Which methods

The shortlist is seven entries, chosen so that each PAIR isolates one variable.
A benchmark of seven unrelated systems answers nothing; these answer four
questions.

| | method | modality | loop closure | isolates |
|---|---|---|---|---|
| **anchor** | **KISS-ICP** | LiDAR / depth | no | the sensor, by being the same estimator on all of them |
| | FAST-LIO2 | LiDAR + IMU | no | what the IMU buys over KISS-ICP |
| | GLIM | LiDAR + IMU | yes | what global optimisation buys over FAST-LIO2 |
| | RTAB-Map (LiDAR) | LiDAR | yes | loop closure *without* an IMU, against KISS-ICP |
| | RTAB-Map (RGB-D) | RGB-D | yes | one full SLAM system held fixed across sensors |
| | ORB-SLAM3 (RGB-D) | RGB-D | yes | features vs. geometry on the same depth stream |
| **ceiling** | reference accumulation | — | — | what the sequence's geometry supports given perfect poses |

**Benchmark on KISS-ICP.** Not because it is the most accurate — it will not be,
if an IMU exists — but because it is the only entry that can be the *reference
point* of this particular study. It needs no IMU, so it runs whatever the bag
turns out to contain; its published claim is to need no per-dataset tuning, so a
poor result indoors is a finding rather than a tuning failure; and it runs
unchanged on the Ouster, the ZED depth and the RealSense depth. That last
property is what makes the multi-modality axis possible at all. Every other
comparison you could draw on coop2 — LiDAR method vs. RGB-D method — varies the
estimator and the sensor in the same step and cannot attribute the difference to
either.

So the structure is: **KISS-ICP across all three geometry streams** gives you the
modality row. **Everything else on `mobile_1.ouster`** gives you the method row.
Do not merge the two into one ranking — the evaluator will not, either: which
comparison block a result lands in comes from the stream it ran on, not from the
method, so `kiss_icp` appears once in the LiDAR block and twice in the RGB-D one.

The cleanest pair on the whole sequence is `mobile_1.ouster` against
`mobile_1.zed_rgbd`: same platform, same motion, same clock, one reference, two
sensors. That is a LiDAR-vs-RGB-D comparison with nothing else varying, and it
is rare — most of that literature compares across datasets, or compares LiDAR
against *predicted* depth. The depth here is measured.

`reference_accumulation` is not a method and is not optional. It accumulates the
raw sweeps along the reference trajectory, and it is both the ceiling for every
map score and the check that the reference, the extrinsics and the deskew agree.
Run it first. If that cloud is smeared, nothing measured against it means
anything yet, and no method should be scheduled until it is sharp.

Radar is out. The IWR6843 returns order 10¹–10² points per frame; scan matching
on that indoors is a research contribution, not a baseline, and a failed radar
odometry row would describe the scoping rather than the sensor. The extrinsics
are in `configs/coop2.yaml` so they are not lost.

## Which environment

**Software.** ROS 2 Humble, one container per method, `--network none`, replay
from the mcap with `--clock`. Every container writes `trajectory.tum` and
`map.ply` into a bind-mounted run directory and that is the whole interface —
see `docker/README.md`. The evaluator is ROS-free Python, so a threshold can be
changed and every run re-scored in seconds without re-running anything. Nothing
needs a GPU except the optional learned entries.

**Site.** The harder question, and the answer is that **MIRC is the right place
to build this and the wrong place to draw the headline conclusion from.**

The room is ~16.6 m corner to corner and the sequence is 156 s. Any competent
LiDAR method will finish it with centimetre-scale drift, and the reference
trajectory's own uncertainty is 3–15 mm depending on which surveyed board you
anchor to. The methods will land inside, or barely outside, the band where the
reference can tell them apart. That is not a reason to skip MIRC — the surveyed
boards, the anchored map, the disciplined clocks and the third static observer
are exactly the instrumentation a SLAM benchmark needs, and almost no indoor
recording has them. It is a reason to know in advance what MIRC can and cannot
decide:

* MIRC can decide **map quality**. Accuracy and completeness against a surveyed
  map at 2 cm resolution discriminate hard, and they discriminate on precisely
  the output a collaborative-perception pipeline consumes.
* MIRC can decide **robustness**: which methods lose tracking on a turning
  pushcart, which need an IMU they do not have, which produce a map at all.
* MIRC **cannot** decide long-horizon drift, and a 156 s single-lap route may
  not be able to evaluate loop closure at all — check the tier-3 revisit count
  before claiming it did.

For the drift and loop-closure claims, record a second sequence designed for
them: the outdoor Guangfu site the dataset paper already scopes, a route of
several hundred metres with at least two deliberate revisits of one place
separated by minutes, and the same board-survey anchoring at the start and the
end. That sequence costs one afternoon and it is the difference between
"comparable on our sequence" and a drift number anyone can cite.

## Three things that decide whether any of this means anything

**1. The reference is a SLAM result.** `/mobile_*/global_pose` is the offline
mapping pipeline's stage-09 output. Scoring a SLAM system against it measures
agreement with that pipeline. A method that matches it perfectly has reproduced
it; a method that disagrees may be the better of the two and tier 1 cannot say.
So every run is reported in three tiers — ATE/RPE against the pipeline, the
independent check against the surveyed boards, and reference-free
self-consistency — and the evaluator daggers any ATE that falls below the
reference's own uncertainty rather than printing it as a win.

**2. Tier 2 is currently silent.** The board positions are in the config; the
*dwell windows* are not, because nothing in the bag states when each platform
stood in front of which board. Reading those three intervals off the run is an
hour of work and it is the highest-value hour in this whole project: without
them, no independent evidence enters the benchmark at all.

**3. Nobody has established whether this bag carries an IMU.** It decides
whether half the shortlist can run. `scripts/run_method.py` refuses to start an
inertial method while `streams.*.imu.present` is null rather than discovering it
forty minutes into a replay. Resolve it first:

```bash
python3 ../rosbag_to_opv2v/scripts/inspect_bag.py --bag <bag> | grep -i imu
```

If there is no IMU, that is a finding about the *recording* to fix before the
next one — not a gap to paper over with a synthetic IMU.

## Relationship to the sibling projects

`../rosbag_to_opv2v` owns the bag: its configs are where every extrinsic, topic
and rate in `configs/coop2.yaml` came from, and `scripts/export_reference.py`
reuses its bag reader rather than decoding the recording a second way. When that
project's config changes, this one changes in the same commit.

`../multimodal_v2x_benchmark` poses the *collaborative* tasks. Its T3
(cross-modal localisation) and the pose input every fusion operator depends on
are the same quantity this bench measures single-agent — which is the argument
for doing this first: T2 and T4 assume relative pose is solved, and nothing here
has yet established how well any method on this platform solves it.
