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

## Three further tracks

The full specification — tasks, protocol, baselines, metrics, reporting rules
and limits — is `docs/BENCHMARK.md`. `configs/tracks.yaml` defines the tracks and
`docs/TRACK_PLANS.md` carries the algorithms, procedures and gates for each. Each is built around a metric the
single-agent table does not have, and each has a precondition that
`scripts/precheck_tracks.py` answers from the reference trajectories alone —
before any container starts.

**A — SLAM without a LiDAR.** The constrained agent is one that carries a depth
camera and nothing else. `mobile_1` carries the Ouster **and** the ZED on one
rigid body, so the headline is `mobile_1.ouster` against `mobile_1.zed_rgbd`:
one platform, one motion, one clock, one reference, and the only difference is
whether the agent had a LiDAR. Almost no dataset offers that pair — the usual
LiDAR-vs-RGB-D comparison is across datasets, or against *predicted* depth.
`mobile_2` is the real LiDAR-free platform and checks the conclusion generalises
off that one body.

KISS-ICP is the bridge, running on both sensors so its two rows differ in the
sensor and nothing else; ORB-SLAM3, RTAB-Map, BAD-SLAM and DROID-SLAM are the
RGB-D field. Nothing in the track needs an IMU, so it runs today in full.

Two instruments, because an RGB-D agent fails two ways and only one is
geometric: `slambench/observability.py` for what the 87° depth wedge stopped
constraining, and ORB feature density for what the image stopped offering. A
textureless wall at 2 m is geometrically fine and visually empty; a textured
scene at 15 m is the opposite.

The ablation grid (`configs/ablations.yaml`) is not the track — it is the
attribution. Once the gap is measured, degrade the Ouster toward the depth
camera's envelope until the bridge's result meets it, and the gap stops being a
number and becomes *"68% of it was the field of view, not the range"*.

**B — collaborative, two mobile agents.** The metric is not each agent's ATE. Two
trajectories can each be excellent and useless together if the transform between
them is wrong, and on coop2 the agents share a frame only because the reference's
anchoring put them there — a collaborative system has to estimate it. So the
headline is the error in `T_ab(t)`, which needs **no alignment at all**. It is
then split into the fixed part (a map-merge or place-recognition failure) and the
varying part (relative drift), because those have different fixes.

Methods: **Swarm-SLAM** (decentralised, LiDAR and RGB-D front-ends — the one
entry that takes this heterogeneous pair as it is) and **decoupled registration**
(each agent alone, then one offline map registration). The second is not
optional: it is the floor a joint system has to beat, and it still produces a
number on the day cross-modal place recognition finds nothing.

**C — infrastructure-anchored localization.** `infra_1` is static, surveyed, on
its own clock, watching the room from 2 m up. Two uses: a benchmark task (ego +
node at a known pose vs. ego alone — the interesting cell is the *constrained*
agent, since a single static observer constrains bearing well and range poorly,
close to complementary to a narrow-FoV depth camera), and instrumentation for
every other track, being the only **continuous** evidence that comes from
neither the reference nor the method.

```bash
python3 scripts/precheck_tracks.py --config configs/coop2.yaml \
    --reference-dir runs/coop2_20260828/reference --cloud sample_scan.ply
python3 scripts/eval_collab.py --config configs/coop2.yaml \
    --method configs/methods/swarm_slam.yaml \
    --run runs/coop2_20260828/swarm_slam \
    --reference-dir runs/coop2_20260828/reference --solo <a single-agent run>
```

Two preconditions are live questions, not formalities. The agents start 16.3 m
apart at opposite corners of a 16.6 m room, so **whether they ever shared a place
at all** decides whether track B measures collaboration or each method's fallback
— and if the answer is no, that is a finding about the *recording*, to fix by
routing both platforms through one shared corridor next session rather than by
widening the radius until the number looks acceptable. And the infrastructure
node's own **pose uncertainty is stated nowhere**; until it is a number, track C
reports improvement over ego-only and not an absolute accuracy.

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

**1. The reference is a pipeline, and its seed is GLIM.** GLIM supplies the seed
trajectory; stage 01a then re-derives every pose by registering each scan into a
frozen reference map, initialising from the *seed* rather than from the previous
scan — which is why it is a reference and not another odometry result: error
cannot accumulate along it. That construction is stronger than any single SLAM
system in the shortlist, and it is why the shortlist is scored against it rather
than the other way round.

The consequence for the table is narrower than plain circularity. Stage 01a
keeps the seed wherever it rejects a correction (> 0.5 m or > 5°), so the
reference inherits GLIM's error exactly where GLIM was worst. A GLIM row would
agree with the reference for the wrong reason — correlated errors — so it runs
as an ablation ("what do refinement and anchoring add to the seed"), in its own
block, never in the peer ranking. Every other method is scored normally.

And the per-round convergence statistic is precision, not accuracy: a trajectory
and a map can converge to a jointly wrong solution, which is what the Tikhonov
damping toward the seed exists to bound in geometrically unobservable
directions. Only the board survey bounds the accuracy. So every run is reported
in three tiers — ATE/RPE against the pipeline, the independent check against the
surveyed boards, and reference-free self-consistency — and the evaluator daggers
any ATE below the reference's stated uncertainty rather than printing it as a
win.

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
