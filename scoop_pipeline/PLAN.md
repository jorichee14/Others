# THE PLAN: SCooP processing

**Goal:** turn raw bags into the release tree of the paper
(`../scoop_paper/main.tex`, *Release Structure*) reproducibly. Every stage
writes to disk and leaves a `stamp.json` recording its config, code and inputs.
A rerun skips whatever is current, and every number in the paper traces back to
the run that produced it.

The paper sets up three levels (site, session, sequence), and they split the
processing into two pipelines that run at different frequencies:

| | Site pipeline | Sequence pipeline |
|---|---|---|
| Runs | once per site, per mapping pass (A, plus B for validation) | once per sequence |
| Input | mapping-session bag + GLIM trajectory | sequence bags + the frozen site reference |
| Output | `release/site_X/reference/` | `release/site_X/session_YY/seq_ZZ/` |

## Layout

```
scoop_pipeline/                 code (in git)
  scoop/                        library: bag, traj, config, layout, stamp, stages, runner
  stages/{pass,site,seq}/       one script per stage, each exposing run(ctx)
  configs/defaults.yaml         frozen parameters
  configs/sites/<site>.yaml     per-site: bags, seed trajectories, human choices
  run.py                        status / run / stages
  notebooks/                    read-only QA and paper tables

<data_root>/                    data (not in git; $SCOOP_DATA)
  raw/      site_1/mapping_A/, site_1/session_02/seq_05/units/...
  work/     site_1/mapping_A/<stage>/, site_1/_site/<stage>/, .../seq_05/<stage>/
  release/  site_1/reference/, site_1/session_02/seq_05/...
```

## Phase 0: Foundations (done)

| Step | What | Status |
|---|---|---|
| 0.1 | `scoop/bag.py`: select-before-decode, direct CDR parsing, per-point time | done |
| 0.2 | `scoop/traj.py`: TUM I/O (slambench-compatible), vectorised nearest / interp / quaternions | done |
| 0.3 | `scoop/config.py`, `layout.py`, `stamp.py`: merged config, raw/work/release paths, stamps | done |
| 0.4 | `scoop/stages.py` + `runner.py` + `run.py`: stage list, skip-if-current, one process per stage | done |

Tests: `scripts/test_bag.py`, `scripts/test_pipeline.py`. On a real bag:
`scripts/bag_check.py`.

## Phase 1: Site pipeline

Port 01/01a/02/03 into `stages/pass/` and `stages/site/`, reading through
`scoop.bag` and `scoop.traj`. **Get identical results first, speed up second.**

| Stage | From | Output | Speed-ups after the gate |
|---|---|---|---|
| s01_seed_map | 01, stop after denoise | `denoised.pcd` | skip colorize/flatten/anchor |
| s02_refine | 01a | refined trajectory, per-round stats | fused rounds (bag reads 2r-1 → r), parallel ICP, vectorised plane table |
| s03_final_map | 01 on refined poses | `map.pcd` | early height cut before denoise/colorize |
| s04_cut | 02 | `cut.pcd`, z histogram | none |
| s05_anchor | 03 | `anchored.pcd`, `frame.json` | none |
| s06_reference | new | `release/.../reference/`, `refcache.npz` | points + normals + plane table built once |
| s07_validate | new | `validate.json` | none |

**Gate:** on the coop2 mapping bag, s01–s05 reproduce the current pipeline's
trajectory and map to within millimetres before any speed-up is kept. Each
speed-up is then checked against the gate output.

**Needed:** `pipeline_common.py`, `pipeline_boards.py`, `pipeline_config.json`,
`calibration.json`, and the path to one mapping bag.

## Phase 2: Sequence pipeline

| Stage | What |
|---|---|
| q01_lidar_loc | mobile_1: 01a `register()` against `refcache.npz`, one pass |
| q02_rgbd_loc | mobile_2: depth registration + board constraints (existing track 08) |
| q03_infra | infrastructure poses through the board chain (the infrastructure-camera equation) |
| q04_board_check | held-out board residuals, pose consistency |
| q05_geometry | the collaboration geometry (ω, visible, observed, β, C, clearance) → `_geom.yaml` |
| q06_extract | `rosbag_to_opv2v` extraction + link/CSI |

Inputs and outputs in `scoop/stages.py` are provisional until each stage is
implemented.

**Needed:** the mapping pipeline's stages 04–09 (the track 08/09 code is not in this repo).

## Phase 3: Paper numbers

Notebooks read `stamp.json` and the validation outputs and fill the pseudo-GT,
sessions and per-sequence tables and the reference-map and sites figures.
Nothing in the paper is typed by hand.

## Rules

1. A stage writes only its own work directory and the release paths it declares.
2. Parameters live in config, never in a stage script. Human choices (cut bounds,
   anchor board) go in the site file.
3. A port changes nothing but I/O until it passes the gate.
