# One pipeline, one shared layer

The problem this solves: every script had grown its own copy of bag reading,
SE(3) maths, ICP, plotting and report writing, and each was launched by hand
with different arguments. Stage 08 alone carried 477 lines that stage 09,
`ntp_analysis.py` and the rest also needed.

## Layout

```
map_stages_20260828/            <- keep your existing directory and filenames
  mircpipe/                     <- the shared layer, imported by everything
    config.py    load_config, Cfg (path resolution, stage lookup, runs), Ctx
    se3.py       Rt inv log_R exp_r jr_inv apply compose_all interp_traj
                 traj_gap report_gap path_length subsample decimate_idx
                 level_parts se3_scale read_tum write_tum make_T_xyzq
    bag.py       iter_topic topic_types topic_frame read_odom read_pose
                 read_imu tf_static_rot pc2_xyzt img_gray img_depth
                 camera_K read_map_xyz
    mapref.py    voxel_centroid deskew depth_to_cloud Reference icp_frame Grid2D
    cache.py     bag -> parquet, one file per topic, shared by all analyses
    report.py    figure save_fig write_markdown write_latex_table write_csv
  pipeline_common.py            <- yours, unchanged (load_pipeline)
  pipeline_boards.py            <- yours, unchanged (Board, read_bag, ...)
  01_build_map.py ... 09_publish_poses.py
  ntp_analysis.py, wifi_analysis.py, ...
  run_pipeline.py               <- the single entry point
  pipeline_config.json          <- one config, one block per stage
```

Nothing is renamed and no script has to move. `mircpipe/` is dropped in
beside them and each script deletes its private copies.

## Running it

```
python3 run_pipeline.py --list                          # the registry
python3 run_pipeline.py --run coop2 --stages 08,09      # a subset, in order
python3 run_pipeline.py --run coop2 --from 08           # 08 and everything after
python3 run_pipeline.py --run coop2 --all --keep-going
python3 run_pipeline.py --run coop2 --stages ntp --dry  # print the command only
```

Stages run in registry order whatever order `--stages` lists them in, so a
consumer can never run before its producer. Each stage's output is teed to
`<out_root>/logs/<stage>.log` and a timing/status table is printed at the end.

A stage is either

* **script** - your existing standalone script, run as a subprocess exactly
  as you run it by hand. This is what lets 01-07 and the analyses join the
  pipeline today with no rewrite. If the script does not take the config as
  `argv[1]`, give it an `args` template with `{config} {bag} {run}
  {out_root} {results_root}`.
* **module** - a python module exposing `run(cfg, ctx)`. These share one
  `Ctx`, so the 1.17 M point reference map and the parquet cache are built
  once for the whole invocation instead of once per stage.

Add or patch stages from the config itself, no code edit:

```json
"pipeline": {
  "python": { "ros":      "/usr/bin/python3",
              "analysis": "~/.venvs/mirc/bin/python3" },
  "extra":    [ { "name": "rtab", "kind": "script", "target": "run_rtabmap.py",
                  "args": ["localize", "--platform", "mobile_1",
                           "--bag", "{bag}", "--db", "rtab_mobile_1.db"] } ],
  "override": [ { "name": "ntp", "target": "analysis/ntp_analysis.py" } ]
}
```

## Config

Two blocks are added; everything you already have stays as it is.

```json
"paths": { "raw_root": "~/workspaces/isaac_ros-dev/data/raw/20260828",
           "out_root": "map_stages_20260828_outputs",
           "results_root": "results" },
"runs":  { "coop2":    { "bag": "mirc_dataset_coop2_20260828_completed" },
           "mapping2": { "bag": "mirc_dataset_mapping2_20260828_merged" } }
```

A stage block then says `"run": "coop2"` instead of repeating a bag path, or
inherits the `--run` given on the command line. Relative paths resolve
against the config file's directory, so the pipeline runs from anywhere.

## Migrating one script (10 minutes each)

1. Add at the top, after `import os, sys`:

   ```python
   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
   from mircpipe.se3 import Rt, inv, interp_traj, read_tum, write_tum
   from mircpipe.bag import iter_topic, read_odom
   ```

2. Delete the local definitions of anything now imported. They are
   character-identical, so nothing changes behaviourally - stage 08 shrank
   from 2875 to 2410 lines and its end-to-end test passed unchanged.
3. For an analysis, replace the private bag-to-table extractor with
   `cache.ensure(bag, out_dir)` once and `cache.load(dir, topic)` per topic;
   replace the figure/markdown/latex writers with `report.*`.
4. Register the script in `run_pipeline.py` (or the config's
   `pipeline.extra`) and check `--dry` prints the command you used to type.

Order to do them in: the analyses first (they share `cache` and `report`,
the biggest duplication), then 01-07 (they share `bag` and `mapref`).

## Two interpreters, on purpose

`module 'numpy' has no attribute '_CopyMode'` is the ROS shell putting the
old apt numpy (1.21, `/usr/lib/python3/dist-packages`) ahead of the newer
`~/.local` one (1.26.4) that scikit-learn needs. Do not delete either; split
them by role, which is what the parquet cache is for:

```
python3 -m venv --system-site-packages ~/.venvs/mirc
~/.venvs/mirc/bin/pip install -U numpy pandas pyarrow matplotlib scikit-learn
```

Then `"python": {"ros": "/usr/bin/python3", "analysis": "~/.venvs/mirc/bin/python3"}`
in the config: bag stages (01-09, extraction) keep the ROS interpreter,
analysis stages read parquet in the venv and never import rclpy.
