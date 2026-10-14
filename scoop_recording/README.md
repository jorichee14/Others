# scoop_recording — on-site recording

Runs on the robots during a pass. Separate from `scoop_pipeline/`, which
processes the recordings afterwards (decode, retime, GLIM, ZED).

- `rig.py <pass>` opens one terminal window per machine (`panes.yaml`).
- `record.sh <machine> <pass> [label]` starts the bag for one machine
  (`record.yaml`, `rosbag_qos_override.yaml`).

Copy these into `<workspace>/Dataset/scripts/` on each machine. Bags land in
the `out_dir` set in `record.yaml`; move them under `data/raw/...` before
running `scoop_pipeline/processing/process_recording.py`.
