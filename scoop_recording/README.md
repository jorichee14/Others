# scoop_recording — on-site recording

Runs on the robots during a pass. Separate from `scoop_pipeline/`, which
processes the recordings afterwards (decode, retime, GLIM, ZED).

- `rig.py <pass>` opens one terminal window per machine (`panes.yaml`).
- `record.sh <machine> <pass> [label]` starts the bag for one machine
  (`record.yaml`, `rosbag_qos_override.yaml`).

Copy these into `<workspace>/Dataset/scripts/` on each machine. Bags land in
the `out_dir` set in `record.yaml`; move them under `data/raw/...` before
running `scoop_pipeline/processing/process_recording.py`.

## Choosing topics per run

`record.yaml` groups each machine's topics (`lidar`, `vslam`, `link`, ...). Which
groups a run records:

```
./record.sh mobile_2 survey_1                 # profile chosen by pass name (survey_*)
PROFILE=coop ./record.sh mobile_2 test_1      # force a profile
ONLY=imu,ntp ./record.sh mobile_2 test_1      # exactly these groups
SKIP=vslam ./record.sh mobile_2 coop_1        # all groups but these
```

Edit the `profiles:` block at the bottom of `record.yaml` to change what a kind
of run records. The banner prints `selected:` before recording starts.
