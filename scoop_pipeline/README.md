# scoop_pipeline

Processing for the SCooP dataset (`../scoop_paper`): per-site reference maps
and fiducial boards, then per-sequence reference poses, geometry and release
files. Code lives here; bags, intermediates and the release tree live outside
git (see the plan in the paper's Release Structure section).

```
scoop/               library: everything reusable lives here
  bag.py             reading bags: scans, images, topic schemas (the only reader)
  bagwrite.py        writing bags: BagWriter (MCAP + metadata.yaml), write / write_raw
  rosmsg.py          building ROS messages: header, PointCloud2 (from a dtype), Imu, String
  ouster.py          Ouster packets -> frames / IMU / scans, clock fit, ouster_ros layout
  replay.py          the replay step: packets bag -> points bag, with remaps
  retime.py          sensor time -> capture time (retime_bag.py without ROS)
  imu_sync.py        how far an IMU's stamps run behind another's on the robot (|w| lined up);
                     retime moves the ZED IMU's back by it (recording.yaml ouster.imu_sync)
  zed.py             ZED SVO2 -> bag on capture time (wrapper replay + restamp)
  recording.py       one recording: raw bag -> decoded -> retimed -> GLIM, SVO -> zed
  session.py         one pass: every machine's bags, checked against record.yaml
  merge.py           several bags -> one, in log-time order (zstd by default)
  tftree.py          calibrated transforms added to a bag's TF tree (one parent per frame)
  channel.py         when a message over the link arrives: ideal, synthetic (V2X-ViT,
                     CoBEVFlow, ...), fitted, or the logged link of a pass
  timing.py          how evenly a topic is spaced: intervals, gaps, repeated frames
configs/recording.yaml  settings for processing a recording (topics, remaps, GLIM config)
configs/zed/         the robot's zed_wrapper config, used to replay SVOs
configs/record.yaml  what each machine records (the robots' record.yaml)
configs/static_tf.yaml  calibrated transforms the merge adds to /tf_static
configs/glim/<machine>/  GLIM's config folder per robot (configs/glim/README.md)
configs/comms.yaml   which TF frame places each machine in the comms tables
environment.yml      conda env for all offline processing
processing/          raw recordings -> processed bags (command lines around scoop/)
  process_recording.py  raw recording -> decoded, retimed, glim, zed (the usual entry point)
  decode_ouster.py   packets bag -> points bag (replaces the ouster_ros replay)
  retime.py          points bag -> retimed bag (same arguments as retime_bag.py)
  imu_sync.py        a processed bag's ZED IMU against the Ouster's IMU; --out a copy with its
                     stamps moved back (for passes processed before the retimed step did it)
  run_glim.sh        GLIM (docker) on a bag, dump next to the bag, owned by you
  svo_to_bag.py      one SVO2 -> bag on capture time (the zed step alone)
  run_zed.sh         ZED wrapper replay + ros2 bag record (here or in the isaac_ros container)
  merge_session.py   check a pass's bags against record.yaml, merge them into one
  merge_bags.py      any bags -> one, with the topics you choose (--list, --topics, --pick)
  comms_tables.py    NTP / Wi-Fi / ping / iperf / CSI -> tables + positions (comms/)
  csi_clean.py       CSI for use: data frames, no pilots, H scaled to RSSI, poses
  link_trace.py      the measured link as traces (pings, iperf, outages) for scoop/channel.py
  tf_edit.py         show or change a bag's TF: drop edges, add transforms, new odometry
  dvlc_pauses.py     a merged bag's pauses (robot still) -> one short bag each, for Koide's
                     direct_visual_lidar_calibration
  run_dvlc.sh        that calibration (docker) on the pause bags: preprocess, guess, calibrate,
                     viewer; output owned by you
  glim_check.py      a GLIM run's trajectory summary and loop closure on the boards
  convert_cloud.py   a point cloud to another format (pcd, ply, xyz, ...), optionally thinned
  rename_frames.py   rename TF frames in a bag (headers, Odometry child, /tf, /tf_static), e.g.
                     --suffix _mobile2 for a bag decoded before --frame-suffix
run_pass.py          one pass from raw to processed, every step in order (below)
clean_pass.py <config> [--apply] [--work]  keep a pass's current results, move older
                     versions, side runs and caches to processed/<date>/<pass>/_old/
                     (a dry run without --apply; clean.keep in the config keeps more)
check.py <command>   every map and trajectory check, one command (check.py -h;
                     check.py <command> -h):
    map     <map.pcd>       surface thickness (walls, floors; a wall's two faces vs
                            one surface mapped twice with --traj; --save patches.ply)
    walls   <config>        the map's wall spots with the scans placed by a trajectory:
                            one scan (the sensor), 1 s ... whole run, the map
    maps    <A> <B>         completeness / accuracy of map B against map A
    scans   <config>        single scans and pairs: noise floor, deskew, time offset, drift
    trajs   <config>        a pass's trajectories: ATE / RPE against --ref, heading vs
                            turning, loop closures
    time    <config>        is GLIM's pose time its scans'? (run_pass.py's time_check)
    imu     <config>        is the IMU on the LiDAR's time?
    loop    <config>        loop closure on the boards (05 --check)
    report  <config>        a mapping pass: GLIM, + time correction, + scan-to-map, + BA
                            against the reference; TSDF and union maps (run_pass's report)
map_stages/          map from LiDAR + GLIM poses, numbered in the order they run
                     (01_build_map.py, 02_bundle_adjust.py, ...);
                     calibration read from the bag (camera_info + /tf_static); outputs in
                     data/processed/<date>/<pass>/{mapping,odometry/<machine>,frames,bags,comms};
                     datasets/<machine>/<format>_<depth>_<pass>_<date> from datasets/
                     mapping pass: 02 bundle adjustment of GLIM's poses (the reference),
                     01 map (TSDF) and its union, 02b scan-to-map (the comparison),
                     05 anchor, 06 view copy, 07/08 cameras+TF, 10 poses;
                     a run in that map (pipeline_config_<run>.json: "extends" the mapping
                     config, dataset.reference_pass): 03 start pose from the anchor board --
                     "source": "lidar_odom" combines every opening view, parked or moving,
                     through the run's GLIM trajectory (board_views.py); optional:
                     03_init.relocalize, LiDAR relocalization (lidar_reloc.py),
                     04 per-scan LiDAR ICP, 05 on the run places every board it sees in
                     map (boards_<tag>.json, nothing re-anchored), 07 infra cameras,
                     09 run cloud, 10 poses bag
datasets/            bags -> datasets for other tools
  mcap_convert.py    replica (RGB + lidar depth), mcd, mcgs (standalone)
  swarm_slam.py      a multi-robot pass as Swarm-SLAM input (RGB-D per robot, /r<i>,
                     offline cuVSLAM (vslam.py --machine), onboard or ground-truth odometry,
                     started where 03's session_anchor.json
                     put each robot) + config; slam/run_swarm_slam.sh runs it
  hydra.py           a swarm_slam.py export as Hydra input: one bag per robot (colour,
                     depth, camera_info, /tf odom -> base -> camera), a hydra_ros config
                     each (scene graph + mesh streamed once a second) and a launch file
                     (visualize:=true for the live scene graph); --configs-only rewrites those
                     --poses kimera --kimera <euroc export>: Kimera-VIO's odometry in /tf and its
                     loop closures that agree with it on /hydra/external_loop_closures
  euroc.py           a multi-robot pass as Kimera-VIO input: each robot's rectified stereo
                     pair and IMU in the EuRoC layout, a parameter folder from the recorded
                     calibration (slam/run_kimera.sh runs it, loop closure on; STRETCHES=1
                     one run per stretch without a dropout, slam/kimera_stretches.py); the IMU
                     moved onto the images' clock (mobile_1's ZED: 2.12 s; --imu-only rewrites it)
  replica_multiagent.py  a multi-robot pass as a ReplicaMultiagent scene for MAGiC-SLAM:
                     one common camera, frames in lockstep, poses in the shared map
analysis/            measuring results; reads outputs, writes nothing back
  headroom.py configs/headroom_terms.json [--traj ref.tum ...]  E1: every testbed error term
                          against the fusion breaking points, like against like, log axis
  maps.py, trajs.py  the map and trajectory checks, run through check.py (above)
  odom/zed_vs_lidar.py  the ZED's own tracking against the LiDAR trajectory
  (slam/channel_relay.py: ROS 2 relay between robot domains through scoop/channel.py;
   slam/run_swarm_slam.sh <export> <channel>: one Swarm-SLAM run per channel;
   slam/run_hydra.sh <hydra export> <channel>: every robot's Hydra streaming its
   scene graph to a server through the channel, one run per channel)
  slam/cslam_eval.py  Swarm-SLAM run against the ground truth: ATE per robot and joint,
                     inter-robot loop closures checked, time to merge, bytes exchanged
  slam/kimera_eval.py  Kimera-VIO (slam/run_kimera.sh) against the ground truth: ATE and
                     drift in map per run, or per stretch without a dropout (STRETCHES=1)
  slam/hydra_eval.py  Hydra against the ground truth as cslam_eval does Swarm-SLAM: ATE per
                     robot and joint, in map, the odometry baseline, loop closures
  slam/imu_check.py  a euroc.py export's IMU timing against its images and ground truth
                     (turn rates lined up by time shift), gravity while still, T_BS rotation
  slam/relay_stats.py  what a channel did to a run's messages (relay.csv), runs side by
                     side: MB, lost, delay, backlog; --to server: map age at the receiver
  slam/dsg_stream.py  Hydra's scene-graph stream: log (in Hydra's container) the size of
                     every hydra/backend/dsg update against bag time; plot size and rate
  odom/zed_vs_lidar.py    drift of the ZED's own tracking (map_zed) against the LiDAR trajectory
  comms/             link analyses on processing/comms_tables.py's tables
  comms/ntp_analysis.py   clock agreement: offsets at polls, error bounds, steps,
                     frequency vs temperature, worst case between machines
  comms/wifi_analysis.py  links: RSSI, PHY rates, ping loss per ICMP seq, iperf,
                     ping during iperf vs not, RSSI along the path
  comms/comms_map.py      RF map: NTP, RSSI, ping, iperf on the map's top view with
                     each moving robot's path and the parked nodes; a coop run
                     gets RSSI and ping panels per robot (comms/maps/)
  comms/csi_analysis.py   CSI links: capture, frame types and their subcarriers, path
                     loss, delay spread, coherence vs speed, RSSI on the map
  comms/csi_view.py       pictures of the clean CSI: amplitude, phase, delay profile
                     over time, snapshots by link length (comms/csi/view/)
  bags/topic_timing.py    intervals, gaps and repeated frames of topics in a bag
  bags/show_topics.py     what topics hold: type, count, definition, first messages
checks/              is the environment / the decoder right? (against real bags)
  env_check.py       is this environment ready?
  bag_check.py       verify + time the fast decoder on a real bag
  ouster_check.py    verify packet decoding against a replayed/retimed bag
  replica_check.py   does a replica dataset fit itself (depth vs image, depth vs poses)
data_quality_validation/  is the recorded data right? (straight from the bag, PASS/FAIL)
  camera_lidar_sync.py <bag> [--robot mobile_1]  camera and LiDAR on one clock: the turn rate
                     from image shifts and from scan azimuth shifts, lined up per segment
tests/               self-tests, no real bag needed
  test_bag.py        reading, writing, messages
  test_ouster.py     packets, clock, replay, remaps (needs ouster-sdk)
  test_zed.py        restamp, run_zed.sh with a fake ros2 and docker
  test_merge.py      machines, topic check, merge, tf_edit
  test_pass_tf.py    the complete TF tree of a pass (merge, 10)
  test_run_pass.py   run_pass.py: status, 10 + finalize, a mapping pass's order
  test_comms.py      comms tables: every message, summary numbers, positions
  test_ntp_analysis.py  ntp_analysis on a synthetic ntp.csv with known events
  test_wifi_analysis.py  wifi_analysis: seq wrap, late replies, contention, goodput vs RSSI
  test_comms_map.py  comms_map: figures written, grid cells hold their samples' median
  test_sim_gap.py    sim_gap: known RTT, goodput and outage give the message delays and losses
  test_channel.py    link_trace + channel: hops, own test not counted twice, cross traffic,
                     outage, stall, queue, synthetic and fitted channels
  test_channel_relay.py  channel_relay: messages held to their arrival, outage dropped,
                     never back to the sender, every message logged; MB updates to a
                     server queued per robot uplink
  test_cslam_eval.py cslam_eval: keys to robots, ATE, wrong loop closure flagged, merge time
  test_swarm_slam.py swarm_slam: topics, intrinsics, registered depth, odometry in the
                     robot's odom frame at the image stamp, ground truth
  test_euroc.py      euroc: stereo pairs at --rate, unpaired frames out, IMU increasing,
                     intrinsics from P, T_BS from /tf_static, flags, --imu-only --imu-delay
  test_hydra_export.py  hydra: a bag per robot, images byte for byte, /tf composing to the
                     odometry with a z-up body frame, config frames, range and mesh stream
  test_hydra_eval.py hydra_eval: odometry shift aligned away, Hydra's offset in map, --cov,
                     loop closures, json and figure
  test_camera_lidar_sync.py  camera_lidar_sync: synced bag passes (Ouster start-of-sweep stamps, robot
                     body and a blocked sector in view), a LiDAR 0.3 s late found and fails
  test_imu_sync.py   imu_sync: a ZED IMU 2.12 s late found against the reference IMU, a still
                     robot not sure; processing/imu_sync.py --out moves only its stamps; retime step 5
  test_kimera_stretches.py  kimera_stretches: a dropout and an IMU-only gap split three stretches;
                     kimera_eval: ATE and in-map error per stretch, a failed stretch, a whole run
  test_hydra_kimera.py  hydra --poses kimera: the camera from Kimera-VIO's IMU at every frame of its
                     runs, a right loop closure sent, one the odometry contradicts left out
  test_imu_check.py  imu_check: a gyro 2.1 s late found against images and ground truth;
                     an IMU turned 90 deg found by gravity and the rotation fit
  test_relay_stats.py  relay_stats: delay, loss, backlog of a queued stream, map age
  test_dsg_stream.py dsg_stream: sizes, rates from bag time between updates, figure
  test_replica_multiagent.py  replica_multiagent: common camera, lockstep frames, poses,
                     colour resampling and depth registration on a two-robot bag
  test_csi_analysis.py  csi_analysis: occupancy, capture, path loss, delay spread, coherence
  test_csi_view.py   csi_view: delay profile puts the paths at their delays; figures
  test_replica.py    replica with lidar depth on a synthetic scene
  test_scan_consistency.py  check.py scans on synthetic sweeps: true, time-
                     shifted, drifting, broken deskew
  test_map_quality.py  check.py map's layer split: a wall's two faces, a surface
                     mapped twice, smear
  test_wall_build.py  check.py walls on a synthetic room: clean, and a 2 % range error
  test_bundle_adjust.py  02 on a synthetic room driven twice with drift: poses
                     back to the truth, planes back to the noise
  test_clean_pass.py  clean_pass on a made-up pass: current results kept, a stale
                     seed map, a cache, an old map folder and a GLIM side run moved
  test_pose_time_check.py  check.py time: a corrected GLIM on time, one 30 ms
                     late flagged with the offset to set; check.py report's rows
  test_compare_trajs.py  check.py trajs on an offset and a drifting trajectory
  test_tsdf.py       01's TSDF fusion against its voxel union on a smeared synthetic
                     room: thinner, and its surfaces where the true ones are
```

## Environment

Two environments, each for its own kind of work:

| | conda env `scoop` (`environment.yml`) | ROS 2 install |
|---|---|---|
| for | decoding, Ouster packets, map building, refinement, boards, evaluation, figures | recording, `ouster_ros` replay, GLIM, `ros2 bag convert` (merge.yaml), `retime_bag.py` |
| Python | 3.11 from conda | the system's (3.10 on Humble) |

```
cd scoop_pipeline
conda env create -f environment.yml
conda activate scoop
python checks/env_check.py        # versions, cv2.aruco, open3d, Ouster SDK, GPU
python tests/test_bag.py && python tests/test_ouster.py
```

After `environment.yml` changes: `conda env update -f environment.yml --prune`.
For CUDA 11 machines, swap `cupy-cuda12x` for `cupy-cuda11x` first
(`nvidia-smi` shows the version); without a GPU, stage 01 runs on the CPU.

Keep them in separate terminals: `source /opt/ros/<distro>/setup.bash` puts
ROS's Python packages on `PYTHONPATH`, where they shadow the env's.
`env_check.py` warns when that has happened.

## Bag decoding (`scoop/bag.py`)

Built on `../rosbag_to_opv2v`'s `BagReader` (MCAP without ROS, `.db3` via
rosbags). What it adds over the stage scripts' own readers:

- **select before decode**: the header stamp is read from the raw CDR bytes and
  passed to `select(stamp_ns)`; scans without a trajectory pose within
  `time_tol` are never deserialised.
- **direct CDR parsing** of `PointCloud2`, `Image`, `CompressedImage`, with a
  strided zero-copy xyz view for the usual float32 x/y/z layout. Anything else
  falls back to the generic decoder.
- **per-point time** as float64 seconds relative to the header stamp, unit
  detected from the sweep span, same mask as the points.

```python
from scoop import bag
reader = bag.open_bag(cfg["dataset"]["bag"])
topic = bag.detect_points_topic(reader, n_poses=len(traj_t))
for scan in bag.iter_scans(reader, topic, select=bag.nearest_pose(traj_t, tol),
                           min_range=lo, max_range=hi, with_time=True):
    T = traj_T[scan.key]          # scan.xyz (N,3) f32, scan.t (N,) s or None
for fr in bag.iter_images(reader, image_topic, mode="gray", period=0.2):
    ...                           # fr.stamp, fr.image
```

Before using it on a new sensor or driver version:

```
python3 tests/test_bag.py
python3 checks/bag_check.py <bag> [--points /ouster/points] [--image <topic>]
```

`bag_check.py` compares the fast path with the generic decoder field by field
on the first 50 messages (exits non-zero on any mismatch) and times both.
On a synthetic uncompressed MCAP of 128x1024 Ouster scans (48-byte points,
6.3 MB each): 67 msg/s for the scripts' decode + extract + range gate, 235
msg/s for `iter_scans` with the same gate, file reading included. Reading
alone runs at ~630 msg/s, so that's the upper limit for a single process.

## A pass from raw to processed (`run_pass.py`)

```
cd ~/workspaces/Others && git pull
python3 scoop_pipeline/run_pass.py ~/workspaces/isaac_ros-dev/data/raw/20260924/survey_1 --status
python3 scoop_pipeline/run_pass.py ~/workspaces/isaac_ros-dev/data/raw/20260924/survey_1
```

runs every step of the pass in order and skips the ones whose output is there
already, so the same command picks up where it stopped. `--status` shows what
is done, `--dry` prints the commands, `--from S` redoes S and everything after
it, `--only S` runs S alone, `--until S` stops after S.

| step | what | output |
|---|---|---|
| process | `process_recording.py` per machine folder with an Ouster bag or SVO2 | `work/<date>/<pass>/<machine>/` |
| merge | `merge_session.py`: check against record.yaml, merge | `work/<date>/<pass>/<prefix>_<pass>_<date>_merged` |
| vslam | only with a `"vslam"` block (coop: mobile_2 drives, its VSLAM was not recorded): `processing/vslam.py`, cuVSLAM offline on its stereo + IMU, in the isaac_ros container | `work/<date>/<pass>/<machine>/vslam/traj_vslam.txt` |
| 02_ba | mapping pass: `02_bundle_adjust.py`, GLIM's poses (moved by `glim.pose_time_offset` on export) bundle-adjusted on the map's planes: the reference trajectory | `odometry/<machine>/traj_lidar_ba.txt` (`02_bundle_adjust.output`) |
| time_check | `check.py time`: GLIM's pose time against the reference (a mapping pass: 02's; a run: 04's track); on time, or OFF with the `pose_time_offset` to set | `odometry/<machine>/pose_time_check.json` |
| 01_seed | mapping pass, for 02b: `01_build_map.py --seed`, the map from GLIM's poses | `mapping/denoised_seed_<tag>.pcd` |
| 02b_s2m | mapping pass, the comparison: `02b_scan_to_map.py`, every scan registered to that map (`"02b_scan_to_map": {"enabled": false}` skips it and 01_seed) | `odometry/<machine>/traj_lidar_refined.txt` |
| 01_map | mapping pass: `01_build_map.py`, the map from `dataset.traj` (02's), fused as `01_build_map.fusion` says | `mapping/map_final_<tag>.pcd` |
| 01_union | mapping pass with a TSDF map: `01_build_map.py --union`, the same poses as a union of points, the comparison map | `mapping/denoised_union_<tag>.pcd` |
| report | mapping pass: `check.py report`, GLIM / + time correction / + scan-to-map / + BA against the reference (ATE, RPE, pose time, loop closure, path) and the TSDF and union maps (thickness, doubled surfaces, completeness) | `mapping_report_<tag>.md`, `.json` |
| 03_init | run: `03_init_from_boards.py`, start pose from the anchor board | `frames/session_anchor.json` |
| 04_reference | run: `04_reference_traj.py`, every scan registered to the anchored map; its `"tracks"` (`map_stages/depth_track.py`): another robot's depth frames registered to it from its own odometry; with `"method": "pgo"` (`map_stages/depth_pgo.py`) one pose graph instead: cuVSLAM between frames, weak across camera dropouts, the start anchor, boards and map fixes (search, then ICP with Hessian information), solved together; each pose's marginal covariance in `cov_<track>.csv` (σx, σy, σyaw, xy correlation, the xy ellipse's major semi-axis) | `odometry/reference_<tag>/` |
| 05_anchor | `05_anchor.py`: the boards (mapping pass: and the anchored map) | `frames/anchor_frame.json`, `boards_<tag>.json` |
| 06_cut | mapping pass, only when `06_cut` has floor/ceil: viewing copy | `mapping/*_noceil.pcd` |
| 07_cameras | `07_build_cameras.py`: infra / parked cameras in map | `frames/cameras_in_map.yaml` |
| 08_tfs | `08_emit_tfs.py` | `frames/publish_tfs_<tag>.sh` |
| 09_cloud | run: `09_build_coop_cloud.py`, the run's cloud in the map | `mapping/cloud_in_ref_<tag>.pcd` |
| 10_poses | `10_publish_poses.py`: poses + the complete TF | `bags/<tag>_best_poses` |
| finalize | the merged bag again, with 10's /tf, /tf_static and poses in it | the merged bag, in place |
| comms | `processing/comms_tables.py`: NTP, Wi-Fi, ping, iperf, CSI with positions | `comms/{ntp,wifi,csi}/`, `comms/summary.json` |
| csi_clean | `processing/csi_clean.py`: QoS data frames, no pilots, H scaled to RSSI, both ends' poses | `comms/csi/csi_<link>_clean.npz` + README |

Stage 02 is plane bundle adjustment of all poses at once (BALM / HBA: every
plane's smallest eigenvalue, the planes solved in closed form, 1 s keyframes,
the input's relative motion as a soft factor, roll / pitch held to the
input's) on GLIM's trajectory, and its loop closure on the boards. A mapping
pass's `dataset.traj` is its output. The keyframes are kept in
`<output>_keyframes.npz`, so a re-run with other settings does not read the
bag. 02b, scan-to-map, is kept as the comparison (on 20261015 a BA started
from it reaches the same poses, 1.0 cm / 0.12 deg ATE).

`01_build_map.fusion: "tsdf"` fuses the scans as a truncated signed distance
field (Curless & Levoy; VDBFusion) instead of keeping every cell any scan hit:
the map is the averaged surface, without the union's ~4 cm of stray cells.
`tsdf.trunc` = 3 x the spread of where the scans put a wall (check.py walls'
"whole run" / 3.29), at least 3 cells.

The map stages need the pass's config in `map_stages/`: `pipeline_config.json`
for the mapping pass, and for a run `pipeline_config_<run>.json` with
`"extends": "pipeline_config.json"` and `dataset.reference_pass` (e.g.
`pipeline_config_survey_1.json`); `--config` names another. Once the merged
bag exists, processing is not redone, so the decoded/retimed/ZED bags may be
deleted (keep `glim/`: 02, 01_seed, the report and 03 read GLIM's trajectory). finalize
needs as much free space as the merged bag takes. A config written before the
stages were renumbered still works: a block under its old number ("08_reference")
is read as the new one ("04_reference"). A stage whose block has `"enabled": false`
is not run.

A coop run (`pipeline_config_coop_2.json`) has both robots moving. mobile_1 is
as in a survey run. mobile_2: `vslam` gives its odometry; 03 places its colour
camera at `rs_anchor` in the opening dwell, with the board's pose from the pass
that measured it (`dataset.boards_from: ["survey_1"]`; boards do not move); 04's
track `mobile_2_depth` (`"method": "pgo"`: one pose graph over every depth
frame, see 04 above) places mobile_2 in the anchored map. The tree gets
`board_rs ── map_realsense` (its first pose) `~~ camera_link`, mobile_1's
`board ── map_zed` where the ZED's own tracking puts the camera at the LiDAR
track's first pose (no 05), and the arducam where survey_1's 07 put it
(`dataset.cameras_from: ["survey_1"]`; the infra camera did not move); 10
publishes `/mobile_2/global_pose` and `/mobile_2/local_pose`. No infra cameras
placed (07/08) or boards re-measured (05) in coop itself.

A later day in the same place (the place is not mapped again; the boards are
put up again, roughly where they were): every run of that day keeps the first
day's map, `"reference_pass": "20260924/mapping_A"`, and says the boards moved,
`"boards_moved": true` in `dataset`. The first run of the day measures where
they are now: `03_init.relocalize` places its start from the LiDAR (the board
dwell would put it where the board used to be; 03 reports that disagreement as
the board's move instead of refusing), 04 registers its LiDAR track to the map,
and 05 on the run measures every board it saw, in that map
(`frames/boards_<tag>.json`). With `boards_moved` the poses measured since the
reference pass -- the run's own, then `boards_from`'s -- replace the
reference's wherever a board is used (03, 04's tracks, 07, the TF tree,
`pipeline_common.Pipeline.boards`); the reference's pose stands only for a
board nobody measured since. The other runs of the day take them with
`"boards_from": ["<first run>"]` (or `"<date>/<run>"`); on the first run,
`--from 03_init` then places its parked robots from the new poses too. Without
`boards_moved` the reference's pose wins, as before.

## Processing a recording

```
python processing/process_recording.py ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A/mobile_1
```

reads the raw recording and writes everything derived to the mirrored folder
below `data/work/`, always under the same names:

```
data/raw/20260924/mapping_A/mobile_1/      as recorded; never written to
    mirc_dataset_survey_1_mapping_20260924_mobile_1/   (the packets bag)
    *.svo2
data/raw/20260924/mapping_A/infra_cameras/<camera>/   one folder per infra camera:
    its recordings + extrinsic_<camera>.yaml saved during the session (map stage 07
    reads the yaml here)
data/work/20260924/mapping_A/mobile_1/
    mirc_dataset_survey_1_mapping_20260924_mobile_1_decoded/   1  points bag, sensor time
    mirc_dataset_survey_1_mapping_20260924_mobile_1_retimed/   2  capture time; the GLIM input
    glim/          3  GLIM dump: traj_lidar.txt, map
    <svo name>_zed/        4  the SVO2 as a bag, capture time (zed_logs/ inside)
    clock.json        the clock fit (drift, residual, packets) for the paper
    process.yaml      the settings the steps ran with
data/processed/20260924/mapping_A/odometry/mobile_1/   what the map stages use of GLIM
    traj_lidar.txt    GLIM's poses of the cloud's frame (TUM), the stages' start
    traj_imu.txt      GLIM's IMU poses, when it wrote them
    glim_config/      the GLIM config it ran with (configs/glim/<machine>)
    glim.yaml         bag, config, image; poses, span, rate, path length; loop closure
    glim_loop_closure.json   the boards' sections and loop closures on GLIM's poses
```

The GLIM step writes that `odometry/<machine>/` export when it finishes (and
on the next run for a GLIM run done before); GLIM's dump, with its map,
stays in `work/.../glim/`. It then measures GLIM's loop closure on the
boards: stage 05 in its read-only check mode (`05_anchor.py --traj
<GLIM's traj_lidar.txt> --check <report>`, with the pass's map-stage config)
finds every board seen at the start and again later and reports how far
apart GLIM puts it, in mm / deg against the detections' own noise:

```
    glim -> .../odometry/mobile_2: 13871 poses over 1387.017 s (10.0 Hz), 543.375 m
    loop closure rs_anchor: s0->s3 over 1380 s: 12.4 mm / 0.210 deg (noise 3.1 mm) -> DRIFT
```

`processing/glim_check.py <raw machine folder>` runs both again on a GLIM run
that is already there. Stage 02 runs the same check on its refined trajectory
when it finishes and prints it next to GLIM's (`check.py loop <config>
[--traj FILE]` on any trajectory):

```
  against GLIM:
    anchor       GLIM   10.8 mm / 1.240 deg  ->  refined    4.2 mm / 0.310 deg
``` Without a map-stage config for the pass (or one that
maps with another robot) the loop closure is not measured.

Settings come from `configs/recording.yaml` (`--settings` for another file).
Its top level is mobile_1's; a machine folder named in its `machines:` block
gets that block merged over it. mobile_2's (`mobile_2/` folder):

```
data/raw/<date>/<pass>/mobile_2/<its bag>/      Ouster packets on /mobile_2/ouster
data/work/<date>/<pass>/mobile_2/
    <bag>_decoded/   points + IMU, every other topic copied (camera, VSLAM, /tf,
                     /tf_static) without the packets; Ouster frames *_mobile2
    <bag>_retimed/   capture time; what the merge takes for mobile_2
    clock.json  process.yaml
```

GLIM runs on it with its own config (`machines.mobile_2.glim.config`:
`configs/glim/mobile_2`, made from `configs/glim/mobile_1` by
`processing/glim_config.py`: topics, and `T_lidar_imu` from the bag's Ouster
frames; see `configs/glim/README.md`). In a pass where mobile_2 maps alone, the map stages
take this retimed bag as `dataset.bag` (no merge; GLIM's `glim/` is found next
to it) -- e.g. `map_stages/pipeline_config_mapping_20261015.json`.
`run_pass.py`, `merge_session.py` and `process_recording.py` all apply the
block by the folder's (machine's) name.

Each robot's TF is made with its bag, not in the merge: decode
(`ouster.static_tf: configs/static_tf.yaml`) puts the calibrations reached
from the robot's Ouster frames into its `/tf_static` -- mobile_1:
`os_lidar_mobile1 -> zed_left_camera_optical_frame -> radar1/2_link`,
mobile_2: `os_lidar_mobile2 -> camera_color_optical_frame` -- with the
Ouster's tree hung below the camera (`<camera optical> -> os_sensor_mobileN`),
also when the camera's frames are in another bag (mobile_1's ZED bag), so a
merge never gives a frame two parents. A processed bag then carries its
robot's whole TF on its own. The merge only adds the pass-level TF (map,
boards, infra cameras, from the map stages) and the calibrations no bag has
yet (infra_1's camera -> radar; bags decoded before this).
The two bags are named `<original bag>_decoded` and `<original bag>_retimed`
(the .mcap inside too), so a copied bag still says where it came from.
A folder with an SVO but no Ouster packets bag (e.g. an older recording)
does the zed step alone; `--only X` runs just step X. Steps already done are
skipped. `--redo X` redoes X and the steps built from
it: `--redo retimed` redoes retimed and glim, `--redo zed` only zed.
`--until retimed` stops before GLIM; `--status` shows what is done. The
decoded bag may be deleted once retimed and glim exist: it is then skipped
("not needed"), and rebuilt only when retimed is redone. A step is
built in `.partial/<step>/` and only moved into place when it succeeded, so a
step folder is always complete. Steps 1-2 run in the `scoop` env, step 3
starts docker (`processing/run_glim.sh`), step 4 needs the ZED wrapper
(below).

## Ouster packets: decode, then retime

```
original bag (<ns>/lidar_packets, imu_packets, metadata, other topics)
   |  python processing/decode_ouster.py <original> <decoded> [--copy-all] [--remap ...]
   v
decoded bag, sensor time (<out-ns>/points, imu, metadata + copied topics)
   |  python processing/retime.py <original> <decoded> <retimed> /ouster /mobile_1/ouster
   v
retimed bag
   |  processing/run_glim.sh <retimed>                                 docker
   v
<folder of the bag>/glim_dump/traj_lidar.txt  -> stage 01
```

Both steps run in the `scoop` env. `processing/retime.py` does what
`retime_bag.py` does, with the same arguments; it rewrites the header stamp in
the message bytes instead of going through rclpy, so it needs no ROS. On
synthetic bags (sensor clock from boot and PTP) its output equals
`retime_bag.py`'s message for message -- topic, log and publish time,
sequence, bytes -- and metadata.yaml is identical.

`decode_ouster.py` replaces `ros2 launch ouster_ros replay.launch.xml` +
`ros2 bag record`: same topics, same organized cloud layout (x y z intensity
t reflectivity ring ambient range), IMU in m/s^2 and rad/s, stamps in sensor
time, `metadata.yaml` included, without waiting for real time. `retime_bag.py`
also still runs on its output unchanged.

```
python processing/decode_ouster.py <original bag> <decoded bag dir> \
    --packets-ns /ouster --out-ns /mobile_1/ouster --driver-min-range 1.30 \
    --frame-suffix _mobile1
```

`--frame-suffix` goes on every Ouster frame -- `os_lidar`, `os_sensor`,
`os_imu` become `os_lidar_mobile1`, `os_sensor_mobile1`, `os_imu_mobile1` --
one suffix per robot, so two robots' sensors stay apart when their bags are
merged. Without it the names are ouster_ros's. The copied topics are renamed
too (their headers, /tf, /tf_static), so the driver's `os_sensor -> os_lidar`
becomes `os_sensor_mobile1 -> os_lidar_mobile1`. `configs/recording.yaml` has
`frame_suffix: _mobile1`, and `configs/static_tf.yaml` hangs each robot's
camera from its own LiDAR frame. A bag decoded before this has the plain
names; rename them in a copy:

```
python processing/rename_frames.py <bag> -o <new bag> --suffix _mobile1
```

`--limit 200` stops after 200 scans, for a quick look. `--remap SRC:=DST`
copies a topic of the original bag into the output under a new name, byte for
byte (what `ros2 bag play --remap` did in the replay launch). The replay
launch's set:

```
    --remap /ouster/metadata:=/mobile_1/ouster/metadata \
    --remap /ouster/imu_packets:=/mobile_1/ouster/imu_packets \
    --remap /ouster/lidar_packets:=/mobile_1/ouster/lidar_packets \
    --remap /zed/zed_node/imu/data:=/mobile_1/zed/imu/data \
    --remap /zed/zed_node/imu/mag:=/mobile_1/zed/imu/mag
```

`--copy-all` copies every other topic of the original too (tf, tf_static,
radar, ntp, ...) under its own name, unless a `--remap` renames it;
`--exclude TOPIC` leaves one out. Copied topics keep their QoS from the
original metadata.yaml, so /tf_static stays latched when the bag is played.
Retime passes them all on (times shifted by the same constant as the replay),
so the retimed bag holds the whole recording. `configs/recording.yaml` has
`copy_all: true`.

From Python (a later pipeline stage calls the same function):

```python
from scoop import replay
replay.decode_ouster_bag(original, out_dir, packets_ns="/ouster", out_ns="/mobile_1/ouster",
                         driver_min_range=1.30,
                         remap=replay.parse_remaps(["/zed/zed_node/imu/data:=/mobile_1/zed/imu/data"]))
```

## Ouster packets in Python (`scoop/ouster.py`)

The mapping bags record the Ouster as `<ns>/lidar_packets` + `<ns>/metadata`.
Instead of `ros2 launch ouster_ros replay.launch.xml` (real time) followed by
`retime_bag.py`, the packets are decoded directly from the original bag:

```python
from scoop import bag, ouster
reader = bag.open_bag(packets_bag)
clock = ouster.fit_clock(reader, "/ouster")        # retime_bag's clock fit, ported
for scan in ouster.iter_ouster_scans(reader, "/ouster", clock=clock,
                                     driver_min_range=1.30, frame="lidar"):
    ...                     # same Scan as bag.iter_scans: stamp_ns, xyz, t
```

- stamp: first non-zero column timestamp (sensor clock) mapped to host capture
  time by the same lower-envelope fit `retime_bag.py` uses;
- per-point `t`: column timestamp minus the stamp; points destaggered, in
  the cloud's row-major order;
- `driver_min_range`: the replay's `min_range`, applied to the RANGE field;
- `frame`: `lidar` (os_lidar) or `sensor` (os_sensor).

Which conventions the replayed clouds used is checked, not assumed:

```
python3 checks/ouster_check.py <packets_bag> --points-bag <retimed bag> \
    [--packets-ns /ouster] [--points-ns /mobile_1/ouster] [--driver-min-range 1.30]
```

It prints the clock fit (compare with `retime_bag.py`'s output), the decode
speed as a multiple of real time, and per scan: stamp difference, point
count, max xyz difference in both frames, and max per-point time difference.
It ends with `MATCH in the <frame> frame` or `NO MATCH`.

## ZED SVO2 -> bag

Step 4 of `process_recording.py`, or alone:

```
python processing/svo_to_bag.py <file.svo2> <out bag dir>
```

Reading an SVO2 needs the ZED SDK, so the SVO is played through the ZED ROS 2
wrapper and recorded (`processing/run_zed.sh`):

```
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i svo_path:=<svo>
    + svo.use_svo_timestamps: true, svo.svo_loop: false, svo.svo_realtime: false
ros2 bag record <zed.topics>          stopped when the wrapper logs "SVO reached the end"
```

recorded in 4 GiB pieces, then restamped (`scoop/zed.py`) piece by piece,
each deleted once read, so the disk needs about one copy of the bag, not two:
log time = header stamp (the SVO's capture
time, the recording PC's clock, as in the raw bag), topics renamed
`/zed/zed_node/...` -> `/mobile_1/zed/...`, QoS kept. The result is on the
same timeline as the retimed bag. The wrapper runs with the robot's own
wrapper config (`configs/zed/common_stereo.yaml`, `configs/zed/zed2i.yaml`:
HD1080 at 15 Hz, NEURAL_PLUS depth, 0.3-10 m, ...), so the replay matches the
live settings; the SVO keys above go on top of it, then `params`. Topics,
camera model and these files are the `zed` section of
`configs/recording.yaml`. `zed.bags` lists one bag per replay; by default
one, `zed`, with the paper's `mobile_1/zed` topics (`tables/topics.tex`: left
image, depth, odom, pose, paths; the IMU is already in the raw bag) and the
right camera. About 100 GB for an 18-minute recording (960x540 at 15 Hz,
stored uncompressed). The
point cloud is left out: it is depth + left camera_info, ~145 GB more.
`svo_to_bag.py --bag zed_right` makes one of them alone. The wrapper log and the parameters used are kept in
`<out>/zed_logs/`.

Where to run: in any terminal. When `ros2` + `zed_wrapper` are not installed
there, `run_zed.sh` runs itself in the running isaac_ros container
(`ZED_CONTAINER`, default `isaac_ros_dev-x86_64-container`) with `docker exec`;
`~/workspaces/isaac_ros-dev` (`ISAAC_ROS_WS`) is `/workspaces/isaac_ros-dev`
in there, so the SVO and the output must be below it (data/raw and data/work
are). Start the container first (`run_dev.sh`).

## Checking and merging a pass

```
python scoop_pipeline/processing/merge_session.py ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A --check
python scoop_pipeline/processing/merge_session.py ~/workspaces/isaac_ros-dev/data/raw/20260924/mapping_A
```

A pass folder holds one folder per machine (`mobile_1/`, `mobile_2/`,
`infra_1/`, ...). Each bag's machine comes from its name
(`..._<machine>`, `_r2` repeats included) and `configs/record.yaml` -- the
robots' own file -- says what it must contain, like `record.sh`'s preflight:
every topic listed must be there and not empty (`MISSING`, `EMPTY`; a topic
in `check.may_be_empty`, like `/*/ntp/events`, which publishes only on a clock
step, must be there but may have no messages: `ok/none`),
except what the pass mode does not record (`check:` in `record.yaml`:
mapping has no wifi/iperf/ntp/diagnostics; the mode comes from the pass
folder's name, `--mode` sets it). A
machine contributes its processed bags where they exist -- mobile_1: the
retimed bag and the ZED bags, checked for the recorded topics renamed plus
the decoded ones -- and its raw bags otherwise. A new machine (mobile_2,
infra_N) needs only its entry in `record.yaml` and its folder in the pass.

`--check` stops after the check. Otherwise, if nothing is missing (or with
`--allow-missing`), every bag is merged in log-time order -- what
`ros2 bag convert` with `all_topics: true` did, without ROS -- into
`data/work/<date>/<pass>/<prefix>_<pass>_<date>_merged/` (named from the folders, not the bags' own labels; prefix from `record.yaml`), byte for byte,
QoS kept, and checked again. Every bag the pipeline writes (decoded, retimed,
ZED, merged, tf_edit / merge_bags output) is zstd-compressed inside the MCAP:
lossless, `ros2 bag play`, GLIM and scoop read it as is, about half the size
for a pass of raw images; `--compression none` writes it uncompressed. It
stops and removes the partial bag before the disk fills. `--out` puts it on another drive, `--name` names it.

The merge also adds the calibrated transforms of `configs/static_tf.yaml`
(e.g. `os_lidar -> zed_left_camera_optical_frame`) to `/tf_static`. All
static transforms -- the inputs' (Ouster, ZED) and these -- go into ONE
latched `/tf_static` message at the start of the bag, as tf2's static
broadcaster does: `ros2 bag play` replays only the last message of a latched
topic to a late subscriber (rviz, tf2_echo), so separate messages would lose
all but one set of transforms. A frame has one parent in TF, and the ZED's
optical frame already has one (its camera chain), so the merge then adds the
same geometry as `zed_left_camera_optical_frame -> os_sensor` (the root of the
Ouster tree), composed through `os_sensor -> os_lidar`: looking up
`os_lidar -> zed_left_camera_optical_frame` gives exactly the calibration.
It prints what it added; `--static-tf ''` adds nothing.

Once the map stages have run on the pass, merging it again gives the complete
TF tree (`map_stages/pass_tf.py`), taken from the config in `map_stages/` whose
dataset is that pass (`--pipeline-config` picks another, `--no-pipeline-tf`
leaves it out; `--check` prints the tree without merging):

```
map
├── board                   05's start board
│  └── map_zed              mobile_1's origin, 05: the ZED's map at the start board
│                           (a run without 05, coop: mobile_1's start)
│     └~~ zed_camera_link   mobile_1 on its LiDAR trajectory (04's in a run, 02's
│        └── ... ZED, os_sensor, radars   through T_N_world in a mapping pass);
│                           replaces the ZED's map_zed ~~ odom_zed ~~ zed_camera_link
│                           (odom_zed is dropped)
├── board_rs                05's RealSense board
│  └── map_realsense        mobile_2's origin (07 camera "origin_frame", below its "board")
│     └── camera_link       parked: where 07 placed it
├── board_anchor_b, ...     05's other boards
└── arducam_optical_frame   07 (its ChArUco pose under map_zed is dropped)
   └── infra1_link          the radar, configs/static_tf.yaml
```

## mobile_2: camera-LiDAR calibration at the pauses

Koide's direct_visual_lidar_calibration (DVLC) wants short bags with the
sensors still. mobile_2's recording has Ouster packets only and stops several
times. Two steps make one complete bag on one clock, then one bag per pause.

```
RAW=mirc_dataset_mapping_2_run_20261015_mobile_2      # the raw rosbag2 folder
W=work/mobile_2_calib

# 1. decode: packets -> points + IMU; every other topic copied (camera, VSLAM,
#    /tf, /tf_static); os_lidar, os_sensor, os_imu -> *_mobile2 everywhere
python processing/decode_ouster.py $RAW $W/decoded \
    --packets-ns /mobile_2/ouster --out-ns /mobile_2/ouster --frame-suffix _mobile2 \
    --static-tf configs/static_tf.yaml \
    --copy-all --exclude /mobile_2/ouster/lidar_packets --exclude /mobile_2/ouster/imu_packets
#    (--static-tf: its calibration, os_lidar_mobile2 -> colour camera, into /tf_static)
# 2. retime: the Ouster on the PC clock, with everything else  ->  the processed bag
python processing/retime.py $RAW $W/decoded $W/retimed /mobile_2/ouster /mobile_2/ouster
# 3. the pauses: look first, then cut
python processing/dvlc_pauses.py $W/retimed $W/pauses --list
python processing/dvlc_pauses.py $W/retimed $W/pauses
# 4. DVLC on every pause together (docker; output in $W/pauses_preprocessed)
processing/run_dvlc.sh preprocess $W/pauses    # then guess, calibrate, viewer
processing/run_dvlc.sh guess      $W/pauses
processing/run_dvlc.sh calibrate  $W/pauses
processing/run_dvlc.sh viewer     $W/pauses
```

Timing check (the RealSense IMU against the retimed Ouster IMU; ~0 s and a
correlation above 0.8 = one clock; 20261015: -0.018 s, 0.97):

```
python processing/imu_sync.py $W/retimed --topic /mobile_2/imu --ref /mobile_2/ouster/imu
```

A pause is where, over +-0.25 s, the VSLAM odometry moves under 0.01 m/s, the
RealSense gyro reads under 0.02 rad/s and its accelerometer shakes under
0.08 m/s^2 (std), for at least 4 s; 1 s comes off both ends and at most the
middle 10 s are kept (`--max-speed --max-gyro --max-acc-std --min-pause
--trim --max-len`). `--list` prints the table with each pause's worst values;
`--only 0,2,5` cuts just those, `--intervals 120.5-130,300-310` (s from the
bag's start) skips the detection. Each pause goes to `pause_NN/` with the
three DVLC topics byte for byte; `pauses.yaml` records what was cut. For the
infra1 camera instead: pass `/mobile_2/infra1/image_rect_raw` and
`/mobile_2/infra1/camera_info` as `--image --camera-info`
(and to `run_dvlc.sh` as `IMAGE_TOPIC=... CAMERA_INFO_TOPIC=...`).
`run_dvlc.sh` mounts only the `pause_*` folders, read-only; `DVLC_IMAGE`
picks another image than `dvlc:humble-mcap`.

## Merging chosen topics of any bags

```
python scoop_pipeline/processing/merge_bags.py <bag> <bag> [...] --list
python scoop_pipeline/processing/merge_bags.py <bag> <bag> [...] -o <out> --topics '/mobile_1/zed/left/*' /tf /tf_static
python scoop_pipeline/processing/merge_bags.py <bag> <bag> [...] -o <out> --pick
```

`--list` shows every topic (type, messages, which bags). `--topics` keeps
the ones matching (shell patterns; quote them), `--exclude` drops some,
`--pick` asks with a numbered list (`1,3,5-8`, `all` or patterns). A chosen
topic that is in more than one bag is refused (every message would be there
twice) unless `--source PATTERN=N` takes it from bag N (the Nth on the command
line) or `--keep-duplicates` keeps all. Same merge
as `merge_session.py`: log-time order, byte for byte, QoS kept, one latched
`/tf_static`, zstd-compressed unless `--compression none`; `--static-tf` adds
calibrations. Only the chosen topics are read.

## TF: showing and changing it

```
python scoop_pipeline/processing/tf_edit.py <bag> --show
python scoop_pipeline/processing/tf_edit.py <bag> -o <out> \
    --drop 'map_zed->odom_zed' 'odom_zed->*' \
    --add os_lidar radar3_link 0.1 0 0.2 0 0 0 1 \
    --add-file scoop_pipeline/configs/static_tf.yaml \
    --odom-tum <glim>/traj_lidar.txt --odom-frame os_lidar --odom-parent map
```

`--show` prints the tree (static edges `──`, /tf edges `~~`). `--drop`
removes edges from /tf and /tf_static (shell patterns). `--add PARENT CHILD x
y z qx qy qz qw` (what `tf2_echo PARENT CHILD` prints) and `--add-file` add
static transforms. `--odom-tum` (a TUM file, e.g. GLIM's `traj_lidar.txt`) or
`--odom-topic` (Odometry / PoseStamped) put in a new odometry: poses of
`--odom-frame` in `--odom-parent`. A frame keeps one parent: a transform whose
child already has one is added the other way round, and the odometry goes in
as `<parent> -> <root of the tree>`, composed through the static chain, so the
lookups give exactly what was given. The result is a new bag with every other
topic copied; all static transforms in one latched /tf_static.

## Datasets: Replica and MCD (`datasets/mcap_convert.py`)

```
B=<merged bag>  P=<data>/processed/<date>/<pass>
python scoop_pipeline/datasets/mcap_convert.py $B --format replica \
    --pose-tum $P/odometry/mobile_1/traj_lidar_refined.txt \
    --depth-source map --map $P/mapping/denoised_<pass>_<date>.pcd
python scoop_pipeline/datasets/mcap_convert.py $B --format mcd \
    --pose-tum $P/odometry/mobile_1/traj_lidar_refined.txt \
    --depth-source map --map $P/mapping/denoised_<pass>_<date>.pcd
python scoop_pipeline/datasets/mcap_convert.py $B --inspect
```

Without an output folder, a dataset goes to
`processed/<date>/<pass>/datasets/<machine>/<format>[_<depth>]_<pass>_<date>/`
(`replica_map_mapping_A_20260924`, `replica_lidar_...`, `mcd_map_...`,
`mcgs_...`), the machine from the colour topic; an MCD sequence is
`<pass>_<date>` (`mapping_A_20260924_merged.bag`). The files inside keep the
names their format requires.

Replica: `results/frameNNNNNN.jpg` + `depthNNNNNN.png` (uint16, metres = value /
6553.5 as Replica, so 0-10 m; `--depth-scale 1000` for mm), `traj.txt`
(camera-to-world, OpenCV axes), `splatam_data_config.yaml`, `report.txt`.
Before a SLAM run, `python checks/replica_check.py <dataset>`: it checks the
config against the files, scores how well the depth edges sit on the colour
image (against the mirrored depth as chance) and how well the poses move one
frame's depth onto the next, and writes `<dataset>/check/frameNNNNNN.png`
(colour | depth | depth edges on the colour). Depth that does not fit the
image means the pose the depth was rendered at is not the image's pose (a
clock offset between images and trajectory, or the wrong extrinsic); no SLAM
config recovers from that.
MCD: `<seq>_merged.bag` (ROS1), `groundtruth/pose_inW.csv`, `gt_tum.txt`,
`calibration.yaml`, `camera.yaml` / `imu.yaml` / `lidar.yaml`. Clouds are as in
MCD's Ouster bags: `ring` uint8, no return = (0, 0, 0).

Gaussian-LIC2 (Coco-LIC front end + Gaussian-LIC) runs on the MCD bag; the MCD
run also writes its configs (`lic2:` in the config; `lic2: false` for none):
`lic2/cocolic/ct_odometry_<seq>.yaml` + `lic2/cocolic/<seq>/` (copy into
`Coco-LIC/config/`) and `lic2/gaussian_lic/<seq>.yaml` (into
`Gaussian-LIC/config/`). Then `roslaunch gaussian_lic mcd.launch
config_path:=config/<seq>.yaml` and `roslaunch cocolic odometry.launch
config_path:=/config/ct_odometry_<seq>.yaml bag_path:=<the bag>`. The IMU is
`yaml.imu` (the Ouster IMU); the bag must start at rest (static IMU init).

Poses: `--pose-tum` takes a TUM trajectory (the refined one, of the LiDAR
frame; `--tum-frame` says otherwise), chained to the camera (replica) or the
MCD body frame through the bag's `/tf_static`; without it, the `--pose` topic.

Depth (`--depth-source`; for MCD it replaces the config's depth sensors):
- `lidar` (replica default): the `--lidar-scans` scans nearest each image,
  every point moved to the image time through the poses at its own time,
  projected with the colour camera_info, nearest point per pixel, 0 where
  there is none; `--occlusion-px` / `--occlusion-margin` drop points the camera
  cannot see. Sparse.
- `map`: rendered from the built map (`--map`, at `--map-voxel`, 2 cm) as
  surfels -- each map point a disc on its local plane, the pixel's depth where
  its ray meets the nearest disc: dense, exact on slanted surfaces, edges
  about one voxel wide. The map is in the frame of the trajectory it was built
  from, so give that trajectory as `--pose-tum`. Moving things are not in the
  map (01 removed them). On the GPU when cupy works (`--cpu` to stay off it).
- `zed`: the ZED depth image.

The file is standalone (no scoop imports).
