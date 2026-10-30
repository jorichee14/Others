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
  convert_cloud.py   a point cloud to another format (pcd, ply, xyz, ...), optionally thinned
  rename_frames.py   rename TF frames in a bag (headers, Odometry child, /tf, /tf_static), e.g.
                     --suffix _mobile2 for a bag decoded before --frame-suffix
run_pass.py          one pass from raw to processed, every step in order (below)
clean_pass.py <config> [--apply] [--work]  keep a pass's current results, move older
                     versions, side runs and caches to processed/<date>/<pass>/_old/
                     (a dry run without --apply; clean.keep in the config keeps more)
clean_pass.py <config> --redo [--apply]  the pass again from GLIM's export: all it
                     makes (maps, frames, bags, figures, comms, report, 02's outputs) to
                     _old/redo_<date>_<time>/; then run_pass.py <raw pass> --from 02_ba
check.py <command>   every check, one command (check.py -h; check.py <command> -h):
    map     <map.pcd>       surface thickness (walls, floors; a wall's two faces vs
                            one surface mapped twice with --traj; --save patches.ply)
    walls   <config>        the map's wall spots with the scans placed by a trajectory:
                            one scan (the sensor), 1 s ... whole run, the map
    maps    <A> <B>         completeness / accuracy of map B against map A
    distances <map> <csv>   control distances (wall to wall, laser meter) against
                            the map: each wall fitted around a picked point (RANSAC),
                            error per distance, RMS, scale
    noise   <bag>           the LiDAR's noise floor from a bag alone (no trajectory,
                            no map): one scan's thickness on walls / floors, by range
                            and incidence
    scans   <config>        single scans and pairs: noise floor, deskew, time offset, drift
    trajs   <config>        a pass's trajectories: ATE / RPE against --ref, heading vs
                            turning, loop closures
    time    <config>        is GLIM's pose time its scans'? (run_pass.py's time_check)
    imu     <config>        is the IMU on the LiDAR's time?
    loop    <config>        loop closure on the boards (05 --check)
    figures <config>        a figure per step that is done, PNG + PDF in processed/<pass>/
                            figures/: timing, ba, trajectories, maps, wall (one wall cut
                            across), loop, stages (needs walls --json), noise (--bag)
    chain   <config>        is the mapping pass's chain each step once, nothing stale:
                            the export's pose-time shift applied once, BA's input /
                            output, the keyframe cache's input, every 01 cloud newer
                            than its poses (01 resumes from clouds on disk), 02b,
                            the report; lists what is in no step
    tf      <config>        a robot's TF in 10's poses bag (--bag the finalized one): the
                            base link -> LiDAR against the calibration, the LiDAR and
                            camera in map at every pose against the reference
                            trajectories, global_pose against the tree
    smooth  <config>        how smooth each trajectory is (GLIM, 02b, BA): jitter about a
                            local quadratic fit (0.5 s, 1 s), jerk, angular acceleration;
                            also while standing still; no bag, no map
    method  <config>        the method on one wall spot, from the pass's own clouds:
                            path and map, the 0.25 m patch and its thickness, the
                            keyframes that saw it before / after BA, each visit as a
                            submap before / after BA, union vs TSDF;
                            PNG + PDF in processed/<pass>/figures/method/ (--spot x,y,z)
    report  <config>        a mapping pass: GLIM, + time correction, + scan-to-map, + BA
                            against the reference; TSDF and union maps (run_pass's report)
  the recording and the environment (scripts in checks/, run by check.py as they are):
    env                     is this environment ready? (versions, cv2.aruco, open3d, GPU)
    decoder <bag>           the fast bag decoder against the generic one, field by field, timed
    ouster  <packets bag>   offline Ouster packet decoding against the replayed clouds
    sync    <bag>           a robot's camera and LiDAR on one clock: the turn rate from
                            image shifts and from scan azimuth shifts, per segment (PASS/FAIL)
    board   <bag>           the LiDAR's noise on a ChArUco board placed at several
                            distances: the board found by the camera, its returns fitted
                            a plane of their own; per station sigma along the beam,
                            5-95 % thickness, the plane's scan-to-scan repeatability
                            standing still (what a fine movement is read above, the one
                            number with no camera in it) and its step from the station
                            before, bias against the camera's plane, and sigma = a + b d
                            fitted over the stations. --runs keeps stations millimetres
                            apart apart, for a board shimmed by a known few mm
    glim    <machine folder> a GLIM run: trajectory summary and loop closure on the boards
    topics  <bag> [PATTERN] what a bag's topics hold: type, count, definition, first messages
    spacing <bag> <topic>.. intervals, gaps and repeated frames of topics in a bag
    euroc   <export>/<robot> a euroc.py export's IMU timing against its images and ground
                            truth, gravity while still, T_BS rotation
    replica <dataset>       a Replica-format dataset fits itself (depth vs image, poses)
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
  swarm_slam_lidar.py  the same in LiDAR mode: each robot's Ouster cloud (/r<i>/pointcloud)
                     and any method's odometry (--odom-tum MACHINE TUM [FRAME]: KISS-ICP,
                     FAST-LIO2, ...; or the ground truth), stage 04's track as ground truth,
                     --start/--duration for a slice; slam/run_swarm_slam.sh runs it too
                     --boards (--boards-only for an export already written): board_start.json,
                     each robot's odom frame in map by its start board (03's session anchor),
                     and its error against the ground truth in report.txt; run_swarm_slam.sh
                     with BOARDS=1 gives cslam that as one inter-robot measurement between the
                     first keyframes (slam/board_rendezvous.py), runs/<channel>_boards
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
  lod3dgs.py         a replica export + the map as a LOD-3DGS (LetsGo) scene: COLMAP
                     layout with the known poses, depth for its depth loss, starting
                     points sampled from the trimmed Poisson mesh of the map
analysis/            measuring results; reads outputs, writes nothing back
  headroom.py configs/headroom_terms.json [--traj ref.tum ...]  E1: every testbed error term
                          against the fusion breaking points, like against like, log axis
  maps.py, trajs.py  the map and trajectory checks, run through check.py (above)
  (slam/channel_relay.py: ROS 2 relay between robot domains through scoop/channel.py;
   slam/run_swarm_slam.sh <export> <channel>: one Swarm-SLAM run per channel;
   slam/run_hydra.sh <hydra export> <channel>: every robot's Hydra streaming its
   scene graph to a server through the channel, one run per channel)
  slam/patches/cslam_gnc_known_inliers.py  run once in the container, then colcon build
                     --packages-select cslam: GNC may reject odometry edges (it cut robot 1's
                     chain on site_2, 3 loop closures); the prior and the odometry become its
                     known inliers, loop closures alone are judged
  slam/patches/cslam_lidar_lc_direction.py  run once in Swarm-SLAM's container: its LiDAR
                     loop closures (TEASER++ + ICP) enter the pose graph inverted; measured
                     against the ground truth, they are right only the other way round
  analysis/slam/relative_pose.py  the robots' relative pose over a Swarm-SLAM run, as collaborative
                     perception uses it: at every second the latest merged graph, carried by
                     each robot's odometry, against the ground truth; share of the run with a
                     shared frame, translation / rotation error, a point at --range m, usable
                     share (<export> --run <run> [--run ...] --out plot.png)
  analysis/slam/relative_pose_loc.py  the same for two robots each localized in the map (two
                     fast2lamaa_loc runs): no graph, no alignment, the map as the shared frame
  analysis/slam/trajectory_video.py  top-view videos over the reference (and --map's floor plan): two
                     robots' odometry each placed at its own first pose (odometry --run <run>
                     --run <run>), or Swarm-SLAM's graph in time order with its loop closures,
                     robot 1 faded in its own frame until the merge (swarm <export> --run <run>);
                     each inter-robot loop closure shown in action: highlighted, the time stopped
                     2 s at the first 3 (--hold, --hold-s), and an inset of its two keyframes'
                     scans from the export's bag, robot b's with no shared frame and by the loop
                     closure (--no-scans to skip reading the bag)
  slam/odometry/      the odometry methods of slam/odometry/PLAN.md (KISS-ICP, RKO-LIO,
                     FAST-LIO2, OpenVINS, RTAB-Map's rgbd_odometry) in one ROS 2 image
                     (build.sh); run_odometry.sh <method> <bag> <machine> <run>: the bag at 1x,
                     a trajectory, and the cost (slam/monitor/: messages.csv, resources.csv,
                     summary.json: latency, drops, Hz, MB/s, CPU, RSS, GPU); make_config.py
                     makes each method's config from the bag; processing/run_vslam.sh does the
                     same for cuVSLAM with VSLAM_MONITOR=1. 2Fast-2Lamaa (fast2lamaa: odometry +
                     mapping; fast2lamaa_loc: localization in the reference pass's anchored map,
                     scored also in map with no fit) in its own Jazzy image (build.sh 2f2l);
                     compare.py tables every run and robot-to-robot result given;
                     processing/vslam_loc.py: cuVSLAM localizing in a map it saved on the mapping
                     pass (map <mapping config> --ref-topic /mobile_2/global_pose, then localize
                     <coop config> --map-from <mapping config>), put in the twin keyframe by
                     keyframe (score/; the one fitted transform in score_fit/), scored the same way
                     rtabmap_map / rtabmap_loc (build.sh rtabmap): a visual map built on the
                     mapping pass's /mobile_2/global_pose (in the twin by construction), each
                     coop run localized in it, scored in map like fast2lamaa_loc
  slam/cslam_eval.py  Swarm-SLAM run against the ground truth: ATE per robot and joint,
                     inter-robot loop closures checked, time to merge, bytes exchanged
  slam/kimera_eval.py  Kimera-VIO (slam/run_kimera.sh) against the ground truth: ATE and
                     drift in map per run, or per stretch without a dropout (STRETCHES=1)
  slam/hydra_eval.py  Hydra against the ground truth as cslam_eval does Swarm-SLAM: ATE per
                     robot and joint, in map, the odometry baseline, loop closures
  slam/relay_stats.py  what a channel did to a run's messages (relay.csv), runs side by
                     side: MB, lost, delay, backlog; --to server: map age at the receiver
  slam/dsg_stream.py  Hydra's scene-graph stream: log (in Hydra's container) the size of
                     every hydra/backend/dsg update against bag time; plot size and rate
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
checks/              the recording and environment checks, run through check.py (above)
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
  test_relative_pose.py  relative_pose: no shared frame before the first merged graph, a graph's
                     0.3 m offset carried by the odometry until the next graph, then 0
  test_cslam_eval.py cslam_eval: keys to robots, ATE, wrong loop closure flagged, merge time
  test_swarm_slam.py swarm_slam: topics, intrinsics, registered depth, odometry in the
                     robot's odom frame at the image stamp, ground truth
  test_cslam_lidar_patch.py  the patch: compute_transform returns the pose of dst in src,
                     a second run changes nothing, unknown code is refused
  test_trajectory_video.py  trajectory_video: the swarm graph fitted onto the ground truth, robot 1
                     faded until the merge, edges sorted, the loop closure's scans overlapping
                     once placed by it; odometry placed at its first pose; both videos written
                     with the frames asked for (and the hold)
  test_rtabmap_map.py rtabmap_map: the pose topic relayed as RTAB-Map's odometry, the database
                     path, /tf_static played, refused without the pose; rtabmap_loc: the database
                     copied, localization mode, refused without a database
  test_vslam_loc.py  vslam_loc: the hook script compiles; keyframes fitted back onto the twin give
                     the map's pose (residual 0); a bent map anchored keyframe by keyframe;
                     map -> odom (latest at each stamp) @ odometry from the localization on
  test_fast2lamaa.py fast2lamaa: the LiDAR in the IMU frame, the IMU as it is, the bag's clock;
                     fast2lamaa_loc: the map as gp_map loads it, the reference's IMU pose at the
                     first scan, refused without a reference; score's in-map error (0, then
                     a 10 cm offset the rigid fit hides); relative_pose_loc 0 and 0.2 m;
                     compare's rows
  test_odometry.py   make_config: frames, beams and extrinsics from the bag, the refusals
                     (no IMU noise, the ZED, depth outside the left camera, a 1 fps camera);
                     summarize: outputs matched to their inputs (FAST-LIO2's late stamp,
                     OpenVINS's early one), drops, repeats, resources over the play
  test_swarm_slam_lidar.py  swarm_slam_lidar: clouds and odometry per robot in the slice,
                     no-returns and the robot itself out, an IMU-frame odometry moved to the
                     LiDAR in its own odom frame (or in map with --start-at-gt), ground truth
  test_euroc.py      euroc: stereo pairs at --rate, unpaired frames out, IMU increasing,
                     intrinsics from P, T_BS from /tf_static, flags, --imu-only --imu-delay
  test_hydra_export.py  hydra: a bag per robot, images byte for byte, /tf composing to the
                     odometry with a z-up body frame, config frames, range and mesh stream
  test_hydra_eval.py hydra_eval: odometry shift aligned away, Hydra's offset in map, --cov,
                     loop closures, json and figure
  test_camera_lidar_sync.py  check.py sync: synced bag passes (Ouster start-of-sweep stamps, robot
                     body and a blocked sector in view), a LiDAR 0.3 s late found and fails
  test_imu_sync.py   imu_sync: a ZED IMU 2.12 s late found against the reference IMU, a still
                     robot not sure; processing/imu_sync.py --out moves only its stamps; retime step 5
  test_kimera_stretches.py  kimera_stretches: a dropout and an IMU-only gap split three stretches;
                     kimera_eval: ATE and in-map error per stretch, a failed stretch, a whole run
  test_hydra_kimera.py  hydra --poses kimera: the camera from Kimera-VIO's IMU at every frame of its
                     runs, a right loop closure sent, one the odometry contradicts left out
  test_imu_check.py  check.py euroc: a gyro 2.1 s late found against images and ground truth;
                     an IMU turned 90 deg found by gravity and the rotation fit
  test_relay_stats.py  relay_stats: delay, loss, backlog of a queued stream, map age
  test_dsg_stream.py dsg_stream: sizes, rates from bag time between updates, figure
  test_replica_multiagent.py  replica_multiagent: common camera, lockstep frames, poses,
                     colour resampling and depth registration on a two-robot bag
  test_lod3dgs.py    lod3dgs on a synthetic open-top room: centred camera, poses, depth,
                     the Poisson surface over the open top trimmed, point colours; mesh:
                     the map's colours, the ceiling found and cut, the walls kept; view:
                     floor holes filled (not under a box, a stair opening, over max), walls flat;
                     mirror ghosts and loose pieces (courtyard, floating) out, furniture kept
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
  test_distances.py  check.py distances on a synthetic room: 8 m and 6 m to 0.1 mm: csv, picks alone, --ask, the --click floor plan with its reading box and json
                     from rough picks (one at the floor), a pick in the air
                     reported, a map scaled 1.002 read as +2000 ppm
  test_noise.py      check.py noise on a synthetic bag: one scan's walls and floors
                     read 3.29 x the points' 1.4 cm noise
  test_board_noise.py  check.py board on a synthetic bag: a 1.00 x 0.75 m board at 2, 3,
                     4 and 5 m, each point ranged along its own beam with 4 mm + 1.2 mm/m
                     of noise and a 10 mm bias, read back per station and as the line;
                     then the board shimmed 5 mm at a time at 2 m (--runs): the step read
                     back to a millimetre, the plane repeating to 0.6 mm scan to scan
  test_figures.py    check.py figures on a made-up pass: seven figures as PNG and
                     PDF, noise skipped without --bag
  test_chain.py      check.py chain on a made-up pass: consistent; then the export
                     shifted twice, a map older than the reference, a cache from
                     another input, each caught
  test_tfcheck.py    check.py tf: a made-up poses bag reads consistent; a base ->
                     os_sensor edge turned 1 deg caught against the calibration
                     and the reference LiDAR trajectory
  test_smooth.py     check.py smooth: a smooth path near zero, the same with 3 mm /
                     0.1 deg jitter read back as 0.72 / 0.89 x it
  test_method_figures.py  check.py method on a synthetic room: a wall seen on two
                     laps, thinner after BA than before, TSDF thinner than union
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
python check.py env               # versions, cv2.aruco, open3d, Ouster SDK, GPU
python tests/test_bag.py && python tests/test_ouster.py
```

After `environment.yml` changes: `conda env update -f environment.yml --prune`.
For CUDA 11 machines, swap `cupy-cuda12x` for `cupy-cuda11x` first
(`nvidia-smi` shows the version); without a GPU, stage 01 runs on the CPU.

Keep them in separate terminals: `source /opt/ros/<distro>/setup.bash` puts
ROS's Python packages on `PYTHONPATH`, where they shadow the env's.
`check.py env` warns when that has happened.

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
python3 tests/test_rename_bag.py   # a renamed bag folder keeps its files named after it
python3 check.py decoder <bag> [--points /ouster/points] [--image <topic>]
```

`check.py decoder` compares the fast path with the generic decoder field by field
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

A pass is the folder below the date: `raw/<date>/<pass>/<machine>/`
(`20260924/mapping`), or, for a pass at one of several sites that day,
`raw/<date>/<pass>/site_<N>/<machine>/` (`20261015/survey/site_1`). Each site is
its own pass: its own `work/` and `processed/` folders, named
`<pass>_site_<N>_<date>` (`survey_site_1_20261015`); a config names it as
`"survey/site_1"` (same date) or `"20261015/survey/site_1"` (`scoop/layout.py`).

A new pass first needs its config. `run_pass.py <raw pass folder> --new-config`
writes it (`map_stages/pipeline_config_<tag>.json`) from what the raw folder
holds, shows it and stops: a mapping pass by the robot with an Ouster, with
20261015's settings; a survey with mobile_1 driving, in the same day's mapping
pass's map, a parked mobile_2 placed by its LiDAR; a coop run with both
robots driving: mobile_2 with an Ouster bag as `pipeline_config_coop_site_1_20261015.json`
(GLIM on it, every scan to the map), without one as `pipeline_config_coop_2.json`
(cuVSLAM and its depth track against the map), the boards and the infra
camera from the day's survey passes (`boards_from`, `cameras_from`), and
mobile_2's start board to check. `--robot` and `--reference` override its choices. It also lists every
bag's time span and flags one that does not overlap the run: a bag from
another run copied into the folder (the site 2 sniffer bag in site 1's, on
20261015) fails later, at the comms step, with no positions.
The robots' topics and cameras come from the base config's `"robots"` block, so
the written config names only the robot. Check it, then run the pass.

`run_pass.py` runs every step of the pass in order and skips the ones whose output is there
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
| 01_seed | mapping pass with `"comparisons": true`, for 02b: `01_build_map.py --seed`, the map from GLIM's poses | `mapping/denoised_seed_<tag>.pcd` |
| 02b_s2m | mapping pass with `"comparisons": true`: `02b_scan_to_map.py`, every scan registered to that map, the comparison | `odometry/<machine>/traj_lidar_refined.txt` |
| 01_map | mapping pass: `01_build_map.py`, the map from `dataset.traj` (02's), fused as `01_build_map.fusion` says | `mapping/map_final_<tag>.pcd` |
| 01_union | mapping pass with a TSDF map and `"comparisons": true`: `01_build_map.py --union`, the same poses as a union of points, the comparison map | `mapping/denoised_union_<tag>.pcd` |
| report | mapping pass: `check.py report`, GLIM / + time correction / + scan-to-map / + BA against the reference (ATE, RPE, pose time, loop closure, path) and the TSDF and union maps (thickness, doubled surfaces, completeness) | `mapping_report_<tag>.md`, `.json` |
| 03_init | run: `03_init_from_boards.py`, start pose from the anchor board | `frames/session_anchor.json` |
| 04_reference | run: `04_reference_traj.py`, every scan registered to the anchored map; its `"tracks"` (`map_stages/depth_track.py`): another robot's depth frames registered to it from its own odometry; with `"method": "pgo"` (`map_stages/depth_pgo.py`) one pose graph instead: cuVSLAM between frames, weak across camera dropouts, the start anchor, boards and map fixes (search, then ICP with Hessian information), solved together; each pose's marginal covariance in `cov_<track>.csv` (σx, σy, σyaw, xy correlation, the xy ellipse's major semi-axis) | `odometry/reference_<tag>/` |
| 05_anchor | `05_anchor.py`: the boards (mapping pass: and the anchored map) | `frames/anchor_frame.json`, `boards_<tag>.json` |
| 06_cut | mapping pass, only when `06_cut` has floor/ceil: viewing copy | `mapping/*_noceil.pcd` |
| 07_cameras | `07_build_cameras.py`: infra / parked cameras in map | `frames/cameras_in_map.yaml` |
| 08_tfs | `08_emit_tfs.py` | `frames/publish_tfs_<tag>.sh` |
| 09_cloud | run: `09_build_coop_cloud.py`, the run's cloud in the map from every LiDAR 04 tracked (the pass's robot and each `kind: lidar` track, on its own trajectory), the movers removed by free-space carving (what the rays saw through; the span test of 01 is off here, `--min-span-s` brings it back), optionally the reference map merged in (`ref_map`) (span and free-space carving as in 01; `remove_dynamic` block, `--keep-dynamic` keeps them) | `mapping/cloud_in_ref_<tag>.pcd`, the dropped points in `cloud_in_ref_<tag>_dynamic.pcd` |
| 10_poses | `10_publish_poses.py`: poses + the complete TF; with a robot's `base_frame`: its base link (on /tf_static, or with `children` the root of its sensors, moved by the trajectory), `/<robot>/global_pose` = the base link in map, `/<robot>/camera_pose` = the camera optical frame | `bags/<tag>_best_poses` |
| finalize | the merged bag again, with 10's /tf, /tf_static and poses in it; a bag finalized in place before is only renamed | `<merged bag>_final`, in its place (the stages read it for dataset.bag from then on) |
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
`pipeline_config_survey_1.json`); `--config` names another.

The robot a pass maps with, and its sensors, are set once, in `dataset`; every
stage reads them from there (a stage block that sets one is refused, naming
where it goes). mobile_1's are the default (`pipeline_config.json`):

```
"dataset": {
  "bag": ..., "traj": ..., "machine": "mobile_1",
  "points_topic": "/mobile_1/ouster/points",            the LiDAR (02, 02b, 04, 09, the TF)
  "image_topic": "/mobile_1/zed/left/image_rect_color",  the camera (01's colours, 05's boards)
  "camera_info_topic": "/mobile_1/zed/left/camera_info",
  "camera_frame": "zed_left_camera_optical_frame",       its optical frame (05, 10, the TF)
  "camera_map_frame": "map_zed",                         the camera's own tracking origin
  "rectified": true                                      false: 01 and 05 undistort the image
}
```

The robot's LiDAR track (`traj_<machine>_lidar.tum`) and its entry in 10's
`robots` (that track, in `camera_frame`) follow from `machine`; 10's `robots`
lists only other robots. Each robot's body is set once for every pass, in the
base config's `"robots"` block: its base link, the root of its sensors
(`mobile_1_base_link` over `zed_camera_link`, `os_sensor_mobile1` and both
radars; `mobile_2_base_link` over `camera_link` and `os_sensor_mobile2`),
which the trajectory moves, and `local_pose`'s frame. 10 gives it to every
robot it publishes (a sensor a pass did not record is left out); the base sits
on the camera body until it is measured, then its pose in `measured_in` goes
there, for every pass at once. The ChArUco boards are the base config's `"boards"`: `board_1`..`board_4` by
dictionary (4x4, 5x5, 6x6, 7x7), 9 x 7 squares. A board's `"was"` lists the
names passes before 20261015 measured it under (`anchor`, `anchor_b`,
`rs_anchor`), so their stored poses, and a config that still says `anchor`, are
read under the new names; a board stays on its own TF frame (`board`,
`board_rs`, `board_3`, `board_4`). A pass uses the ones in `05_anchor.boards_used`;
one whose `square_len` / `marker_len` is not set stops 05 with its name.
A section needs `05_anchor.min_section_samples` (4) sightings to count; a
board that only seeds a placement, which 07's LiDAR registration then
decides, may pass on fewer: `"min_section_samples_by_board": {"board_4": 1}`
(the survey of 20261015: mobile_1's ZED passed board_4 once, the parked
mobile_2's RealSense faces it from 0.5 m).
A section is refused when its sightings spread more than `fuse.max_std_mm`
(20 mm) about their mean. That spread is the TRAJECTORY's jitter as much as
the detector's: a run whose scans were each registered to the map
independently wobbles, and a board seen from one spot over a few seconds
inherits it (site_2's survey, 20261015: 12 to 22 mm, where the same
sightings on GLIM's own smooth trajectory gave 2 to 8 mm). Loosen it for the
one board that needs it, not globally:
`"fuse": {"max_std_mm_by_board": {"board_3": 25.0}}`, and say in the pass's
results what the spread was.

Why a board was seen, or not (05 scans only the dataset's camera, so a board
meant for a parked robot's camera is scanned here):

```
python3 scoop_pipeline/map_stages/board_scan.py <config> --board board_4 [--camera realsense]
    [--from S --to S] [--period 0.2] [--save found|all|none]
```

scans one stream, the dataset's camera or a `07_build_cameras` entry's
(`--camera NAME`: its bag, topics and distortion; `--image-topic`,
`--camera-info`, `--bag` for any other), every `--period` s, and prints a line
per frame with a marker of the board: how many markers (the board's own, their
side in px), the chessboard corners found, and the pose verdict (ok, or
no_corners / few_corners / pnp_fail / ambiguous / reproj, the detector's own
gates as in 05); frames without a marker of the dictionary are absent. Then
the totals, when the board was in view, and whether 05's section gate (4
frames within 3 s) could pass. Written to `frames/board_scan_<board>_<camera>/`:
`contact_sheet.jpg`, the whole stream in thumbnails (green border: pose,
yellow: markers only), so what the camera looked at is seen at once, and the
frames with a marker annotated (`--save all`: every sampled frame).

A pass mapped by another robot copies this block with its values:
`pipeline_config_mapping_20261015.json` (mobile_2) and
`pipeline_config_mapping_20260924.json` (mobile_1). Once the merged
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

`check.py glim <raw machine folder>` runs both again on a GLIM run
that is already there. Stage 02 runs the same check on its refined trajectory
when it finishes and prints it next to GLIM's (`check.py loop <config>
[--traj FILE]` on any trajectory):

```
  against GLIM:
    anchor       GLIM   10.8 mm / 1.240 deg  ->  refined    4.2 mm / 0.310 deg
``` Without a map-stage config for the pass (or one that
maps with another robot) the loop closure is not measured.

Settings come from `configs/recording.yaml` (`--settings` for another file).
`base:` holds what every robot shares (driver ranges, copying, the GLIM image
and its pose-time offset); `machines:` holds each robot's own block (its Ouster
namespaces and frames, topic remaps, GLIM config, ZED), merged over `base` for
the machine folder of that name. A folder without a block, and a tool that names
no robot, get `default_machine`'s (mobile_1). mobile_2's (`mobile_2/` folder):

```
data/raw/<date>/<pass>/mobile_2/<its bag>/      Ouster packets on /mobile_2/ouster
data/work/<date>/<pass>/mobile_2/
    <bag>_decoded/   points + IMU, every other topic copied (camera, VSLAM, /tf,
                     /tf_static, the packets); Ouster frames *_mobile2
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
python3 check.py ouster <packets_bag> --points-bag <retimed bag> \
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
The stamps are surveyed first: a message stamped on another clock (the
wrapper's startup messages carry the wall clock, days from the SVO's
frames) is dropped and counted, and the reorder window grows to the
largest step back in time among the rest (up to 60 s; beyond, the error
names the topic and the step), so a topic the replay delivered late is
still written in order.

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

A short bag recorded for calibration (the three DVLC topics, `.db3` or MCAP,
no odometry or IMU) goes through `--intervals` as well, one output folder per
bag. Cut its first seconds: DVLC keeps only the first image of each bag, and
the latched image topic hands a new recording the camera's last frame from
before it started (20261030: bag 2's image showed bag 1's place).

```
for b in calib_bag calib_bag2 calib_bag3; do
  python processing/dvlc_pauses.py $W/$b $W/${b}_cut --intervals 2-20 &&
  mv $W/${b}_cut/pause_00 $W/pauses/pause_$b
done
```

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
Before a SLAM run, `python check.py replica <dataset>`: it checks the
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

## Running Sionna RT on the exported scene (`analysis/comms/sionna_run.py`)

```
python3 scoop_pipeline/analysis/comms/sionna_run.py <scene.xml> --tx X Y Z [--rx X Y Z ...]
    [--freq 5.745e9] [--max-depth 3] [--power 20] [--radio-map [--cell 0.5] [--height 1.2]]
    [--samples 1000000] [--out DIR] [--frame frames/anchor_frame.json]
```

loads the scene (`sionna.rt.load_scene`), sets the frequency (channel 149:
5.745 GHz) and isotropic antennas, and for each receiver traces the paths
up to `--max-depth` bounces: their number, the total path gain, the
line-of-sight delay, the RMS delay spread, the strongest paths, and the
channel impulse response written as `cir_<rx>.npz` (gains and delays per
path). `--radio-map` computes the path gain over a plane `--height` m above
the scene's lowest point in `--cell` m cells and writes `radio_map.npy`,
`.csv` and `.png`. Positions are in the scene's frame, the mesh's: a scene
made from `mesh_view_<tag>.ply` is in the mapping pass's GLIM world. With
`--frame` (that pass's `frames/anchor_frame.json`) they are given in map
coordinates, where the survey's trajectories, boards and cameras are, and
moved into the scene's frame; the radio map's csv then carries both. Test:
`tests/test_sionna_run.py` (Sionna's built-in box).

## One figure of a survey pass (`analysis/survey_figure.py`)

```
python3 scoop_pipeline/analysis/survey_figure.py <config> [--gif] [--ap X Y Z] [--link infra_1]
    [--near 3.0] [--band 0.3 1.8] [--seconds 30 --fps 10] [--out DIR]
```

writes `figures/survey/<tag>_overview.png` (300 dpi) and `.pdf` under the
pass's processed folder: the anchored map from above with the LiDAR
trajectory coloured by the RSSI to the access point (by time without comms
tables), start and end, the boards (05) with their normals, the parked robot
and the infra camera (07), the AP with `--ap`, a scale bar; Wi-Fi RSSI, PHY
rate, ping RTT and iperf goodput over the run, grey bands where the robot was
within `--near` m of an anchor; RSSI against link length per CSI link with
the path-loss fit; the CSI amplitude of `--link` per subcarrier over the run
with its delay spread; and 04's residual and correction per scan with the
boards' spreads and loop closures. Panels whose inputs are not there yet are
left empty with a note. `--gif` adds `<tag>_overview.gif` (and `.mp4` with
ffmpeg): the path growing with a heading arrow, cursors sweeping the time
panels. Test: `tests/test_survey_figure.py`.

## Gaussian splatting: LOD-3DGS (`datasets/lod3dgs.py`)

Block diagram of everything made from the reference map, each block with the
test that covers it: `docs/lod3dgs_pipeline.svg` (and `.png`), drawn by
`python3 scoop_pipeline/analysis/lod3dgs_diagram.py`.

LetsGo (Cui et al., ACM TOG 2024; github.com/zhaofuq/LOD-3DGS) trains
level-of-detail Gaussians from a LiDAR scan: known poses instead of SfM, a
depth loss, and starting points sampled from a mesh of the scan rather than
the raw points (2.5-3.4 dB PSNR better in their Table 3, against raw LiDAR;
01's TSDF map is already the averaged surface).

The mesh alone, to look at first (20261015: mobile_2, the BA poses):

```
P=<data>/processed/20261015/mapping
python scoop_pipeline/datasets/lod3dgs.py mesh $P/mapping/map_final_<tag>.pcd \
    --traj $P/odometry/mobile_2/traj_lidar_ba_glim.txt --ceil auto [--voxel 0.04] [--show]
```

writes `mapping/mesh_<tag>.ply` in the map's colours (colorize's) and
`mesh_<tag>_noceil.ply`, a viewing copy without the ceiling: every face
above the highest peak of the height histogram (printed) less
`--ceil-margin` (0.10 m), or above `--ceil Z`. `--voxel 0.04` meshes a
quarter of the points for a quick look (Poisson depth one less). Every face
is split in four (`--subdivide 1`) and every vertex takes the colour of the
map's nearest point (Poisson's own colours are averages, blurred and
greyed); the mesh is written without normals (`--normals` to keep them), so
a viewer shows its colours unlit. The mirror ghosts and loose pieces go as
in `view` (`--keep-ghosts`, `--min-piece 0` keep them). The `view` copy can be meshed the same way
(`mesh view_<tag>.ply` -> `mesh_view_<tag>.ply`): walls flat, floor filled.

A viewing copy of the map as points, its colours as crisp as the map's
(the mesh's are interpolated):

```
python scoop_pipeline/datasets/lod3dgs.py view $P/mapping/map_final_<tag>.pcd \
    --traj $P/odometry/mobile_2/traj_lidar_ba_glim.txt [--ceil Z] [--max-hole 20]
```

writes `mapping/view_<tag>.pcd`: stray points out (fewer than 8 others within
8 cm), the mirror ghosts out (the room the LiDAR saw in polished stone or
glass, put behind it: points no direct ray from the path reaches, cast from
1000 poses against a coarse mesh; `--keep-ghosts`), every piece not joined
to the building (a piece of `--min-piece`, 50 m2, or more) out when open to
the sky (a courtyard, facades and sheets seen through glass) or floating
1.2 m or more above the floor (lamps, signs; furniture under a ceiling,
down at the floor, stays), every flat 0.4 m cell's points put onto its plane (walls and floors
thin), and the floor's holes filled at the height and in the colours of the
floor around them. The floor is found from the trajectory (sensor height
from the floor under the poses); filled are the holes enclosed by floor and
walls and gaps up to 0.4 m, never under furniture or walls (points 0.15-0.6
m above the floor), never over a stair opening or an atrium (points below
it), never a courtyard (a hole over 1 m2 with a ceiling over less than
`--roof`, 50 %, of it), each up to `--max-hole` m2. For looking at only: the reference map,
and every measurement, stays `map_final_<tag>.pcd`.

On request, both clear what the robot drove through (`--body 0.25`): nothing
solid is within 0.25 m of the LiDAR's track from the floor to just over the
LiDAR, and a flat surface the robot crossed (poses on both sides of it) is
taken whole, its edges in the cells it shares with a wall too (the points on
its plane facing as it does): a sheet seen in glass that cuts across the way
and joins the walls it crosses. A wall the robot drove along stays.

With `--sky`, `view` also clears what is over open sky: a column
of 0.4 m with no ceiling of the building over it (horizontal surfaces 2 m or
more above the floor, within a column) and more than 0.6 m from the robot's
way: a courtyard's inside, whatever it touches, a sheet in it joined to the
wall it crosses included; the courtyard's walls stay, the ceiling reaching
them.

With `--bag <the bag the map was made from>` (`--topic`, the points;
`--scan-every 5`) `view` also clears what the LiDAR's own scans saw through:
every 5th scan as a range image, each 10 cm cell of the map in its view
seen through when the scan's return in that direction lies beyond it; a cell
seen through 3 times or more and over twice as often as seen at is empty
space (a wall's reflection in glass, there only as seen in the glass). The
trajectory must be the map's and the bag's (map_final with the BA
trajectory). A cell no scan saw is kept.

`--ceil auto` takes off the ceiling over each place: in every 0.5 m column
the lowest horizontal surface 2.3 m or more above the floor there (and the
next columns', so the wall it meets goes with it), and nothing above the
highest histogram peak, each less `--ceil-margin`. A low arcade beside a tall
hall loses its own ceiling; a courtyard's walls keep their height. `--ceil Z`
cuts at Z everywhere.

Anchored: make the view and mesh from `map_final_<tag>.pcd` with the BA
trajectory (GLIM's world), then move each into 05's frame:
`lod3dgs.py anchor <view or mesh> --frame $P/frames/anchor_frame.json`
-> `<name>_anchored` (T_N_world, the move the anchored map was made with).

`sionna` makes a Sionna RT scene of the mesh with its ceiling (not
`_noceil`: the waves bounce off it): the mesh's own faces, nothing rebuilt,
split into floor, walls, ceiling and objects. Sionna loads `scene.xml`; it
reads the `.ply` files in `meshes/` beside it, so keep the folder whole:

    <mesh>_sionna/scene.xml                     sionna.rt.load_scene(".../scene.xml")
    <mesh>_sionna/meshes/{floor,walls,ceiling,objects,glass}.ply   each in the mesh's colours, floor grey
    <mesh>_sionna/parts.json                    faces and m2 per part, and left out
    <mesh>_sionna.ply                           to look at: its own colours, floor grey
    <mesh>_sionna_labels.ply                    to check: floor grey, walls beige, ceiling white, objects orange
    <mesh>_sionna_left_out.ply                  what --traj left out (hanging from the ceiling), to look at

Every face of the mesh is in one part or left out: checked on every run
(the run stops when not) and printed as faces in = floor + walls + ceiling
+ objects + left out. Sionna ignores the parts' colours (it shows its
materials'); Mitsuba warns that they are integers, harmlessly.

The labels are found on an even copy of about 4,000,000 faces (vertex clustering: faces of one size, so a flat wall keeps enough of them; flatness is judged by area, not face count) and every face takes its
nearest one's: floor, the horizontal faces within 0.12 m of the floor (with
`--traj` the nearest pose less the sensor height, so a reflection under the
floor cannot pull it down; without, each 1 m cell's lowest); with `--traj`
what is more than 0.25 m under the floor (the room mirrored in a polished
floor, joined to the walls it mirrors) is left out; ceiling,
horizontal 2 m or more above it and the highest such overhead (a tall
rack's top under it is not); walls, the flat sheets standing on the
floor: a flat near-vertical surface (each cell's faces taken apart by the
way they face, so the two sides of a thin wall are two surfaces) of 0.5 m2
or more reaching 2.2 m or higher, every face on its plane (within 5 cm,
facing it, inside its extent, down to the floor behind the furniture in
front of it), when it comes down to the floor (0.3 m) somewhere; what is
left of a wall's shape (a pillar's rounded edges, 70 % within 10 cm of wall
faces) and round pillars (round in plan) with them. A rack or stacked
chairs (no flat sheet) stay objects. The seams facing sideways go with the
floor, wall or ceiling they touch; everything else is objects, but stairs
and ramps (all their faces climbing along the piece, 0.25-1.5 m per m,
over 0.8 m) and all that is against a flight (its sides, the landing, the
floor above it), and low walls between pillars (from the
floor, 1.5 m long or more, thin across, both ends at a wall) are walls;
and all that is attached to the walls (touching a wall face and nowhere
more than 0.2 m from them, or 90 % facing sideways, or off the floor, over
0.3 m, or reaching 2.2 m with 30 % of it above 1.8 m: door and window
frames, sills, panels, cabinets and signs on a wall, a pillar's curve,
pieces of wall, a wall cluttered) is wall too, and so are a thin upright sheet
within 0.5 m of a wall (a door set in its frame or ajar, a strip of wall), a
doorway's frame whose wall is not in the map (on the floor at both ends,
nothing under 1.7 m in its middle), what floats (off the floor, over 0.3 m) within
0.3 m of a wall or of what is wall by these rules (a ledge, a lintel, a door's part), what touches a wall, reaches 2.2 m and is
nowhere farther than 0.6 m from the walls (a wall cluttered), and of what is
left objects the faces within 0.1 m of a wall plane (the wall's own faces fused
with the furniture against it, a door set in its frame) and those over 2 m
within 0.5 m of a wall (the furniture under them stays objects); of the
fragments that leaves, one floating within 0.3 m of a wall, or with 90 % of it
within 0.25 m of one (a frame, a strip; a chair reaches out more), is wall. The
distance to a wall counts the wall's plane where the map has no wall (hidden by
what stood against it), so a cluttered wall is wall though the wall behind it
is missing and a wall's top (facing up, wall along both its
long edges; a table by a wall has it on one side): objects are only what stands out or
apart (a chair or desk pushed to a wall reaches out 0.4 m and more);
floor bumps (no higher than 0.12 m) are floor and scraps under 0.05 m2 the
part next to them. The floor is then flattened onto its own level (the
median of its flat faces per 1 m, blended between cells; a vertex moved 0.4 m at most, a floor at another level is not a bump): the bumps under
furniture go (`--keep-floor` keeps them). Every hole up to 0.8 m across is closed (its rim triangulated: the glass
panes the lidar saw through, which become `glass`, and the gaps the meshing
left, which take the part round them; a doorway is larger and stays open).
The floor's holes (where its reflection was cut away, or furniture hid it;
closed all round by floor, or walls and furniture standing on it, up to
100 m2) are filled with flat patches at its
level, and the ceiling's the same way (what hung from it cut away). The ceiling is flattened onto its
levels the same way (0.15 m at most). The walls are smoothed too: their faces
within 6 cm of a wall plane put on it (`--keep-walls` leaves them as meshed);
what stands off the planes (pillars, stairs, frames) keeps its shape but is
smoothed (Taubin), its ripples gone. And the wall hidden behind
furniture pushed against it, which the mesh does not have, is filled on its
plane from the floor (where something stands within 0.4 m of it and no wall
face is, on the plane or set back from it like a door), and so are the wall's own holes, closed all round by it (the
floor closing them below, its top above) and up to 1 m2: the walls
continuous. A larger opening that is not a doorway (off the floor: a window;
or wider than 2.5 m: a glass wall), which the lidar saw through, is filled
the same but as `glass` (`meshes/glass.ply`, ITU glass 0.01 m); doorways stay
open, and so does an opening at the top of its wall with ceiling on both
sides of the wall (over a parapet or a counter, inside the building: a glass
wall to the outside has ceiling on one side only). `--open X,Y` (repeat it)
keeps the opening at that plan point open by hand: no patch within 1 m of it. Filled faces are
counted apart (`--no-fill-walls`). With `--traj` (the mesh's
frame; anchored: `odometry/reference_<tag>/traj_mobile_2_lidar.tum`) what
the `_noceil` copy drops as hanging from the ceiling only (a reflection
joined to a mirrored ceiling: loose once the ceiling is cut, and reaching up
to the cut, and joined to the building before it) is left out too;
furniture, loose but well under the ceiling or joined to nothing, and
pillars standing on the floor, stay. `--flat-colours [W]` makes the colours one per segment, so the mesh
looks clean: each wall plane the median colour of its own vertices (the
median leaves the glints out), each pillar, frame or flight of stairs its
own, the ceiling one, each object one, glass a light tint; blended in by W
(default 1, flat; 0.7 keeps a trace of signs and posters). `--faces N` simplifies the floor, walls and ceiling together to about N faces (flat stays flat, in few faces) and `--object-faces M` the objects apart (0: as meshed), so objects keep finer detail than the walls. A light scene for Sionna (it loads a few
hundred thousand faces in seconds; 40 M takes minutes and GBs of GPU memory):
`--faces 200000 --object-faces 200000 --no-colours` (the parts and
`<out>.ply` in greys, one shade per part, floor dark to glass light, and
each object its own: Sionna shows its materials' own colours, and a viewer
these rather than its default, CloudCompare's green). Each part an
ITU material, `--material PART=TYPE[:THICKNESS m]` (default concrete 0.2 m,
objects wood 0.05 m); `--normals` writes the parts with vertex normals (a
viewer shades them as surfaces; without, the per-vertex colours make a
mesh look like points); `--why-objects N` logs the N largest objects with what each rule saw of
them (centre, m2, height, share facing sideways, size, distance to the
walls); `--why X,Y` (repeat it) logs, for the pieces left within 1 m of that plan
point (pick it in CloudCompare), what each rule saw: height, share facing
sideways, length and thickness across, how far its ends are from a wall,
how it climbs; `--split-objects` writes each object to
`meshes/objects/obj_NNN.ply` (load with `merge_shapes=False`; edit an
object's material in `scene.xml`).

    python3 scoop_pipeline/datasets/lod3dgs.py sionna mapping/mesh_view_<tag>_anchored.ply \
        --traj odometry/reference_<tag>/traj_mobile_2_lidar.tum

`fly` renders a video flying once round the area inside a mesh (`<file>_fly.mp4`,
H.264 through ffmpeg; its first frame beside it as `_frame0.png`): the free
floor (its holes up to 10 m2 filled, furniture's footprints, a courtyard not;
less every cell with anything at walking height: a wall however thin, a
pillar, furniture) shrunk `--inset` 1.5 m (less where that leaves nothing), the largest piece's outline the
path (a room behind a door narrower than 3 m falls off: the main area only);
from `--start` X,Y (the map's origin; `loop`: on the loop), a turn round there
over `--spin` 8 s, then straight to the loop's nearest point and round it; at
`--eye` 1.6 m over the floor, `--speed` 1 m/s, looking `--ahead` 2 m along
the way; or `--path "X,Y;X,Y;..."`, waypoints joined in a loop. Recorded through Open3D's OpenGL window (the one whose
frames read back on every driver; `--hidden`: not shown), the flight seen
as it goes, the mesh unlit in its colours made vivid, all of it however far; H.264 when ffmpeg is installed (`sudo apt install ffmpeg`), else
OpenCV's mp4v, which browsers show black (VLC plays it); `--fps`, `--size WxH` (1280x720), `--seconds N`
for the first N seconds only; the path flown is written beside the video
as `<out>_path.json` (eye and look-at per frame), for `splat_walk.py --fly`. `splat_walk.py --fly ... --eye-shift -1.0` flies the same loop
with the eye 1 m lower: the splats were trained from the robot's camera at
0.3 m, and seen from 1.6 m the ceiling, never observed that close, is blobs.

    python3 scoop_pipeline/datasets/lod3dgs.py fly mapping/mesh_view_<tag>_anchored_sionna.ply

After training, `splats-clean` removes the Gaussians training was unsure
about: farther than `--far` 0.15 m from the mesh's surface (floaters, wisps
in the air), opacity under `--min-opacity` 0.05, longer than `--max-scale`
0.5 m along an axis (a blob smeared over a wall) -> `<ply>_clean.ply`; and
`recolour --renders <dir>` gives the mesh the model's colours: the scene's
cameras rendered from the splats first (`splat_walk.py --render-views`, in
the LOD-3DGS environment: rgb and depth per view), then each vertex the mean
of the rendered pixels that see it, depth-checked, as the camera colouring
did with the photos; the Gaussians blended as training meant, so the colours
are clean (one by one their colours cancel each other out: confetti, which
is what `recolour` without `--renders` gives) -> `<mesh>_gs.ply`, to `anchor`
and `sionna` as before. Both take the unanchored mesh (the model's frame).

    python3 scoop_pipeline/datasets/lod3dgs.py splats-clean <run>/point_cloud/iteration_300000/level_5.ply \
        mapping/mesh_view_<tag>.ply
    # lod3dgs env
    python3 scoop_pipeline/datasets/splat_walk.py <run>/point_cloud/iteration_300000/level_5_clean.ply \
        --render-views $P/datasets/mobile_2/lod3dgs_map_<tag> <run>/renders --every 2
    # scoop env
    python3 scoop_pipeline/datasets/lod3dgs.py recolour mapping/mesh_view_<tag>.ply - --renders <run>/renders

`datasets/splat_psnr.py <renders> --scene <scene>` scores a model on the
scene's own images: PSNR and L1 of `splat_walk.py --render-views`' renders
against the images they were rendered for, per camera (each export's) and
all together, a row per view in `psnr.csv`. Two models rendered on the same
scene's cameras compare on the same views; a camera a model never saw
scores low, by design.

`datasets/splat_walk.py` walks inside the trained splats (or flies them round
the mesh flight's loop) with LOD-3DGS's own rasteriser, in its environment
(`conda activate lod3dgs; export PYTHONNOUSERSITE=1`): the same keys as
`walk` (arrows or the mouse to turn, W S A D Q Z to move, R, Escape).
`--scene <scene>` starts at the scene's first camera, `--at X,Y` elsewhere;
`--fly <mesh>_fly_path.json out.mp4` renders the loop `fly` flew (the path
it writes beside the video), `--frame frames/anchor_frame.json` moving it
from the anchored frame into the model's. The level files beside the one
given (`level_0.ply` the coarsest ... `level_5.ply` the finest) are all
loaded, and each Gaussian is drawn from the coarsest level whose cell is
still small on screen at its distance (`--px` 2 pixels: at 720p the finest
to 20 m, each coarser one over the next doubling), as LOD-3DGS's own
renderer does; `--no-lod` draws the one file.

    python3 scoop_pipeline/datasets/splat_walk.py <run>/point_cloud/iteration_300000/level_5.ply \
        --scene $P/datasets/mobile_2/lod3dgs_map_<tag>
    python3 scoop_pipeline/datasets/splat_walk.py <run>/point_cloud/iteration_300000/level_5.ply \
        --fly mapping/sionna_full_fly_path.json splats_fly.mp4 --frame $P/frames/anchor_frame.json

`walk` opens a window to walk inside a mesh (the one with its ceiling, or
its `_sionna.ply`): Open3D's fly mode, started 1.6 m (`--eye`) over the floor
at the first pose of `--traj` facing along the path (or `--at X,Y`), the
mesh unlit in its colours with the speckle smoothed over (a vertex much
brighter or darker than its neighbours, a glint of the camera's colouring,
takes their colour), smoothed over the surface to `--smooth` 0.06 m with
edges kept (a poster's or a door's outline stays, the stipple inside a wall
goes; 0: not) and made vivid (contrast up a little; `--gamma` under 1 lifts
the shadows, default 1; white balanced, the green of
fluorescent light gone, `--cast` keeps it; stretched over their range, the
camera's haze and grey gone; the saturation up `--sat` 1.2; `--plain`: as
stored) and seen from both sides (`--one-side`: as stored, half the load),
cut into blocks of `--cell` 4 m in plan with only the blocks within
`--near` 15 m of the eye drawn (and of those only the ones in view), so all
of a 40 M face mesh walks smoothly (`--faces N` simplifies it to N first;
default 0: all of it). Turn with the arrow keys (left/right turn, up/down tilt,
`--turn` 4 degrees each) or by dragging with the mouse; move with W
forward, S back, A left, D right, Q up, Z down (`--step` 0.1 m each; hold a
key to keep going: each key event moves that far whatever the frame rate,
where Open3D's own fly mode moves per frame and sticks when a big mesh
draws slowly); R goes back to the start; Escape or closing the window ends
it. `--shot view.png` writes the start view and closes.

    python3 scoop_pipeline/datasets/lod3dgs.py walk mapping/mesh_view_<tag>_anchored.ply \
        --traj odometry/reference_<tag>/traj_mobile_2_lidar.tum

The scene, from a replica export with map depth made with the same poses:

```
python scoop_pipeline/datasets/mcap_convert.py <merged bag> --format replica \
    --pose-tum $P/odometry/mobile_2/traj_lidar_ba_glim.txt \
    --color /mobile_2/color/image_raw --info /mobile_2/color/camera_info \
    --lidar /mobile_2/ouster/points --depth-source map --map $P/mapping/map_final_<tag>.pcd
python scoop_pipeline/datasets/lod3dgs.py scene $P/datasets/mobile_2/replica_map_<tag> \
    --map $P/mapping/map_final_<tag>.pcd [--mesh $P/mapping/mesh_<tag>.ply] --every 3
# in LOD-3DGS (git clone --recursive, its environment, pip install laspy plyfile), once:
python3 <Others>/scoop_pipeline/datasets/lod3dgs_patch.py .   # (or git apply <Others>/scoop_pipeline/datasets/lod3dgs_lod_levels.patch)
python train.py -s $P/datasets/mobile_2/lod3dgs_map_<tag> --use_lod --sh_degree 2 \
    --depths depths --densification_interval 10000 --iterations 300000 \
    --scaling_lr 0.0015 --position_lr_init 0.000016 --opacity_reset_interval 300000 \
    --densify_until_iter 200000 --data_device cpu -r 1      # its README's flags
```

The map is meshed (Poisson, `--poisson-depth`, default cells about two
`--voxel` across; normals turned to the nearest pose), faces farther than
`--trim` (0.10 m, the paper's) from a map point are removed (Poisson closes
open sides and holes with surfaces nothing saw), and the mesh is sampled
every `--spacing` (4 cm, the paper's) into `sparse/0/points3D.ply`, each
point the mean colour of the `--color-views` frames that see it (`mesh.ply`
is kept). The scene keeps the ceiling: the cameras see it. LOD-3DGS ignores
the principal point (it takes the image centre), so images and depth are
resampled onto the largest window centred on it; depth goes into `depths/`
as LOD-3DGS reads it (8-bit warped grey, alpha where there is depth). The
colour images are undistorted first when the camera has distortion (the
export writes its coefficients; for an older export `--bag <bag>` reads them
from its camera_info, or `--dist k1,k2,p1,p2`): the depth is pinhole already
and LOD-3DGS assumes a pinhole camera. The
map must be the one the export's depth was rendered from, so poses, depth
and points share a frame. Several exports make one scene (`scene <export 1>
<export 2> ... --bag <bag 1> --bag -`), each with its own camera in
`cameras.txt` (a ZED's intrinsics differ from a RealSense's), its frames
named `c2_frame...` from the second on; an export of another session, in
its own frame, is moved into the first's with `--transform - --transform
<align json>` (one per export, like `--bag`)
after `align` has registered that session's map onto this one's (its depth
is its own map's, consistent with its poses):

    python3 scoop_pipeline/datasets/lod3dgs.py align <other session>/mapping/map_final_<tag2>.pcd \
        $P/mapping/map_final_<tag>.pcd            # -> map_final_<tag2>_to_map_final_<tag>.json, _aligned.ply
    python3 scoop_pipeline/datasets/lod3dgs.py scene <export of this session> <export of the other> \
        <scene folder> --map $P/mapping/map_final_<tag>.pcd --mesh $P/mapping/mesh_view_<tag>.ply \
        --every 8 --bag <this session's bag> --bag - --transform - --transform <the align json>

`align` thins both to 5 cm, finds a global registration on FPFH features
(or takes `--init X,Y,YAW` by hand) and refines it with ICP point to plane;
its fitness is the share of the source within 5 cm of the target (under
0.5, look at `_aligned.ply` over the target in CloudCompare). The place must
not have changed much between the sessions: furniture moved shows the model
two things at once there. `--every N` thins the frames. `--mesh` samples a
mesh made beforehand (`lod3dgs.py mesh`: ghosts, loose pieces and objects
handled) instead of meshing the map again.

The level-of-detail start is written too: `lod/level_<k>.ply`, coarsest
first, as the paper makes it (Sec. 4.2): the finest level every starting
point (4 cm), each coarser one a point per cell twice the last one's size,
until a level has fewer than `--lod-min` (10,000) points. LOD-3DGS makes
its levels with its own PotreeConverter, a Windows build (a prebuilt LASzip
DLL, a `Converter.exe` run through the shell);
`datasets/lod3dgs_lod_levels.patch` (against zhaofuq/LOD-3DGS eec40cf) has
its loader read `lod/` instead, so no converter is needed on Linux;
`datasets/lod3dgs_patch.py <LOD-3DGS>` makes the same change whatever the
file's line numbers or endings. Its environment.yml pins PyTorch for CUDA
11.6 and Python 3.7, which no current nvcc or gcc builds; on Ubuntu 24.04
make the environment by hand: Python 3.10, torch 2.4.1 cu124, the CUDA 12.4
toolkit from the nvidia conda channel (`CUDA_HOME=$CONDA_PREFIX`), and the two
submodules with `pip install --no-build-isolation`, with `PYTHONNOUSERSITE=1`
so nothing in `~/.local` leaks in.

### Stage 04: when to turn `carry` off

`carry` starts each scan from its seed moved by the last well-fitting scan's
correction. That is right while corrections are the centimetres of GLIM's own
drift, and wrong once one is metres: the gate (`max_shift`) is measured from
the CARRIED seed, so a slide into a parallel corridor, geometry that fits the
map as well as the true one does, is handed to every scan after it and never
trips the gate. The run arrives bent and displaced with a healthy residual,
and the boards measured through it scatter far past 05's gates (site_2's
survey, 20261015: a 10 m median correction, 13 m of lateral error, every
board refused, while GLIM's own trajectory placed the same boards to 2-8 mm).

Round 1's own numbers decide it, and the stage now says so:

| what round 1 shows | what it means |
|---|---|
| median correction within a few times `max_shift` | GLIM's drift; leave `carry` on (the default) |
| median correction past 2 x `max_shift` | the carry accumulated a slide; check where the run came out, then re-run with `"carry": false, "max_shift": 0.3, "max_rot_deg": 2.0, "prior_beta": 0.5` |
| with `carry` off, a fifth or more of the scans keeping GLIM's pose | the gates are too tight for the seed, or the map does not cover where the run went |

`check.py glim <raw machine folder>` tells the two apart before any of this:
it measures the boards on GLIM's own trajectory, so millimetre spreads there
with metre corrections in 04 mean the registration is at fault, not the run.

### Sensor sync quality: `check.py sensors`

Do a robot's sensors agree on time, per frame? `checks/sync_quality.py` reads
the merged (final) bag and assesses every stream of every robot that has an
Ouster: the LiDAR, the RGB cameras, the depth images and the IMUs.

```
python3 check.py sensors <final bag> --config map_stages/pipeline_config_coop_site_1_20261015.json \
    [--robot mobile_1] [--out <dir>] [--dataset <OPV2V split dir>]
```

Two layers per stream. *Stamps*: rate, jitter (p95 of the period's
deviation), gaps, and per LiDAR frame the nearest message's distance from the
frame time (median and p95). *Content*: the robot's turn rate about the
vertical measured from the stream itself (an image's horizontal shift between
frames over fx, a depth image's range-per-column profile likewise, a scan's
shift in azimuth, an IMU's |angular velocity|), lined up against the
registered LiDAR trajectory's yaw rate (stage 04, from `--config`; without a
config, the scans). The lag is how far the stream's stamps run behind the
reference, over the whole run and per `--segment` (default 60 s), so a drift
shows as a spread. A stream passes when its lag is within `--tol` (30 ms) and
it correlates (above 0.5) over a run in which the robot turns. `--dataset`
adds the converter's per-frame `ros_sync` offsets of the dataset as built.
Writes `sync_quality.{json,md,png}`; the figure shows each stream's turn rate
shifted by its lag over the 40 s where the robot turns most, and the lags as
bars. Test: `tests/test_sync_quality.py` (a turning robot with an on-time
LiDAR, RGB and Ouster IMU, a depth camera 50 ms late and a ZED IMU 80 ms late).

