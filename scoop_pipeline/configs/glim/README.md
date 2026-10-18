# GLIM configs, one folder per robot

GLIM (`koide3/glim_ros2`, run by `processing/run_glim.sh`) reads every setting
from the config folder it is given: topics in `config_ros.json`, the LiDAR-IMU
extrinsic `T_lidar_imu` in `config_sensors.json`, and odometry, sub-mapping,
global mapping and viewer settings in the others. The files carry `//`
comments, so they are not plain JSON.

```
configs/glim/mobile_1/   /mobile_1/ouster/points, /mobile_1/ouster/imu
configs/glim/mobile_2/   /mobile_2/ouster/points, /mobile_2/ouster/imu
```

`configs/recording.yaml` picks the folder: `glim.config` (mobile_1's, the top
level) and `machines.<machine>.glim.config`. `$GLIM_CONFIG` overrides it.

mobile_2's folder is mobile_1's with only the topics and `T_lidar_imu`
changed, the latter taken from a processed mobile_2 bag's `/tf_static` (the
Ouster's own `os_sensor -> os_lidar / os_imu`):

```
python processing/glim_config.py configs/glim/mobile_1 configs/glim/mobile_2 \
    <work>/<date>/<pass>/mobile_2/<bag>_retimed
```

So a change to GLIM's tuning goes into mobile_1's folder and is carried over
by making mobile_2's again.

`imu_time_offset` (config_ros.json) stays 0. The Ouster IMU runs about 11 ms
behind its LiDAR (`check.py imu` on 20261015: 8 / 12 / 14 ms
against the BA, 02 and reference trajectories), but GLIM run with -0.011 gave
the same trajectory (ATE 3.41 vs 3.42 cm, the same 46 ms heading pattern).
That 46 ms is between GLIM's output poses and the scans, and is corrected on
export instead: `machines.mobile_2.glim.pose_time_offset` in
`configs/recording.yaml` (scoop/recording.py export_glim).
