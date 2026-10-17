#!/usr/bin/env python3
"""A GLIM config folder for another robot: a copy of an existing one with its
points / IMU topics and T_lidar_imu set from that robot's processed bag.

    python3 processing/glim_config.py <src config dir> <new config dir> <bag>

    e.g. python3 processing/glim_config.py configs/glim/mobile_1 configs/glim/mobile_2 \
             <work>/<date>/<pass>/mobile_2/<bag>_retimed
        [--points /mobile_2/ouster/points] [--imu /mobile_2/ouster/imu]

GLIM's config files carry // comments, so they are edited as text: every
file is copied as it is, except in config_ros.json the values of
"imu_topic" and "points_topic", and in config_sensors.json the "T_lidar_imu"
array, written as [x, y, z, qx, qy, qz, qw] (GLIM's TUM form). T_lidar_imu
comes from the bag's /tf_static -- the Ouster driver's os_sensor -> os_lidar
and os_sensor -> os_imu, i.e. the sensor's own metadata -- for the frames
of the points and IMU messages (os_lidar_mobile2, os_imu_mobile2 on mobile_2).
The old and new T_lidar_imu are printed side by side.
"""
import argparse
import os
import re
import shutil
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", FutureWarning)

from scoop import bag, tftree                                       # noqa: E402
from scoop.bag import BagError                                      # noqa: E402

_ARRAY = re.compile(r'("T_lidar_imu"\s*:\s*)\[[^\]]*\]', re.S)


def frame_of(reader, topic):
    for _, _, msg in reader.iter_messages([topic]):
        return msg.header.frame_id
    raise BagError(f"no messages on {topic}")


def t_lidar_imu(bag_dir, lidar_frame, imu_frame):
    """[x, y, z, qx, qy, qz, qw]: points in the IMU frame -> the LiDAR frame."""
    edges = tftree.tf_edges([bag_dir])
    T = tftree.static_between(lidar_frame, imu_frame, edges)   # below os_sensor: static
    t, q = tftree.to_tq(T)
    return [float(v) for v in (*t, *q)]


def set_topic(text, key, value):
    new, n = re.subn(rf'("{key}"\s*:\s*)"[^"]*"', lambda m: f'{m.group(1)}"{value}"', text)
    if n != 1:
        raise BagError(f'config_ros.json: "{key}" found {n} times (expected once, uncommented)')
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("bag")
    ap.add_argument("--points", default="/mobile_2/ouster/points")
    ap.add_argument("--imu", default="/mobile_2/ouster/imu")
    a = ap.parse_args()
    src, dst = os.path.expanduser(a.src), os.path.expanduser(a.dst)
    if os.path.exists(dst):
        sys.exit(f"{dst} exists; pick a new folder")
    try:
        r = bag.open_bag(a.bag)
        lf, imf = frame_of(r, a.points), frame_of(r, a.imu)
        T = t_lidar_imu(a.bag, lf, imf)
        ros = open(os.path.join(src, "config_ros.json")).read()
        sens = open(os.path.join(src, "config_sensors.json")).read()
        ros = set_topic(set_topic(ros, "imu_topic", a.imu), "points_topic", a.points)
        old = _ARRAY.search(sens)
        if old is None or len(_ARRAY.findall(sens)) != 1:
            raise BagError('config_sensors.json: no single "T_lidar_imu": [...]')
        sens = _ARRAY.sub(lambda m: m.group(1) + "[" + ", ".join(f"{v:.9f}" for v in T) + "]",
                          sens)
    except (BagError, OSError) as e:
        sys.exit(f"error: {e}")
    shutil.copytree(src, dst)
    open(os.path.join(dst, "config_ros.json"), "w").write(ros)
    open(os.path.join(dst, "config_sensors.json"), "w").write(sens)
    print(f"{dst}: a copy of {src}")
    print(f"    points_topic  {a.points}  (frame {lf})")
    print(f"    imu_topic     {a.imu}  (frame {imf})")
    print(f"    T_lidar_imu   was {' '.join(old.group(0).split(':', 1)[1].split())}")
    print(f"                  now [{', '.join(f'{v:.6f}' for v in T)}]  (from the bag's /tf_static)")


if __name__ == "__main__":
    main()
