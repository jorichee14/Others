# scoop_pipeline

Processing for the SCooP dataset (`../scoop_paper`): per-site reference maps
and fiducial boards, then per-sequence reference poses, geometry and release
files. Code lives here; bags, intermediates and the release tree live outside
git (see the plan in the paper's Release Structure section).

```
scoop/          library imported by every stage
  bag.py        bag decoding: the only module that opens a bag
scripts/
  test_bag.py   self-tests (no bag needed)
  bag_check.py  verify + time the fast decoder on a real bag
```

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
python3 scripts/test_bag.py
python3 scripts/bag_check.py <bag> [--points /ouster/points] [--image <topic>]
```

`bag_check.py` compares the fast path with the generic decoder field by field
on the first 50 messages (exits non-zero on any mismatch) and times both.
On a synthetic uncompressed MCAP of 128x1024 Ouster scans (48-byte points,
6.3 MB each): 67 msg/s for the scripts' decode + extract + range gate, 235
msg/s for `iter_scans` with the same gate, file reading included. Reading
alone runs at ~630 msg/s, so that's the upper limit for a single process.
