# Containers

One image per method. The evaluator never imports a method's code and no method
ever imports the evaluator: the whole interface is three files in `/out`.

## The contract

A container is given:

| | |
|---|---|
| `/bag` | the recording, read-only |
| `/out` | a writable run directory |
| `$SLAM_TOPICS` | comma-separated topics for this stream |
| `$SLAM_RATE` | bag replay rate (1.0 unless a table says otherwise) |
| `$SLAM_PARAMS` | the method config's `params` block as JSON |

and must write:

| | |
|---|---|
| `/out/trajectory.tum` | `timestamp tx ty tz qx qy qz qw`, seconds, metres, **on the bag's clock** |
| `/out/map.ply` | the dense map, if the method makes one, in the same world frame as the trajectory |
| `/out/timing.json` | `{frames, wall_s, cpu_s, peak_rss_mb}` |

Three rules, and each one exists because breaking it produces a run that
completes and is wrong:

1. **Poses describe the sensor frame the method was given, in the method's own
   world frame.** Do not apply an extrinsic. Do not try to output `map`. The
   evaluator does both, from the dataset config, identically for every method —
   a container that "helpfully" pre-transforms is a container whose extrinsic is
   applied twice and nothing downstream can see it.
2. **Stamps are the bag's, not the wall clock.** Replay with `--clock` and run
   every node with `use_sim_time: true`. Wall-clock stamps still associate
   against the reference if the run happens to start near the bag's epoch, which
   is the worst possible failure mode.
3. **`--network none`.** A benchmark run that can reach the network is a
   benchmark run that can download a different checkpoint than the one the
   manifest records.

## Building

```bash
docker build -f docker/Dockerfile.base   -t slambench/base:humble .
docker build -f docker/Dockerfile.kiss-icp -t slambench/kiss-icp:humble .
docker build -f docker/Dockerfile.rtabmap  -t slambench/rtabmap:humble .
```

Pin every method to a commit in its Dockerfile, not to a branch. `run.json`
records the image tag; the tag is only useful if it means one thing.
