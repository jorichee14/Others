# Configs

`coop2.yaml` describes the sequence. `methods/*.yaml` describe the methods. The
split is deliberate: the sequence's geometry is a property of the recording and
is written once; a method's parameters are a property of the comparison and
change per experiment.

## Adding a method

1. Copy the closest `methods/*.yaml`. Fill `modality`, `needs_imu`,
   `loop_closure` and `outputs` honestly — `modality` decides which comparison
   block the row lands in, and blocks are never merged.
2. List in `streams` the stream keys from `coop2.yaml` the method runs on — a
   list, so one method can run on several streams of one agent. If a stream does
   not exist yet, add it there with its `reference_frame_from_sensor`; do not
   carry an extrinsic in the method file unless it is genuinely method-specific
   (`extrinsic_by_stream` exists for that case).
3. State every parameter changed from the published default, with the default
   named in a comment. A value that cannot be derived stays `null`;
   `run_method.py` refuses to start rather than substituting one.
4. Write a Dockerfile against the contract in `docker/README.md` and pin it to a
   commit.
5. `python3 scripts/run_method.py ... ` without `--execute` first, and read the
   preflight output.

## Adding a sequence

Copy `coop2.yaml`. The three blocks that need real work are `reference` (what it
is, where it came from, and its uncertainty — the worst anchor, not the best),
`streams` (one `reference_frame_from_sensor` per sensor, taken from the
converter config rather than re-derived), and `eval.volume`. Everything else
carries over.
