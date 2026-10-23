# rosbag_to_opv2v

Convert a multi-agent ROS 2 recording (`.mcap` or `.db3`) into the OPV2V directory
layout that OpenCOOD reads — with the synchronisation between agents measured and
written into every frame rather than assumed.

No ROS installation and no GPU. Custom message types are decoded from the schemas
embedded in the bag.

```bash
pip install -r requirements.txt
python scripts/test_ros2opv2v.py                                  # 90 self-tests, no bag needed

python scripts/inspect_bag.py    --bag <bag> --emit-config configs/mine.yaml   # stage A: look first
python scripts/convert_rosbag.py --config configs/mine.yaml --dry-run          # stage B: plan, write nothing
python scripts/convert_rosbag.py --config configs/mine.yaml                    #          convert
python scripts/validate_opv2v.py --root <output>/test                          # stage C: check it as OpenCOOD will read it
python scripts/label_static.py   --pcd map.pcd --dataset <output>/test         # stage D: ground truth from the run's own map
```

| | |
|---|---|
| `ros2opv2v/` | the converter: bag reading, cross-host clock reconciliation, frame synchronisation, cloud decoding + sweep deskew, pose parameterisation, writers |
| `scripts/` | the stages above, plus `label_static.py` (label static objects once in the map frame), `make_view.py` (the frames that have an optional camera's image, as a second tree), `make_incop_sidecars.py` (InCoP's modality assignment, per agent with `--assign 0=m1 --assign 1=m2`, and class map, `--extra-class robot`) and the self-tests |
| `configs/mirc_coop2.yaml` | a complete, runnable config for the MIRC coop2 recording (two pushcarts + one static infrastructure node) |
| `configs/mirc_site1_20261015_incop.yaml` | coop site_1, 20261015: both robots with Ousters, extrinsics looked up from `/tf_static` (`extrinsic: tf`), the ZED as an optional camera, the link state in every frame (`comms`); plan in `docs/SITE1_20261015_PLAN.md` |
| `docs/ROS2OPV2V.md` | the format, the synchronisation protocol, what the operator must supply, and what converted data can and cannot answer |

## What it does that a plain converter does not

- **Three robots means three clocks.** Per-host offsets are estimated from NTP
  monitor topics and from the delivery floor (`log_time − header.stamp`) and
  cross-checked against each other. By default nothing is shifted — a chrony-
  disciplined stamp is already corrected, and the daemon's reported residual is
  what gets carried into every frame — with a `correct` mode for hosts that were
  not disciplined at record time. A clock step in the daemon's event log is
  surfaced as the discontinuity it is.
- **The residual is part of the data.** Every agent's frame yaml carries a
  `ros_sync` block: its signed skew from the frame time and the clock residual
  correction could not remove. The report carries a tightness curve and the
  structural floor (half the slowest required stream's period), so the tolerance
  is chosen from data.
- **A sweep is not an instant.** Spinning-LiDAR clouds are deskewed to the frame
  instant using the agent's own odometry.
- **Extrinsics come from the bag when it has them.** `extrinsic: tf` with a
  `frame_id` composes the base -> sensor chain from `/tf_static`, refuses a
  missing frame or a moving edge, and records what it resolved.
- **The link is part of the frame.** `comms.streams` writes each agent's Wi-Fi
  status, RTT, goodput, NTP residual and the sniffer's CSI into every frame
  yaml, summarised around the frame time, with the full-rate tables beside the
  scenario.
- **It refuses rather than guesses.** A null extrinsic, a pose source that does
  not start at its declared anchor, a tolerance below the structural floor — each
  fails loudly, because the alternative is a dataset that converts, validates,
  and is geometrically wrong.

Everything the operator must decide is in `docs/ROS2OPV2V.md`.
