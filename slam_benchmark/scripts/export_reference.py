#!/usr/bin/env python3
"""Export an agent's reference trajectory from the bag to TUM, once.

Reuses `rosbag_to_opv2v`'s bag reader rather than reimplementing it: that reader
already decodes custom schemas out of the bag and is covered by its own tests,
and two decoders for one recording is two chances to disagree about a stamp.

    python3 scripts/export_reference.py --config configs/coop2.yaml \
        --agent mobile_1 --out runs/coop2_20260828/reference/mobile_1.tum

It refuses if the first pose is further than the tolerance from the config's
`expected_start`. That check is not ceremony: an optical-frame trajectory and
the same trajectory republished at a body frame under the sensor have identical
shape and timing and differ by a metre, and nothing downstream can tell them
apart once the file is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "rosbag_to_opv2v"))

from slambench import Trajectory, load_dataset, save_tum, se3   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bag", default=None, help="override dataset.bag")
    ap.add_argument("--tolerance-m", type=float, default=0.25)
    args = ap.parse_args()

    try:
        from ros2opv2v.bagreader import BagReader
    except ImportError as e:
        print(f"error: cannot import ros2opv2v ({e}).\n"
              f"  It is expected at {ROOT.parent / 'rosbag_to_opv2v'}; "
              f"install its requirements with\n"
              f"      pip install -r {ROOT.parent / 'rosbag_to_opv2v' / 'requirements.txt'}",
              file=sys.stderr)
        return 2

    cfg = load_dataset(args.config)
    ref = cfg.agent_reference(args.agent)
    bag = args.bag or cfg.raw["dataset"]["bag"]
    reader = BagReader(str(Path(bag).expanduser()), stamp_source="header")

    stamps, poses = [], []
    # BagReader.iter_messages yields (topic, stamp_ns, msg) — in that order.
    # This loop once unpacked it as (topic, msg, stamp_ns) and would have
    # crashed on the first message; it had never met a real bag.
    for topic, stamp_ns, msg in reader.iter_messages([ref["topic"]]):
        p = getattr(msg, "pose", msg)
        p = getattr(p, "pose", p)                    # PoseStamped or PoseWithCovarianceStamped
        t = [p.position.x, p.position.y, p.position.z]
        q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        stamps.append(stamp_ns * 1e-9)
        poses.append(se3.pose_from_quat(t, q))

    if not stamps:
        print(f"error: no messages on {ref['topic']}", file=sys.stderr)
        return 1

    stamps = np.asarray(stamps)
    order = np.argsort(stamps, kind="stable")
    stamps, poses = stamps[order], [poses[i] for i in order]
    uniq = np.concatenate([[True], np.diff(stamps) > 0])
    traj = Trajectory(stamps[uniq], np.stack(poses)[uniq], name=f"reference/{args.agent}",
                      frame=ref.get("frame", ""), world=cfg.raw["dataset"]["world"]["frame_id"])

    exp = ref.get("expected_start")
    if exp is not None:
        d = float(np.linalg.norm(traj.positions[0] - np.asarray(exp, dtype=float)))
        if d > args.tolerance_m:
            print(f"error: {args.agent} starts at {traj.positions[0].round(4).tolist()}, "
                  f"{d:.3f} m from the declared anchor {exp} (tolerance "
                  f"{args.tolerance_m} m).\n"
                  f"  Either the topic is not the frame the config says it is, or the "
                  f"anchor is stale. Do not widen the tolerance to get past this.",
                  file=sys.stderr)
            return 1
        print(f"start check: {d * 1e3:.1f} mm from the declared anchor — ok")

    save_tum(traj, args.out)
    print(f"{len(traj)} poses, {traj.duration:.1f} s, {traj.path_length():.2f} m of path "
          f"-> {args.out}\n  frame={traj.frame} world={traj.world}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
