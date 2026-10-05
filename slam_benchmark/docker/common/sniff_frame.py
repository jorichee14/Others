#!/usr/bin/env python3
"""Read the frame_id off the first message of a topic, straight out of the bag.

RTAB-Map has to be told which frame its poses are in before it starts, and the
only authoritative answer is the one the recording itself carries: the header of
the image the method will actually consume. Namespaced or not, remapped or not,
the header is what every downstream consumer sees.

So this does not replay anything and does not guess. It deserialises exactly one
message and prints its frame_id. If the topic carries none, it exits non-zero
and the run does not start -- a wrong frame_id produces a trajectory that is
internally consistent, passes every check, and is expressed in a frame nobody
declared, which is the worst kind of wrong (CLAUDE.md rule 3).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", required=True)
    args = ap.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in types:
        print(f"topic {args.topic!r} is not in the bag; it has "
              f"{len(types)} topics", file=sys.stderr)
        return 2
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
    msg_cls = get_message(types[args.topic])

    while reader.has_next():
        _, data, _ = reader.read_next()
        frame = getattr(deserialize_message(data, msg_cls), "header", None)
        frame = getattr(frame, "frame_id", "")
        if frame:
            print(frame)
            return 0
        print(f"{args.topic} carries an EMPTY frame_id", file=sys.stderr)
        return 3
    print(f"{args.topic} has no messages", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
