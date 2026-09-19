#!/usr/bin/env python3
"""$SLAM_PARAMS (JSON) -> the `--Key value` string RTAB-Map takes on its command line.

Only keys that look like RTAB-Map parameters (`Group/Name`) are emitted. Anything
else in the params block is a knob for a different image and is skipped here
rather than passed through, because RTAB-Map's response to an unknown parameter
is a warning on stderr that a forty-minute run will bury.

Booleans are emitted as RTAB-Map spells them -- lowercase `true`/`false` -- and
YAML's `Grid/3D: "true"` is already a string, so both spellings arrive here and
both have to come out the same. A `Grid/3D True` would be parsed as false.
"""
from __future__ import annotations

import json
import sys


def to_args(params: dict) -> list[str]:
    out: list[str] = []
    for key in sorted(params):
        if "/" not in key:
            continue
        v = params[key]
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower()
        out += [f"--{key}", str(v)]
    return out


if __name__ == "__main__":                                   # pragma: no cover
    print(" ".join(to_args(json.loads(sys.argv[1] if len(sys.argv) > 1
                                      else sys.stdin.read() or "{}"))))


def to_ros_params(params: dict, drop: tuple[str, ...] = ()) -> list[str]:
    """The other half: plain keys -> `-p key:=value` for a ROS 2 node.

    The complement of `to_args`: slash keys are RTAB-Map's, plain keys are ROS
    parameters. Split this way, one params block feeds both images and neither
    silently swallows the other's knobs.

    `drop` names keys the CALLER is overriding for a structural reason -- not
    tuning, which rule 7 forbids per run, but a property of the stream, such as
    a depth camera having no per-point timestamps to deskew with. The caller is
    expected to say so on stdout; dropping silently is the thing to avoid.
    """
    out: list[str] = []
    for key in sorted(params):
        if "/" in key or key in drop:
            continue
        v = params[key]
        if isinstance(v, bool):
            v = "true" if v else "false"
        out += ["-p", f"{key}:={v}"]
    return out
