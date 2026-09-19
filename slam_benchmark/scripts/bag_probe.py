#!/usr/bin/env python3
"""What the bag actually carries: the static TF tree, and whether streams share a clock.

Two questions that only the recording can answer, and that both produced a
failure in the first real container run:

  1. RTAB-Map refused to initialise its IMU because `zed_imu_link` is not in the
     replayed tf tree. Is the edge missing, or is it there under another name?
     Guessing at a frame name is how a transform becomes plausible and wrong.
  2. rgb and depth stamps came out 67 ms apart -- one whole frame at 14.7 Hz --
     with the SIGN alternating, which is what an off-by-one pairing looks like.
     Either the two streams carry different stamps or they do not, and the bag
     knows which.

    python3 scripts/bag_probe.py --config configs/coop2.yaml --stream mobile_1.zed_rgbd
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slambench.config import load_dataset                          # noqa: E402


def components(edges: list[tuple[str, str]]) -> list[set[str]]:
    """Connected components of the tf tree, treating edges as undirected.

    tf resolves a lookup through any path, in either direction, so two frames in
    the same component are mutually transformable and two in different ones are
    not -- which is exactly the question a `lookupTransform` failure asks.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    out: list[set[str]] = []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack += [m for m in adj[n] if m not in comp]
        seen |= comp
        out.append(comp)
    return sorted(out, key=len, reverse=True)


def nearest_dt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each stamp in `a`, the SIGNED gap to the closest stamp in `b`.

    Signed, so a systematic lead or lag is visible rather than folded away. But
    the DIAGNOSTIC is the magnitude against the frame period: jitter is a small
    fraction of one, while a gap near half a period means the closest partner a
    matcher can find is half a frame of real motion away, and no sync interval
    recovers a pairing that was never recorded. A sign that alternates is a
    symptom of sitting near that tie point, not an independent fault.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0)
    b = np.sort(b)
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    left, right = b[idx - 1], b[idx]
    closer = np.where(np.abs(a - left) <= np.abs(a - right), left, right)
    return a - closer


def describe(name: str, t: np.ndarray) -> str:
    if len(t) < 2:
        return f"  {name:<46} {len(t)} msgs"
    d = np.diff(np.sort(t))
    return (f"  {name:<46} {len(t):>6} msgs  {1.0 / np.median(d):6.2f} Hz  "
            f"span {t.max() - t.min():7.2f} s")


def main() -> int:                                                 # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--bag", default=None)
    args = ap.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    cfg = load_dataset(args.config)
    stream = cfg.stream(args.stream)
    agent = stream.get("agent")
    bag = str(Path(args.bag or cfg.raw["dataset"]["bag"]).expanduser())

    wanted = {k: v for k, v in stream.items()
              if k.endswith("topic") and isinstance(v, str) and v.startswith("/")}
    imu_key = f"{agent}.imu"
    imu = cfg.raw.get("streams", {}).get(imu_key) or {}
    if imu.get("topic"):
        wanted["imu_topic"] = imu["topic"]

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics = [t for t in list(wanted.values()) + ["/tf_static"] if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    classes = {t: get_message(types[t]) for t in topics}

    stamps: dict[str, list[float]] = defaultdict(list)
    frames: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, classes[topic])
        if topic == "/tf_static":
            for tr in msg.transforms:
                e = (tr.header.frame_id, tr.child_frame_id)
                if e not in edges:
                    edges.append(e)
            continue
        h = getattr(msg, "header", None)
        if h is None:
            continue
        stamps[topic].append(h.stamp.sec + h.stamp.nanosec * 1e-9)
        frames.setdefault(topic, h.frame_id)

    print(f"\n=== {args.stream}  ({agent})\n")
    print("TOPICS")
    for key, topic in wanted.items():
        if topic not in stamps:
            print(f"  {topic:<46} NOT IN BAG")
            continue
        print(describe(topic, np.array(stamps[topic])) + f"   frame_id={frames.get(topic)}")

    print(f"\nSTATIC TF — {len(edges)} edges, {len(components(edges))} disconnected tree(s)")
    for i, comp in enumerate(components(edges)):
        print(f"  tree {i}: {len(comp)} frames")
        for f in sorted(comp):
            print(f"      {f}")

    # The two frames RTAB-Map has to connect, named by the config rather than
    # guessed: the stream's own sensor frame and the IMU's.
    need = [stream.get("sensor_frame"), imu.get("sensor_frame")]
    if all(need):
        comps = components(edges)
        where = [next((i for i, c in enumerate(comps) if f in c), None) for f in need]
        print(f"\n  {need[0]!r} -> tree {where[0]}")
        print(f"  {need[1]!r} -> tree {where[1]}")
        if None in where:
            missing = [f for f, w in zip(need, where) if w is None]
            print(f"  VERDICT: {missing} absent from tf_static entirely. The edge has to "
                  f"be supplied from configs/coop2.yaml, not recovered from the bag.")
        elif where[0] != where[1]:
            print("  VERDICT: both present but in DISCONNECTED trees — no path exists.")
        else:
            print("  VERDICT: connected. A lookup between them will resolve.")

    depth, color = stream.get("depth_topic"), stream.get("color_topic")
    if depth in stamps and color in stamps:
        d, c = np.array(stamps[depth]), np.array(stamps[color])
        dt = nearest_dt(c, d)
        print(f"\nCOLOUR vs DEPTH stamp alignment  ({len(c)} colour, {len(d)} depth)")
        print(f"  exact matches            {int(np.sum(dt == 0))} / {len(dt)}")
        print(f"  |dt| median              {np.median(np.abs(dt)) * 1e3:8.2f} ms")
        print(f"  |dt| p95 / max           {np.percentile(np.abs(dt), 95) * 1e3:8.2f} / "
              f"{np.abs(dt).max() * 1e3:.2f} ms")
        print(f"  signed dt median         {np.median(dt) * 1e3:+8.2f} ms")
        print(f"  sign flips               {int(np.sum(np.diff(np.sign(dt[dt != 0])) != 0))}"
              f" of {int(np.sum(dt != 0)) - 1} steps")
        period = float(np.median(np.diff(np.sort(d)))) if len(d) > 1 else float("nan")
        print(f"  depth frame period       {period * 1e3:8.2f} ms")
        if np.all(dt == 0):
            v = ("every pair is stamped identically. Run approx_sync:=false — exact "
                 "sync cannot mispair, and approximate sync can.")
        elif np.median(np.abs(dt)) > 0.4 * period:
            v = (f"the closest colour/depth pair is {np.median(np.abs(dt)) * 1e3:.0f} ms "
                 f"apart against a {period * 1e3:.0f} ms frame period. The streams are "
                 f"INTERLEAVED, not merely jittered: no sync interval recovers a pairing "
                 f"that does not exist, and each pair carries that much real motion.")
        else:
            v = (f"jitter well inside one frame. Keep approx_sync with "
                 f"approx_sync_max_interval around {np.percentile(np.abs(dt), 95) * 2e3:.0f} ms "
                 f"to reject the tail.")
        print(f"  VERDICT: {v}")
    return 0


if __name__ == "__main__":                                         # pragma: no cover
    raise SystemExit(main())
