#!/usr/bin/env python3
"""
Run YOLO over the bag's camera stream ONCE and cache the per-frame results, so
stage 01 never has to import torch.

WHY THIS EXISTS
---------------
`pip install ultralytics` into a ROS/Open3D environment resolves torch, which
drags numpy to 2.x. Open3D and the system scipy are compiled against the numpy
1.x ABI, so they stop importing and the whole pipeline dies -- for a dependency
that is only needed to turn images into label masks. Isolating that one step is
the fix: this script runs in its own venv with numpy 2 and torch, writes a
small cache to disk, and stage 01 reads the cache with nothing but numpy+cv2.

    ROS / Open3D env  ->  01_build_map.py   reads  detections/
    isolated venv     ->  detect_cache.py   writes detections/

USAGE
-----
    python3 -m venv ~/yolo-env                # no --system-site-packages:
    ~/yolo-env/bin/pip install ultralytics rosbags opencv-python
    ~/yolo-env/bin/python detect_cache.py pipeline_config.json

    # then, back in the normal environment, with detect_objects.enable = true
    python3 01_build_map.py pipeline_config.json

CACHE LAYOUT (out_dir/detections/)
----------------------------------
    index.json      model, class names, topic, geometry, and one entry per
                    PROCESSED frame: {n, t, insts:[[class_id, conf], ...]}
    000123.png      label image, uint8, 0 = background and k+1 = instance k

Frames with no detections are still recorded, with an empty `insts` and no
png. That is not an optimisation detail -- stage 01 counts every frame a point
was visible in as the denominator of its multi-view agreement test, so
silently dropping empty frames would inflate every vote ratio.

Label images are stored as PNG because they are almost entirely background and
compress to a few kB each; a raw int16 array would be 460 kB per frame.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from yolo_labels import decode_img, load_model, class_filter, label_image

TS = get_typestore(Stores.ROS2_HUMBLE)


def resolve_topic(cfg, d, override):
    """Find the camera topic without importing pipeline_common, which would
    pull open3d into this venv."""
    if override:
        return override
    if d.get("image_topic"):
        return d["image_topic"]
    calib = cfg["dataset"].get("calib_json")
    if calib and os.path.exists(calib):
        with open(calib) as f:
            c = json.load(f)
        for k, v in _walk(c):
            if isinstance(v, str) and "image" in k.lower() and "topic" in k.lower():
                return v
        for k, v in _walk(c):        # last resort: any topic-looking string
            if isinstance(v, str) and v.startswith("/") and "image" in v.lower():
                return v
    raise SystemExit(
        "could not determine the camera topic. Pass --image-topic /your/topic, "
        "or add \"image_topic\" to 01_build_map.detect_objects in the config.")


def _walk(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, k)
            yield k, v
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v, prefix)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg_path = args[0] if args else "pipeline_config.json"
    topic_override = None
    limit = 0
    for a in sys.argv[1:]:
        if a.startswith("--image-topic="):
            topic_override = a.split("=", 1)[1]
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])

    with open(cfg_path) as f:
        cfg = json.load(f)
    st = cfg["01_build_map"]
    d = st.get("detect_objects", {})
    bag = cfg["dataset"]["bag"]
    out_dir = cfg["dataset"]["out_dir"]
    cache_dir = os.path.join(out_dir, d.get("cache", "detections"))
    os.makedirs(cache_dir, exist_ok=True)
    topic = resolve_topic(cfg, d, topic_override)
    W, H = int(st["image_width"]), int(st["image_height"])
    stride = max(1, int(d.get("img_stride", 5)))

    model, names = load_model(d)
    ok = class_filter(d)
    names = {int(k): v for k, v in dict(names).items()}
    print(f"model {d.get('model')} on {d.get('device') or 'auto'}; "
          f"topic {topic}; every {stride}th image -> {cache_dir}/")

    entries = []
    n = n_proc = n_det = 0
    with AnyReader([Path(bag)], default_typestore=TS) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"topic {topic} not found in {bag}")
        for conn, _, raw in r.messages(connections=conns):
            n += 1
            if n % stride:
                continue
            msg = r.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            img = decode_img(msg)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (W, H):
                img = cv2.resize(img, (W, H))
            lab, insts = label_image(model, names, img, d, W, H, ok)
            e = {"n": n, "t": t, "insts": [[c, round(cf, 4)] for c, cf in insts]}
            if insts:
                fn = f"{n:07d}.png"
                # +1 so background is 0; uint8 caps at 254 instances per frame,
                # far above anything a detector returns for one image
                cv2.imwrite(os.path.join(cache_dir, fn),
                            np.clip(lab + 1, 0, 255).astype(np.uint8))
                e["png"] = fn
                n_det += len(insts)
            entries.append(e)
            n_proc += 1
            if n_proc % 200 == 0:
                print(f"    {n_proc} frames, {n_det} detections", flush=True)
            if limit and n_proc >= limit:
                break

    index = {
        "model": str(d.get("model", "")),
        "device": str(d.get("device", "")),
        "topic": topic,
        "width": W, "height": H,
        "img_stride": stride,
        "conf": d.get("conf", 0.35),
        "names": {str(k): v for k, v in names.items()},
        "frames": entries,
    }
    with open(os.path.join(cache_dir, "index.json"), "w") as f:
        json.dump(index, f)
    size = sum(os.path.getsize(os.path.join(cache_dir, f))
               for f in os.listdir(cache_dir))
    print(f"cached {n_proc} frames ({n_det} detections, "
          f"{size / 2**20:.1f} MiB) -> {cache_dir}/index.json")
    print("now run 01_build_map.py in the normal environment with "
          "detect_objects.enable = true")


if __name__ == "__main__":
    main()
