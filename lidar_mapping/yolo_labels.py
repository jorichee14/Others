#!/usr/bin/env python3
"""
Shared, DEPENDENCY-LIGHT image + detection helpers.

Imported by both 01_build_map.py (which lives in the ROS/Open3D environment)
and detect_cache.py (which lives in an isolated YOLO venv). It deliberately
imports nothing beyond numpy and cv2 -- no open3d, no scipy, no torch -- so it
can be loaded from either side of that boundary. `load_model` is the single
exception and imports ultralytics lazily, inside the call.
"""
import numpy as np
import cv2


def decode_img(msg):
    """ROS Image message -> BGR uint8 array, or None for an encoding we do not
    handle."""
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    h, w = msg.height, msg.width
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(h, w, 4)[:, :, :3]
        if enc == "rgba8":
            img = img[:, :, ::-1]
    elif enc in ("bgr8", "rgb8"):
        img = buf.reshape(h, w, 3)
        if enc == "rgb8":
            img = img[:, :, ::-1]
    else:
        return None
    return np.ascontiguousarray(img)


def load_model(d):
    """Import ultralytics and open the checkpoint. The ONLY torch touchpoint."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not importable in this environment.\n"
            "  pip install ultralytics --no-deps\n"
            "or run detect_cache.py from the isolated venv (see README).")
    m = YOLO(d.get("model", "yolo11n-seg.pt"))
    return m, m.names


def class_filter(d):
    """allowlist AND denylist by class NAME (empty allowlist = everything)."""
    allow = set(d.get("classes") or [])
    deny = set(d.get("exclude", ["person"]) or [])
    def ok(name):
        return (not allow or name in allow) and name not in deny
    return ok


def label_image(model, names, img, d, W, H, ok):
    """One frame -> (int16 image of per-instance ids, [(class_id, conf), ...]).

    Segmentation masks are used when the weights provide them and are eroded a
    couple of pixels, because the outermost mask ring straddles the silhouette
    and would label whatever is behind the object. Box-only weights fall back
    to the box shrunk by bbox_shrink per side, for the same reason.

    Background is -1; instance k occupies value k, matching the index into the
    returned list."""
    res = model.predict(img, conf=d.get("conf", 0.35), iou=d.get("iou", 0.5),
                        imgsz=d.get("imgsz", 640), device=d.get("device"),
                        verbose=False)[0]
    lab = np.full((H, W), -1, np.int16)
    insts = []
    if res.boxes is None or len(res.boxes) == 0:
        return lab, insts
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    masks = None
    if getattr(res, "masks", None) is not None:
        masks = res.masks.data.cpu().numpy()
    shrink = float(d.get("bbox_shrink", 0.12))
    erode = int(d.get("mask_erode", 2))
    for i in range(len(cls)):
        nm = names[int(cls[i])]
        if not ok(nm):
            continue
        k = len(insts)
        if masks is not None and i < len(masks):
            m = cv2.resize(masks[i].astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST)
            if erode > 0:
                m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode)
            m = m > 0
            if not m.any():
                continue
            lab[m] = k
        else:
            x0, y0, x1, y1 = xyxy[i]
            dx = shrink * (x1 - x0); dy = shrink * (y1 - y0)
            x0 = int(max(0, x0 + dx)); x1 = int(min(W, x1 - dx))
            y0 = int(max(0, y0 + dy)); y1 = int(min(H, y1 - dy))
            if x1 <= x0 or y1 <= y0:
                continue
            lab[y0:y1, x0:x1] = k
        insts.append((int(cls[i]), float(conf[i])))
    return lab, insts
