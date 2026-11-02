#!/usr/bin/env python3
"""Complete, metric object Gaussians from a ROS 2 bag: SAM3 mask + depth pointmap -> SAM 3D Objects.

Four stages, each a subcommand. They exchange files only, because SAM3 and SAM 3D Objects need
different Python environments (see objects/README.md):

  topics       list the bag's topics, types and counts                         (any env with rosbags)
  extract      RGB + registered depth + camera_info -> frame dirs with pointmap (any env with rosbags)
  segment      SAM3 text prompts -> one mask per object instance               (sam3 env)
  reconstruct  SAM 3D Objects with the metric pointmap -> .ply per object      (sam3d-objects env)

Frame directory written by `extract` and filled by the later stages:

  out/frame_0003/
    rgb.png            image as recorded (RGB)
    depth.npy          float32 metres, same size as rgb.png, NaN = no depth
    pointmap.npy       float32 HxWx3, metres, camera optical frame (x right, y down, z forward)
    meta.json          stamp, frame_id, K, topics, image-depth time offset, objects (after segment)
    masks/<prompt>_<k>.png, overlay.png                          (segment)
    objects/<prompt>_<k>_cam.ply, _local.ply, _pose.json         (reconstruct)

`_cam.ply` is the object posed in the camera optical frame (same frame as pointmap.npy, metres);
`_local.ply` is SAM 3D Objects' normalized object frame.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ZED = '/zed/zed_node'
DEFAULTS = {
    'image_topic': f'{ZED}/left/image_rect_color',
    'depth_topic': f'{ZED}/depth/depth_registered',
    'info_topic': f'{ZED}/left/camera_info',
}


# ----------------------------------------------------------------------------- bag reading

def open_reader(bags):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    return AnyReader([Path(b) for b in bags], default_typestore=get_typestore(Stores.ROS2_HUMBLE))


def conns_for(reader, topic):
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        sys.exit(f'topic {topic} not in the bag; run `topics` to see what is there')
    return conns


def stamp_of(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def decode_image(msg, msgtype):
    """sensor_msgs/Image or CompressedImage -> RGB uint8 HxWx3."""
    import cv2
    if msgtype.endswith('CompressedImage'):
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_UNCHANGED)
        enc = 'bgra8' if img.ndim == 3 and img.shape[2] == 4 else ('bgr8' if img.ndim == 3 else 'mono8')
    else:
        enc = msg.encoding.lower()
        ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc)
        if ch is None:
            sys.exit(f'unsupported image encoding {msg.encoding}')
        img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
        img = img.reshape(msg.height, msg.width, ch) if ch > 1 else img
    if enc == 'mono8':
        return np.repeat(img[..., None], 3, axis=2)
    if enc.startswith('bgr'):
        img = img[..., [2, 1, 0]]
    return np.ascontiguousarray(img[..., :3])


def decode_depth(msg, msgtype):
    """sensor_msgs/Image (32FC1 m, 16UC1 mm) or compressedDepth -> float32 metres, NaN invalid."""
    import cv2
    if msgtype.endswith('CompressedImage'):
        raw = bytes(msg.data)
        fmt = msg.format.lower()
        if 'compresseddepth' not in fmt:
            sys.exit(f'depth CompressedImage with format "{msg.format}" is not compressedDepth')
        # compressed_depth_image_transport: 12-byte header (int32 format, float depthQuantA, depthQuantB) + PNG
        _, qa, qb = struct.unpack('<iff', raw[:12])
        png = cv2.imdecode(np.frombuffer(raw[12:], np.uint8), cv2.IMREAD_UNCHANGED)
        if fmt.startswith('16uc1'):
            depth = png.astype(np.float32) / 1000.0
        else:  # 32FC1 is stored as quantized inverse depth
            v = png.astype(np.float32)
            depth = np.where(v > 0, qa / (v - qb), np.nan).astype(np.float32)
    else:
        enc = msg.encoding.upper()
        if enc == '32FC1':
            depth = np.frombuffer(bytes(msg.data), '<f4').reshape(msg.height, msg.step // 4)[:, :msg.width]
            depth = depth.astype(np.float32)
        elif enc in ('16UC1', 'MONO16'):
            depth = np.frombuffer(bytes(msg.data), '<u2').reshape(msg.height, msg.step // 2)[:, :msg.width]
            depth = depth.astype(np.float32) / 1000.0
        else:
            sys.exit(f'unsupported depth encoding {msg.encoding}')
    depth = depth.copy()
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1).astype(np.float32)


# ----------------------------------------------------------------------------- stages

def cmd_topics(a):
    with open_reader(a.bag) as r:
        dur = (r.end_time - r.start_time) * 1e-9
        print(f'duration {dur:.1f} s, start {r.start_time * 1e-9:.3f}')
        for c in sorted(r.connections, key=lambda c: c.topic):
            print(f'{c.msgcount:8d}  {c.topic:60s} {c.msgtype}')


def cmd_extract(a):
    import cv2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open_reader(a.bag) as r:
        img_conns = conns_for(r, a.image_topic)
        dep_conns = conns_for(r, a.depth_topic)
        info_conns = conns_for(r, a.info_topic)
        t0 = r.start_time

        # camera_info: first message
        for conn, _, raw in r.messages(connections=info_conns):
            info = r.deserialize(raw, conn.msgtype)
            break
        K_info = np.array(info.k, dtype=np.float64).reshape(3, 3)
        info_wh = (info.width, info.height)

        # pass 1: log times only (no image decoding)
        img_t = np.array([t for _, t, _ in r.messages(connections=img_conns)], dtype=np.int64)
        dep_t = np.array([t for _, t, _ in r.messages(connections=dep_conns)], dtype=np.int64)
        if len(img_t) == 0 or len(dep_t) == 0:
            sys.exit('no image or no depth messages')

        if a.times:
            wanted = [t0 + int(s * 1e9) for s in a.times]
            pick = sorted({int(np.argmin(np.abs(img_t - w))) for w in wanted})
        else:
            pick, last = [], None
            for i, t in enumerate(img_t):
                if t < t0 + a.start * 1e9:
                    continue
                if last is None or t - last >= a.every * 1e9:
                    pick.append(i)
                    last = t
                if len(pick) >= a.max_frames:
                    break
        pick_t = {int(img_t[i]) for i in pick}
        near_dep = {int(img_t[i]): int(dep_t[np.argmin(np.abs(dep_t - img_t[i]))]) for i in pick}
        want_dep = set(near_dep.values())

        # pass 2: decode only the picked images and their nearest depth messages
        images, depths = {}, {}
        for conn, t, raw in r.messages(connections=img_conns + dep_conns):
            if conn.topic == a.image_topic and t in pick_t and t not in images:
                images[t] = r.deserialize(raw, conn.msgtype), conn.msgtype
            elif conn.topic == a.depth_topic and t in want_dep and t not in depths:
                depths[t] = r.deserialize(raw, conn.msgtype), conn.msgtype

    n = 0
    for k, t in enumerate(sorted(images)):
        imsg, itype = images[t]
        dmsg, dtype = depths[near_dep[t]]
        dt = stamp_of(dmsg) - stamp_of(imsg)
        if abs(dt) > a.slop:
            print(f'skip {(t - t0) * 1e-9:7.2f} s: nearest depth is {dt * 1e3:.0f} ms away (> --slop)')
            continue
        rgb = decode_image(imsg, itype)
        depth = decode_depth(dmsg, dtype)
        h, w = rgb.shape[:2]
        K = K_info.copy()
        if (w, h) != info_wh:  # image published at another resolution than camera_info
            K[0] *= w / info_wh[0]
            K[1] *= h / info_wh[1]
        if depth.shape != (h, w):  # registered depth at another resolution, same field of view
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        depth[depth > a.max_depth] = np.nan
        pm = backproject(depth, K)

        d = out / f'frame_{k:04d}'
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / 'rgb.png'), rgb[..., ::-1])
        np.save(d / 'depth.npy', depth)
        np.save(d / 'pointmap.npy', pm)
        meta = {
            'bag': [str(b) for b in a.bag], 'image_topic': a.image_topic, 'depth_topic': a.depth_topic,
            'stamp': stamp_of(imsg), 't_from_bag_start': (t - t0) * 1e-9, 'depth_dt': dt,
            'frame_id': imsg.header.frame_id, 'K': K.tolist(), 'width': w, 'height': h,
            'valid_depth_fraction': float(np.isfinite(depth).mean()),
        }
        (d / 'meta.json').write_text(json.dumps(meta, indent=2))
        print(f'{d.name}: t={meta["t_from_bag_start"]:.2f} s  depth dt={dt * 1e3:+.0f} ms  '
              f'valid depth {meta["valid_depth_fraction"]:.0%}')
        n += 1
    print(f'{n} frames in {out}')


def frame_dirs(out):
    dirs = sorted(p for p in Path(out).glob('frame_*') if (p / 'meta.json').exists())
    if not dirs:
        sys.exit(f'no frame_* dirs in {out}; run extract first')
    return dirs


def cmd_segment(a):
    import cv2
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    prompts = [p.strip() for p in a.prompts.split(',') if p.strip()]
    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=a.threshold)
    rng = np.random.default_rng(0)
    for d in frame_dirs(a.out):
        meta = json.loads((d / 'meta.json').read_text())
        image = Image.open(d / 'rgb.png').convert('RGB')
        rgb = np.array(image)
        depth = np.load(d / 'depth.npy')
        (d / 'masks').mkdir(exist_ok=True)
        overlay = rgb.astype(np.float32)
        objects = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(image)
            for prompt in prompts:
                o = processor.set_text_prompt(state=state, prompt=prompt)
                masks = o['masks'].squeeze(1).cpu().numpy().astype(bool)
                scores = o['scores'].float().cpu().numpy()
                boxes = o['boxes'].float().cpu().numpy()
                for k in np.argsort(-scores):
                    m = masks[k]
                    area = int(m.sum())
                    depth_ok = float(np.isfinite(depth[m]).mean()) if area else 0.0
                    if area < a.min_area or depth_ok < a.min_depth_fraction:
                        continue
                    name = f'{prompt.replace(" ", "_")}_{len(objects)}'
                    cv2.imwrite(str(d / 'masks' / f'{name}.png'), m.astype(np.uint8) * 255)
                    colour = rng.integers(64, 255, 3)
                    overlay[m] = 0.5 * overlay[m] + 0.5 * colour
                    x0, y0, x1, y1 = boxes[k].astype(int)
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), colour.tolist(), 2)
                    cv2.putText(overlay, f'{name} {scores[k]:.2f}', (x0, max(12, y0 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour.tolist(), 1)
                    objects.append({'name': name, 'prompt': prompt, 'score': float(scores[k]),
                                    'box_xyxy': boxes[k].tolist(), 'area_px': area,
                                    'depth_fraction': depth_ok})
        cv2.imwrite(str(d / 'overlay.png'), overlay.astype(np.uint8)[..., ::-1])
        meta['objects'] = objects
        (d / 'meta.json').write_text(json.dumps(meta, indent=2))
        print(f'{d.name}: ' + (', '.join(f'{o["name"]} ({o["score"]:.2f})' for o in objects) or 'nothing'))


def ply_to_optical(path_in, path_out):
    """PyTorch3D camera frame (x left, y up, z forward) -> optical frame (x right, y down, z forward).
    That is a 180 deg rotation about z: negate x, y; rotate each Gaussian by q_z180 (w,x,y,z)=(0,0,0,1)."""
    from plyfile import PlyData
    ply = PlyData.read(str(path_in))
    v = ply['vertex'].data
    v['x'] *= -1
    v['y'] *= -1
    w, x, y, z = (v[f'rot_{i}'].copy() for i in range(4))
    v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'] = -z, -y, x, w  # q_z180 * q
    ply.write(str(path_out))
    return np.stack([v['x'], v['y'], v['z']], axis=1)


def cmd_reconstruct(a):
    import os
    import torch
    from PIL import Image
    repo = Path(a.sam3d_repo).resolve()
    os.environ.setdefault('CONDA_PREFIX', sys.prefix)  # notebook/inference.py reads it
    sys.path.insert(0, str(repo / 'notebook'))
    from inference import Inference, make_scene

    model = Inference(str(repo / 'checkpoints' / a.tag / 'pipeline.yaml'), compile=False)
    only = set(a.only.split(',')) if a.only else None
    for d in frame_dirs(a.out):
        meta = json.loads((d / 'meta.json').read_text())
        if not meta.get('objects'):
            print(f'{d.name}: no objects, run segment first')
            continue
        rgb = np.array(Image.open(d / 'rgb.png').convert('RGB'))
        pm = np.load(d / 'pointmap.npy')
        # SAM 3D Objects wants its pointmap in the PyTorch3D camera convention (x left, y up, z forward)
        pm_p3d = torch.from_numpy(pm * np.array([-1, -1, 1], np.float32))
        (d / 'objects').mkdir(exist_ok=True)
        for obj in meta['objects']:
            if only and obj['name'] not in only and obj['prompt'] not in only:
                continue
            mask = np.array(Image.open(d / 'masks' / f'{obj["name"]}.png')) > 127
            out = model(rgb, mask, seed=a.seed, pointmap=pm_p3d)
            base = d / 'objects' / obj['name']
            out['gs'].save_ply(f'{base}_local.ply')
            scene = make_scene(out)  # Gaussians posed in the camera (PyTorch3D convention)
            scene.save_ply(f'{base}_p3d.ply')
            xyz = ply_to_optical(f'{base}_p3d.ply', f'{base}_cam.ply')
            Path(f'{base}_p3d.ply').unlink()

            # visible-side check: distance from each measured point in the mask to the nearest Gaussian
            seen = pm[mask]
            seen = seen[np.isfinite(seen).all(axis=1)]
            err = None
            if len(seen):
                from scipy.spatial import cKDTree
                dist, _ = cKDTree(xyz).query(seen[:: max(1, len(seen) // 20000)])
                err = {'median_m': float(np.median(dist)), 'p90_m': float(np.percentile(dist, 90))}
            pose = {
                'frame': 'camera optical frame of meta.json frame_id (x right, y down, z forward), metres',
                'frame_id': meta['frame_id'], 'stamp': meta['stamp'],
                'center_m': xyz.mean(axis=0).tolist(),
                'aabb_extent_m': (xyz.max(axis=0) - xyz.min(axis=0)).tolist(),
                'scale': float(out['scale'][0, 0]),
                'raw_p3d': {k: out[k].detach().cpu().numpy().tolist() for k in ('rotation', 'translation', 'scale')},
                'visible_side_to_gaussians': err,
            }
            Path(f'{base}_pose.json').write_text(json.dumps(pose, indent=2))
            ext = ' x '.join(f'{e:.2f}' for e in pose['aabb_extent_m'])
            chk = f'visible-side median {err["median_m"] * 100:.1f} cm' if err else 'no depth in mask'
            print(f'{d.name}/{obj["name"]}: {xyz.shape[0]} Gaussians, extent {ext} m, {chk}')
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('topics', help='list topics')
    s.add_argument('bag', nargs='+', help='ROS 2 bag directory (db3 or mcap); several are read together')
    s.set_defaults(func=cmd_topics)

    s = sub.add_parser('extract', help='bag -> frame dirs (rgb, depth, pointmap, meta)')
    s.add_argument('bag', nargs='+')
    s.add_argument('--out', required=True)
    for k, v in DEFAULTS.items():
        s.add_argument(f'--{k.replace("_", "-")}', dest=k, default=v)
    s.add_argument('--times', type=float, nargs='*', help='seconds from bag start (overrides --every)')
    s.add_argument('--start', type=float, default=0.0, help='seconds from bag start')
    s.add_argument('--every', type=float, default=5.0, help='seconds between frames')
    s.add_argument('--max-frames', type=int, default=5)
    s.add_argument('--slop', type=float, default=0.05, help='max image-depth stamp difference, s')
    s.add_argument('--max-depth', type=float, default=10.0, help='drop depth beyond this, m')
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser('segment', help='SAM3 text prompts -> masks')
    s.add_argument('--out', required=True)
    s.add_argument('--prompts', default='robot', help='comma separated, e.g. "robot,chair,table"')
    s.add_argument('--threshold', type=float, default=0.5)
    s.add_argument('--min-area', type=int, default=1500, help='px')
    s.add_argument('--min-depth-fraction', type=float, default=0.3,
                   help='skip masks with less valid depth than this')
    s.set_defaults(func=cmd_segment)

    s = sub.add_parser('reconstruct', help='SAM 3D Objects -> .ply per object')
    s.add_argument('--out', required=True)
    s.add_argument('--sam3d-repo', required=True, help='clone of facebookresearch/sam-3d-objects')
    s.add_argument('--tag', default='hf', help='checkpoints/<tag>/pipeline.yaml')
    s.add_argument('--only', help='comma separated object names or prompts')
    s.add_argument('--seed', type=int, default=42)
    s.set_defaults(func=cmd_reconstruct)

    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
