#!/usr/bin/env python
"""Step 4.3 — spatial decomposition of failures into ego-visible vs occluded zones.

For selected (method, condition) cells, reruns inference with the CommChannel and
scores each frame's predictions against clean GT split into two zones:

    ego-visible GT : boxes containing >= MIN_PTS of the ego's OWN lidar points
    occluded GT    : all other GT (only collaborators could reveal these)

Reported per cell:
    recall_visible / recall_occluded   (IoU 0.7, greedy by score)
    fp_per_frame                       (all false positives)
    fp_egovis_per_frame                (FPs claiming an object where the ego's own
                                        sensor sees >= MIN_PTS points — direct
                                        evidence of fusion contamination inside
                                        ego's field of view)

Predictions: delivery failures should depress recall_occluded only; content
failures should depress recall_visible and raise fp_egovis too.

Run on the GPU machine:

    cd ~/cpfa/OpenCOOD
    python ~/cpfa/Others/collab_perception_failure_analysis/scripts/run_phase43.py \
        --checkpoint-root ~/cpfa/checkpoints --out ~/cpfa/results/spatial
"""
import argparse
import json
import os
import sys
import time
import zlib
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from commchannel import ChannelConfig, CommChannel        # noqa: E402
from run_phase1 import METHODS                             # noqa: E402
from run_phase3 import predict_boxes                       # noqa: E402

MIN_PTS = 5
IOU_T = 0.7

# name -> ChannelConfig kwargs; chosen to cover one delivery + three content flavors
CONDITIONS = OrderedDict([
    ('identity', {}),
    ('loss90',   {'loss_p': 0.9}),
    ('latency200ms', {'latency_frames': 2}),
    ('ghosts8',  {'ghost_p': 1.0, 'ghost_count': 8}),
    ('swap50',   {'swap_p': 0.5}),
])
DEFAULT_METHODS = ['attfuse', 'coalign', 'fcooper']


def crc(*parts):
    return zlib.crc32('/'.join(str(p) for p in parts).encode('utf8')) & 0xFFFFFFFF


def points_in_box(points, corners):
    """points (N,3), corners (8,3) with faces per OpenCOOD convention: bottom 0-3,
    top 4-7. Returns boolean mask of points inside the box."""
    origin = corners[0]
    u = corners[1] - origin
    v = corners[3] - origin
    w = corners[4] - origin
    d = points - origin
    eps = 1e-6
    m = np.ones(len(points), dtype=bool)
    for axis in (u, v, w):
        L2 = float(axis @ axis)
        if L2 < eps:
            return np.zeros(len(points), dtype=bool)
        proj = d @ axis
        m &= (proj > -eps) & (proj < L2 + eps)
    return m


def match_greedy(pred_np, scores_np, gt_np):
    """Greedy score-ordered matching at BEV IoU >= IOU_T (OpenCOOD polygon IoU).
    Returns (matched_gt_indices set, fp_pred_indices list)."""
    from opencood.utils import common_utils
    if pred_np is None or len(pred_np) == 0:
        return set(), []
    gt_polys = list(common_utils.convert_format(gt_np)) if len(gt_np) else []
    pred_polys = list(common_utils.convert_format(pred_np))
    order = np.argsort(-scores_np)
    remaining = dict(enumerate(gt_polys))
    matched, fps = set(), []
    for pi in order:
        best_iou, best_gi = 0.0, None
        for gi, gp in remaining.items():
            iou = float(common_utils.compute_iou(pred_polys[pi], [gp])[0])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi is not None and best_iou >= IOU_T:
            matched.add(best_gi)
            del remaining[best_gi]
        else:
            fps.append(pi)
    return matched, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='*', default=DEFAULT_METHODS,
                    choices=[m for m in METHODS if m != 'nocomm'])
    ap.add_argument('--conditions', nargs='*', default=list(CONDITIONS.keys()),
                    choices=list(CONDITIONS.keys()))
    ap.add_argument('--stride', type=int, default=3)
    args = ap.parse_args()

    import torch
    from opencood.tools import train_utils
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import pcd_utils

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_root = os.path.expanduser(args.checkpoint_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for method in args.methods:
        folder, fusion, _ = METHODS[method]
        ckpt_dir = os.path.join(ckpt_root, folder)
        pending = [c for c in args.conditions if not os.path.isfile(
            os.path.join(out_dir, '%s__%s.json' % (method, c)))]
        if not pending:
            print('[%s] all conditions done — skipping' % method)
            continue

        print('[%s] building datasets + model (%d conditions)' % (method, len(pending)))
        hypes = yaml_utils.load_yaml(os.path.join(ckpt_dir, 'config.yaml'))
        ds_clean = build_dataset(hypes, visualize=False, train=False)
        ds_chan = build_dataset(hypes, visualize=False, train=False)
        model = train_utils.create_model(hypes)
        if torch.cuda.is_available():
            model.cuda()
        _, model = train_utils.load_saved_model(ckpt_dir, model)
        model.eval()

        frame_indices = list(range(0, len(ds_clean), args.stride))

        # per-frame clean context: GT corners + zone labels (from ego's own cloud).
        # Ego points are 1-in-3 subsampled (MIN_PTS scaled accordingly) to keep the
        # cache ~3x lighter; zone labels are insensitive to this at these densities.
        print('[%s] building GT + zone cache (%d frames)'
              % (method, len(frame_indices)))
        t_gt = time.time()
        gt_cache = {}
        min_pts_sub = max(2, MIN_PTS // 3)
        for n_done, i in enumerate(frame_indices):
            np.random.seed(crc('gt', method, i))
            base = ds_clean.retrieve_base_data(i)
            ego_entry = next(v for v in base.values() if v['ego'])
            ego_pts = pcd_utils.mask_ego_points(
                ego_entry['lidar_np'])[::3, :3].astype(np.float32)
            sample = ds_clean[i]
            batch = ds_clean.collate_batch_test([sample])
            gt = ds_clean.post_processor.generate_gt_bbx(batch).cpu().numpy()
            visible = np.array([points_in_box(ego_pts, gt[k]).sum() >= min_pts_sub
                                for k in range(len(gt))], dtype=bool)
            gt_cache[i] = (gt, visible, ego_pts)
            if (n_done + 1) % 100 == 0:
                print('[%s]   cache %d/%d (%.0fs)' % (
                    method, n_done + 1, len(frame_indices), time.time() - t_gt))
        print('[%s] cache ready (%.0fs)' % (method, time.time() - t_gt))

        for cond in pending:
            t0 = time.time()
            cfg = ChannelConfig(seed=crc(method, cond), name=cond,
                                **CONDITIONS[cond])
            channel = CommChannel(cfg)
            channel.attach(ds_chan)
            tp_vis = tp_occ = n_vis = n_occ = 0
            fp_total = fp_egovis = 0
            try:
                with torch.no_grad():
                    for i in frame_indices:
                        np.random.seed(crc(method, cond, i))
                        sample = ds_chan[i]
                        batch = ds_chan.collate_batch_test([sample])
                        batch = train_utils.to_device(batch, device)
                        pred, score = predict_boxes(fusion, ds_chan, model, batch)
                        gt, visible, ego_pts = gt_cache[i]
                        pred_np = pred.cpu().numpy() if pred is not None else None
                        score_np = score.cpu().numpy() if pred is not None else None
                        matched, fps = match_greedy(pred_np, score_np, gt)
                        n_vis += int(visible.sum())
                        n_occ += int((~visible).sum())
                        tp_vis += sum(1 for g in matched if visible[g])
                        tp_occ += sum(1 for g in matched if not visible[g])
                        fp_total += len(fps)
                        for pi in fps:
                            if points_in_box(ego_pts,
                                             pred_np[pi]).sum() >= min_pts_sub:
                                fp_egovis += 1
            finally:
                channel.detach()
            nf = len(frame_indices)
            record = {
                'method': method, 'condition': cond,
                'channel_config': cfg.to_dict(), 'frames': nf,
                'gt_visible': n_vis, 'gt_occluded': n_occ,
                'recall_visible': round(tp_vis / n_vis, 4) if n_vis else None,
                'recall_occluded': round(tp_occ / n_occ, 4) if n_occ else None,
                'fp_per_frame': round(fp_total / nf, 3),
                'fp_egovis_per_frame': round(fp_egovis / nf, 3),
                'runtime_s': round(time.time() - t0, 1),
            }
            with open(os.path.join(out_dir, '%s__%s.json' % (method, cond)),
                      'w') as f:
                json.dump(record, f, indent=2)
            print('[%s__%s] R_vis %.3f | R_occ %.3f | FP/f %.2f | '
                  'FP_egovis/f %.2f | %.0fs'
                  % (method, cond, record['recall_visible'],
                     record['recall_occluded'], record['fp_per_frame'],
                     record['fp_egovis_per_frame'], record['runtime_s']))

        # release per-method state before the next model (caches are ~GB-scale;
        # keeping several alive drives the machine into swap)
        import gc
        del gt_cache, ds_clean, ds_chan, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print('done — results in %s' % out_dir)


if __name__ == '__main__':
    main()
