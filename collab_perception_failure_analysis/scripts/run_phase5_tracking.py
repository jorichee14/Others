#!/usr/bin/env python
"""Phase 5 — cooperative detection + multi-object tracking under channel impairments.

Tests the two standing predictions from the detection study (results/ANALYSIS.md):
  P1: burstiness, irrelevant for single-frame detection at matched loss rate,
      SHOULD matter for tracking (temporal state: bursts break tracks).
  P2: staleness verdicts may shift when a motion-model tracker smooths detections.

Design: detections from the impaired pipeline (stride 1, contiguous frames, per
scenario) are fed to a world-frame constant-velocity Kalman tracker (AB3DMOT-style:
Hungarian association on BEV center distance, 2-hit confirmation, 3-miss deletion).
GT tracks come from the clean dataset's object ids. MOT accounting per scenario:
misses (FN), false positives (FP), identity switches (IDSW), fragmentations;
MOTA = 1 - (FN+FP+IDSW)/GT.

Run on the GPU machine (intermediate-fusion methods only):

    cd ~/cpfa/OpenCOOD
    python ~/cpfa/Others/collab_perception_failure_analysis/scripts/run_phase5_tracking.py \
        --checkpoint-root ~/cpfa/checkpoints --out ~/cpfa/results/tracking
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

DT = 0.1              # OPV2V is 10 Hz
ASSOC_GATE = 3.0      # tracker association gate (m)
EVAL_GATE = 2.0       # GT<->track matching gate for MOT metrics (m)
CONFIRM_HITS = 2
MAX_MISSES = 3

def ge(loss):
    p_bg = 0.3
    return {'ge_p_bad_to_good': p_bg, 'ge_p_good_to_bad': loss * p_bg / (1 - loss)}

CONDITIONS = OrderedDict([
    ('clean',    {}),
    ('iid30',    {'loss_p': 0.3}),
    ('burst30',  ge(0.3)),
    ('iid70',    {'loss_p': 0.7}),
    ('burst70',  ge(0.7)),
    ('stale4',   {'stale_period': 4}),
    ('latency2', {'latency_frames': 2}),
])
DEFAULT_METHODS = ['coalign', 'cobevt', 'fcooper']


def crc(*parts):
    return zlib.crc32('/'.join(str(p) for p in parts).encode('utf8')) & 0xFFFFFFFF


class Track:
    _next_id = 0

    def __init__(self, xy):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, 10.0, 10.0])
        self.hits = 1
        self.misses = 0

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    def predict(self):
        F = np.eye(4); F[0, 2] = DT; F[1, 3] = DT
        Q = np.diag([0.05, 0.05, 0.5, 0.5])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, xy):
        H = np.zeros((2, 4)); H[0, 0] = 1; H[1, 1] = 1
        R = np.eye(2) * 0.3
        y = np.asarray(xy) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(self):
        self.tracks = []

    def step(self, det_xy):
        from scipy.optimize import linear_sum_assignment
        for t in self.tracks:
            t.predict()
        matched_t, matched_d = set(), set()
        if self.tracks and len(det_xy):
            cost = np.linalg.norm(
                np.array([t.x[:2] for t in self.tracks])[:, None, :]
                - det_xy[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= ASSOC_GATE:
                    self.tracks[r].update(det_xy[c])
                    matched_t.add(r); matched_d.add(c)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]
        for c in range(len(det_xy)):
            if c not in matched_d:
                self.tracks.append(Track(det_xy[c]))
        return [(t.id, t.x[0], t.x[1]) for t in self.tracks if t.confirmed]


class MotAccumulator:
    def __init__(self):
        self.n_gt = self.fn = self.fp = self.idsw = self.frag = 0
        self.last_track_for_gt = {}   # gt id -> track id of last match
        self.gt_was_tracked = {}      # gt id -> was matched on previous frame

    def reset_sequence(self):
        self.last_track_for_gt = {}
        self.gt_was_tracked = {}

    def step(self, gt_items, trk_items):
        """gt_items: list of (gt_id, x, y); trk_items: list of (trk_id, x, y)."""
        from scipy.optimize import linear_sum_assignment
        self.n_gt += len(gt_items)
        matched_g = {}
        if gt_items and trk_items:
            g = np.array([[a[1], a[2]] for a in gt_items])
            t = np.array([[a[1], a[2]] for a in trk_items])
            cost = np.linalg.norm(g[:, None, :] - t[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= EVAL_GATE:
                    matched_g[gt_items[r][0]] = trk_items[c][0]
        self.fn += len(gt_items) - len(matched_g)
        self.fp += len(trk_items) - len(matched_g)
        for gid, _, _ in gt_items:
            tid = matched_g.get(gid)
            if tid is not None:
                prev = self.last_track_for_gt.get(gid)
                if prev is not None and prev != tid:
                    self.idsw += 1
                if self.gt_was_tracked.get(gid) is False:
                    self.frag += 1
                self.last_track_for_gt[gid] = tid
            self.gt_was_tracked[gid] = tid is not None

    def summary(self):
        mota = 1.0 - (self.fn + self.fp + self.idsw) / self.n_gt if self.n_gt else None
        return {'gt': self.n_gt, 'fn': self.fn, 'fp': self.fp,
                'idsw': self.idsw, 'frag': self.frag,
                'mota': round(mota, 4) if mota is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='*', default=DEFAULT_METHODS)
    ap.add_argument('--conditions', nargs='*', default=list(CONDITIONS.keys()),
                    choices=list(CONDITIONS.keys()))
    args = ap.parse_args()

    import torch
    from opencood.tools import train_utils
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.utils.transformation_utils import x1_to_x2

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_root = os.path.expanduser(args.checkpoint_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for method in args.methods:
        folder, fusion, _ = METHODS[method]
        assert fusion == 'intermediate', 'Phase 5 targets intermediate methods'
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

        n_total = len(ds_clean)
        # scenario boundaries: [start, end) index ranges
        bounds, start = [], 0
        for end in ds_clean.len_record:
            bounds.append((start, end)); start = end

        # per-frame clean context (shared by all conditions): world-frame GT tracks
        print('[%s] building GT-track + ego-pose cache' % method)
        gt_cache = {}
        for i in range(n_total):
            np.random.seed(crc('gt', method, i))
            sample = ds_clean[i]
            ego = sample['ego']
            mask = ego['object_bbx_mask'] == 1
            centers = np.asarray(ego['object_bbx_center'])[mask][:, :3]
            ids = list(ego['object_ids'])
            base = ds_clean.retrieve_base_data(i)
            ego_pose = next(v for v in base.values()
                            if v['ego'])['params']['lidar_pose']
            M = x1_to_x2(ego_pose, [0, 0, 0, 0, 0, 0])   # ego frame -> world
            pts = centers @ M[:3, :3].T + M[:3, 3]
            gt_cache[i] = ([(gid, float(p[0]), float(p[1]))
                            for gid, p in zip(ids, pts)], M)

        for cond in pending:
            t0 = time.time()
            cfg = ChannelConfig(seed=crc(method, cond), name=cond,
                                **CONDITIONS[cond])
            channel = CommChannel(cfg)
            channel.attach(ds_chan)
            acc = MotAccumulator()
            try:
                with torch.no_grad():
                    for (s, e) in bounds:
                        tracker = Tracker()
                        acc.reset_sequence()
                        for i in range(s, e):
                            np.random.seed(crc(method, cond, i))
                            sample = ds_chan[i]
                            batch = ds_chan.collate_batch_test([sample])
                            batch = train_utils.to_device(batch, device)
                            pred, score = predict_boxes(fusion, ds_chan,
                                                        model, batch)
                            gt_items, M = gt_cache[i]
                            if pred is not None and len(pred):
                                c_ego = pred.cpu().numpy().mean(axis=1)  # (N,3)
                                c_w = c_ego @ M[:3, :3].T + M[:3, 3]
                                det_xy = c_w[:, :2]
                            else:
                                det_xy = np.zeros((0, 2))
                            trk = tracker.step(det_xy)
                            acc.step(gt_items, [(t[0], t[1], t[2]) for t in trk])
            finally:
                channel.detach()
            record = {'method': method, 'condition': cond,
                      'channel_config': cfg.to_dict(),
                      'frames': n_total, 'scenarios': len(bounds),
                      'metrics': acc.summary(),
                      'runtime_s': round(time.time() - t0, 1)}
            with open(os.path.join(out_dir, '%s__%s.json' % (method, cond)),
                      'w') as f:
                json.dump(record, f, indent=2)
            m = record['metrics']
            print('[%s__%s] MOTA %.3f | FN %d FP %d IDSW %d FRAG %d | %.0fs'
                  % (method, cond, m['mota'], m['fn'], m['fp'], m['idsw'],
                     m['frag'], record['runtime_s']))
    print('done — results in %s' % out_dir)


if __name__ == '__main__':
    main()
