#!/usr/bin/env python
"""Phase 2 identity gate: an attached CommChannel with the identity config must be
perfectly transparent.

OpenCOOD's test-time pipeline is stochastic (shuffle_points draws a fresh
np.random.permutation in every __getitem__), so two passes over the same frame differ
even with NO channel involved. The gate therefore seeds numpy identically before each
side's __getitem__ and compares the COLLATED INPUT BATCHES bitwise, tensor by tensor —
the strongest possible statement of channel transparency — and additionally checks that
model predictions on the two (identical) batches agree.

Run on the GPU machine:

    cd ~/cpfa/OpenCOOD
    python ~/cpfa/Others/collab_perception_failure_analysis/scripts/run_phase2_identity.py \
        --checkpoint-root ~/cpfa/checkpoints --method attfuse --frames 100
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commchannel import ChannelConfig, CommChannel     # noqa: E402
from run_phase1 import METHODS                          # noqa: E402

SEED0 = 20260805


def build(ckpt_dir):
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    hypes = yaml_utils.load_yaml(os.path.join(ckpt_dir, 'config.yaml'))
    return build_dataset(hypes, visualize=True, train=False), hypes


def compare(a, b, path=''):
    """Recursively compare two collated batch structures bitwise. Returns a list of
    paths that differ."""
    import torch
    diffs = []
    if type(a) is not type(b):
        return [path + ' (type %s vs %s)' % (type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return [path + ' (keys %s vs %s)' % (sorted(map(str, a)),
                                                 sorted(map(str, b)))]
        for k in a:
            diffs += compare(a[k], b[k], '%s.%s' % (path, k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + ' (len %d vs %d)' % (len(a), len(b))]
        for i, (x, y) in enumerate(zip(a, b)):
            diffs += compare(x, y, '%s[%d]' % (path, i))
    elif isinstance(a, torch.Tensor):
        if a.shape != b.shape or not torch.equal(a, b):
            diffs.append(path)
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or not np.array_equal(a, b):
            diffs.append(path)
    else:
        if a != b:
            diffs.append(path)
    return diffs


def predict(batch_data, model, dataset, fusion):
    from opencood.tools import inference_utils
    if fusion == 'late':
        return inference_utils.inference_late_fusion(batch_data, model, dataset)
    if fusion in ('early', 'intermediate'):
        return inference_utils.inference_early_fusion(batch_data, model, dataset)
    raise ValueError(fusion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-root', required=True)
    ap.add_argument('--method', default='attfuse',
                    choices=[m for m in METHODS if m != 'nocomm'])
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--model-check-frames', type=int, default=10,
                    help='frames on which to also compare model predictions')
    args = ap.parse_args()

    import torch
    from opencood.tools import train_utils

    folder, fusion, _ = METHODS[args.method]
    ckpt_dir = os.path.join(os.path.expanduser(args.checkpoint_root), folder)

    print('building clean and channel-wrapped datasets')
    clean_ds, hypes = build(ckpt_dir)
    wrapped_ds, _ = build(ckpt_dir)
    channel = CommChannel(ChannelConfig())   # identity
    channel.attach(wrapped_ds)

    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, model = train_utils.load_saved_model(ckpt_dir, model)
    model.eval()

    n = min(args.frames, len(clean_ds))
    bad_frames = 0
    pred_bad = 0
    pred_checked = 0
    with torch.no_grad():
        for i in range(n):
            # pin the pipeline's own randomness (shuffle_points) identically
            # pin the pipeline's randomness identically on both sides — BOTH
            # __getitem__ (shuffle_points) and collate (downsample_lidar_minimum
            # for the visualization cloud) consume the global numpy RNG.
            np.random.seed(SEED0 + i)
            sample_c = clean_ds[i]
            np.random.seed(SEED0 + i)
            sample_w = wrapped_ds[i]
            np.random.seed(SEED0 + i)
            batch_c = clean_ds.collate_batch_test([sample_c])
            np.random.seed(SEED0 + i)
            batch_w = wrapped_ds.collate_batch_test([sample_w])

            diffs = compare(batch_c, batch_w)
            if diffs:
                bad_frames += 1
                print('frame %d INPUT MISMATCH at: %s%s'
                      % (i, ', '.join(diffs[:4]),
                         ' (+%d more)' % (len(diffs) - 4) if len(diffs) > 4 else ''))
                continue

            if pred_checked < args.model_check_frames:
                bc = train_utils.to_device(batch_c, device)
                bw = train_utils.to_device(batch_w, device)
                pc, sc, gc = predict(bc, model, clean_ds, fusion)
                pw, sw, gw = predict(bw, model, wrapped_ds, fusion)
                ok = (pc is None) == (pw is None)
                if ok and pc is not None:
                    ok = (pc.shape == pw.shape and torch.allclose(pc, pw)
                          and torch.allclose(sc, sw))
                ok = ok and gc.shape == gw.shape and torch.allclose(gc, gw)
                pred_checked += 1
                if not ok:
                    pred_bad += 1
                    print('frame %d PREDICTION MISMATCH (inputs were identical)' % i)

    print('\n=== Phase 2 identity gate (%s) ===' % args.method)
    print('input batches: %d/%d bitwise identical' % (n - bad_frames, n))
    print('model outputs: %d/%d identical (on input-identical frames)'
          % (pred_checked - pred_bad, pred_checked))
    if bad_frames == 0 and pred_bad == 0:
        print('PASS — identity channel is perfectly transparent')
        sys.exit(0)
    print('FAIL')
    sys.exit(1)


if __name__ == '__main__':
    main()
