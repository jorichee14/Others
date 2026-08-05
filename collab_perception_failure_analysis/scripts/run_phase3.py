#!/usr/bin/env python
"""Phase 3 — constrained-link sweep runner.

Executes the configs/matrix.yaml grid: for every (method, impairment, level, seed)
cell, evaluates the pretrained model with the CommChannel applied and writes one JSON
of aggregate metrics to <out>/. Fully resumable (existing cell JSONs are skipped).

Key invariants:
  * Predictions come from the impaired dataset; ground truth ALWAYS comes from a clean
    parallel dataset (cached per frame per method), so GT never shrinks when messages
    are dropped or go stale.
  * All pipeline randomness (OpenCOOD's test-time point shuffling) is seeded per
    (cell, frame); all channel randomness is seeded from the cell id. Cells are
    exactly reproducible.
  * Bandwidth cells use feature-quantization hooks (intermediate fusion only) and
    record the measured bits/frame.

Run on the GPU machine:

    cd ~/cpfa/OpenCOOD
    python ~/cpfa/Others/collab_perception_failure_analysis/scripts/run_phase3.py \
        --checkpoint-root ~/cpfa/checkpoints --out ~/cpfa/results/sweeps

Useful flags:
    --methods attfuse fcooper     # subset of methods
    --impairments latency ghosts  # subset of impairments
    --seeds 0                     # subset of seeds
    --stride 10                   # override frame stride (pilot runs)
    --max-cells 5                 # stop after N cells (smoke test)
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

from commchannel import ChannelConfig, CommChannel            # noqa: E402
from run_phase1 import METHODS, compute_metrics, IOUS         # noqa: E402

INTERMEDIATE_ONLY_IMPAIRMENTS = {'bandwidth'}


def crc(*parts):
    return zlib.crc32('/'.join(str(p) for p in parts).encode('utf8')) & 0xFFFFFFFF


def cell_channel_config(imp_name, imp_spec, level_value, cell_seed):
    """Translate a matrix entry into a ChannelConfig (bandwidth returns identity —
    it is applied via hooks, not the dataset channel)."""
    kw = {'seed': cell_seed, 'name': imp_name}
    param = imp_spec['param']
    if param == 'ge_mean_loss':
        p_bg = float(imp_spec.get('p_bad_to_good', 0.3))
        loss = float(level_value)
        kw['ge_p_bad_to_good'] = p_bg
        kw['ge_p_good_to_bad'] = loss * p_bg / (1.0 - loss)
    elif param == 'bandwidth_bits':
        pass  # hooks handle it
    elif param == 'ghost_count':
        kw['ghost_count'] = int(level_value)
        kw['ghost_p'] = float(imp_spec.get('ghost_p', 1.0))
    elif param == 'pose_xyz_std':
        kw['pose_xyz_std'] = float(level_value)
        kw['pose_yaw_std_deg'] = float(level_value) * \
            float(imp_spec.get('yaw_deg_per_m', 0.0))
    else:
        kw[param] = type(ChannelConfig.__dataclass_fields__[param].default)(level_value)
    return ChannelConfig(**kw)


def predict_boxes(fusion, dataset, model, batch):
    """Predictions only (no GT) — GT comes from the clean cache."""
    if fusion == 'late':
        output = OrderedDict()
        for cav_id, cav_content in batch.items():
            output[cav_id] = model(cav_content)
        return dataset.post_processor.post_process(batch, output)
    output = OrderedDict(ego=model(batch['ego']))
    return dataset.post_processor.post_process(batch, output)


def collaborators_in(fusion, batch):
    if fusion == 'late':
        return len(batch) - 1
    return int(batch['ego']['record_len'].sum().item()) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--matrix', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'matrix.yaml'))
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--impairments', nargs='*')
    ap.add_argument('--seeds', nargs='*', type=int)
    ap.add_argument('--stride', type=int)
    ap.add_argument('--max-cells', type=int, default=0)
    args = ap.parse_args()

    import yaml
    import torch
    from opencood.tools import train_utils
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils

    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    stride = args.stride or int(matrix.get('frame_stride', 1))
    seeds = args.seeds if args.seeds is not None else matrix['seeds']
    methods = args.methods or matrix['methods']
    impairments = {k: v for k, v in matrix['impairments'].items()
                   if not args.impairments or k in args.impairments}

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_root = os.path.expanduser(args.checkpoint_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    done = skipped = failed = 0
    ran_cells = 0
    for method in methods:
        folder, fusion, _ = METHODS[method]
        ckpt_dir = os.path.join(ckpt_root, folder)

        # cells for this method (so we only build model/datasets when needed)
        cells = []
        for imp_name, imp_spec in impairments.items():
            if imp_name in INTERMEDIATE_ONLY_IMPAIRMENTS \
                    and fusion != 'intermediate':
                continue
            for li, level in enumerate(imp_spec['levels']):
                for seed in seeds:
                    cid = '%s__%s__L%d__s%d' % (method, imp_name, li, seed)
                    if not os.path.isfile(os.path.join(out_dir, cid + '.json')):
                        cells.append((cid, imp_name, imp_spec, li, level, seed))
                    else:
                        skipped += 1
        if not cells:
            continue

        print('[%s] building model + datasets (%d cells pending)'
              % (method, len(cells)))
        hypes = yaml_utils.load_yaml(os.path.join(ckpt_dir, 'config.yaml'))
        ds_clean = build_dataset(hypes, visualize=True, train=False)
        ds_chan = build_dataset(hypes, visualize=True, train=False)
        model = train_utils.create_model(hypes)
        if torch.cuda.is_available():
            model.cuda()
        _, model = train_utils.load_saved_model(ckpt_dir, model)
        model.eval()

        n_total = len(ds_clean)
        frame_indices = list(range(0, n_total, stride))
        gt_cache = {}

        def gt_for(i):
            if i not in gt_cache:
                np.random.seed(crc('gt', method, i))
                sample = ds_clean[i]
                batch = ds_clean.collate_batch_test([sample])
                gt_cache[i] = ds_clean.post_processor.generate_gt_bbx(batch)
            return gt_cache[i]

        for (cid, imp_name, imp_spec, li, level, seed) in cells:
            if args.max_cells and ran_cells >= args.max_cells:
                break
            t0 = time.time()
            cell_seed = crc(cid)
            cfg = cell_channel_config(imp_name, imp_spec, level, cell_seed)

            channel = CommChannel(cfg)
            channel.attach(ds_chan)
            hooks, meter = [], None
            if imp_spec['param'] == 'bandwidth_bits':
                from commchannel.feature_hooks import (attach_bandwidth_hooks,
                                                       BandwidthMeter)
                meter = BandwidthMeter()
                hooks = attach_bandwidth_hooks(model, int(level), meter)

            stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                    for iou in IOUS}
            collab_sum = 0
            try:
                with torch.no_grad():
                    for i in frame_indices:
                        np.random.seed(crc(cid, i))
                        sample = ds_chan[i]
                        batch = ds_chan.collate_batch_test([sample])
                        batch = train_utils.to_device(batch, device)
                        pred_box, pred_score = \
                            predict_boxes(fusion, ds_chan, model, batch)
                        gt_box = gt_for(i)
                        collab_sum += collaborators_in(fusion, batch)
                        for iou in IOUS:
                            eval_utils.caluclate_tp_fp(
                                pred_box, pred_score, gt_box, stat, iou)
                record = {
                    'cell': cid,
                    'method': method,
                    'fusion': fusion,
                    'impairment': imp_name,
                    'level_index': li,
                    'level_value': level,
                    'sweep_seed': seed,
                    'channel_config': cfg.to_dict(),
                    'frames': len(frame_indices),
                    'frame_stride': stride,
                    'mean_collaborators': round(
                        collab_sum / len(frame_indices), 3),
                    'mean_bits_per_frame': meter.mean_bits if meter else None,
                    'metrics': {str(iou): compute_metrics(stat, iou)
                                for iou in IOUS},
                    'runtime_s': round(time.time() - t0, 1),
                    'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                with open(os.path.join(out_dir, cid + '.json'), 'w') as f:
                    json.dump(record, f, indent=2)
                m = record['metrics']
                print('[%s] AP@0.5/0.7 = %.3f/%.3f | P@0.7 %.3f R@0.7 %.3f | '
                      'collab %.2f | %.0fs'
                      % (cid, m['0.5']['ap'], m['0.7']['ap'],
                         m['0.7']['precision'], m['0.7']['recall'],
                         record['mean_collaborators'], record['runtime_s']))
                done += 1
            except Exception as e:  # noqa: BLE001 - a bad cell must not kill the sweep
                import traceback
                traceback.print_exc()
                print('[%s] FAILED — %s: %s' % (cid, type(e).__name__, e))
                failed += 1
            finally:
                channel.detach()
                for h in hooks:
                    h.remove()
            ran_cells += 1
        if args.max_cells and ran_cells >= args.max_cells:
            break

    print('\nsweep finished: %d done, %d skipped (already existed), %d failed'
          % (done, skipped, failed))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
