#!/usr/bin/env python
"""Phase 1 — perfect-channel baseline runner.

Evaluates every checkpoint on the OPV2V test split with stock OpenCOOD inference and
records AP@0.3/0.5/0.7 plus overall precision/recall per method into one JSON per method,
then regenerates a markdown summary table. Also provides the No-Comm ego-only floor
(late-fusion checkpoint, ego input only, ground truth still built from the FULL
collaborator set so the floor is scored against the same GT as everything else).

Run on the GPU machine, inside the `opencood` conda env, from the OpenCOOD repo root
(so `import opencood` resolves):

    cd ~/cpfa/OpenCOOD
    python ~/cpfa/Others/collab_perception_failure_analysis/scripts/run_phase1.py \
        --checkpoint-root ~/cpfa/checkpoints \
        --out ~/cpfa/results/phase1

Useful flags:
    --methods nocomm late attfuse   # run a subset (default: all)
    --max-frames 100                # quick sanity pass on first N frames
    --force                         # re-run methods whose JSON already exists
    --num-workers 8

Each finished method writes <out>/<method>.json immediately, so an interrupted run
resumes by skipping completed methods. <out>/baseline.md is rebuilt after every method.
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Method registry: short name -> (checkpoint folder, fusion mode, published AP@0.7)
# Published numbers: OpenCOOD README OPV2V LiDAR-track benchmark (Default Towns);
# nocomm reference is the OPV2V paper's "No Fusion" baseline.
# ---------------------------------------------------------------------------
METHODS = OrderedDict([
    ('nocomm',       ('pointpillar_late_fusion',                 'nocomm',       0.602)),
    ('late',         ('pointpillar_late_fusion',                 'late',         0.781)),
    ('early',        ('pointpillar_early_fusion',                'early',        0.800)),
    ('attfuse',      ('pointpillar_attentive_fusion',            'intermediate', 0.815)),
    ('attfuse_comp', ('pointpillar_attentive_fusion_compression','intermediate', None)),
    ('fcooper',      ('pointpillar_fcooper',                     'intermediate', 0.790)),
    ('v2vnet',       ('pointpillar_v2vnet',                      'intermediate', 0.822)),
    ('coalign',      ('point_pillar_coalign',                    'intermediate', 0.833)),
    ('coalign_comp', ('point_pillar_coalign_compression',        'intermediate', None)),
    ('cobevt',       ('pointpillar_CoBEVT_nocompression',        'intermediate', 0.861)),
    ('cobevt_comp',  ('cobevt_compression',                      'intermediate', None)),
])

IOUS = (0.3, 0.5, 0.7)


def compute_metrics(result_stat, iou):
    """AP (VOC2010, accumulation order — matches OpenCOOD's default non-global-sort)
    plus overall precision/recall at the deployed operating point (the postprocessor's
    score threshold + NMS). Works on copies; never mutates result_stat."""
    tp = list(result_stat[iou]['tp'])
    fp = list(result_stat[iou]['fp'])
    gt = result_stat[iou]['gt']
    n_det = len(tp)
    tp_total = sum(tp)

    precision = tp_total / n_det if n_det else 0.0
    recall = tp_total / gt if gt else 0.0

    cum_tp, cum_fp, rec, prec = 0, 0, [], []
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        rec.append(cum_tp / gt if gt else 0.0)
        prec.append(cum_tp / (cum_tp + cum_fp))

    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return {'ap': ap, 'precision': precision, 'recall': recall,
            'tp': tp_total, 'fp': n_det - tp_total, 'gt': gt, 'detections': n_det}


def run_method(name, ckpt_dir, fusion, args):
    import torch
    from torch.utils.data import DataLoader, Subset

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils, inference_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils

    hypes = yaml_utils.load_yaml(os.path.join(ckpt_dir, 'config.yaml'))

    print('[%s] building dataset' % name)
    # visualize=True matches the stock inference.py code path validated in Phase 0.
    dataset = build_dataset(hypes, visualize=True, train=False)
    n_total = len(dataset)
    eval_set = dataset
    if args.max_frames and args.max_frames < n_total:
        eval_set = Subset(dataset, range(args.max_frames))
    loader = DataLoader(eval_set, batch_size=1, num_workers=args.num_workers,
                        collate_fn=dataset.collate_batch_test, shuffle=False,
                        pin_memory=False, drop_last=False)

    print('[%s] creating model + loading checkpoint' % name)
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch, model = train_utils.load_saved_model(ckpt_dir, model)
    model.eval()

    result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in IOUS}

    from tqdm import tqdm
    t0 = time.time()
    n_frames = 0
    with torch.no_grad():
        for batch_data in tqdm(loader, desc=name):
            batch_data = train_utils.to_device(batch_data, device)
            if fusion == 'late':
                pred_box, pred_score, gt_box = \
                    inference_utils.inference_late_fusion(batch_data, model, dataset)
            elif fusion == 'early':
                pred_box, pred_score, gt_box = \
                    inference_utils.inference_early_fusion(batch_data, model, dataset)
            elif fusion == 'intermediate':
                pred_box, pred_score, gt_box = \
                    inference_utils.inference_intermediate_fusion(batch_data, model,
                                                                  dataset)
            elif fusion == 'nocomm':
                # Ego-only floor: model + box post-processing see ONLY the ego vehicle,
                # but ground truth is generated from the full collaborator set so the
                # floor is scored against identical GT as the collaborative methods.
                output_dict = OrderedDict(ego=model(batch_data['ego']))
                ego_batch = OrderedDict(ego=batch_data['ego'])
                pred_box, pred_score = \
                    dataset.post_processor.post_process(ego_batch, output_dict)
                gt_box = dataset.post_processor.generate_gt_bbx(batch_data)
            else:
                raise ValueError('unknown fusion mode: %s' % fusion)

            for iou in IOUS:
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box,
                                           result_stat, iou)
            n_frames += 1
    runtime = time.time() - t0

    record = {
        'method': name,
        'checkpoint_folder': os.path.basename(ckpt_dir.rstrip('/')),
        'fusion_method': fusion,
        'loaded_epoch': epoch,
        'frames': n_frames,
        'frames_total_in_split': n_total,
        'runtime_s': round(runtime, 1),
        'fps': round(n_frames / runtime, 2) if runtime else None,
        'metrics': {str(iou): compute_metrics(result_stat, iou) for iou in IOUS},
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    return record


def write_baseline_md(out_dir):
    rows = []
    for name, (folder, fusion, ref) in METHODS.items():
        path = os.path.join(out_dir, name + '.json')
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            r = json.load(f)
        m50, m70 = r['metrics']['0.5'], r['metrics']['0.7']
        rows.append('| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s |' % (
            name, fusion, r['frames'],
            r['metrics']['0.3']['ap'], m50['ap'], m70['ap'],
            m70['precision'], m70['recall'], m50['recall'],
            ('%.3f' % ref) if ref is not None else '—'))
    lines = [
        '# Phase 1 — Perfect-channel baseline (OPV2V test split)',
        '',
        'Generated by `scripts/run_phase1.py`. Precision/recall are overall values at the',
        'deployed operating point (score threshold + NMS), cumulative over the split.',
        'The `nocomm` row is the ego-only floor — the reference line for Phase 4',
        'failure attribution.',
        '',
        '| method | fusion | frames | AP@0.3 | AP@0.5 | AP@0.7 | P@0.7 | R@0.7 | R@0.5 | published AP@0.7 |',
        '|--------|--------|--------|--------|--------|--------|-------|-------|-------|------------------|',
    ] + rows + ['']
    with open(os.path.join(out_dir, 'baseline.md'), 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='*', default=list(METHODS.keys()),
                    choices=list(METHODS.keys()))
    ap.add_argument('--max-frames', type=int, default=0,
                    help='evaluate only the first N frames (0 = all)')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--force', action='store_true',
                    help='re-run methods whose JSON already exists')
    args = ap.parse_args()

    ckpt_root = os.path.expanduser(args.checkpoint_root)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    failures = {}
    for name in args.methods:
        folder, fusion, _ = METHODS[name]
        out_json = os.path.join(out_dir, name + '.json')
        if os.path.isfile(out_json) and not args.force:
            print('[%s] already done (%s) — skipping' % (name, out_json))
            continue
        ckpt_dir = os.path.join(ckpt_root, folder)
        if not os.path.isdir(ckpt_dir):
            failures[name] = 'checkpoint folder missing: %s' % ckpt_dir
            print('[%s] SKIP — %s' % (name, failures[name]))
            continue
        try:
            record = run_method(name, ckpt_dir, fusion, args)
        except Exception as e:  # noqa: BLE001 - one bad method must not kill the sweep
            import traceback
            traceback.print_exc()
            failures[name] = '%s: %s' % (type(e).__name__, e)
            print('[%s] FAILED — %s' % (name, failures[name]))
            continue
        with open(out_json, 'w') as f:
            json.dump(record, f, indent=2)
        write_baseline_md(out_dir)
        m = record['metrics']
        print('[%s] AP@0.3/0.5/0.7 = %.3f/%.3f/%.3f | P@0.7 %.3f R@0.7 %.3f | %.0fs'
              % (name, m['0.3']['ap'], m['0.5']['ap'], m['0.7']['ap'],
                 m['0.7']['precision'], m['0.7']['recall'], record['runtime_s']))

    write_baseline_md(out_dir)
    print('\nSummary written to %s' % os.path.join(out_dir, 'baseline.md'))
    if failures:
        print('FAILED methods:')
        for k, v in failures.items():
            print('  %s: %s' % (k, v))
        sys.exit(1)


if __name__ == '__main__':
    main()
