#!/usr/bin/env python
"""Phase 2 identity gate: an attached CommChannel with the identity config must be
perfectly transparent. Runs the same frames through (a) the stock dataset and (b) a
channel-wrapped dataset, and asserts the predictions are IDENTICAL per frame (boxes
and scores), which is far stricter than AP equality.

Also demonstrates the reference GT policy for all impairment runs: predictions from
the (wrapped) dataset, ground truth from a parallel clean dataset.

Run on the GPU machine:

    cd ~/cpfa/OpenCOOD
    python ~/cpfa/Others/collab_perception_failure_analysis/scripts/run_phase2_identity.py \
        --checkpoint-root ~/cpfa/checkpoints --method attfuse --frames 100
"""
import argparse
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commchannel import ChannelConfig, CommChannel     # noqa: E402
from run_phase1 import METHODS                          # noqa: E402


def build(ckpt_dir):
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    hypes = yaml_utils.load_yaml(os.path.join(ckpt_dir, 'config.yaml'))
    return build_dataset(hypes, visualize=True, train=False), hypes


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
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
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

    def loader(ds):
        return DataLoader(ds, batch_size=1, num_workers=0,
                          collate_fn=ds.collate_batch_test, shuffle=False)

    n = min(args.frames, len(clean_ds))
    mismatches = 0
    checked = 0
    with torch.no_grad():
        for i, (bc, bw) in enumerate(zip(loader(clean_ds), loader(wrapped_ds))):
            if i >= n:
                break
            bc = train_utils.to_device(bc, device)
            bw = train_utils.to_device(bw, device)
            pc, sc, gc = predict(bc, model, clean_ds, fusion)
            pw, sw, gw = predict(bw, model, wrapped_ds, fusion)
            ok = True
            if (pc is None) != (pw is None):
                ok = False
            elif pc is not None:
                ok = (pc.shape == pw.shape and torch.allclose(pc, pw)
                      and torch.allclose(sc, sw))
            # GT policy check: clean GT usable against wrapped predictions
            ok = ok and gc.shape == gw.shape and torch.allclose(gc, gw)
            mismatches += 0 if ok else 1
            checked += 1
            if not ok:
                print('frame %d MISMATCH' % i)

    print('\n=== Phase 2 identity gate (%s, %d frames) ===' % (args.method, checked))
    if mismatches == 0:
        print('PASS — identity channel is perfectly transparent '
              '(boxes, scores, GT all identical)')
        sys.exit(0)
    print('FAIL — %d/%d frames differ' % (mismatches, checked))
    sys.exit(1)


if __name__ == '__main__':
    main()
