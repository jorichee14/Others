#!/usr/bin/env python
"""Aggregate Phase 3 sweep cell JSONs into per-(method, impairment, level) rows with
mean±std over seeds, plus the Phase 4 floor-test classification.

    python aggregate_sweeps.py --sweeps ~/cpfa/results/sweeps --out ~/cpfa/results

Outputs:
    <out>/sweep_summary.csv   one row per (method, impairment, level)
    <out>/sweep_summary.md    same, human-readable, grouped by method

Floor defaults come from the frozen Phase 1 baseline (results/baseline.md): the
ego-only nocomm floor. Classification at AP@0.7:
    above_floor   mean > floor + margin   (collaboration still net-positive)
    at_floor      within +/- margin       (benefit fully lost: delivery-type endpoint)
    below_floor   mean < floor - margin   (collaboration actively harmful: content-type)
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

FLOOR = {'ap70': 0.575, 'p70': 0.825, 'r70': 0.666}


def classify(ap70_mean, floor_ap, margin):
    if ap70_mean < floor_ap - margin:
        return 'below_floor'
    if ap70_mean > floor_ap + margin:
        return 'above_floor'
    return 'at_floor'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweeps', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--floor-ap70', type=float, default=FLOOR['ap70'])
    ap.add_argument('--margin', type=float, default=0.02)
    args = ap.parse_args()

    sweep_dir = os.path.expanduser(args.sweeps)
    groups = defaultdict(list)
    for fn in sorted(os.listdir(sweep_dir)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(sweep_dir, fn)) as f:
            r = json.load(f)
        groups[(r['method'], r['impairment'], r['level_index'])].append(r)

    rows = []
    for (method, imp, li), cells in sorted(groups.items()):
        def agg(pick):
            vals = [pick(c) for c in cells]
            return (statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ap50 = agg(lambda c: c['metrics']['0.5']['ap'])
        ap70 = agg(lambda c: c['metrics']['0.7']['ap'])
        p70 = agg(lambda c: c['metrics']['0.7']['precision'])
        r70 = agg(lambda c: c['metrics']['0.7']['recall'])
        rows.append({
            'method': method,
            'impairment': imp,
            'level_index': li,
            'level_value': cells[0]['level_value'],
            'seeds': len(cells),
            'ap50_mean': round(ap50[0], 4), 'ap50_std': round(ap50[1], 4),
            'ap70_mean': round(ap70[0], 4), 'ap70_std': round(ap70[1], 4),
            'p70_mean': round(p70[0], 4), 'p70_std': round(p70[1], 4),
            'r70_mean': round(r70[0], 4), 'r70_std': round(r70[1], 4),
            'mean_collaborators': (round(statistics.mean(collabs), 3)
                                   if (collabs := [c['mean_collaborators']
                                                   for c in cells
                                                   if c.get('mean_collaborators')
                                                   is not None]) else None),
            'mean_bits_per_frame': cells[0].get('mean_bits_per_frame'),
            'floor_class': classify(ap70[0], args.floor_ap70, args.margin),
        })

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'sweep_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(out_dir, 'sweep_summary.md')
    with open(md_path, 'w') as f:
        f.write('# Sweep summary (mean ± std over seeds)\n\n')
        f.write('Floor (ego-only) AP@0.7 = %.3f, margin ±%.2f\n' %
                (args.floor_ap70, args.margin))
        methods = sorted({r['method'] for r in rows})
        for method in methods:
            f.write('\n## %s\n\n' % method)
            f.write('| impairment | level | AP@0.7 | P@0.7 | R@0.7 | collab | floor test |\n')
            f.write('|---|---|---|---|---|---|---|\n')
            for r in rows:
                if r['method'] != method:
                    continue
                collab = ('%.2f' % r['mean_collaborators']
                          if r['mean_collaborators'] is not None else 'n/a')
                f.write('| %s | L%d (%s) | %.3f ± %.3f | %.3f | %.3f | %s | %s |\n' % (
                    r['impairment'], r['level_index'], r['level_value'],
                    r['ap70_mean'], r['ap70_std'], r['p70_mean'], r['r70_mean'],
                    collab, r['floor_class']))
    print('wrote %s (%d rows) and %s' % (csv_path, len(rows), md_path))


if __name__ == '__main__':
    main()
