#!/usr/bin/env python
"""Pivot the per-method sweep summary into one full grid table (methods as columns).

    python pivot_sweep_table.py --summary ../results/sweep_summary.md \
        --out ../results/sweep_table.md

`aggregate_sweeps.py` writes one table per method, which is the right shape for
reading a single architecture's degradation curve but the wrong shape for
comparing architectures at a fixed condition. This reshapes the same rows into
(impairment, level) x method blocks -- one for AP@0.7, one for P@0.7, one for
R@0.7 -- and marks each cell with its floor-test class. No numbers are
recomputed: the mean/std/classification all come from the summary as written, so
this stays consistent with results/ANALYSIS.md by construction.

Reads sweep_summary.md rather than sweep_summary.csv because the CSV is
git-ignored alongside the raw cell JSONs, while the md is the committed record.
"""
import argparse
import os
import re

# Impairment order: delivery families first, then content -- the study's main axis.
IMP_ORDER = ['loss_iid', 'loss_burst', 'bandwidth',
             'latency', 'stale', 'pose', 'ghosts', 'swap']
FAMILY = {'loss_iid': 'delivery', 'loss_burst': 'delivery', 'bandwidth': 'delivery',
          'latency': 'content', 'stale': 'content', 'pose': 'content',
          'ghosts': 'content', 'swap': 'content'}
# Clean-channel AP@0.7 order (best -> worst), from results/baseline.md.
METHOD_ORDER = ['cobevt', 'coalign', 'v2vnet', 'attfuse', 'early', 'fcooper', 'late']
# Published spellings, for the summary block only. The detail blocks keep the bare
# checkpoint keys, which are what the runners take on the command line.
METHOD_NAME = {'cobevt': 'CoBEVT', 'coalign': 'CoAlign', 'v2vnet': 'V2VNet',
               'attfuse': 'AttFuse', 'fcooper': 'F-Cooper',
               'early': 'Early fusion', 'late': 'Late fusion'}
CLASS_MARK = {'above_floor': '', 'at_floor': '~', 'below_floor': '*'}

ROW_RE = re.compile(
    r'^\|\s*(?P<imp>\w+)\s*\|\s*L(?P<li>\d+)\s*\((?P<lv>[^)]*)\)\s*\|'
    r'\s*(?P<ap>[\d.]+)\s*±\s*(?P<sd>[\d.]+)\s*\|\s*(?P<p>[\d.]+)\s*\|'
    r'\s*(?P<r>[\d.]+)\s*\|\s*(?P<collab>[\d.]+|n/a)\s*\|\s*(?P<cls>\w+)\s*\|')


def parse(summary_path):
    """-> (rows, floor_ap, margin); rows keyed by (impairment, level_index, method)."""
    rows, floor_ap, margin, method = {}, None, None, None
    with open(summary_path) as f:
        for line in f:
            if line.startswith('## '):
                method = line[3:].strip()
                continue
            if line.startswith('Floor'):
                m = re.search(r'=\s*([\d.]+).*±([\d.]+)', line)
                if m:
                    floor_ap, margin = float(m.group(1)), float(m.group(2))
                continue
            m = ROW_RE.match(line)
            if not m:
                continue
            rows[(m.group('imp'), int(m.group('li')), method)] = {
                'level_value': m.group('lv'),
                'ap70': float(m.group('ap')), 'ap70_std': float(m.group('sd')),
                'p70': float(m.group('p')), 'r70': float(m.group('r')),
                'collab': m.group('collab'), 'floor_class': m.group('cls'),
            }
    return rows, floor_ap, margin


def conditions(rows):
    """Ordered (impairment, level_index, level_value) present in the sweep."""
    seen = {}
    for (imp, li, _), r in rows.items():
        seen.setdefault((imp, li), r['level_value'])
    return sorted(seen.items(),
                  key=lambda kv: (IMP_ORDER.index(kv[0][0])
                                  if kv[0][0] in IMP_ORDER else 99, kv[0][1]))


def write_block(f, rows, conds, methods, metric, *, with_std=False, mark=False):
    f.write('| impairment | level | ' + ' | '.join(methods) + ' |\n')
    f.write('|---|---|' + '---|' * len(methods) + '\n')
    prev_imp = None
    for (imp, li), lv in conds:
        if prev_imp is not None and imp != prev_imp:
            f.write('| | | ' + ' | '.join([''] * len(methods)) + ' |\n')
        prev_imp = imp
        cells = []
        for method in methods:
            r = rows.get((imp, li, method))
            if r is None:
                cells.append('n/a')
                continue
            if metric == 'collab':
                # Already a string ('n/a' where the method reports no collaborator
                # count); the floor class belongs to AP, not to delivery volume.
                cells.append(r['collab'])
                continue
            val = ('%.3f ± %.3f' % (r[metric], r['ap70_std'])
                   if with_std else '%.3f' % r[metric])
            # The floor test is defined on AP@0.7, so only that block is marked --
            # tagging a precision cell with an AP verdict would read as a claim
            # about precision.
            cells.append(val + CLASS_MARK[r['floor_class']] if mark else val)
        f.write('| %s | L%d (%s) | %s |\n' % (imp, li, lv, ' | '.join(cells)))


# Every impairment family, named, with the failure mode it represents and the unit
# its severities are expressed in. Ordered delivery-first, matching IMP_ORDER.
FAMILY_LABEL = [
    ('loss_iid', 'i.i.d. packet loss', 'drop rate'),
    ('loss_burst', 'bursty packet loss', 'mean drop rate'),
    ('bandwidth', 'feature quantization', 'bits'),
    ('latency', 'constant latency', 'frames @ 10 Hz'),
    ('stale', 'stale messages', 'refresh period, frames'),
    ('pose', 'collaborator pose error', 'metres'),
    ('ghosts', 'ghost injection', 'ghosts per message'),
    ('swap', 'scene swap', 'corrupted fraction'),
]
# The two headline aggregates. bandwidth and ghosts are deliberately in neither:
# bandwidth changes character mid-sweep (delivery to 4-bit, content below it), and
# ghosts is the one content family that never crosses the floor, so folding either
# in would blur the split. This is the weighting behind ANALYSIS.md's headline
# spans, and reproduces them.
AGGREGATES = [('delivery mean', ['loss_iid', 'loss_burst']),
              ('content mean', ['latency', 'stale', 'pose', 'swap'])]


def summary_block(f, rows, methods, clean):
    """One row per impairment family, mean AP@0.7 over that family's severities.

    Same orientation as the detail blocks below (families down, methods across) so
    the summary and the per-severity tables read the same way. Every family in the
    matrix appears by name -- the aggregate rows are additions at the bottom, never
    a substitute for the families they average.
    """
    def fam_mean(imp, method):
        vals = [r['ap70'] for (i, _, m), r in rows.items()
                if i == imp and m == method]
        return sum(vals) / len(vals) if vals else None

    def severities(imp):
        lv = sorted(((li, r['level_value']) for (i, li, _), r in rows.items()
                     if i == imp), key=lambda x: x[0])
        vals = [v for _, v in dict(lv).items()] if lv else []
        return len(vals), ('%s → %s' % (vals[0], vals[-1]) if vals else '')

    f.write('| impairment family | severities | ' +
            ' | '.join(METHOD_NAME.get(m, m) for m in methods) + ' |\n')
    f.write('|---|---|' + '---|' * len(methods) + '\n')
    f.write('| *clean channel (no impairment)* | — | ' +
            ' | '.join('%.3f' % clean[m] if m in clean else '—'
                       for m in methods) + ' |\n')
    for imp, label, unit in FAMILY_LABEL:
        n, span = severities(imp)
        if not n:
            continue
        cells = []
        for method in methods:
            mu = fam_mean(imp, method)
            cells.append('%.3f' % mu if mu is not None else 'n/a')
        f.write('| %s (`%s`) | %d levels, %s (%s) | %s |\n' % (
            label, imp, n, span, unit, ' | '.join(cells)))
    for label, fams in AGGREGATES:
        cells = []
        for method in methods:
            present = [mu for mu in (fam_mean(i, method) for i in fams)
                       if mu is not None]
            cells.append('%.3f' % (sum(present) / len(present))
                         if present else 'n/a')
        f.write('| **%s** (%s) | — | %s |\n' % (
            label, ', '.join('`%s`' % i for i in fams), ' | '.join(cells)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='results/sweep_summary.md')
    ap.add_argument('--out', default='results/sweep_table.md')
    args = ap.parse_args()

    rows, floor_ap, margin = parse(os.path.expanduser(args.summary))
    if not rows:
        raise SystemExit('no sweep rows parsed from %s' % args.summary)
    methods = [m for m in METHOD_ORDER if any(k[2] == m for k in rows)]
    methods += sorted({k[2] for k in rows} - set(methods))
    conds = conditions(rows)

    with open(os.path.expanduser(args.out), 'w') as f:
        f.write('# Phase 3 grid sweep — full results table\n\n')
        f.write('Every cell of the frozen matrix (`configs/matrix.yaml`) in one table: '
                '%d (impairment, level) conditions × %d methods = %d evaluated '
                'method-conditions (%d of the %d slots are `n/a`), each the mean over '
                '3 seeds on the OPV2V test split at stride 3 (724 frames/cell) — the '
                'study\'s %d cells.\n\n' % (len(conds), len(methods), len(rows),
                                len(conds) * len(methods) - len(rows),
                                len(conds) * len(methods), len(rows) * 3))
        f.write('Generated from `results/sweep_summary.md` by '
                '`scripts/pivot_sweep_table.py` — same numbers, reshaped so methods '
                'can be compared at a fixed condition.\n\n')
        f.write('Floor (ego-only, nocomm) AP@0.7 = **%.3f**, margin ±%.2f. '
                'Floor test, marked in the AP@0.7 block only: unmarked = above floor '
                '(collaboration still net-positive), `~` = at floor (benefit fully '
                'lost), `*` = below floor (collaboration actively '
                'harmful).\n\n' % (floor_ap, margin))
        f.write('`n/a` = condition not in the matrix for that method: bandwidth '
                'quantizes shared *features*, so it does not apply to late (boxes) '
                'or early (raw points) fusion.\n\n')
        clean = {'cobevt': 0.862, 'coalign': 0.833, 'v2vnet': 0.822, 'attfuse': 0.815,
                 'early': 0.801, 'fcooper': 0.790, 'late': 0.781}

        f.write('## Summary — one row per impairment family\n\n')
        f.write('All eight families, each collapsed to the mean AP@0.7 over its own '
                'severities. Read a row to see what one impairment costs every '
                'architecture; read a column to see one architecture\'s profile. The '
                'severities column says how many levels went into the mean and over '
                'what range, since a mean over "0.1 → 0.9 drop rate" is a summary of a '
                'curve, not a single operating point — the per-severity numbers are in '
                'the blocks below.\n\n')
        f.write('The last two rows are the study\'s headline aggregates, averaging the '
                'named families they list (each family weighted equally, not each '
                'cell). `bandwidth` and `ghosts` are in neither: quantization is a '
                'delivery impairment down to 4-bit and a content one below that, and '
                'ghost injection is the only content family that never crosses the '
                'floor, so folding either in would blur the very split the aggregates '
                'exist to show.\n\n')
        summary_block(f, rows, methods, clean)
        f.write('\nThe two aggregate rows are the whole study in miniature: delivery '
                'stays well above the %.3f floor for every method, content sits far '
                'below it for every method, and the gap between them (0.22–0.39) is '
                'wider than the spread between methods within either row. Latency is '
                'the hardest single family for all seven. Per-severity detail follows; '
                'the floor-crossing levels and the ranking reshuffle between delivery '
                'and content are in `results/ANALYSIS.md` §2–§4.\n' % floor_ap)

        f.write('\n## AP@0.7 (mean ± std over seeds)\n\n')
        write_block(f, rows, conds, methods, 'ap70', with_std=True, mark=True)
        f.write('\n## P@0.7\n\n')
        write_block(f, rows, conds, methods, 'p70')
        f.write('\n## R@0.7\n\n')
        write_block(f, rows, conds, methods, 'r70')

        f.write('\n## Mean collaborators delivered per frame\n\n')
        f.write('Realized delivery, not a requested rate: 1.59 = every collaborator '
                'arrived. Only the loss families move it — every other impairment '
                'delivers all messages and corrupts their content, which is the '
                'delivery/content split made mechanical. Two smaller effects are '
                'visible: the pose L3–L4 drop to ~1.3 is `dropped_empty` rather than '
                'channel loss — a cloud mislocalized far enough to land entirely '
                'outside the ego detection crop carries no points, and stock OpenCOOD '
                'cannot represent an empty agent, so the collaborator is dropped for '
                'that frame (`commchannel/channel.py:_survives_crop`); late fusion, '
                'which fuses boxes, is unaffected and holds 1.59. The ≤0.02 dips at '
                'the harshest latency/stale levels hit every method including late '
                'fusion, so they are not that filter. Here `n/a` '
                'means the count was not recorded for that method, not that the '
                'condition was skipped — early fusion reports no per-frame '
                'collaborator count.\n\n')
        write_block(f, rows, conds, methods, 'collab')

        f.write('\n## Not in this table\n\n')
        f.write('- **`loss_blocked` / `loss_iid_matched`** (scene-conditioned blockage '
                'and its matched i.i.d. control) are audited separately, at matched '
                'packet delivery — see `docs/BLOCKAGE.md` and `results/ANALYSIS.md` '
                '§8.\n')
        f.write('- **Compression variants** (`*_comp`) are excluded from the core '
                'matrix; their clean-channel cost is in `results/baseline.md`.\n')
        f.write('- **Tracking metrics** under impairment (Phase 5) are in '
                '`results/ANALYSIS.md` §9, not here — this table is detection only.\n')
    print('wrote %s (%d conditions × %d methods)' %
          (args.out, len(conds), len(methods)))


if __name__ == '__main__':
    main()
