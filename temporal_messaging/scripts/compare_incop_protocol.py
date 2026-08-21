#!/usr/bin/env python
"""Protocol-matched comparison against V2X-INCOP's headline OPV2V number.

WHY THIS EXISTS. V2X-INCOP (Ren, Lei, Wang, Dianati, Wang, Chen, Zhang;
IEEE T-IV 9(4) 2024; arXiv:2304.11821) has no public code release — verified
2026-08-21 across arXiv, the authors' GitHub, and general search (see
docs/BASELINE_AVAILABILITY.md). Its algorithm therefore cannot be run inside
`commchannel`. What CAN be done is speak its metric.

INCOP's headline is "cooperative perception gains up to 14.06% over individual
perception on average across different packet drop rates" on OPV2V. Our parent
study computes exactly that quantity natively, because the ego-only floor test
IS individual perception: the same detector, the same GT, the collaborators
withheld. So a protocol-matched table is free.

    gain(method) = mean over drop rates p of AP(method, p)  -  AP(ego-only)

reported both as absolute AP points and as a percentage relative to the floor,
because the paper's "%" is ambiguous from the abstract alone.

WHAT THIS DOES NOT SHOW. This is a protocol tension, not a refutation. Four
things must be checked against the paper's full text before any claim is made:

  1. Is their "%" relative gain or absolute mAP points?
  2. Which AP threshold — we only have AP@0.7 in the sweep summary.
  3. Which drop rates, and is their "interruption" a multi-frame outage rather
     than per-frame loss? Sustained outage is closer to our loss_burst axis with
     a low p_bg (the burst30_long / burst70_long conditions), not to loss_iid.
  4. What is their "individual perception" baseline detector, and what absolute
     AP does it reach? A weaker single-agent baseline inflates relative gain.

Usage:
    python scripts/compare_incop_protocol.py --sweep <sweep_summary.md> [--axis loss_iid]
"""
import argparse
import re

# Their reported headline, for the printed comparison line.
INCOP_OPV2V_GAIN_PCT = 14.06


def parse(path, axis):
    """-> ({method: [(level, ap), ...]}, floor_ap)."""
    floor, rows, method = None, {}, None
    with open(path) as f:
        for line in f:
            if floor is None:
                m = re.search(r'Floor \(ego-only\) AP@0\.7 = ([\d.]+)', line)
                if m:
                    floor = float(m.group(1))
            if line.startswith('## '):
                method = line[3:].strip()
                rows[method] = []
            elif method and line.startswith('| ' + axis + ' '):
                c = [x.strip() for x in line.strip('|\n').split('|')]
                rows[method].append((float(re.search(r'\(([\d.]+)\)', c[1]).group(1)),
                                     float(c[2].split('±')[0])))
    if floor is None:
        raise SystemExit('no floor line found in %s' % path)
    return {m: sorted(r) for m, r in rows.items() if r}, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True, help='results/sweep_summary.md')
    ap.add_argument('--axis', default='loss_iid', help='loss_iid | loss_burst')
    args = ap.parse_args()

    rows, floor = parse(args.sweep, args.axis)
    levels = [p for p, _ in next(iter(rows.values()))]

    print('=== V2X-INCOP protocol match: %s, OPV2V, AP@0.7 ===' % args.axis)
    print('individual perception (ego-only floor) = %.3f\n' % floor)

    hdr = ['method'] + ['p=%g' % p for p in levels] + ['mean', 'gain (pts)', 'gain (rel %)']
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))

    out = []
    for m, r in rows.items():
        mean = sum(a for _, a in r) / len(r)
        out.append((m, r, mean, mean - floor, (mean - floor) / floor * 100))
    for m, r, mean, pts, rel in sorted(out, key=lambda t: -t[4]):
        print('| %s | %s | %.3f | %+.3f | %+.2f%% |'
              % (m, ' | '.join('%.3f' % a for _, a in r), mean, pts, rel))

    best = max(out, key=lambda t: t[4])
    worst = min(out, key=lambda t: t[4])
    print('\nV2X-INCOP reports +%.2f%% on OPV2V averaged over drop rates.' % INCOP_OPV2V_GAIN_PCT)
    print('Unmodified baselines here span %+.2f%% (%s) to %+.2f%% (%s).'
          % (worst[4], worst[0], best[4], best[0]))
    if worst[4] > INCOP_OPV2V_GAIN_PCT:
        print('\nEVERY unmodified baseline exceeds the reported figure. That is a')
        print('protocol tension, not a refutation — resolve the four questions in')
        print('this file\'s docstring against the paper before claiming anything.')


if __name__ == '__main__':
    main()
