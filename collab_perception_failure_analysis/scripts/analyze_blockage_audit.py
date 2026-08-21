#!/usr/bin/env python
"""Stratified re-analysis of the Phase 6.1 blockage audit.

WHY THIS EXISTS. The audit's verdict reads a POOLED comparison: E[U|blocked] vs
E[U|clear] over all links. Pooling across scenarios is confounded, because dense
traffic raises BOTH quantities at once — more vehicles means more chord blockers
AND more objects the ego cannot see by itself. A pooled correlation is therefore
consistent with zero within-scene coupling (Simpson's paradox), which is exactly
the failure mode the audit's own per-scenario table was added to expose.

This script decides the question the way it has to be decided: WITHIN scenario.

    stratified effect = Σ_s w_s (E[U|blocked,s] − E[U|clear,s]) / Σ_s w_s

with w_s = links in scenario s, over scenarios that actually contain BOTH blocked
and clear links (a scenario at 0% or 100% blockage carries no information about
the conditional and must not silently vote through the pooled mean).

Three robustness checks, because a "GO" that rides on one intersection is not a GO:

  * leave-one-scenario-out jackknife — is any single scenario carrying the sign?
  * sign test over scenarios       — do most scenes agree, or is it a coin flip?
  * permutation test               — shuffle blocked/clear WITHIN each scenario,
                                     10k times; how often does chance beat the
                                     observed stratified effect?

Usage:
    python scripts/analyze_blockage_audit.py \
        --audit ~/cpfa/results/blockage/blockage_audit.json [--clearance 1.0]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def stratified(groups):
    """groups: {scenario: (u_blocked[], u_clear[])} -> (effect, usable weight,
    per-scenario [(s, n, diff)])."""
    num = den = 0.0
    per = []
    for s, (ub, uc) in sorted(groups.items()):
        if len(ub) == 0 or len(uc) == 0:
            continue                       # no within-scenario contrast
        diff = float(np.mean(ub) - np.mean(uc))
        w = len(ub) + len(uc)
        num += w * diff
        den += w
        per.append((s, w, diff))
    return (num / den if den else float('nan')), den, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', required=True, help='blockage_audit.json')
    ap.add_argument('--clearance', type=float, default=1.0)
    ap.add_argument('--min-blockers', type=int, default=1)
    ap.add_argument('--permutations', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(os.path.expanduser(args.audit)) as f:
        blob = json.load(f)
    links = blob['links']
    key = 'n_blockers_%g' % args.clearance
    if links and key not in links[0]:
        raise SystemExit('clearance %g not in audit (have: %s)' % (
            args.clearance, [k for k in links[0] if k.startswith('n_blockers')]))

    groups = defaultdict(lambda: ([], []))
    for r in links:
        blocked = r[key] >= args.min_blockers
        groups[r['scenario']][0 if blocked else 1].append(r['unique_value'])

    # ---- pooled (what the audit's verdict uses) ----
    ub_all = [u for s in groups for u in groups[s][0]]
    uc_all = [u for s in groups for u in groups[s][1]]
    pooled = float(np.mean(ub_all) - np.mean(uc_all))

    # ---- stratified ----
    effect, weight, per = stratified(groups)
    informative = len(per)
    total_scen = len(groups)

    # ---- jackknife ----
    jack = []
    for s_drop, _, _ in per:
        e, _, _ = stratified({s: v for s, v in groups.items() if s != s_drop})
        jack.append((s_drop, e))
    worst = min(jack, key=lambda t: t[1])
    flips = [s for s, e in jack if (e > 0) != (effect > 0)]

    # ---- sign test ----
    pos = sum(1 for _, _, d in per if d > 0)
    neg = sum(1 for _, _, d in per if d < 0)

    # ---- permutation (within scenario) ----
    rng = np.random.RandomState(args.seed)
    obs = effect
    ge = 0
    for _ in range(args.permutations):
        shuffled = {}
        for s, (ub, uc) in groups.items():
            pool = np.array(ub + uc, dtype=float)
            if len(pool) == 0:
                continue
            idx = rng.permutation(len(pool))
            nb = len(ub)
            shuffled[s] = (list(pool[idx[:nb]]), list(pool[idx[nb:]]))
        e, _, _ = stratified(shuffled)
        if not np.isnan(e) and e >= obs:
            ge += 1
    p_perm = (ge + 1) / (args.permutations + 1)

    print('=== Phase 6.1 stratified re-analysis (clearance %.1f m) ===' % args.clearance)
    print('links %d | scenarios %d (%d with both blocked and clear links)'
          % (len(links), total_scen, informative))
    print()
    print('POOLED   E[U|blocked] - E[U|clear] = %+.4f   <- audit verdict uses this'
          % pooled)
    print('STRATIFIED (within-scenario, weighted) = %+.4f' % effect)
    print()
    print('per-scenario differences (positive = blocked links are MORE valuable):')
    for s, w, d in per:
        print('   scenario %-3s n=%-4d diff=%+.4f' % (s, w, d))
    print()
    print('sign test:  %d scenarios positive, %d negative' % (pos, neg))
    print('jackknife:  most influential scenario = %s -> effect becomes %+.4f'
          % (worst[0], worst[1]))
    if flips:
        print('            REMOVING scenario(s) %s FLIPS THE SIGN'
              % ', '.join(str(s) for s in flips))
    print('permutation (%d shuffles within scenario): p = %.4f'
          % (args.permutations, p_perm))
    print()
    strong = (effect > 0 and not flips and p_perm < 0.05 and pos > neg)
    print('VERDICT: %s' % (
        'effect survives stratification and robustness checks'
        if strong else
        'POOLED EFFECT IS NOT ROBUST — do not claim scene-conditioned value coupling'))
    if not strong:
        print('  The matched-PDR sweep is still worth running: it tests whether')
        print('  geometrically clustered/persistent loss hurts more than i.i.d. at')
        print('  equal delivery. But the "you lose the messages you needed" framing')
        print('  is not supported by this data and should not be asserted.')


if __name__ == '__main__':
    main()
