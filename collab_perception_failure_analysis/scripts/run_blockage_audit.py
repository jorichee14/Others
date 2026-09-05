#!/usr/bin/env python
"""Step 6.1 — is link blockage correlated with collaborator value? (model-free)

The whole study so far — and every robustness result in the collaborative
perception literature — drops messages INDEPENDENTLY of the scene. The physical
objection is that the vehicles occluding an agent's lidar are the same vehicles
obstructing its radio, so the messages you lose are disproportionately the ones
that would have filled your blind spots:

    P(message arrives | you needed it)  <  P(message arrives)

This script tests that claim with labels and geometry only. No detector, no
checkpoint, no GPU, no propagation model — so the finding cannot be manufactured
by a modelling choice downstream. Per (frame, collaborator j) it measures:

    B_j  blockage : does a labeled vehicle sit on the ego<->j chord, at each of
                    several clearances (Fresnel-radius proxies)?
    U_j  unique value : how many GT boxes are NOT ego-visible but ARE visible to
                    j — the objects only j can reveal. Visibility uses the study's
                    existing definition (>= MIN_PTS of that agent's own returns
                    inside the box, per run_phase43.py), so the numbers are
                    directly comparable to the published spatial decomposition.

and reports the two quantities that decide the direction:

    E[U | blocked]  vs  E[U | clear]      independence predicts these are equal
    availability A  vs  1 - mean(B)       independence predicts these are equal

where A = sum(U * delivered) / sum(U) is the fraction of unique occluded-object
coverage that actually survives the channel.

GO / NO-GO (fixed before the numbers, so the result cannot be rationalised):
    * NO-GO if the blockage base rate is < --min-base-rate (default 0.10): the
      effect will not survive contact with a detector.
    * NO-GO if E[U|blocked] <= E[U|clear]: the common-cause premise is false in
      this data, and the direction dies for the cost of one script.
    * GO otherwise, and the printed matched-PDR block goes into configs/matrix.yaml.

Run on the machine holding the dataset (no GPU needed):

    python scripts/run_blockage_audit.py \
        --config ~/cpfa/checkpoints/<any>/config.yaml \
        --out ~/cpfa/results/blockage --stride 10

Geometry-only self-test (no opencood, no dataset):

    python scripts/run_blockage_audit.py --selftest
"""
import argparse
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from commchannel.blockage import (BlockageTable, DEFAULT_CLEARANCES,  # noqa: E402
                                  DEFAULT_ENDPOINT_MARGIN, locate_frame)

MIN_PTS = 5          # identical to run_phase43.py — one visibility convention
DEFAULT_MIN_BASE_RATE = 0.10


def crc(*parts):
    return zlib.crc32('/'.join(str(p) for p in parts).encode('utf8')) & 0xFFFFFFFF


def points_in_box(points, corners):
    """points (N,3), corners (8,3), OpenCOOD convention: bottom 0-3, top 4-7.
    Verbatim from run_phase43.py so both scripts share one definition."""
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


def point_biserial(binary, values):
    """Correlation between a 0/1 label and a continuous value. Returns None when
    either side is degenerate (all blocked, all clear, or U constant)."""
    b = np.asarray(binary, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if len(b) < 3 or b.std() < 1e-12 or v.std() < 1e-12:
        return None
    return float(np.corrcoef(b, v)[0, 1])


def summarise(rows, clearance, min_blockers=1):
    """Aggregate per-link records into the decision statistics for one clearance."""
    key = 'n_blockers_%g' % clearance
    blocked = np.array([r[key] >= min_blockers for r in rows], dtype=bool)
    u = np.array([r['unique_value'] for r in rows], dtype=np.float64)
    n = len(rows)
    if n == 0:
        return None
    base_rate = float(blocked.mean())
    u_blocked = float(u[blocked].mean()) if blocked.any() else None
    u_clear = float(u[~blocked].mean()) if (~blocked).any() else None
    total_u = float(u.sum())
    # availability: share of unique-object coverage that survives the channel
    availability = float(u[~blocked].sum() / total_u) if total_u > 0 else None
    independent = 1.0 - base_rate
    return {
        'clearance_m': clearance,
        'min_blockers': min_blockers,
        'links': n,
        'blockage_base_rate': round(base_rate, 4),
        'mean_unique_blocked': round(u_blocked, 4) if u_blocked is not None else None,
        'mean_unique_clear': round(u_clear, 4) if u_clear is not None else None,
        'unique_value_ratio': (round(u_blocked / u_clear, 4)
                               if u_blocked is not None and u_clear else None),
        'availability': round(availability, 4) if availability is not None else None,
        'availability_if_independent': round(independent, 4),
        'availability_gap': (round(availability - independent, 4)
                             if availability is not None else None),
        'point_biserial_r': (round(point_biserial(blocked, u), 4)
                             if point_biserial(blocked, u) is not None else None),
        'total_unique_gt': int(total_u),
    }


def matched_iid_levels(rows, clearance, blockage_levels, min_blockers=1):
    """Realized loss rate for each `blockage_p` level, i.e. the loss_p an i.i.d.
    control arm must use to be matched on packet delivery."""
    key = 'n_blockers_%g' % clearance
    base = float(np.mean([r[key] >= min_blockers for r in rows])) if rows else 0.0
    return [round(base * float(p), 4) for p in blockage_levels]


def _selftest_stats():
    """The decision statistics, on synthetic links whose answers are known by hand.
    These four numbers ARE the contribution, so they are checked, not trusted."""
    ok = True

    def check(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print('  FAIL %-46s got %r want %r' % (name, got, want))
        else:
            print('  ok   %-46s %r' % (name, got))

    def mk(n_blockers, u):
        return {'n_blockers_1': n_blockers, 'unique_value': u}

    # Null case: blockage carries no information about value. 2 of 4 links
    # blocked, U identical everywhere -> availability must equal 1 - base rate.
    null = [mk(1, 3), mk(1, 3), mk(0, 3), mk(0, 3)]
    s = summarise(null, 1.0)
    check('null: base rate', s['blockage_base_rate'], 0.5)
    check('null: E[U|blocked] == E[U|clear]',
          (s['mean_unique_blocked'], s['mean_unique_clear']), (3.0, 3.0))
    check('null: availability == 1 - base rate',
          (s['availability'], s['availability_if_independent']), (0.5, 0.5))
    check('null: gap is zero', s['availability_gap'], 0.0)

    # The effect: blocked links are the valuable ones. Half the links are lost but
    # they carry 9/12 of the unique coverage, so availability falls well below the
    # independence prediction of 0.5.
    eff = [mk(1, 5), mk(1, 4), mk(0, 2), mk(0, 1)]
    s = summarise(eff, 1.0)
    check('effect: E[U|blocked] > E[U|clear]',
          (s['mean_unique_blocked'], s['mean_unique_clear']), (4.5, 1.5))
    check('effect: value ratio', s['unique_value_ratio'], 3.0)
    check('effect: availability', s['availability'], 0.25)
    check('effect: availability gap is negative', s['availability_gap'], -0.25)
    check('effect: correlation is positive', s['point_biserial_r'] > 0.9, True)

    # Inverted case: blocked links are the worthless ones — direction refuted.
    inv = [mk(1, 1), mk(1, 2), mk(0, 4), mk(0, 5)]
    s = summarise(inv, 1.0)
    check('inverted: E[U|blocked] < E[U|clear]',
          s['mean_unique_blocked'] < s['mean_unique_clear'], True)
    check('inverted: availability beats independence',
          s['availability_gap'] > 0, True)

    # Degenerate guards
    check('all-blocked: no clear mean', summarise([mk(1, 2)], 1.0)['mean_unique_clear'],
          None)
    check('no GT anywhere: availability undefined',
          summarise([mk(1, 0), mk(0, 0)], 1.0)['availability'], None)
    check('constant U: correlation undefined',
          summarise(null, 1.0)['point_biserial_r'], None)

    # Matched-PDR arithmetic: realized loss = P(drop|blocked) x base rate.
    check('matched i.i.d. levels at base rate 0.5',
          matched_iid_levels(null, 1.0, [0.2, 0.5, 1.0]), [0.1, 0.25, 0.5])
    check('matched levels are zero when nothing is blocked',
          matched_iid_levels([mk(0, 3)], 1.0, [1.0]), [0.0])

    print('\nstats selftest: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='any OpenCOOD checkpoint config.yaml '
                                     '(the audit is model-free; only the dataset '
                                     'section is used)')
    ap.add_argument('--out', help='output directory')
    ap.add_argument('--stride', type=int, default=10,
                    help='evaluate every Nth frame (default 10)')
    ap.add_argument('--clearances', type=float, nargs='*',
                    default=list(DEFAULT_CLEARANCES))
    ap.add_argument('--endpoint-margin', type=float,
                    default=DEFAULT_ENDPOINT_MARGIN)
    ap.add_argument('--min-blockers', type=int, default=1)
    ap.add_argument('--min-base-rate', type=float, default=DEFAULT_MIN_BASE_RATE)
    ap.add_argument('--decision-clearance', type=float, default=1.0,
                    help='clearance the GO/NO-GO verdict is read at (default 1.0 m, '
                         'the ~5.9GHz first-Fresnel-radius proxy)')
    ap.add_argument('--blockage-levels', type=float, nargs='*',
                    default=[0.2, 0.4, 0.6, 0.8, 1.0],
                    help='P(drop|blocked) levels to emit matched i.i.d. rates for')
    ap.add_argument('--selftest', action='store_true',
                    help='run geometry checks only (no dataset, no opencood)')
    args = ap.parse_args()

    if args.selftest:
        from commchannel.blockage import _selftest
        print('--- geometry ---')
        rc_geom = _selftest()
        print('\n--- decision statistics ---')
        rc_stats = _selftest_stats()
        return rc_geom or rc_stats
    if not args.config or not args.out:
        ap.error('--config and --out are required unless --selftest is given')

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import pcd_utils

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(os.path.expanduser(args.config))
    ds = build_dataset(hypes, visualize=False, train=False)
    print('[audit] dataset: %d frames, %d scenarios'
          % (len(ds), len(ds.scenario_database)))

    clearances = tuple(sorted(set(float(c) for c in args.clearances)))
    if args.decision_clearance not in clearances:
        clearances = tuple(sorted(set(clearances) | {args.decision_clearance}))

    t0 = time.time()
    table = BlockageTable.load_or_build(ds, clearances=clearances,
                                        endpoint_margin=args.endpoint_margin)
    print('[audit] blockage table: %d links (%.0fs)'
          % (len(table.counts), time.time() - t0))

    frame_indices = list(range(0, len(ds), args.stride))
    rows = []
    t1 = time.time()
    for n_done, i in enumerate(frame_indices):
        # seed exactly as the other runners do: OpenCOOD's __getitem__ shuffles
        # points, and an unseeded call would make the audit irreproducible.
        np.random.seed(crc('audit', i))
        base = ds.retrieve_base_data(i)
        ego_entry = next((v for v in base.values() if v['ego']), None)
        if ego_entry is None:
            continue
        ego_pts = pcd_utils.mask_ego_points(
            ego_entry['lidar_np'])[:, :3].astype(np.float32)

        np.random.seed(crc('audit', i))
        sample = ds[i]
        batch = ds.collate_batch_test([sample])
        gt = ds.post_processor.generate_gt_bbx(batch).cpu().numpy()
        if len(gt) == 0:
            continue
        ego_visible = np.array(
            [points_in_box(ego_pts, gt[k]).sum() >= MIN_PTS
             for k in range(len(gt))], dtype=bool)

        si, ti = locate_frame(ds, i)
        for cav_id, entry in base.items():
            if entry['ego']:
                continue
            # collaborator points, own-vehicle returns masked, projected to ego
            pts = pcd_utils.mask_ego_points(
                entry['lidar_np'])[:, :3].astype(np.float64)
            tm = np.asarray(entry['params']['transformation_matrix'],
                            dtype=np.float64)
            proj = pts @ tm[:3, :3].T + tm[:3, 3]
            cav_visible = np.array(
                [points_in_box(proj, gt[k]).sum() >= MIN_PTS
                 for k in range(len(gt))], dtype=bool)

            row = {
                'frame': i, 'scenario': si, 't_index': ti, 'cav_id': str(cav_id),
                'gt_total': int(len(gt)),
                'gt_ego_visible': int(ego_visible.sum()),
                'gt_occluded': int((~ego_visible).sum()),
                # the objects ONLY this collaborator can reveal
                'unique_value': int((~ego_visible & cav_visible).sum()),
            }
            for c in clearances:
                row['n_blockers_%g' % c] = table.n_blockers(si, ti, cav_id, c)
            rows.append(row)

        if (n_done + 1) % 50 == 0:
            print('[audit]   %d/%d frames, %d links (%.0fs)'
                  % (n_done + 1, len(frame_indices), len(rows), time.time() - t1))

    if not rows:
        print('[audit] no links collected — is this a single-agent split?')
        return 1

    summaries = [summarise(rows, c, args.min_blockers) for c in clearances]
    decision = next(s for s in summaries
                    if s['clearance_m'] == args.decision_clearance)

    # per-scenario breakdown, so a single intersection cannot carry the result
    per_scenario = defaultdict(list)
    for r in rows:
        per_scenario[r['scenario']].append(r)
    scen_rows = {int(s): summarise(rs, args.decision_clearance, args.min_blockers)
                 for s, rs in sorted(per_scenario.items())}

    matched = matched_iid_levels(rows, args.decision_clearance,
                                 args.blockage_levels, args.min_blockers)

    base_ok = decision['blockage_base_rate'] >= args.min_base_rate
    corr_ok = (decision['mean_unique_blocked'] is not None
               and decision['mean_unique_clear'] is not None
               and decision['mean_unique_blocked'] > decision['mean_unique_clear'])
    verdict = 'GO' if (base_ok and corr_ok) else 'NO-GO'

    record = {
        'frames': len(frame_indices), 'stride': args.stride,
        'links': len(rows), 'min_pts': MIN_PTS,
        'endpoint_margin': args.endpoint_margin,
        'min_blockers': args.min_blockers,
        'decision_clearance_m': args.decision_clearance,
        'min_base_rate': args.min_base_rate,
        'by_clearance': summaries,
        'by_scenario': scen_rows,
        'matched_iid': {'blockage_levels': list(args.blockage_levels),
                        'loss_p_levels': matched},
        'verdict': verdict,
        'verdict_reason': {
            'base_rate_ok': bool(base_ok),
            'unique_value_higher_when_blocked': bool(corr_ok),
        },
        'runtime_s': round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, 'blockage_audit.json'), 'w') as f:
        json.dump({'summary': record, 'links': rows}, f, indent=2)

    lines = [
        '# Blockage audit — is lost mail the mail you needed?',
        '',
        'Model-free: labels and geometry only, no detector and no propagation model.',
        '',
        '- frames %d (stride %d) | links %d | visibility >= %d pts'
        % (len(frame_indices), args.stride, len(rows), MIN_PTS),
        '- endpoint margin %.1f m | min blockers %d'
        % (args.endpoint_margin, args.min_blockers),
        '',
        '| clearance (m) | blocked | E[U\\|blocked] | E[U\\|clear] | ratio | '
        'availability | if independent | gap | r |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for s in summaries:
        lines.append('| %.1f | %.3f | %s | %s | %s | %s | %.3f | %s | %s |' % (
            s['clearance_m'], s['blockage_base_rate'],
            s['mean_unique_blocked'], s['mean_unique_clear'],
            s['unique_value_ratio'], s['availability'],
            s['availability_if_independent'], s['availability_gap'],
            s['point_biserial_r']))
    lines += [
        '',
        '`U` = GT boxes the ego cannot see but this collaborator can.',
        '`availability` = share of that coverage surviving the channel; under the',
        'independence assumption it would equal `1 - blocked`. A negative gap is the',
        'effect: the messages lost are worth more than the messages kept.',
        '',
        '## Verdict at %.1f m clearance: **%s**' % (args.decision_clearance, verdict),
        '',
        '- base rate %.3f (>= %.2f required): %s'
        % (decision['blockage_base_rate'], args.min_base_rate,
           'ok' if base_ok else 'FAILS — effect will not survive a detector'),
        '- E[U|blocked] %s vs E[U|clear] %s: %s'
        % (decision['mean_unique_blocked'], decision['mean_unique_clear'],
           'ok' if corr_ok else 'FAILS — common-cause premise not supported here'),
        '',
        '## Matched-PDR levels for configs/matrix.yaml',
        '',
        'The `loss_blocked` family sweeps P(drop|blocked); realized loss is that',
        'times the geometric base rate. `loss_iid_matched` must use these levels so',
        'the two arms are compared at equal packet delivery:',
        '',
        '```yaml',
        '  loss_iid_matched:',
        '    param: loss_p',
        '    levels: [%s]' % ', '.join('%.4f' % v for v in matched),
        '```',
        '',
        '## Per-scenario (decision clearance)',
        '',
        '| scenario | links | blocked | E[U\\|blocked] | E[U\\|clear] | gap |',
        '|---|---|---|---|---|---|',
    ]
    for s, srow in scen_rows.items():
        lines.append('| %d | %d | %.3f | %s | %s | %s |' % (
            s, srow['links'], srow['blockage_base_rate'],
            srow['mean_unique_blocked'], srow['mean_unique_clear'],
            srow['availability_gap']))
    with open(os.path.join(out_dir, 'blockage_audit.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines[5:]))
    print('\n[audit] wrote %s' % os.path.join(out_dir, 'blockage_audit.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
