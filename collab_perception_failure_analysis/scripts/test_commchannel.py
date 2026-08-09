#!/usr/bin/env python
"""Unit tests for the commchannel package that need no GPU, dataset, or OpenCOOD —
pure numpy/torch logic: schedule statistics, determinism, quantizer behavior, ghost
geometry, decision composition. Run:

    python scripts/test_commchannel.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commchannel.config import ChannelConfig          # noqa: E402
from commchannel.schedule import Schedule             # noqa: E402
from commchannel.channel import CommChannel, GHOST_LWH, GHOST_RANGE_XY  # noqa: E402
from commchannel.blockage import BlockageTable         # noqa: E402

FAILED = []


def check(name, fn):
    try:
        fn()
        print('[PASS] %s' % name)
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print('[FAIL] %s — %s: %s' % (name, type(e).__name__, e))


def test_identity_is_noop():
    s = Schedule(ChannelConfig())
    for t in range(50):
        d = s.decide(0, t, '650', is_ego=False)
        assert not d.drop and d.delay_frames == 0 and d.pose_noise is None
        assert d.ghost_seed is None and d.swap_seed is None


def test_ego_never_impaired():
    cfg = ChannelConfig(loss_p=1.0, latency_frames=5, pose_xyz_std=10.0,
                        ghost_p=1.0, swap_p=1.0, stale_period=10)
    s = Schedule(cfg)
    for t in range(20):
        d = s.decide(0, t, '641', is_ego=True)
        assert not d.drop and d.delay_frames == 0 and d.pose_noise is None
        assert d.ghost_seed is None and d.swap_seed is None


def test_bernoulli_rate():
    for p in (0.1, 0.3, 0.7):
        s = Schedule(ChannelConfig(loss_p=p, seed=7))
        n, drops = 20000, 0
        for t in range(n):
            drops += s.decide(0, t, 'cav2', False).drop
        rate = drops / n
        assert abs(rate - p) < 0.02, 'p=%s measured=%s' % (p, rate)


def test_determinism_and_independence():
    s1 = Schedule(ChannelConfig(loss_p=0.5, pose_xyz_std=1.0, seed=3))
    s2 = Schedule(ChannelConfig(loss_p=0.5, pose_xyz_std=1.0, seed=3))
    s3 = Schedule(ChannelConfig(loss_p=0.5, pose_xyz_std=1.0, seed=4))
    same = diff = 0
    for t in range(500):
        a, b, c = (s.decide(2, t, '1023', False) for s in (s1, s2, s3))
        assert a.drop == b.drop
        if a.pose_noise is not None:
            assert np.allclose(a.pose_noise, b.pose_noise)
        same += a.drop == c.drop
        diff += a.drop != c.drop
    assert diff > 50, 'different seeds should give different realizations'


def test_gilbert_elliott_burstiness():
    # bursty channel: rare entry to Bad, sticky once there
    cfg = ChannelConfig(ge_p_good_to_bad=0.05, ge_p_bad_to_good=0.3, seed=11)
    s = Schedule(cfg)
    seq = [s.decide(0, t, 'cavx', False).drop for t in range(5000)]
    loss_rate = sum(seq) / len(seq)
    # stationary Bad occupancy = 0.05/(0.05+0.3) ~ 0.143
    assert 0.10 < loss_rate < 0.19, loss_rate
    # burstiness: P(loss at t | loss at t-1) must far exceed marginal rate
    both = sum(1 for i in range(1, len(seq)) if seq[i] and seq[i - 1])
    cond = both / max(1, sum(seq[:-1]))
    assert cond > loss_rate * 2.5, 'cond=%s marginal=%s' % (cond, loss_rate)
    # i.i.d. control shows no such correlation
    s_iid = Schedule(ChannelConfig(loss_p=loss_rate, seed=11))
    seq2 = [s_iid.decide(0, t, 'cavx', False).drop for t in range(5000)]
    both2 = sum(1 for i in range(1, len(seq2)) if seq2[i] and seq2[i - 1])
    cond2 = both2 / max(1, sum(seq2[:-1]))
    assert cond2 < loss_rate * 1.5


def test_latency_and_stale():
    s = Schedule(ChannelConfig(latency_frames=3))
    assert all(s.decide(0, t, 'c', False).delay_frames == 3 for t in range(20))
    s = Schedule(ChannelConfig(stale_period=5))
    ages = [s.decide(0, t, 'c', False).delay_frames for t in range(15)]
    assert ages == [0, 1, 2, 3, 4] * 3, ages  # sawtooth: refresh every 5 frames
    s = Schedule(ChannelConfig(latency_frames=2, stale_period=4))
    ages = [s.decide(0, t, 'c', False).delay_frames for t in range(8)]
    assert ages == [2, 3, 4, 5, 2, 3, 4, 5], ages  # compose additively


def test_pose_noise_stats():
    s = Schedule(ChannelConfig(pose_xyz_std=0.5, pose_yaw_std_deg=2.0, seed=5))
    xs, yaws = [], []
    for t in range(4000):
        d = s.decide(0, t, 'c', False)
        xs.append(d.pose_noise[0])
        yaws.append(d.pose_noise[3])
    assert abs(np.std(xs) - 0.5) < 0.05
    assert abs(np.std(yaws) - 2.0) < 0.2
    assert abs(np.mean(xs)) < 0.05
    # varies across frames (regression test for OpenCOOD's constant-offset bug)
    assert len({round(x, 6) for x in xs[:100]}) > 90


def test_ghost_geometry():
    ch = CommChannel(ChannelConfig())
    pts = ch._ghost_points(seed=42, count=3)
    assert pts.shape == (3 * 220, 4) and pts.dtype == np.float32
    l, w, h = GHOST_LWH
    half_diag = np.hypot(l / 2, w / 2)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    assert radii.min() > GHOST_RANGE_XY[0] - half_diag - 0.5
    assert radii.max() < GHOST_RANGE_XY[1] + half_diag + 0.5
    assert pts[:, 2].min() > -2.0 and pts[:, 2].max() < h - 1.9 + 0.1
    assert (pts[:, 3] >= 0).all() and (pts[:, 3] <= 1).all()
    assert np.allclose(pts, ch._ghost_points(seed=42, count=3))  # deterministic


def test_quantizer():
    import torch
    from commchannel.feature_hooks import quantize, _quantize_rows
    x = torch.randn(2, 8, 4, 4)
    assert torch.equal(quantize(x, 32), x)                    # 32 bits = no-op
    for bits in (1, 2, 4, 8):
        q = quantize(x, bits)
        assert len(torch.unique(q)) <= 2 ** bits
        assert q.min() >= x.min() - 1e-6 and q.max() <= x.max() + 1e-6
    err8 = (quantize(x, 8) - x).abs().mean()
    err2 = (quantize(x, 2) - x).abs().mean()
    assert err8 < err2                                        # monotone degradation
    y = _quantize_rows(x.clone(), 2, None)
    assert torch.equal(y[0], x[0])                            # ego row untouched
    assert not torch.equal(y[1], x[1])


def test_identity_config_flag():
    assert ChannelConfig().is_identity
    assert not ChannelConfig(loss_p=0.1).is_identity
    assert not ChannelConfig(stale_period=5).is_identity
    assert not ChannelConfig(bandwidth_bits=8).is_identity
    assert not ChannelConfig(blockage_p=0.5).is_identity


# ---------------------------------------------------------------- blockage
def _table(rows, clearances=(0.0, 1.0, 2.0)):
    return BlockageTable(rows, clearances)


def test_blockage_drops_only_blocked_links():
    """Blockage is scene-conditioned: WHICH links drop is geometry, not a draw."""
    tbl = _table({(0, t, 'cav2'): [0, 1, 1] for t in range(0, 100, 2)})
    cfg = ChannelConfig(blockage_p=1.0, blockage_clearance=1.0, seed=5)
    s = Schedule(cfg, tbl)
    for t in range(100):
        blocked_now = (t % 2 == 0)
        assert s.decide(0, t, 'cav2', False).drop is blocked_now, t
    # a link absent from the table is never blocked
    for t in range(100):
        assert not s.decide(0, t, 'cav9', False).drop


def test_blockage_clearance_selects_grid_column():
    tbl = _table({(0, 0, 'cav2'): [0, 1, 3]})
    for clearance, want in ((0.0, False), (1.0, True), (2.0, True)):
        s = Schedule(ChannelConfig(blockage_p=1.0, blockage_clearance=clearance),
                     tbl)
        assert s.decide(0, 0, 'cav2', False).drop is want, clearance
    # min_blockers raises the bar: 1 blocker at c=1 is no longer enough
    s = Schedule(ChannelConfig(blockage_p=1.0, blockage_clearance=1.0,
                               blockage_min_blockers=2), tbl)
    assert not s.decide(0, 0, 'cav2', False).drop
    s = Schedule(ChannelConfig(blockage_p=1.0, blockage_clearance=2.0,
                               blockage_min_blockers=2), tbl)
    assert s.decide(0, 0, 'cav2', False).drop
    # a clearance outside the built grid must fail loudly, not silently read 0
    bad = Schedule(ChannelConfig(blockage_p=1.0, blockage_clearance=1.7), tbl)
    try:
        bad.decide(0, 0, 'cav2', False)
        raise AssertionError('expected KeyError for off-grid clearance')
    except KeyError:
        pass


def test_blockage_realized_rate_is_p_times_base_rate():
    """The family's headline arithmetic: realized loss = P(drop|blocked) x base
    rate. This is why the matched i.i.d. control has to be measured, not guessed."""
    n = 20000
    tbl = _table({(0, t, 'cav2'): [0, 1 if t % 4 == 0 else 0, 0]
                  for t in range(n)})           # base rate exactly 0.25
    for p in (0.2, 0.5, 1.0):
        s = Schedule(ChannelConfig(blockage_p=p, blockage_clearance=1.0, seed=11),
                     tbl)
        drops = sum(s.decide(0, t, 'cav2', False).drop for t in range(n))
        assert abs(drops / n - 0.25 * p) < 0.01, (p, drops / n)


def test_blockage_is_deterministic_and_ego_exempt():
    tbl = _table({(0, t, 'cav2'): [0, 1, 1] for t in range(200)})
    a = Schedule(ChannelConfig(blockage_p=0.5, blockage_clearance=1.0, seed=2), tbl)
    b = Schedule(ChannelConfig(blockage_p=0.5, blockage_clearance=1.0, seed=2), tbl)
    assert [a.decide(0, t, 'cav2', False).drop for t in range(200)] == \
           [b.decide(0, t, 'cav2', False).drop for t in range(200)]
    assert not any(a.decide(0, t, 'cav2', True).drop for t in range(200))


def test_blockage_inactive_without_table_or_p():
    """Config without a table must not silently drop, and a table without
    blockage_p must not activate."""
    tbl = _table({(0, 0, 'cav2'): [1, 1, 1]})
    assert not Schedule(ChannelConfig(blockage_p=1.0), None).decide(
        0, 0, 'cav2', False).drop
    assert not Schedule(ChannelConfig(), tbl).decide(0, 0, 'cav2', False).drop


def test_channel_stats_accounting():
    """End-to-end through CommChannel on a mocked dataset: the realized drop rate
    the sweep records must equal what the geometry dictates."""
    tbl = _table({(0, t, 'cav2'): [0, 1, 1] for t in range(0, 10, 2)})
    cfg = ChannelConfig(blockage_p=1.0, blockage_clearance=1.0, seed=1)
    ch = CommChannel(cfg, blockage=tbl)
    for t in range(10):
        dec_ego = ch.schedule.decide(0, t, 'ego', True)
        dec_cav = ch.schedule.decide(0, t, 'cav2', False)
        assert not dec_ego.drop
        ch.stats['messages'] += 1
        if tbl.is_blocked(0, t, 'cav2', 1.0):
            ch.stats['blocked'] += 1
        if dec_cav.drop:
            ch.stats['dropped_channel'] += 1
    d = ch.stats_dict()
    assert d['messages'] == 10 and d['blocked'] == 5 and d['dropped_channel'] == 5
    assert d['realized_drop_rate'] == 0.5 and d['blocked_rate'] == 0.5
    ch.reset_stats()
    assert ch.stats_dict()['realized_drop_rate'] is None


def test_blockage_composes_with_other_families():
    """Drop wins over content impairments (an unarrived message has no content),
    and i.i.d. loss short-circuits before the blockage branch."""
    tbl = _table({(0, 0, 'cav2'): [1, 1, 1]})
    cfg = ChannelConfig(blockage_p=1.0, blockage_clearance=1.0,
                        latency_frames=4, pose_xyz_std=1.0, ghost_p=1.0)
    d = Schedule(cfg, tbl).decide(0, 0, 'cav2', False)
    assert d.drop and d.delay_frames == 0 and d.pose_noise is None
    # unblocked link still receives the content impairments
    d2 = Schedule(cfg, tbl).decide(0, 1, 'cav2', False)
    assert not d2.drop and d2.delay_frames == 4 and d2.pose_noise is not None


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
