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


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(tests) - len(FAILED), len(tests)))
    sys.exit(1 if FAILED else 0)
