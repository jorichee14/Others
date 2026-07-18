#!/usr/bin/env python3
"""Unit-test the pure co-visibility / time-suggestion logic in find_chair.py.

The detector and bag I/O are environment-specific; this covers the part that
decides *which* timestamps to use.

Run: python3 test_find_chair.py
"""

from __future__ import annotations

import numpy as np

from find_chair import covisibility_windows, suggest_times


def test_single_overlap_window():
    # zed sees the chair 10..30, realsense 12..25, arducam 5..40
    dets = {
        "zed": list(np.arange(10, 30, 0.3)),
        "realsense": list(np.arange(12, 25, 0.3)),
        "arducam": list(np.arange(5, 40, 0.3)),
    }
    wins = covisibility_windows(dets, bin_s=0.5, min_sensors=2)
    # >=2 sensors requires zed OR realsense to overlap arducam -> ~10..30
    assert len(wins) == 1, wins
    a, b, sset = wins[0]
    assert 9.5 <= a <= 10.5 and 29.5 <= b <= 30.5, (a, b)
    assert sset == frozenset({"zed", "realsense", "arducam"})
    print(f"A. single window recovered: [{a:.1f}, {b:.1f}]  sensors={sorted(sset)}   OK")


def test_three_sensor_core():
    dets = {
        "zed": list(np.arange(10, 30, 0.3)),
        "realsense": list(np.arange(12, 25, 0.3)),
        "arducam": list(np.arange(5, 40, 0.3)),
    }
    wins = covisibility_windows(dets, bin_s=0.5, min_sensors=3)
    # all three only overlap 12..25
    assert len(wins) == 1
    a, b, _ = wins[0]
    assert 11.5 <= a <= 12.5 and 24.5 <= b <= 25.5, (a, b)
    print(f"B. 3-sensor core recovered: [{a:.1f}, {b:.1f}]   OK")


def test_two_disjoint_windows():
    # chair seen twice (platform passes it twice), gap in the middle
    dets = {
        "zed": list(np.arange(10, 18, 0.3)) + list(np.arange(40, 48, 0.3)),
        "arducam": list(np.arange(5, 55, 0.3)),
    }
    wins = covisibility_windows(dets, bin_s=0.5, min_sensors=2)
    assert len(wins) == 2, wins
    assert wins[0][1] < wins[1][0], wins  # ordered, separated
    print(f"C. two disjoint passes recovered: "
          f"{[(round(a,1),round(b,1)) for a,b,_ in wins]}   OK")


def test_suggest_spread_and_covered():
    dets = {
        "zed": list(np.arange(10, 30, 0.3)),
        "realsense": list(np.arange(12, 25, 0.3)),
        "arducam": list(np.arange(5, 40, 0.3)),
    }
    times = suggest_times(dets, bin_s=0.5, min_sensors=2, n=5)
    assert len(times) == 5, times
    assert times == sorted(times)
    assert min(times) >= 9.5 and max(times) <= 30.5, times      # inside coverage
    assert min(np.diff(times)) > 0.5, times                     # spread out
    print(f"D. suggested times spread & in-window: {times}   OK")


def test_empty():
    assert covisibility_windows({}, 0.5, 2) == []
    assert suggest_times({}, 0.5, 2, 5) == []
    print("E. empty input -> empty output   OK")


def test_required_all_three_cameras():
    from find_chair import _present
    # zed 10..30, realsense 12..25, arducam 5..40.
    # All three overlap only on 12..25 -> requiring all three must confine here,
    # and EVERY suggested instant must have all three present.
    dets = {
        "zed": list(np.arange(10, 30, 0.3)),
        "realsense": list(np.arange(12, 25, 0.3)),
        "arducam": list(np.arange(5, 40, 0.3)),
    }
    req = ("zed", "realsense", "arducam")
    wins = covisibility_windows(dets, bin_s=0.5, min_sensors=3, required=req)
    assert len(wins) == 1
    a, b, sset = wins[0]
    assert 11.5 <= a <= 12.5 and 24.5 <= b <= 25.5, (a, b)
    assert sset == frozenset(req)                      # all three hold everywhere
    times = suggest_times(dets, bin_s=0.5, min_sensors=3, n=6, required=req)
    assert times, "should find times with all three"
    for t in times:                                    # the hard guarantee
        assert set(req).issubset(_present(dets, t, 0.25)), (t, _present(dets, t, 0.25))
    assert all(11.5 <= t <= 25.5 for t in times), times
    print(f"F. require all three cameras -> {times}, each has all three   OK")


def test_required_excludes_two_only_instants():
    from find_chair import _present
    # realsense drops out entirely after t=20; instants at t>20 have only 2 cams.
    dets = {
        "zed": list(np.arange(10, 40, 0.3)),
        "realsense": list(np.arange(10, 20, 0.3)),
        "arducam": list(np.arange(10, 40, 0.3)),
    }
    req = ("zed", "realsense", "arducam")
    times = suggest_times(dets, bin_s=0.5, min_sensors=3, n=8, required=req)
    assert times and max(times) <= 20.5, times         # nothing past realsense's end
    for t in times:
        assert set(req).issubset(_present(dets, t, 0.25)), t
    print(f"G. two-only instants excluded when three required -> max t={max(times):.1f}   OK")


if __name__ == "__main__":
    test_single_overlap_window()
    test_three_sensor_core()
    test_two_disjoint_windows()
    test_suggest_spread_and_covered()
    test_empty()
    test_required_all_three_cameras()
    test_required_excludes_two_only_instants()
    print("\nall find_chair logic tests passed")
