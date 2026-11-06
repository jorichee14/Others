import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sync_scene import make_scene                      # noqa: E402

from ed305.sync import motion_signal, pair_offset, solve_offsets   # noqa: E402


def test_offsets_from_motion(tmp_path):
    fps = 30.0
    truth = make_scene(tmp_path, [0.0, 1.234, 2.5], seconds=30.0, fps=fps,
                       views=[(0, 0), (120, 60), (200, 120)])
    sig = {n: motion_signal(tmp_path / f"{n}.avi")[1:] for n in truth}
    names = list(truth)
    pairs = {(a, b): pair_offset(*sig[a], *sig[b], max_lag=5) for i, a in enumerate(names) for b in names[i + 1:]}
    off, res, used = solve_offsets(names, pairs)
    for n, o in truth.items():
        actual = round(o * fps) / fps                   # the scene starts cameras on whole frames
        assert off[n] is not None and abs(off[n] - actual) < 0.01, (n, off[n], actual)


def test_solver_drops_a_wrong_pair_and_flags_unlinked():
    names = ["a", "b", "c", "d"]
    true = {"a": 0.0, "b": 0.5, "c": -1.0}
    pairs = {(x, y): (true[x] - true[y], 20.0) for x in true for y in true if x < y}
    pairs[("a", "c")] = (3.0, 8.0)                      # one pair locked onto the wrong peak
    pairs[("a", "d")] = (0.2, 2.0)                      # too weak to use
    pairs[("b", "c")] = (true["b"] - true["c"], 20.0)
    off, res, used = solve_offsets(names, pairs)
    assert abs(off["b"] - 0.5) < 1e-6 and abs(off["c"] + 1.0) < 1e-6
    assert ("a", "c") not in used and off["d"] is None
