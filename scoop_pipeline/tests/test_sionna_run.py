#!/usr/bin/env python
"""Self-test for analysis/comms/sionna_run.py on Sionna's built-in box: a
transmitter and two receivers inside it, the line-of-sight delay of each
equal to its distance over c, reflections behind it, the radio map covering
the floor with the cells nearest the transmitter the strongest; the files
written.

    python scoop_pipeline/tests/test_sionna_run.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "comms"))


def main():
    try:
        import sionna.rt as rt
    except ImportError:
        print("skip: no sionna")
        return
    import sionna_run as SR
    tx, rx1, rx2 = (0.0, 0.0, 1.5), (3.0, 0.0, 1.5), (0.0, -4.0, 1.5)
    sc = SR.build(rt.scene.box, 5.745e9, 20.0, tx, [rx1, rx2])
    lines = []
    recs = SR.paths_report(sc, 2, 200000, log=lines.append)
    print("\n".join(lines))
    for rec, p in zip(recs, (rx1, rx2)):
        d = np.linalg.norm(np.subtract(p, tx))
        assert rec["n_paths"] >= 2, rec
        assert abs(rec["los_delay_ns"] - d / SR.C0 * 1e9) < 0.3, (rec["los_delay_ns"], d / SR.C0 * 1e9)
        assert rec["rms_delay_ns"] > 0 and rec["gain_db"] < -30, rec
    g, cen = SR.radio_map(sc, 1.0, 1.5, 2, 200000, log=lines.append)
    assert g.ndim == 2 and np.isfinite(g).mean() > 0.8, (g.shape, np.isfinite(g).mean())
    d = np.hypot(cen[..., 0] - tx[0], cen[..., 1] - tx[1])
    near, far = g[d < 2], g[d > 4]
    best = np.unravel_index(np.nanargmax(g), g.shape)
    assert d[best] < 2.5 and np.nanmedian(near) > np.nanmedian(far), \
        (d[best], np.nanmedian(near), np.nanmedian(far))     # a closed box: reflections flatten the map
    tmp = tempfile.mkdtemp(prefix="sionna_run_")
    try:
        # map coordinates in, through 05's frame: the map is the scene turned 90 deg and moved
        import json
        T = np.array([[0, -1, 0, 2.0], [1, 0, 0, -1.0], [0, 0, 1, 0.3], [0, 0, 0, 1]], float)
        fj = os.path.join(tmp, "anchor_frame.json")
        json.dump({"T_N_world": T.tolist()}, open(fj, "w"))
        Ts, Tm = SR.map_to_scene(fj)
        p_map = SR.moved(T, [3.0, 0.0, 1.5])
        assert np.allclose(SR.moved(Ts, p_map), [3.0, 0.0, 1.5]) and np.allclose(Tm, T), "frame"
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "sionna_run.py"), "box",
                            "--tx", "0", "0", "1.5", "--rx", "3", "0", "1.5", "--radio-map", "--cell", "1",
                            "--samples", "200000", "--max-depth", "2", "--out", tmp, "--frame", fj],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2000:]
        for f in ("cir_rx0.npz", "radio_map.npy", "radio_map.csv", "radio_map.png"):
            assert os.path.exists(os.path.join(tmp, f)), (f, os.listdir(tmp))
        assert "moved from map coordinates" in r.stdout and "x_map" in open(os.path.join(tmp, "radio_map.csv")).readline()
        print("ok sionna_run: %d and %d paths, line of sight on time, the map %d cells with the near ones "
              "%.0f dB above the far; files written"
              % (recs[0]["n_paths"], recs[1]["n_paths"], g.size, np.nanmedian(near) - np.nanmedian(far)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
