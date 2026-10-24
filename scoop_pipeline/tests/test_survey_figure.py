#!/usr/bin/env python
"""Self-test for analysis/survey_figure.py on a synthetic survey: a room's
plan, a 10 minute loop of the robot, three boards and two parked anchors,
Wi-Fi and ping over time, iperf tests, two CSI links with a path loss, a
cleaned CSI array and 04's quality rows. The composite (PNG, PDF) and the
animation (GIF, a few frames) are written; the panels find the near-anchor
bands and the path-loss fit; with the comms missing the figure is still
made, its panels noting what is not there.

    python scoop_pipeline/tests/test_survey_figure.py
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "comms"))
sys.path.insert(0, os.path.join(ROOT, "map_stages"))

import survey_figure as SF                                      # noqa: E402


def synthetic(rng, with_comms=True):
    t = 1.7e9 + np.arange(0, 600, 0.1)
    s = np.linspace(0, 2 * np.pi, len(t))
    x, y = -8 + 12 * np.cos(s), 5 + 8 * np.sin(s)
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, 0, 3], T[:, 1, 3], T[:, 2, 3] = x, y, 0.2
    plan = np.zeros((300, 400))
    plan[5, :] = plan[-5, :] = plan[:, 5] = plan[:, -5] = 1
    site = {"tag": "survey_x", "machine": "mobile_1", "ap": (-8.0, 5.0, 2.5),
            "plan": plan, "extent": (-24, 8, -6, 18), "t": t, "T": T,
            "boards": [{"name": "board_1", "xyz": [4.0, 5.0, 0.0], "yaw": math.pi, "n_views": 50,
                        "std_mm": 2.0, "sections": [{"gated_out": []}, {"gated_out": ["n<4"]}],
                        "lc": {"delta_mm": 1.4, "delta_deg": 0.3}},
                       {"name": "board_2", "xyz": [-8.0, 13.0, 0.0], "yaw": -math.pi / 2, "n_views": 8,
                        "std_mm": 2.2, "sections": [{"gated_out": []}], "lc": None},
                       {"name": "board_4", "xyz": [-20.0, 5.0, 0.1], "yaw": 0.0, "n_views": 1,
                        "std_mm": 0.0, "sections": [{"gated_out": []}], "lc": None}],
            "statics": [{"name": "mobile_2", "xyz": [-20.0, 4.5, 0.3], "source": "lidar_avg"},
                        {"name": "infra_1", "xyz": [-4.0, 7.8, 1.7], "source": "extrinsic_yaml"}]}
    q_t = t[::10]
    site["quality"] = {"t": q_t, "residual_cm": 1.2 + 0.2 * rng.random(len(q_t)),
                       "correction_cm": np.r_[8 * rng.random(50), 0.05 * rng.random(len(q_t) - 50)],
                       "ok": np.ones(len(q_t), bool)}
    if not with_comms:
        site.update(wifi=None, ping=None, iperf=None, links=None, csi=None)
        return site
    d_ap = np.hypot(x + 8, y - 5)
    wt = t[::5]
    site["wifi"] = {"t": wt, "rssi": -40 - 20 * np.log10(np.maximum(d_ap[::5], 1)) + rng.normal(0, 1, len(wt)),
                    "rate": 400 + 50 * rng.random(len(wt))}
    site["ping"] = {"t": t[::2], "rtt": np.where(rng.random(len(t[::2])) < 0.9, 10 + 5 * rng.random(len(t[::2])), np.nan)}
    it = t[::600]
    site["iperf"] = {"t": it, "mbps": 200 + 100 * rng.random(len(it)), "up": np.arange(len(it)) % 2 == 0}
    d1 = np.hypot(x + 4, y - 7.8)
    d2 = np.hypot(x + 20, y - 4.5)
    site["links"] = {"infra_1": (d1, -40 - 20 * np.log10(np.maximum(d1, 0.5)) + rng.normal(0, 2, len(t))),
                     "mobile_2": (d2, -45 - 25 * np.log10(np.maximum(d2, 0.5)) + rng.normal(0, 2, len(t)))}
    sub = np.arange(-122, 123)
    sub = sub[np.abs(sub) > 1]
    ct = t[::4]
    H = (rng.normal(size=(len(ct), len(sub))) + 1j * rng.normal(size=(len(ct), len(sub)))).astype(np.complex64)
    H *= (1 + 0.5 * np.cos(2 * np.pi * sub / 60))[None]
    site["csi"] = {"name": "infra_1", "t": ct, "sub": sub, "amp_db": 20 * np.log10(np.abs(H) + 1e-6), "H": H}
    return site


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="survey_fig_")
    try:
        site = synthetic(rng)
        anchors = [(b["name"], b["xyz"]) for b in site["boards"]] + [(s["name"], s["xyz"]) for s in site["statics"]]
        bands = SF.near_intervals(site["t"], site["T"], anchors, 3.0)
        names = {b[0] for b in bands}
        assert {"board_1", "board_2", "mobile_2", "board_4"} <= names, bands   # on the loop; infra_1 is inside it
        assert "infra_1" not in names, bands
        files = SF.compose(site, os.path.join(tmp, "full"))
        assert all(os.path.exists(f) and os.path.getsize(f) > 50000 for f in files), files
        gif = SF.animate(site, os.path.join(tmp, "full"), seconds=0.6, fps=5, mp4=False)
        assert gif and os.path.getsize(gif[0]) > 20000, gif
        # a coop run: a second robot, on its own track and starting later
        coop = synthetic(rng)
        coop["tag"] = "coop_x"
        ot = coop["t"][300:]
        oT = np.tile(np.eye(4), (len(ot), 1, 1))
        oT[:, 0, 3] = -8 + 10 * np.cos(np.linspace(np.pi, 3 * np.pi, len(ot)))
        oT[:, 1, 3] = 5 + 6 * np.sin(np.linspace(np.pi, 3 * np.pi, len(ot)))
        coop["others"] = [{"name": "mobile_2", "t": ot, "T": oT}]
        cfiles = SF.compose(coop, os.path.join(tmp, "coop"))
        assert all(os.path.exists(f) and os.path.getsize(f) > 50000 for f in cfiles), cfiles
        assert "mobile_2: " in SF.caption(coop), SF.caption(coop)
        cgif = SF.animate(coop, os.path.join(tmp, "coop"), seconds=0.6, fps=5, mp4=False)
        assert cgif and os.path.getsize(cgif[0]) > 20000, cgif

        bare = synthetic(rng, with_comms=False)
        files2 = SF.compose(bare, os.path.join(tmp, "bare"))
        assert all(os.path.exists(f) for f in files2), files2
        print("ok survey_figure: composite %d kB, %d near-anchor bands (%s), animation %d kB; "
              "a coop run's second robot %d kB + %d kB gif; without comms tables %d kB"
              % (os.path.getsize(files[0]) // 1000, len(bands), ", ".join(sorted(names)),
                 os.path.getsize(gif[0]) // 1000, os.path.getsize(cfiles[0]) // 1000,
                 os.path.getsize(cgif[0]) // 1000, os.path.getsize(files2[0]) // 1000))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
