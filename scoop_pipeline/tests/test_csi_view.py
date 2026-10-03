#!/usr/bin/env python
"""Self-test for analysis/comms/csi_view.py: a three-path link (0, 100 and
250 ns) in the cleaned format. The power delay profile must put the paths
at those delays, and the figures (whole pass, zoom) must be written.

    python scoop_pipeline/tests/test_csi_view.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "comms"))
from scoop import comms                                             # noqa: E402
import csi_view                                                     # noqa: E402


def main():
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp()
    f = []
    try:
        sub = np.arange(-122, 123)
        sub = sub[(np.abs(sub) >= 2) & ~np.isin(np.abs(sub), (11, 39, 75, 103))]
        n, T0 = 3000, 1_790_229_700.0
        t = T0 + np.arange(n) * 0.02
        x = np.where(t - T0 < 20, 2.0, 2.0 + 0.4 * (t - T0 - 20))
        paths = (1 + 0.5 * np.exp(-2j * np.pi * sub * 100e-9 * 312.5e3)
                 + 0.25 * np.exp(-2j * np.pi * sub * 250e-9 * 312.5e3))
        H = (1000 * paths[None, :] * (1 + 0.05 * (rng.normal(size=(n, len(sub)))
                                                 + 1j * rng.normal(size=(n, len(sub)))))).astype(np.complex64)
        amp, ph, pw, _ = comms.csi_features(sub, H)
        rssi = -30 - 25 * np.log10(x)
        pose = np.tile(np.eye(4), (n, 1, 1))
        pose[:, 0, 3] = x
        d = os.path.join(tmp, "proc", "comms", "csi")
        os.makedirs(d)
        np.savez_compressed(os.path.join(d, "csi_infra_1_to_mobile_1_sniffer_clean.npz"), t=t, t_log=t,
                            subcarrier=sub, H=H, H_rssi=(H * (10 ** ((rssi - pw) / 20))[:, None]).astype(np.complex64),
                            amp_db=amp, phase=ph, rssi_dbm=rssi, rx_pose=pose,
                            tx_pose=np.tile(np.eye(4), (n, 1, 1)), distance_m=x, bandwidth_mhz=np.array(80))
        P = csi_view.pdp(H, sub)
        prof = np.median(P, axis=0)
        peaks = sorted(np.argsort(prof)[-3:])
        if [(p - 8) * 12.5 for p in peaks] != [0.0, 100.0, 250.0]:
            f.append("pdp paths at %s ns" % [(p - 8) * 12.5 for p in peaks])
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "csi_view.py"),
                            os.path.join(tmp, "proc")], capture_output=True, text=True)
        print(r.stdout[-800:])
        if r.returncode:
            f.append(r.stderr[-1500:])
        for nme in ("csi_view_infra_1_to_mobile_1_sniffer.png", "csi_view_infra_1_to_mobile_1_sniffer.pdf",
                    "csi_view_infra_1_to_mobile_1_sniffer_zoom.png"):
            if not os.path.exists(os.path.join(d, "view", nme)):
                f.append("no " + nme)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "csi view ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
