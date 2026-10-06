#!/usr/bin/env python
"""Self-test for analysis/comms/csi_analysis.py on a synthetic CSI link.

infra_1 -> sniffer at 100 Hz for 80 s: the sniffer stands for 20 s, then
moves away at 0.4 m/s. The channel has two paths, 0 and 100 ns, the second
6 dB down (RMS delay spread 40 ns), and fades: frame to frame it keeps 99.9 %
of its shape standing and 60 % moving. RSSI = -30 - 25 log10(d) (n = 2.5).
Every fourth QoS data frame is missed (75 % caught). Block Acks come as a
20 MHz legacy duplicate: 4 x 52 subcarriers. The report must give these
back.

    python scoop_pipeline/tests/test_csi_analysis.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scoop import comms                                             # noqa: E402

T0 = 1_790_229_700.0


def legacy_mask(sub):
    m = np.zeros(len(sub), bool)
    for c in (-96, -32, 32, 96):
        m |= (np.abs(sub - c) >= 1) & (np.abs(sub - c) <= 26)
    return m


def case(f, tx_moves=False):
    """tx_moves: the sniffer stands and the transmitter moves (a coop link):
    coherence goes by the faster end's speed all the same."""
    rng = np.random.default_rng(7)
    tmp = tempfile.mkdtemp()
    tag = "transmitter moving: " if tx_moves else ""
    try:
        out = os.path.join(tmp, "processed", "20260101", "survey_9", "comms", "csi")
        os.makedirs(out)
        sub = np.arange(-122, 123)
        sub = sub[np.abs(sub) >= 2]
        base = 1000 * (1 + 0.5 * np.exp(-2j * np.pi * sub * 100e-9 * 312.5e3))   # 2 paths, 100 ns
        n = 8000
        t = T0 + np.arange(n) * 0.01
        x = np.where(t - T0 < 20, 2.0, 2.0 + 0.4 * (t - T0 - 20))
        g = np.zeros(len(sub), complex)
        H = np.zeros((n, len(sub)), np.complex64)
        fc = np.where(np.arange(n) % 3 == 2, 0x94, 0x88)
        rows, seq = [], 0
        for i in range(n):
            rho = 0.999 if t[i] - T0 < 20 else 0.6
            g = rho * g + np.sqrt(1 - rho ** 2) * 0.3 * (rng.normal(size=len(sub)) + 1j * rng.normal(size=len(sub)))
            h = base * (1 + g)
            if fc[i] == 0x94:
                h = np.where(legacy_mask(sub), h, 1e-3)
            H[i] = h
            if fc[i] == 0x88:
                seq += 2 if (i // 3) % 4 == 3 else 1                 # one in four data frames missed
            d = x[i]
            rows.append({"machine": "mobile_1/sniffer", "peer": "infra_1",
                         "topic": "/mobile_1/sniffer/infra_1/csi", "t": t[i], "t_log": t[i] + 0.004,
                         "rssi": -30 - 25 * np.log10(d) + rng.normal(0, 1.0),
                         "frame_control": int(fc[i]), "seq": (seq % 4096) << 4 if fc[i] == 0x88 else 65535,   # Sequence Control
                         **({"x": 0.0, "y": 0.0, "peer_x": d, "peer_y": 0.0} if tx_moves
                            else {"x": d, "y": 0.0}), "distance_m": d})
        amp_db, phase, power, spread = comms.csi_features(sub, H)
        for r, p in zip(rows, power):
            r["csi_power_db"] = p
        with open(os.path.join(out, "csi.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(os.path.join(out, "csi_infra_1_to_mobile_1_sniffer.npz"),
                            t=t, t_log=t + 0.004, subcarrier=sub, H=H, amp_db=amp_db, phase=phase,
                            bandwidth_mhz=np.array(80))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "comms", "csi_analysis.py"),
                            os.path.join(tmp, "processed", "20260101", "survey_9")],
                           capture_output=True, text=True)
        print(r.stdout[-3500:])
        if r.returncode:
            raise AssertionError(r.stderr[-2500:])
        rep = json.load(open(os.path.join(out, "csi_report.json")))["infra_1_to_mobile_1_sniffer"]
        fr = rep["frames"]
        if fr["0x88 QoS Data"]["active_subcarriers"] != 242 or fr["0x94 Block Ack"]["active_subcarriers"] != 208:
            f.append("occupancy: %s" % {k: v["active_subcarriers"] for k, v in fr.items()})
        q = rep["qos_data_capture"]
        if q["median_step"] != 1 or abs(q["step_1_percent"] - 75.0) > 1.0:   # 3 steps of 1 in 4
            f.append("capture: %s" % q)
        pl = rep["path_loss_rssi"]
        if abs(pl["exponent"] - 2.5) > 0.05 or abs(pl["spread_db"] - 1.0) > 0.1:
            f.append("path loss: %s" % pl)
        ds = rep["rms_delay_spread_ns"]["median"]
        if abs(ds - 40) > 6:
            f.append("delay spread %.1f ns (40)" % ds)
        cs = {tuple(c["speed_mps"]): c["median_correlation"] for c in rep["coherence_by_speed"]}
        if not (cs.get((0.0, 0.05), 0) > 0.95 and cs.get((0.2, 0.5), 1) < cs[(0.0, 0.05)] - 0.05):
            f.append(tag + "coherence by speed: %s" % cs)
        for png in ("csi_timeline.png", "csi_occupancy.png", "csi_pathloss.png", "csi_coherence.png",
                    "csi_amplitude_infra_1_to_mobile_1_sniffer.png"):
            try:
                import matplotlib                                        # noqa: F401
            except ImportError:
                break
            if not os.path.exists(os.path.join(out, png)):
                f.append("no " + png)
    except AssertionError as e:
        f.append(tag + str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    f = []
    case(f)
    case(f, tx_moves=True)
    print("\n".join(f) or "csi analysis ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
