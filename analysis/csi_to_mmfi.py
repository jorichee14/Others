#!/usr/bin/env python3
"""Write the CSI stream in MM-Fi's per-frame .mat layout.

    python3 csi_to_mmfi.py --run coop2 --packets 10

MM-Fi stores one .mat per frame holding `CSIamp` and `CSIphase`, each shaped
(antennas, subcarriers, packets): amplitude as 20*log10 of the chip's raw
integers, phase as the raw wrapped angle, nothing sanitised. This writes the
same thing from our extraction: (1, 56, packets) per frame for a one-antenna
Nexmon capture, packets grouped consecutively, plus one CSV per agent that
gives each frame its time span and the robot's ground-truth pose at the
frame's midpoint, which MM-Fi carries in its other modalities and we carry
in the pose topic.

Only the slots that carry a subcarrier are written (the empty guard, DC and
leakage slots are dropped, as MM-Fi's 114 are already the used subcarriers of
a 40 MHz channel), and nothing else is altered: no gain normalisation, no
receiver-shape removal. A consumer of MM-Fi files applies its own
preprocessing and should be able to apply it here unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_poses  # noqa: E402
from csi_analysis import load_csi, stack_H  # noqa: E402
from csi_core import band_mask, band_outliers, occupied_band, usable_subcarriers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run")
    ap.add_argument("--extracts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--packets", type=int, default=10, help="packets per frame, as MM-Fi")
    ap.add_argument("--pose-topic", default="global_pose")
    ap.add_argument("--null-floor-db", type=float, default=20.0)
    ap.add_argument("--band-gap", type=int, default=12)
    args = ap.parse_args()

    extracts = args.extracts or Path("extracts") / args.run
    out = args.out or Path("results") / args.run / "mmfi"
    csi = load_csi(extracts)
    if not csi:
        raise SystemExit(f"no *csi.parquet in {extracts}; run csi_analysis.py --bag first")
    poses = load_poses(extracts, args.pose_topic)
    t0_ns = min(int(d["log_time_ns"].min()) for d in csi.values())

    for agent in sorted(csi):
        df = csi[agent].sort_values("log_time_ns").reset_index(drop=True)
        H_all, idx_all, keep = stack_H(df)
        df = df.loc[keep].reset_index(drop=True)
        use = usable_subcarriers(H_all, args.null_floor_db)
        lo, span = occupied_band(idx_all, use, args.band_gap)
        use &= band_mask(idx_all, lo, span)
        use[use] &= ~band_outliers(H_all[:, use])
        H = H_all[:, use]                       # raw chip values, complex
        amp = 20 * np.log10(np.abs(H) + 1e-12)  # MM-Fi's convention
        pha = np.angle(H)
        t_ns = df["log_time_ns"].to_numpy()

        agent_dir = out / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        n_frames = len(df) // args.packets
        rows = []
        pt, pxy = poses.get(agent, (None, None))
        for k in range(n_frames):
            s = slice(k * args.packets, (k + 1) * args.packets)
            sio.savemat(agent_dir / f"frame{k + 1:03d}.mat",
                        {"CSIamp": amp[s].T[None, :, :],     # (1 antenna, subcarriers, packets)
                         "CSIphase": pha[s].T[None, :, :]},
                        do_compression=True)
            tm = 0.5 * (t_ns[s][0] + t_ns[s][-1])
            row = {"frame": k + 1, "t_start_s": (t_ns[s][0] - t0_ns) / 1e9,
                   "t_end_s": (t_ns[s][-1] - t0_ns) / 1e9,
                   "rssi_dbm": float(df["rssi"].iloc[s].median())}
            if pt is not None:
                o = np.argsort(pt)
                row["x_m"] = float(np.interp(tm, pt[o], pxy[o, 0]))
                row["y_m"] = float(np.interp(tm, pt[o], pxy[o, 1]))
            rows.append(row)
        pd.DataFrame(rows).to_csv(agent_dir / "frames.csv", index=False)
        sio.savemat(agent_dir / "subcarriers.mat", {"fft_slot": idx_all[use][None, :]})
        print(f"{agent}: {n_frames} frames of {args.packets} packets, "
              f"CSIamp shape (1, {int(use.sum())}, {args.packets}), "
              f"amplitude {amp.min():.0f} to {amp.max():.0f} dB -> {agent_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
