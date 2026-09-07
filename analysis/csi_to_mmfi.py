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

The chip's automatic gain control is left in by default, as in MM-Fi: the
level of one packet against another is then the gain setting, not the
received power. The per-packet RSSI is written alongside so that level can be
restored, and --absolute does it at export time.

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
    ap.add_argument("--absolute", action="store_true",
                    help="scale each packet so its mean subcarrier power equals its RSSI, giving "
                         "amplitude in dBm (the Intel 5300 tool's scaling). Default is raw, as "
                         "MM-Fi ships: the chip's automatic gain is left in, so the level of one "
                         "packet against another is the gain setting, not the received power")
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
        n_bag = len(df)
        H_all, idx_all, keep = stack_H(df)   # frames with a different subcarrier layout are dropped
        df = df.loc[keep].reset_index(drop=True)
        use = usable_subcarriers(H_all, args.null_floor_db)
        lo, span = occupied_band(idx_all, use, args.band_gap)
        use &= band_mask(idx_all, lo, span)
        use[use] &= ~band_outliers(H_all[:, use])
        H = H_all[:, use]                       # raw chip values, complex
        amp = 20 * np.log10(np.abs(H) + 1e-12)  # MM-Fi's convention: gain control left in
        if args.absolute:
            # level restored from the packet's RSSI; the shape across subcarriers is untouched
            mean_pw_db = 10 * np.log10(np.maximum((np.abs(H) ** 2).mean(axis=1, keepdims=True), 1e-24))
            amp = amp - mean_pw_db + df["rssi"].to_numpy(float)[:, None]
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
                         "CSIphase": pha[s].T[None, :, :],
                         # extra keys MM-Fi does not have; its loaders index by name and skip them
                         "t_s": ((t_ns[s] - t0_ns) / 1e9)[None, :],
                         "rssi_dbm": df["rssi"].to_numpy(float)[s][None, :]},
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
        print(f"{agent}: {n_frames} frames of {args.packets} packets -> {agent_dir}")
        print(f"  CSIamp shape (1, {int(use.sum())}, {args.packets}), amplitude "
              f"{amp.min():.0f} to {amp.max():.0f} " + ("dBm, RSSI-scaled" if args.absolute
                                                       else "dB of raw chip values, gain control in"))
        print(f"  not written: {n_bag - len(df)} packets with a different subcarrier layout, "
              f"{len(df) - n_frames * args.packets} tail packets short of a frame, "
              f"{H_all.shape[1] - int(use.sum())} of {H_all.shape[1]} slots per packet that carry no subcarrier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
