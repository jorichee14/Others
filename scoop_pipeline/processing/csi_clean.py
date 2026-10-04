#!/usr/bin/env python3
"""The CSI of a pass, cleaned for use: one file per link.

    python scoop_pipeline/processing/csi_clean.py <processed pass folder>
        [--config FILE] [--frames configs/comms.yaml] [--comms DIR]

From comms/csi/ (processing/comms_tables.py: csi.csv, csi_<link>.npz) to
comms/csi/csi_<tx>_to_<rx>_clean.npz:

  frames      QoS data frames only (frame_control 0x88): the ones sent as VHT
              80 MHz, whose CSI covers the band. Block Acks and Action frames
              are legacy 20 MHz PPDUs (duplicated on four blocks, or on the
              primary 20 MHz only): their CSI is noise outside them
  subcarriers the data subcarriers, +-2..+-122 without the pilots (+-11, +-39,
              +-75, +-103): pilots are sent at the transmitter's own level
  H           as measured: after the receiver's gain control, so relative
  H_rssi      H scaled per frame so that its mean power over the subcarriers is
              the frame's RSSI (mean |H_rssi|^2 = 10^(rssi_dbm / 10) mW)
  amp_db      20 log10 |H| (relative)
  phase       per frame, the phase with its linear part across subcarriers
              removed (timing / frequency offsets); not continuous across frames
  rx_pose, tx_pose   4x4 poses in map of the receiver's and the transmitter's
              frames (configs/comms.yaml) at the frame's time; NaN where the
              trajectory does not reach
  distance_m  link length; rssi_dbm, seq (802.11 sequence number), src_mac,
              t (header stamp, the receiver's clock), t_log

and csi_clean_README.txt describing them.
"""
import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "map_stages"))
warnings.simplefilter("ignore", FutureWarning)

from scoop import comms, tftree                                     # noqa: E402

DATA_FC = 0x88
PILOTS = {20: (7, 21), 40: (11, 25, 53), 80: (11, 39, 75, 103), 160: (25, 53, 89, 117)}

README = """CSI, cleaned (scoop_pipeline/processing/csi_clean.py), one file per link
csi_<transmitter>_to_<receiver>_clean.npz, N frames x K subcarriers:

  t            (N,)     header stamp, s (receiver's clock; NTP-aligned with the others)
  t_log        (N,)     when the receiver's bag logged it, s
  subcarrier   (K,)     subcarrier index, +-2..+-122 at 80 MHz, pilots removed
  H            (N, K)   complex CSI as measured (after gain control: relative)
  H_rssi       (N, K)   H scaled so mean |H_rssi|^2 over the subcarriers = 10^(rssi_dbm/10) mW
  amp_db       (N, K)   20 log10 |H|, relative
  phase        (N, K)   phase of H with its linear part across subcarriers removed, per frame
  rssi_dbm     (N,)     the receiver's RSSI of the frame
  seq          (N,)     802.11 sequence number (12 bit); several counters (TIDs) interleave
  src_mac      (N,)     the transmitter's MAC
  rx_pose      (N,4,4)  receiver frame in map at t (NaN where the trajectory does not reach)
  tx_pose      (N,4,4)  transmitter frame in map at t
  distance_m   (N,)     |rx - tx|
  rx_frame, tx_frame    the TF frames those poses are of
  bandwidth_mhz, channel

Only QoS data frames (frame_control 0x88): sent as VHT 80 MHz, their CSI covers
the band. Block Acks / Action frames are legacy 20 MHz and left out. Pilots
(+-11, +-39, +-75, +-103) are sent at the transmitter's own level and left out.
One antenna, one stream: no angle of arrival.
"""


def load_rows(path):
    keep = ["machine", "peer", "t", "t_log", "rssi", "frame_control", "seq", "src_mac", "channel",
            "bandwidth_mhz"]
    with open(path) as fh:
        rd = csv.reader(fh)
        head = next(rd)
        idx = {c: head.index(c) for c in keep if c in head}
        cols = {c: [] for c in idx}
        for row in rd:
            for c, i in idx.items():
                cols[c].append(row[i])
    out = {}
    for c, v in cols.items():
        if c in ("machine", "peer", "src_mac"):
            out[c] = np.array(v)
        else:
            out[c] = np.array([float(x) if x != "" else np.nan for x in v])
    return out


def positions(cfg):
    import pass_tf
    from pipeline_common import load_traj
    P = pass_tf.load_pipeline(cfg)
    edges = tftree.tf_edges([P.dataset["bag"]])
    traj_t = traj_T = body = None
    p = P.lidar_track_traj("_in_cam")
    if os.path.exists(p):
        traj_t, traj_T = load_traj(p)
        body = P.cfg.get("05_anchor", {}).get("cam_frame", "zed_left_camera_optical_frame")
    return comms.Positions(edges, traj_t, traj_T, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed")
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames", default=os.path.join(ROOT, "configs", "comms.yaml"))
    ap.add_argument("--comms", default=None, help="the comms folder (default <processed>/comms)")
    a = ap.parse_args()
    import yaml
    import pass_tf
    proc = os.path.abspath(os.path.expanduser(a.processed))
    out = os.path.join(a.comms or os.path.join(proc, "comms"), "csi")
    path = os.path.join(out, "csi.csv")
    if not os.path.exists(path):
        sys.exit(f"no {path} (run processing/comms_tables.py)")
    cfg = a.config or pass_tf.find_config(proc)
    frames = (yaml.safe_load(open(a.frames)) or {}).get("frames", {})
    pos = positions(cfg) if cfg else None
    if pos is None:
        print("(no pipeline config: no poses)")
    rows = load_rows(path)
    summary = {}
    for peer in sorted(set(rows["peer"])):
        for machine in sorted(set(rows["machine"][rows["peer"] == peer])):
            name = "%s_to_%s" % (peer, machine.replace("/", "_"))
            zp = os.path.join(out, f"csi_{name}.npz")
            if not os.path.exists(zp):
                continue
            m = (rows["peer"] == peer) & (rows["machine"] == machine)
            r = {k: v[m] for k, v in rows.items()}
            o = np.argsort(r["t_log"], kind="stable")
            r = {k: v[o] for k, v in r.items()}
            z = np.load(zp)
            zo = np.argsort(z["t_log"], kind="stable")
            if len(zo) != len(o) or not np.allclose(z["t_log"][zo], r["t_log"], atol=1e-6):
                sys.exit(f"{name}: csi.csv and {zp} disagree; run processing/comms_tables.py again")
            keep = r["frame_control"].astype(int) == DATA_FC
            sub = z["subcarrier"]
            bw = int(z["bandwidth_mhz"]) if "bandwidth_mhz" in z.files else 80
            data_sc = ~np.isin(np.abs(sub), PILOTS.get(bw, ()))
            sub = sub[data_sc]
            H = z["H"][zo][keep][:, data_sc]
            amp_db, phase, power_db, _ = comms.csi_features(sub, H)
            rssi = r["rssi"][keep]
            scale = 10 ** ((rssi - power_db) / 20)
            res = {"t": r["t"][keep], "t_log": r["t_log"][keep], "subcarrier": sub,
                   "H": H.astype(np.complex64),
                   "H_rssi": (H * scale[:, None]).astype(np.complex64),
                   "amp_db": amp_db, "phase": phase, "rssi_dbm": rssi,
                   "seq": (r["seq"][keep].astype(int) >> 4) & 0xFFF
                   if np.nanmax(r["seq"][keep]) > 4095 else r["seq"][keep].astype(int) % 4096,
                   "src_mac": r.get("src_mac", np.array([""] * len(o)))[keep],
                   "bandwidth_mhz": np.array(bw),
                   "channel": np.array(np.nanmedian(r["channel"]) if "channel" in r else np.nan)}
            fr_rx, fr_tx = frames.get(machine), frames.get(peer)
            if pos is not None and fr_rx and fr_tx:
                prx, ptx = pos.pose_of(fr_rx), pos.pose_of(fr_tx)
                if prx is not None and ptx is not None:
                    res["rx_pose"] = prx(res["t"])
                    res["tx_pose"] = ptx(res["t"])
                    res["distance_m"] = np.linalg.norm(res["rx_pose"][:, :3, 3]
                                                       - res["tx_pose"][:, :3, 3], axis=1)
                    res["rx_frame"], res["tx_frame"] = np.array(fr_rx), np.array(fr_tx)
            p = os.path.join(out, f"csi_{name}_clean.npz")
            np.savez_compressed(p, **res)
            d = res.get("distance_m")
            summary[name] = {"frames": int(keep.sum()), "of": int(len(o)), "subcarriers": int(len(sub)),
                             "posed": int(np.isfinite(d).sum()) if d is not None else 0}
            print(f"  {p}: {keep.sum()} of {len(o)} frames (QoS data), {len(sub)} subcarriers"
                  + (f", posed {summary[name]['posed']}, link {np.nanmin(d):.1f}-{np.nanmax(d):.1f} m"
                     if d is not None and np.isfinite(d).any() else ", no poses"))
    with open(os.path.join(out, "csi_clean_README.txt"), "w") as fh:
        fh.write(README)
    with open(os.path.join(out, "csi_clean_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  {os.path.join(out, 'csi_clean_README.txt')}")


if __name__ == "__main__":
    main()
