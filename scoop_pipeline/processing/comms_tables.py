#!/usr/bin/env python3
"""A pass's link and time-base measurements as tables: NTP, Wi-Fi status,
ping, iperf and CSI.

    python scoop_pipeline/processing/comms_tables.py <merged bag> [--config FILE]
        [--out DIR] [--frames configs/comms.yaml] [--no-positions]

writes to data/processed/<date>/<pass>/comms/ (--out for elsewhere):
    ntp.csv  wifi.csv  ping.csv  iperf.csv  csi.csv  csi_status.csv
                       one row per message, every machine (a `machine` column;
                       CSI also `peer`, the transmitter); t = header stamp
                       (that machine's clock), t_log = when the bag logged it
    csi_<tx>_to_<rx>.npz   a CSI link, one row per packet: t, t_log, subcarrier (K,)
                       signed, H (N, K) complex on the occupied subcarriers
                       (+-2..+-122 at 80 MHz: no DC, no guards), amp_db, phase
                       (its linear part across subcarriers removed); csi.csv
                       has csi_power_db and csi_amp_spread_db per packet
    summary.json       per machine / link: NTP offset and steps, ping loss and
                       RTT, iperf throughput up / down, RSSI and PHY rates, CSI rate
With the pass's pipeline config (found in map_stages/ like run_pass.py does)
every row gets its machine's position in map (x, y, z: configs/comms.yaml says
which frame), and a CSI packet the transmitter's too and the link length
(distance_m). Reads the bag once (comms topics only, but they are spread over
the whole bag). The work is in scoop/comms.py.
"""
import argparse
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
from scoop.bag import BagError                                      # noqa: E402


def positions(bag, cfg, log=print):
    """comms.Positions for the pass: the static tree of the bag, the robot's
    trajectory in map (04's track; a mapping pass: 10's from the refined one)."""
    import pass_tf
    from pipeline_common import load_traj
    edges = tftree.tf_edges([bag])
    if not any(e.parent == "map" for e in edges.values()):
        log("    (the bag's TF has no map: merged before the map stages; positions skipped)")
        return None
    traj_t = traj_T = body = None
    if cfg:
        P = pass_tf.load_pipeline(cfg)
        p = P.lidar_track_traj("_in_cam")
        if os.path.exists(p):
            traj_t, traj_T = load_traj(p)
            body = P.cfg.get("05_anchor", {}).get("cam_frame", "zed_left_camera_optical_frame")
            log(f"    moving robot: {len(traj_t)} poses of {body} from {p}")
        else:
            log(f"    (no {p}: the moving robot gets no position)")
    return comms.Positions(edges, traj_t, traj_T, body)


def add_positions(tables, pos, frames, log=print):
    missing = set()
    for kind, tb in tables.items():
        for who, prefix in (("machine", ""), ("peer", "peer_")):
            if who == "peer" and kind != "csi":
                continue
            for name in sorted({r[who] for r in tb.rows if r[who]}):
                fr = frames.get(name)
                fn = pos.of(fr) if fr else None
                if fn is None:
                    missing.add(f"{name} ({fr or 'no frame in comms.yaml'})")
                    continue
                rows = [r for r in tb.rows if r[who] == name]
                xyz = fn(np.array([r["t"] for r in rows], float))
                for r, p in zip(rows, xyz):
                    r[prefix + "x"], r[prefix + "y"], r[prefix + "z"] = (float(v) for v in p)
                for c in ("x", "y", "z"):
                    if prefix + c not in tb.columns:
                        tb.columns.append(prefix + c)
        if kind == "csi" and "peer_x" in tb.columns and "x" in tb.columns:
            for r in tb.rows:
                if "x" in r and "peer_x" in r:
                    r["distance_m"] = float(np.linalg.norm(
                        [r["x"] - r["peer_x"], r["y"] - r["peer_y"], r["z"] - r["peer_z"]]))
            tb.columns.append("distance_m")
    for m in sorted(missing):
        log(f"    (no position for {m})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--config", default=None, help="the pass's pipeline config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", default=os.path.join(ROOT, "configs", "comms.yaml"))
    ap.add_argument("--no-positions", action="store_true")
    a = ap.parse_args()
    import yaml
    import pass_tf
    from pipeline_common import processed_dir_for
    bag = os.path.abspath(os.path.expanduser(a.bag))
    out = a.out or os.path.join(processed_dir_for(bag), "comms")
    cfg = a.config or pass_tf.find_config(os.path.dirname(bag))
    try:
        print(f"comms topics of {bag}:")
        tables = comms.extract(bag)
        if not a.no_positions:
            print("positions:" + (f" (config {cfg})" if cfg else " (no pipeline config)"))
            pos = positions(bag, cfg)
            if pos is not None:
                frames = (yaml.safe_load(open(a.frames)) or {}).get("frames", {})
                add_positions(tables, pos, frames)
        print(f"tables -> {out}")
        comms.write(tables, out)
    except BagError as e:
        sys.exit(f"error: {e}")
    s = comms.summary(tables)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=2)
    print(f"\nsummary ({os.path.join(out, 'summary.json')}):")
    for kind, per in s.items():
        for who, v in per.items():
            print(f"  {kind:10s} {who}: " + _short(kind, v))


def _short(kind, v):
    f = lambda st, k="median": "-" if not st else "%.3g" % st[k]          # noqa: E731
    if kind == "ntp":
        return (f"{v['messages']} msgs, synchronized {v['synchronized_percent']:.0f}%, "
                f"|offset| median {f(v['offset_ms'])} ms (measured {f(v['measured_offset_ms'])} "
                f"ms, max {f(v['measured_offset_ms'], 'max')}), clock steps {v['clock_steps']}, "
                f"source {', '.join(v['sources'])}")
    if kind == "ping":
        return (f"{v['messages']} pings, loss {v['loss_percent']:.1f}%, RTT median "
                f"{f(v['rtt_ms'])} ms, p95 {f(v['rtt_ms'], 'p95')} ms")
    if kind == "iperf":
        return ", ".join(f"{d}: {v[d]['tests']} tests ({v[d]['failed']} failed), median "
                         f"{f(v[d]['mbps'])} Mbit/s (min {f(v[d]['mbps'], 'min')})"
                         for d in ("up", "down") if v[d]["tests"])
    if kind == "wifi":
        return (f"RSSI median {f(v['signal_dbm'])} dBm (min {f(v['signal_dbm'], 'min')}), "
                f"tx {f(v['tx_bitrate_mbps'])} / rx {f(v['rx_bitrate_mbps'])} Mbit/s, "
                f"channel {', '.join(v['channels'])}, roams {v['roams']}")
    if kind == "csi":
        d = v.get("distance_m")
        fc = ", ".join("0x%02x x%d" % (int(k), n) for k, n in v.get("frame_control", {}).items()
                       if k not in ("None", ""))
        return (f"{v['messages']} packets, {v['rate_hz'] or 0:.1f} Hz, longest gap "
                f"{v['longest_gap_s'] or 0:.2f} s, RSSI median {f(v.get('rssi_dbm'))} dBm"
                + (f", distance {f(d, 'min')}-{f(d, 'max')} m" if d else "")
                + (f"; frames {fc}" if fc else ""))
    if kind == "csi_status":
        return (f"{v['messages']} msgs, {f(v.get('frames_per_sec'))} frames/s, monitor mode "
                f"{v.get('monitor_mode_percent', 0):.0f}%, dropped {v.get('dropped_total')}")
    return f"{v['messages']} messages"


if __name__ == "__main__":
    main()
