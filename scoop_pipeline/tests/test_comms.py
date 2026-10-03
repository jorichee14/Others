#!/usr/bin/env python
"""Self-test for processing/comms_tables.py (scoop/comms.py).

A pass like survey_1 (tests/test_pass_tf.py's scene, merged with its complete
TF) gets the robots' comms topics, with their own message types as the bag
carries them: NtpStatus, PingStat, IperfResult, WifiLinkStatus, and a CSI
type. Every table must hold every message, the summary the numbers put in
(ping loss, iperf medians, clock steps, RSSI), each row its machine's
position in map, and a CSI packet the length of its link.

    python scoop_pipeline/tests/test_comms.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import test_pass_tf as T                                            # noqa: E402
from rosbags.typesys import get_types_from_msg                      # noqa: E402
from scoop import merge, rosmsg                                     # noqa: E402
from scoop.bag import TopicSchema                                   # noqa: E402
from scoop.bagwrite import BagWriter                                # noqa: E402

DEFS = {
    "comms_msgs/msg/NtpStatus": "std_msgs/Header header\nstring role\nstring hostname\n"
    "bool synchronized\nstring sync_source\nint32 stratum\nfloat64 offset_seconds\n"
    "float64 last_offset_seconds\nfloat64 frequency_error_ppm\n"
    "builtin_interfaces/Time reference_time\nstring[] warnings\nbool clock_stepped\n",
    "comms_msgs/msg/PingStat": "std_msgs/Header header\nstring target\nuint32 icmp_seq\n"
    "bool reply\nfloat64 rtt_ms\nuint32 window\nfloat64 loss_percent\n",
    "comms_msgs/msg/IperfResult": "std_msgs/Header header\nstring server_address\n"
    "string protocol\nbool reverse\nbool success\nstring error\nfloat64 bitrate_mbps\n"
    "float32[] interval_mbps\n",
    "comms_msgs/msg/WifiLinkStatus": "std_msgs/Header header\nstring interface\nstring bssid\n"
    "int32 channel\nfloat64 signal_dbm\nfloat64 tx_bitrate_mbps\nfloat64 rx_bitrate_mbps\n",
    "csi_msgs/msg/Csi": "std_msgs/Header header\nstring tx_mac\nint32 rssi\nuint16 channel\n"
    "int16[] csi_real\nint16[] csi_imag\n",
}


def comms_bag(path, t0, dur):
    ts = rosmsg.typestore()
    for name, txt in DEFS.items():
        ts.register(get_types_from_msg(txt, name))
    hdr = "\n" + "=" * 80 + "\nMSG: std_msgs/Header\n" + rosmsg.msgdef("std_msgs/msg/Header")
    sch = {n: TopicSchema(n, "ros2msg", (txt + hdr).encode(), "cdr") for n, txt in DEFS.items()}
    M = lambda n, **kw: ts.types[n](**kw)                            # noqa: E731
    H = lambda t, fr="wifi": rosmsg.header(int(round(t * 1e9)), fr)  # noqa: E731
    Time = ts.types["builtin_interfaces/msg/Time"]
    out = []
    for i, t in enumerate(np.arange(t0, t0 + dur, 0.1)):              # ping 10 Hz, 1 in 5 lost
        for m, rtt in (("mobile_1", 3.0), ("mobile_2", 5.0)):
            reply = i % 5 != 0
            out.append((t, f"/{m}/wifi/ping", "comms_msgs/msg/PingStat", M(
                "comms_msgs/msg/PingStat", header=H(t), target="192.168.51.11", icmp_seq=i,
                reply=reply, rtt_ms=rtt if reply else float("nan"), window=20, loss_percent=20.0)))
    for i, t in enumerate(np.arange(t0, t0 + dur, 1.0)):              # 1 Hz: ntp, wifi
        for m, host in (("mobile_1", "mobile1"), ("mobile_2", "ubuntu"), ("infra_1", "pi2"),
                        ("mobile_1/sniffer", "pi1")):
            out.append((t, f"/{m}/ntp/client/status", "comms_msgs/msg/NtpStatus", M(
                "comms_msgs/msg/NtpStatus", header=H(t, host), role="client", hostname=host,
                synchronized=True, sync_source="mobile1", stratum=9, offset_seconds=1e-4,
                last_offset_seconds=-2e-4, frequency_error_ppm=20.0,
                reference_time=Time(sec=int(t), nanosec=0),
                warnings=["a", "b"], clock_stepped=(m == "infra_1" and i == 3))))
        for m in ("mobile_1", "mobile_2"):
            out.append((t, f"/{m}/wifi/status", "comms_msgs/msg/WifiLinkStatus", M(
                "comms_msgs/msg/WifiLinkStatus", header=H(t), interface="wlx", bssid="AA" if i < 5 else "BB",
                channel=149, signal_dbm=-60.0 - (i % 3), tx_bitrate_mbps=325.0, rx_bitrate_mbps=650.0)))
    for i, t in enumerate(np.arange(t0 + 0.5, t0 + dur, 2.0)):        # iperf down / up
        out.append((t, "/mobile_1/wifi/iperf_down", "comms_msgs/msg/IperfResult", M(
            "comms_msgs/msg/IperfResult", header=H(t), server_address="192.168.51.11",
            protocol="TCP", reverse=True, success=True, error="", bitrate_mbps=200.0 + i,
            interval_mbps=np.array([190.0, 210.0], np.float32))))
        out.append((t + 1, "/mobile_1/wifi/iperf_up", "comms_msgs/msg/IperfResult", M(
            "comms_msgs/msg/IperfResult", header=H(t + 1), server_address="192.168.51.11",
            protocol="TCP", reverse=False, success=i != 1, error="" if i != 1 else "refused",
            bitrate_mbps=100.0 if i != 1 else float("nan"), interval_mbps=np.zeros(0, np.float32))))
    for i, t in enumerate(np.arange(t0, t0 + dur, 0.05)):             # CSI 20 Hz per link
        for tx in ("infra_1", "mobile_2"):
            out.append((t, f"/mobile_1/sniffer/{tx}/csi", "csi_msgs/msg/Csi", M(
                "csi_msgs/msg/Csi", header=H(t, "pi1"), tx_mac=tx, rssi=-50, channel=149,
                csi_real=np.arange(64, dtype=np.int16) + i % 7,
                csi_imag=-np.arange(64, dtype=np.int16))))
    out.sort(key=lambda x: x[0])
    with BagWriter(path) as w:
        for t, topic, typ, msg in out:
            w.write_raw(topic, sch[typ], bytes(ts.serialize_cdr(msg, typ)), int(round(t * 1e9)))


def rows(path):
    return list(csv.DictReader(open(path)))


def main():
    tmp = tempfile.mkdtemp()
    f = []
    try:
        data = os.path.join(tmp, "data")
        stages = os.path.join(tmp, "stages")
        os.makedirs(stages)
        static_yaml = os.path.join(tmp, "static_tf.yaml")
        T_cam_sensor = T.outputs(data)
        open(static_yaml, "w").write(
            "transforms:\n  - {parent: zed_left_camera_optical_frame, child: os_sensor, "
            "translation: %s, rotation_xyzw: %s}\n" % tuple(map(json.dumps, T.tq(T_cam_sensor)))
            + "  - {parent: arducam_optical_frame, child: infra1_link, translation: %s, "
            "rotation_xyzw: %s}\n" % tuple(map(json.dumps, T.tq(T.T_RADAR))))
        T.configs(stages, data)
        raw = os.path.join(data, "raw", "20260101", "survey_1")
        inputs, stamps = T.write_bags(raw)
        tfbag = os.path.join(tmp, "tf_merged")
        T.merged_tree(inputs, os.path.join(stages, "pipeline_config_survey_1.json"), tfbag,
                      static_yaml)
        cb = os.path.join(tmp, "comms")
        comms_bag(cb, T.T0, 10.0)
        merged = os.path.join(data, "work", "20260101", "survey_1", "x_merged")
        merge.merge_bags([tfbag, cb], merged, log=lambda *_: None)
        out = os.path.join(tmp, "out")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "processing", "comms_tables.py"),
                            merged, "--config", os.path.join(stages, "pipeline_config_survey_1.json"),
                            "--out", out], capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode:
            raise AssertionError(r.stdout[-1500:] + r.stderr[-2500:])

        ping = rows(os.path.join(out, "ping.csv"))
        if len(ping) != 200:
            f.append("ping rows: %d" % len(ping))
        ntp = rows(os.path.join(out, "ntp.csv"))
        if len(ntp) != 40 or ntp[0]["warnings"] != "a | b" or "reference_time" not in ntp[0]:
            f.append("ntp: %d rows, columns %s" % (len(ntp), list(ntp[0])))
        ip = rows(os.path.join(out, "iperf.csv"))
        if len(ip) != 10 or not any(x["interval_mbps"] == "190;210" for x in ip):
            f.append("iperf rows / interval_mbps: %d %s" % (len(ip), ip[0].get("interval_mbps")))
        s = json.load(open(os.path.join(out, "summary.json")))
        if abs(s["ping"]["mobile_1"]["loss_percent"] - 20.0) > 0.1 or \
                s["ping"]["mobile_2"]["rtt_ms"]["median"] != 5.0:
            f.append("ping summary: %s" % s["ping"])
        iu, idn = s["iperf"]["mobile_1"]["up"], s["iperf"]["mobile_1"]["down"]
        if (iu["tests"], iu["failed"], idn["tests"]) != (5, 1, 5) or idn["mbps"]["median"] != 202.0:
            f.append("iperf summary: %s" % s["iperf"])
        if s["ntp"]["infra_1"]["clock_steps"] != 1 or s["ntp"]["mobile_2"]["clock_steps"] != 0 or \
                abs(s["ntp"]["mobile_1"]["measured_offset_ms"]["median"] - 0.2) > 1e-9:
            f.append("ntp summary: %s" % s["ntp"])
        w1 = s["wifi"]["mobile_1"]
        if w1["roams"] != 1 or w1["signal_dbm"]["min"] != -62.0:
            f.append("wifi summary: %s" % w1)
        csi = rows(os.path.join(out, "csi.csv"))
        if len(csi) != 400:
            f.append("csi rows: %d" % len(csi))
        z = np.load(os.path.join(out, "csi_infra_1_to_mobile_1_sniffer.npz"))
        if z["csi_real"].shape != (200, 64) or len(z["t"]) != 200:
            f.append("csi npz: %s" % {k: z[k].shape for k in z.files})

        # positions: mobile_1 moves with its trajectory, the others are fixed
        T_link_cam = T.CHAIN[0][2] @ T.CHAIN[1][2] @ T.CHAIN[2][2]
        p_infra = (T.T_ARDU @ T.T_RADAR)[:3, 3]
        for x in ping[::23]:
            if x["machine"] != "mobile_1":
                continue
            t = float(x["t"]) - T.T0
            want = (T.true_cam(t) @ np.linalg.inv(T_link_cam))[:3, 3]
            got = np.array([float(x[k]) for k in "xyz"])
            if np.linalg.norm(got - want) > 0.002:
                f.append("mobile_1 position at %.2f s off by %.1f mm"
                         % (t, np.linalg.norm(got - want) * 1000))
        m2 = [x for x in ping if x["machine"] == "mobile_2"][0]
        want = (T.T_RS @ np.linalg.inv(T.RS_CHAIN[0][2] @ T.RS_CHAIN[1][2]))[:3, 3]
        if np.linalg.norm(np.array([float(m2[k]) for k in "xyz"]) - want) > 0.002:
            f.append("mobile_2 position: %s vs %s" % ([m2[k] for k in "xyz"], want))
        for x in csi[::37]:
            if x["peer"] != "infra_1" or not x.get("distance_m"):
                continue
            t = float(x["t"]) - T.T0
            want = np.linalg.norm((T.true_cam(t) @ np.linalg.inv(T_link_cam))[:3, 3] - p_infra)
            if abs(float(x["distance_m"]) - want) > 0.002:
                f.append("csi distance at %.2f s: %s vs %.3f" % (t, x["distance_m"], want))
        if not any(x.get("distance_m") for x in csi):
            f.append("csi: no distance_m")
    except AssertionError as e:
        f.append(str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "comms ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
