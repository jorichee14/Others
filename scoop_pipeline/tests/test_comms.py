#!/usr/bin/env python
"""Self-test for processing/comms_tables.py (scoop/comms.py).

A pass like survey_1 (tests/test_pass_tf.py's scene, merged with its complete
TF) gets the robots' comms topics, with their own message types as the bag
carries them: NtpStatus, PingStat, IperfResult, WifiLinkStatus, CsiFrame
(an 80 MHz nexmon dump: 256 raw slots, junk on DC and the guards) and
CsiStatus. CSI must come out on the 242 occupied subcarriers with the
amplitude and the phase put in (its linear part removed). Every table must hold every message, the summary the numbers put in
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
    "comms_msgs/msg/CsiFrame": "std_msgs/Header header\nstring src_mac\nint8 rssi\n"
    "uint8 frame_control\nuint16 seq\nuint8 core\nuint8 spatial_stream\nuint16 chanspec\n"
    "uint16 chip_version\nuint16 channel\nuint16 bandwidth_mhz\nint32[] subcarrier_index\n"
    "float32[] csi_real\nfloat32[] csi_imag\nbool trimmed\nuint32 raw_slots\n",
    "comms_msgs/msg/CsiStatus": "std_msgs/Header header\nstring interface\nbool monitor_mode\n"
    "uint16 chanspec\nuint16 channel\nuint16 bandwidth_mhz\nstring firmware\n"
    "float32 frames_per_sec\nuint64 frames_total\nuint64 dropped_total\nstring mac_filter\n",
}
SLOTS = np.arange(256)
SUB = np.where(SLOTS < 128, SLOTS, SLOTS - 256)


def csi_slots(i):
    """80 MHz nexmon dump: amplitude 100 and a linear phase on the occupied
    subcarriers (+-2..+-122), junk on DC and the guards, as the 43455 gives."""
    H = 100 * np.exp(1j * (0.05 * SUB + 0.3 + 0.01 * i))
    junk = (np.abs(SUB) < 2) | (np.abs(SUB) > 122)
    H[junk] = -32640 - 18944j
    return H.real.astype(np.float32), H.imag.astype(np.float32)


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
        re, im = csi_slots(i)
        for tx in ("infra_1", "mobile_2"):
            out.append((t, f"/mobile_1/sniffer/{tx}/csi", "comms_msgs/msg/CsiFrame", M(
                "comms_msgs/msg/CsiFrame", header=H(t, "wlan0"), src_mac=tx, rssi=-74,
                frame_control=148 if i % 2 else 136, seq=i, core=0, spatial_stream=0,
                chanspec=57499, chip_version=101, channel=149, bandwidth_mhz=80,
                subcarrier_index=SLOTS.astype(np.int32), csi_real=re, csi_imag=im,
                trimmed=False, raw_slots=256)))
    for t in np.arange(t0, t0 + dur, 2.0):
        out.append((t, "/mobile_1/sniffer/csi_publisher/status", "comms_msgs/msg/CsiStatus", M(
            "comms_msgs/msg/CsiStatus", header=H(t, "wlan0"), interface="wlan0", monitor_mode=True,
            chanspec=57499, channel=149, bandwidth_mhz=80, firmware="nexmon", frames_per_sec=400.0,
            frames_total=100, dropped_total=0, mac_filter="")))
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

        ping = rows(os.path.join(out, "wifi", "ping.csv"))
        if len(ping) != 200:
            f.append("ping rows: %d" % len(ping))
        ntp = rows(os.path.join(out, "ntp", "ntp.csv"))
        if len(ntp) != 40 or ntp[0]["warnings"] != "a | b" or "reference_time" not in ntp[0]:
            f.append("ntp: %d rows, columns %s" % (len(ntp), list(ntp[0])))
        ip = rows(os.path.join(out, "wifi", "iperf.csv"))
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
        csi = rows(os.path.join(out, "csi", "csi.csv"))
        if len(csi) != 400:
            f.append("csi rows: %d" % len(csi))
        z = np.load(os.path.join(out, "csi", "csi_infra_1_to_mobile_1_sniffer.npz"))
        if z["H"].shape != (200, 242) or len(z["t"]) != 200 or "csi_real" in z.files:
            f.append("csi npz: %s" % {k: z[k].shape for k in z.files})
        else:
            sub = z["subcarrier"]
            if sub[0] != -122 or sub[-1] != 122 or 0 in sub or 1 in sub or -1 in sub:
                f.append("csi subcarriers: %s .. %s" % (sub[:3], sub[-3:]))
            if np.abs(z["amp_db"] - 40).max() > 1e-3 or np.abs(z["phase"]).max() > 1e-3:
                f.append("csi amp / phase: %.4f dB, %.5f rad" % (np.abs(z["amp_db"] - 40).max(),
                                                                  np.abs(z["phase"]).max()))
            if np.abs(np.angle(z["H"][0, sub == 10][0]) - (0.5 + 0.3)) > 1e-4:
                f.append("csi H at subcarrier 10: %s" % z["H"][0, sub == 10])
        if abs(float(csi[0]["csi_power_db"]) - 40) > 1e-3 or csi[0]["n_subcarriers"] != "242":
            f.append("csi columns: %s" % {k: csi[0][k] for k in ("csi_power_db", "n_subcarriers")})
        sc = s["csi"]["mobile_1/sniffer <- infra_1"]
        if sc["frame_control"] != {"136": 100, "148": 100} or \
                s["csi_status"]["mobile_1/sniffer"]["frames_per_sec"]["median"] != 400.0:
            f.append("csi summary: %s / %s" % (sc, s.get("csi_status")))

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

        # the clean export: data frames, no pilots, H scaled to RSSI, full poses
        r = subprocess.run([sys.executable, os.path.join(ROOT, "processing", "csi_clean.py"),
                            os.path.join(data, "processed", "20260101", "survey_1"), "--comms", out,
                            "--config", os.path.join(stages, "pipeline_config_survey_1.json")],
                           capture_output=True, text=True)
        print(r.stdout[-1200:])
        if r.returncode:
            raise AssertionError("csi_clean: " + r.stdout[-1000:] + r.stderr[-2000:])
        c = np.load(os.path.join(out, "csi", "csi_infra_1_to_mobile_1_sniffer_clean.npz"))
        sub = c["subcarrier"]
        if c["H"].shape != (100, 234) or np.isin(np.abs(sub), (11, 39, 75, 103)).any():
            f.append("clean: %s, pilots in: %s" % (c["H"].shape, np.isin(np.abs(sub), (11, 39, 75, 103)).any()))
        pw = 10 * np.log10(np.mean(np.abs(c["H_rssi"]) ** 2, axis=1))
        if np.abs(pw - c["rssi_dbm"]).max() > 1e-3 or np.abs(c["amp_db"] - 40).max() > 1e-3:
            f.append("clean: H_rssi power %s vs rssi %s" % (pw[:3], c["rssi_dbm"][:3]))
        if not os.path.exists(os.path.join(out, "csi", "csi_clean_README.txt")):
            f.append("clean: no README")
        for i in range(0, 100, 23):
            t = c["t"][i] - T.T0
            want = T.true_cam(t) @ np.linalg.inv(T_link_cam)
            if np.abs(c["rx_pose"][i] - want).max() > 2e-3:
                f.append("clean: rx_pose at %.2f s off by %.4f" % (t, np.abs(c["rx_pose"][i] - want).max()))
            if np.abs(c["tx_pose"][i] - T.T_ARDU @ T.T_RADAR).max() > 1e-4:
                f.append("clean: tx_pose off")
        dd = np.linalg.norm(c["rx_pose"][:, :3, 3] - (T.T_ARDU @ T.T_RADAR)[:3, 3], axis=1)
        if np.abs(dd - c["distance_m"]).max() > 1e-6 or str(c["tx_frame"]) != "infra1_link":
            f.append("clean: distance / frames")
        # two robots moving (coop): each frame through its own robot's track
        from scoop import tftree as tt, comms
        E = tt.Edge
        Ta = np.eye(4)
        Ta[:3, 3] = (0.1, 0, 0.2)
        edges = {"zed_camera_link": E("map", "zed_camera_link", False),
                 "zed_left_camera_optical_frame": E("zed_camera_link",
                                                    "zed_left_camera_optical_frame", True, Ta),
                 "map_realsense": E("map", "map_realsense", True, np.eye(4)),
                 "camera_link": E("map_realsense", "camera_link", False)}
        tt_ = np.array([0.0, 1.0, 2.0])
        T1 = np.tile(np.eye(4), (3, 1, 1))
        T1[:, 0, 3] = (0, 1, 2)                       # mobile_1's camera along x
        T2 = np.tile(np.eye(4), (3, 1, 1))
        T2[:, 1, 3] = (5, 6, 7)                       # mobile_2 along y
        pos = comms.Positions(edges, tt_, T1, "zed_left_camera_optical_frame",
                              tracks=[(tt_, T2, "camera_link")])
        p1 = pos.of("zed_camera_link")(np.array([0.5]))[0]
        p2 = pos.of("camera_link")(np.array([1.5]))[0]
        if np.abs(p1 - (0.4, 0, -0.2)).max() > 1e-9 or np.abs(p2 - (0, 6.5, 0)).max() > 1e-9:
            f.append("positions, two robots moving: %s %s" % (p1, p2))
        if np.abs(pos.pose_of("camera_link")(np.array([1.0]))[0][:3, 3] - (0, 6, 0)).max() > 1e-9:
            f.append("pose_of, the second robot")
        # its /tf edge not seen (it starts later than tf_edges looks): still placed
        late = dict(edges)
        late.pop("camera_link")
        pl = comms.Positions(late, tt_, T1, "zed_left_camera_optical_frame",
                             tracks=[(tt_, T2, "camera_link")])
        if pl.of("camera_link") is None or pl.pose_of("camera_link") is None or \
                np.abs(pl.of("camera_link")(np.array([1.5]))[0] - (0, 6.5, 0)).max() > 1e-9:
            f.append("positions, a robot whose /tf starts late")
    except AssertionError as e:
        f.append(str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(f) or "comms ok")
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
