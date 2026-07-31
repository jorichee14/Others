"""Unit tests for iperf_parse against representative iperf3 -J output."""

import json

from wifi_monitor.iperf_parse import (
    parse_iperf_json, parse_interval_line, parse_ss_rtt,
)

# Trimmed but structurally faithful iperf3 TCP JSON (Linux client -> RTT).
TCP_JSON = json.dumps({
    "start": {"test_start": {"protocol": "TCP"}},
    "end": {
        "streams": [
            {"sender": {"min_rtt": 1200, "mean_rtt": 3400, "max_rtt": 9000}}
        ],
        "sum_sent": {"bytes": 45000000, "bits_per_second": 180000000.0,
                     "retransmits": 12},
        "sum_received": {"bytes": 44800000, "bits_per_second": 179200000.0},
    },
})

UDP_JSON = json.dumps({
    "start": {"test_start": {"protocol": "UDP"}},
    "end": {
        "sum": {"bytes": 30000000, "bits_per_second": 120000000.0,
                "jitter_ms": 0.45, "lost_packets": 7, "packets": 20000,
                "lost_percent": 0.035},
    },
})

ERROR_JSON = json.dumps({
    "error": "error - unable to connect to server: Connection refused"
})


def test_tcp_parse():
    r = parse_iperf_json(TCP_JSON)
    assert r["success"] is True
    assert r["protocol"] == "TCP"
    assert abs(r["bitrate_mbps"] - 179.2) < 1e-6      # receiver goodput
    assert r["bytes"] == 44800000
    assert r["retransmits"] == 12
    # microseconds -> ms
    assert abs(r["rtt_ms_mean"] - 3.4) < 1e-6
    assert abs(r["rtt_ms_min"] - 1.2) < 1e-6
    assert abs(r["rtt_ms_max"] - 9.0) < 1e-6


def test_udp_parse():
    r = parse_iperf_json(UDP_JSON)
    assert r["success"] is True
    assert r["protocol"] == "UDP"
    assert abs(r["bitrate_mbps"] - 120.0) < 1e-6
    assert r["lost_packets"] == 7
    assert r["total_packets"] == 20000
    assert abs(r["jitter_ms"] - 0.45) < 1e-6
    assert abs(r["lost_percent"] - 0.035) < 1e-6


def test_error_json():
    r = parse_iperf_json(ERROR_JSON)
    assert r["success"] is False
    assert "Connection refused" in r["error"]


def test_malformed_json():
    r = parse_iperf_json("not json at all")
    assert r["success"] is False
    assert "iperf3" in r["error"]


def test_missing_end():
    r = parse_iperf_json(json.dumps({"start": {}}))
    assert r["success"] is False


# --- streaming per-second interval lines ------------------------------------
def test_interval_single_stream():
    line = "[  5]   1.00-2.00   sec  5.97 MBytes  50.0 Mbits/sec    0    269 KBytes"
    r = parse_interval_line(line, parallel=1)
    assert r is not None
    assert abs(r["mbps"] - 50.0) < 1e-6
    assert abs(r["seconds"] - 1.0) < 1e-6
    assert r["retransmits"] == 0
    assert r["bytes"] == int(5.97 * 1024 ** 2)   # MBytes are binary


def test_interval_units():
    # Kbits/sec and Gbits/sec convert to Mbit/s
    assert abs(parse_interval_line(
        "[  5]   0.00-1.00   sec  100 KBytes  800 Kbits/sec")["mbps"] - 0.8) < 1e-9
    assert abs(parse_interval_line(
        "[  5]   0.00-1.00   sec  1.10 GBytes  9.50 Gbits/sec")["mbps"] - 9500.0) < 1e-6


def test_interval_parallel_uses_sum():
    stream = "[  5]   1.00-2.00   sec  6.5 MBytes  54.9 Mbits/sec    0    260 KBytes"
    summ = "[SUM]   1.00-2.00   sec  13.1 MBytes   110 Mbits/sec    0"
    # With parallel>1, per-stream lines are ignored; only SUM is used.
    assert parse_interval_line(stream, parallel=4) is None
    r = parse_interval_line(summ, parallel=4)
    assert abs(r["mbps"] - 110.0) < 1e-6
    # With a single stream, the per-stream line is used and stray SUM ignored.
    assert parse_interval_line(stream, parallel=1) is not None
    assert parse_interval_line(summ, parallel=1) is None


def test_interval_ignores_noise():
    assert parse_interval_line("Connecting to host 192.168.233.142, port 5201") is None
    assert parse_interval_line(
        "[  5]   0.00-10.00  sec  60.0 MBytes  50.0 Mbits/sec    0  sender") is None
    assert parse_interval_line("[ ID] Interval           Transfer") is None


# --- ss -ti TCP RTT parsing (real output) -----------------------------------
SS_SAMPLE = """State  Recv-Q  Send-Q   Local Address:Port   Peer Address:Port
ESTAB  0  496664  192.168.233.106:41402  192.168.233.142:5201
\t cubic wscale:8,7 rto:256 rtt:54.518/4.764 mss:1448 cwnd:94 minrtt:9.73
ESTAB  0  687800  192.168.233.106:41398  192.168.233.142:5201
\t cubic wscale:8,7 rto:256 rtt:55.602/5.36 mss:1448 cwnd:95 minrtt:9.211
ESTAB  0  0  192.168.233.106:41368  192.168.233.142:5201
\t cubic rto:224 rtt:23.585/17.488 ato:40 app_limited busy:60ms minrtt:3.998
ESTAB  0  522728  192.168.233.106:41390  192.168.233.142:5201
\t cubic wscale:8,7 rto:264 rtt:61.382/9.118 mss:1448 cwnd:96 minrtt:9.653
ESTAB  0  503904  192.168.233.106:41380  192.168.233.142:5201
\t cubic wscale:8,7 rto:264 rtt:61.98/6.042 mss:1448 cwnd:95 minrtt:5.047
"""


def test_ss_rtt_excludes_app_limited():
    r = parse_ss_rtt(SS_SAMPLE)
    # data streams: 54.518, 55.602, 61.382, 61.98 -> mean 58.37; control
    # (23.585, app_limited) excluded.
    assert abs(r["rtt_ms_mean"] - (54.518 + 55.602 + 61.382 + 61.98) / 4) < 1e-6
    assert r["rtt_ms_max"] == 61.98
    assert r["rtt_ms_min"] == 5.047       # min of data-stream minrtt values


def test_ss_rtt_all_app_limited_fallback():
    text = ("ESTAB 0 0 a:1 b:5201\n"
            "\t cubic rtt:12.5/3.0 app_limited minrtt:4.0\n")
    r = parse_ss_rtt(text)
    assert abs(r["rtt_ms_mean"] - 12.5) < 1e-6   # falls back to all sockets


def test_ss_rtt_none_when_empty():
    assert parse_ss_rtt("State Recv-Q Send-Q ...\n") is None
