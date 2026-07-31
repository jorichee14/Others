"""Unit tests for iperf_parse against representative iperf3 -J output."""

import json

from wifi_monitor.iperf_parse import parse_iperf_json

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
