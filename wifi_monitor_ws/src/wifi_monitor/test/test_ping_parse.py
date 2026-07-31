"""Unit tests for ping_parse."""

from wifi_monitor.ping_parse import parse_ping_line


def test_normal_reply():
    line = "64 bytes from 192.168.233.142: icmp_seq=7 ttl=64 time=3.45 ms"
    r = parse_ping_line(line)
    assert r == {"seq": 7, "reply": True, "rtt_ms": 3.45}


def test_reply_no_space_before_ms():
    r = parse_ping_line(
        "64 bytes from 10.0.0.1: icmp_seq=2 ttl=63 time=12 ms")
    assert r["seq"] == 2 and r["reply"] is True and r["rtt_ms"] == 12.0


def test_no_answer_yet():
    r = parse_ping_line("no answer yet for icmp_seq=9")
    assert r == {"seq": 9, "reply": False, "rtt_ms": None}


def test_unreachable():
    r = parse_ping_line(
        "From 192.168.233.1 icmp_seq=3 Destination Host Unreachable")
    assert r["seq"] == 3 and r["reply"] is False and r["rtt_ms"] is None


def test_banner_and_summary_ignored():
    assert parse_ping_line("PING 192.168.233.142 (192.168.233.142) 56 data bytes") is None
    assert parse_ping_line("--- 192.168.233.142 ping statistics ---") is None
    assert parse_ping_line("rtt min/avg/max/mdev = 1/2/3/0.5 ms") is None
    assert parse_ping_line("") is None
