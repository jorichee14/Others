"""Tests for comms.ntp.ntp_parser against real chronyc output formats.

The original parser had five silent defects: unsigned offsets, unsigned
frequency, delay and jitter stuck at 0.0 from reading the wrong columns of
`sources -v`, and no chrony poll time. These pin the fixes. The canned
outputs are real chrony 4.x formats, including the bracketed "Last sample"
tokens that raised ValueError in the original.
"""

import pytest

from comms.ntp import ntp_parser as p


# --------------------------------------------------------------- _parse_secs
@pytest.mark.parametrize("text, want", [
    ("0.000123456 seconds slow of NTP time", -0.000123456),
    ("0.000005107 seconds fast of NTP time", +0.000005107),
    ("-123us",                               -0.000123),
    ("-123us[",                              -0.000123),     # sources token
    ("-156us]",                              -0.000156),
    ("+0.000000456 seconds",                 +0.000000456),  # Last offset
    ("-0.000012345 seconds",                 -0.000012345),
    ("45ms",                                  0.045),
    ("1.234 seconds",                         1.234),
    ("12ns",                                  1.2e-8),
    ("",                                      0.0),
    ("garbage",                               0.0),
])
def test_parse_secs_sign_and_units(text, want):
    assert p._parse_secs(text) == pytest.approx(want, abs=1e-12)


# --------------------------------------------------------------- canned chrony
TRACKING = """Reference ID    : C0A83302 (192.168.51.2)
Stratum         : 3
Ref time (UTC)  : Sun Sep 13 10:00:00 2026
System time     : 0.000123456 seconds slow of NTP time
Last offset     : -0.000098765 seconds
RMS offset      : 0.000150000 seconds
Frequency       : 27.237 ppm slow
Residual freq   : +0.003 ppm
Skew            : 0.042 ppm
Root delay      : 0.000812345 seconds
Root dispersion : 0.000456789 seconds
Update interval : 8.1 seconds
Leap status     : Normal
"""
SOURCES = """  .-- Source mode  '^' = server, '=' = peer, '#' = local clock.
MS Name/IP address         Stratum Poll Reach LastRx Last sample
===============================================================================
^* 192.168.51.2                  2   3   377     6   -123us[ -156us] +/-   45ms
"""
NTPDATA_ROOT = """Remote address  : 192.168.51.2 (C0A83302)
Peer delay      : 0.000412345 seconds
Peer dispersion : 0.000000123 seconds
Response time   : 0.000045678 seconds
"""
NTPDATA_UNPRIV = ""          # what an ordinary user gets: nothing
STATS = """Name/IP Address            NP  NR  Span  Frequency  Freq Skew  Offset  Std Dev
==============================================================================
192.168.51.2               12   7   89m     -0.012      0.156   -145us    73us
"""


def _chrony(monkeypatch, ntpdata):
    outs = {"tracking": TRACKING, "sources": SOURCES,
            "ntpdata": ntpdata, "sourcestats": STATS}
    monkeypatch.setattr(p, "_run", lambda cmd: outs.get(cmd[1], ""))
    return p._parse_chronyc()


def test_chrony_signed_offsets_and_frequency(monkeypatch):
    m = _chrony(monkeypatch, NTPDATA_ROOT)
    assert m["offset_seconds"] == pytest.approx(-0.000123456)
    assert m["last_offset_seconds"] == pytest.approx(-0.000098765)
    assert m["frequency_error_ppm"] == pytest.approx(-27.237)
    assert m["synchronized"] is True


def test_chrony_delay_jitter_skew_from_the_right_commands(monkeypatch):
    m = _chrony(monkeypatch, NTPDATA_ROOT)
    assert m["delay_seconds"] == pytest.approx(0.000412345)   # ntpdata
    assert m["jitter_seconds"] == pytest.approx(0.000073)     # sourcestats
    assert m["skew_ppm"] == pytest.approx(0.156)
    assert m["fit_samples"] == 12
    assert not any("upper bound" in w for w in m["warnings"])


def test_chrony_poll_and_reach_from_sources(monkeypatch):
    m = _chrony(monkeypatch, NTPDATA_ROOT)
    assert m["poll_interval_seconds"] == 8
    assert m["reach_register"] == 0o377
    assert m["error_bound_seconds"] == pytest.approx(0.045)


def test_chrony_ref_time_is_the_daemon_poll_time(monkeypatch):
    m = _chrony(monkeypatch, NTPDATA_ROOT)
    # Sun Sep 13 10:00:00 2026 UTC
    assert int(m["ref_time_epoch"]) == 1789293600


def test_unprivileged_user_gets_bounded_delay_with_a_warning(monkeypatch):
    m = _chrony(monkeypatch, NTPDATA_UNPRIV)
    assert m["delay_seconds"] == pytest.approx(2 * 0.045)
    assert any("upper bound" in w for w in m["warnings"])
    # everything else still arrives
    assert m["jitter_seconds"] == pytest.approx(0.000073)
    assert m["last_offset_seconds"] == pytest.approx(-0.000098765)


# `chronyc clients` reports activity, not offsets (13 Sep: the 501 line crashed
# the server node by becoming a topic called /ntp/clients/501/status)

CLIENTS_HEADER = ("Hostname                      NTP   Drop Int IntL Last     Cmd   Drop Int  Last\n"
                  "===============================================================================\n")
CLIENTS_DENIED = CLIENTS_HEADER + "501 Not authorised\n"
CLIENTS_REAL = CLIENTS_HEADER + (
    "192.168.25.31                 120      0   3   -    5       0      0   -     -\n"
    "wicoms-robot1                  88      0   7   -   12       0      0   -     -\n")


def _clients(monkeypatch, out):
    monkeypatch.setattr(p, "_run", lambda cmd: out if cmd[:2] == ["chronyc", "clients"] else "")


def test_clients_status_line_is_not_a_client(monkeypatch):
    _clients(monkeypatch, CLIENTS_DENIED)
    assert p.parse_chronyc_clients() == []
    assert p.get_connected_clients() == -1          # unknown, not zero and not one


def test_clients_real_rows_give_activity_not_offsets(monkeypatch):
    _clients(monkeypatch, CLIENTS_REAL)
    c = p.parse_chronyc_clients()
    assert [x["ip"] for x in c] == ["192.168.25.31", "wicoms-robot1"]
    assert c[0]["poll_interval_seconds"] == 8       # Int column is log2 of the interval
    assert c[1]["poll_interval_seconds"] == 128
    assert c[0]["ntp_packets"] == 120 and c[0]["last_seen_seconds"] == 5
    assert "offset_seconds" not in c[0]             # a server cannot measure it
    assert p.get_connected_clients() == 2


def test_clients_header_only_means_none_connected(monkeypatch):
    _clients(monkeypatch, CLIENTS_HEADER)
    assert p.parse_chronyc_clients() == []
    assert p.get_connected_clients() == 0
