"""Unit tests for wifi_parsers using the real sample output supplied.

These tests exercise the regex parsers directly (no hardware or ROS
required) by monkeypatching the command runner and the sysfs/proc readers.
"""

import wifi_monitor.wifi_parsers as wp

IWCONFIG_SAMPLE = """wlx8876b9eae0ff  IEEE 802.11  ESSID:"BML"
          Mode:Managed  Frequency:5.18 GHz  Access Point: 82:2A:A8:CB:D4:34
          Bit Rate=270 Mb/s   Tx-Power=12 dBm
          Retry short limit:7   RTS thr:off   Fragment thr:off
          Power Management:off
          Link Quality=70/70  Signal level=-37 dBm
          Rx invalid nwid:0  Rx invalid crypt:0  Rx invalid frag:0
          Tx excessive retries:0  Invalid misc:0   Missed beacon:0
"""

PROC_WIRELESS_SAMPLE = (
    "Inter-| sta-|   Quality        |   Discarded packets\n"
    " face | tus | link level noise |  nwid  crypt   frag\n"
    "wlx8876b9eae0ff: 0000   70.  -37.  -60.        0      0      0\n"
)


def test_iwconfig_parsing(monkeypatch):
    monkeypatch.setattr(wp, "_run", lambda cmd: IWCONFIG_SAMPLE)
    data = wp.collect_iwconfig("wlx8876b9eae0ff")

    assert data["essid"] == "BML"
    assert data["mode"] == "Managed"
    assert data["frequency_ghz"] == 5.18
    assert data["bssid"] == "82:2A:A8:CB:D4:34"
    assert data["associated"] is True
    assert data["bit_rate_mbps"] == 270.0
    assert data["tx_power_dbm"] == 12.0
    assert data["link_quality"] == 70
    assert data["link_quality_max"] == 70
    assert data["signal_dbm"] == -37.0
    assert data["rx_invalid_nwid"] == 0
    assert data["missed_beacon"] == 0
    assert data["tx_excessive_retries"] == 0


def test_proc_wireless_parsing(tmp_path, monkeypatch):
    f = tmp_path / "wireless"
    f.write_text(PROC_WIRELESS_SAMPLE)
    monkeypatch.setattr(wp, "PROC_WIRELESS", str(f))

    data = wp.collect_proc_wireless("wlx8876b9eae0ff")
    assert data["link_quality"] == 70
    assert data["signal_dbm"] == -37.0
    assert data["noise_dbm"] == -60.0
    assert data["noise_valid"] is True


def test_proc_wireless_invalid_noise(tmp_path, monkeypatch):
    f = tmp_path / "wireless"
    f.write_text(
        "junk\njunk\nwlx8876b9eae0ff: 0000   70.  -37.  -256\n"
    )
    monkeypatch.setattr(wp, "PROC_WIRELESS", str(f))

    data = wp.collect_proc_wireless("wlx8876b9eae0ff")
    assert data["signal_dbm"] == -37.0
    assert "noise_dbm" not in data  # sentinel treated as no-data
    assert "noise_valid" not in data


def test_snr_derived_in_collect_all(monkeypatch):
    monkeypatch.setattr(wp, "collect_sysfs", lambda i: {})
    monkeypatch.setattr(
        wp,
        "collect_proc_wireless",
        lambda i: {"signal_dbm": -37.0, "noise_dbm": -60.0,
                   "noise_valid": True, "link_quality": 70},
    )
    monkeypatch.setattr(
        wp,
        "collect_iwconfig",
        lambda i: {"link_quality": 70, "link_quality_max": 70,
                   "frequency_ghz": 5.18},
    )
    monkeypatch.setattr(wp, "collect_iw_link", lambda i: {})

    data = wp.collect_all("wlx8876b9eae0ff")
    assert data["snr_db"] == 23.0            # -37 - (-60)
    assert data["link_quality_ratio"] == 1.0
    assert data["channel"] == 36             # 5.18 GHz -> ch 36


def test_channel_from_freq():
    assert wp.channel_from_freq_ghz(2.412) == 1
    assert wp.channel_from_freq_ghz(2.484) == 14
    assert wp.channel_from_freq_ghz(5.18) == 36
    assert wp.channel_from_freq_ghz(5.5) == 100
    assert wp.channel_from_freq_ghz(None) == -1
