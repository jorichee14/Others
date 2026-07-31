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


# --- iw dev <iface> link (the real supplied output) ------------------------
IW_LINK_SAMPLE = """Connected to 82:2a:a8:cb:d4:34 (on wlx8876b9eae0ff)
	SSID: BML
	freq: 5180
	RX: 170807384 bytes (62867 packets)
	TX: 6279838 bytes (44776 packets)
	signal: -66 dBm
	rx bitrate: 162.0 MBit/s VHT-MCS 4 40MHz VHT-NSS 2
	tx bitrate: 240.0 MBit/s VHT-MCS 5 40MHz short GI VHT-NSS 2
"""


def test_iw_link_full_parse(monkeypatch):
    monkeypatch.setattr(wp, "_run", lambda cmd: IW_LINK_SAMPLE)
    d = wp.collect_iw_link("wlx8876b9eae0ff")

    assert d["associated"] is True
    assert d["bssid"] == "82:2A:A8:CB:D4:34"
    assert d["essid"] == "BML"
    assert d["frequency_ghz"] == 5.18
    assert d["signal_dbm"] == -66.0

    # rx bitrate: 162.0 MBit/s VHT-MCS 4 40MHz VHT-NSS 2
    assert d["rx_bitrate_mbps"] == 162.0
    assert d["rx_mcs"] == 4
    assert d["rx_nss"] == 2
    assert d["rx_width_mhz"] == 40
    assert d["rx_phy_mode"] == "VHT"

    # tx bitrate: 240.0 MBit/s VHT-MCS 5 40MHz short GI VHT-NSS 2
    assert d["tx_bitrate_mbps"] == 240.0
    assert d["tx_mcs"] == 5
    assert d["tx_nss"] == 2
    assert d["tx_short_gi"] is True
    assert d["bit_rate_mbps"] == 240.0  # backward-compat mirrors tx

    # per-association byte/packet counters
    assert d["sta_rx_bytes"] == 170807384
    assert d["sta_rx_packets"] == 62867
    assert d["sta_tx_bytes"] == 6279838
    assert d["sta_tx_packets"] == 44776


def test_parse_bitrate_variants():
    ht = wp._parse_bitrate("130.0 MBit/s MCS 15 40MHz short GI")
    assert ht["mbps"] == 130.0 and ht["mcs"] == 15
    assert ht["phy_mode"] == "HT" and ht["short_gi"] is True

    he = wp._parse_bitrate("1201.0 MBit/s HE-MCS 11 80MHz HE-NSS 2 HE-GI 0")
    assert he["mcs"] == 11 and he["nss"] == 2
    assert he["width"] == 80 and he["phy_mode"] == "HE"


IW_STATION_SAMPLE = """Station 82:2a:a8:cb:d4:34 (on wlx8876b9eae0ff)
	inactive time:	40 ms
	rx bytes:	170807384
	rx packets:	62867
	tx bytes:	6279838
	tx packets:	44776
	tx retries:	123
	tx failed:	4
	signal:  	-66 [-67, -71] dBm
	signal avg:	-65 dBm
	tx bitrate:	240.0 MBit/s VHT-MCS 5 40MHz short GI VHT-NSS 2
	rx bitrate:	162.0 MBit/s VHT-MCS 4 40MHz VHT-NSS 2
	expected throughput:	114.688Mbps
	connected time:	3600 seconds
"""


def test_iw_station_parse(monkeypatch):
    monkeypatch.setattr(wp, "_run", lambda cmd: IW_STATION_SAMPLE)
    d = wp.collect_iw_station("wlx8876b9eae0ff")
    assert d["tx_retries"] == 123
    assert d["tx_failed"] == 4
    assert d["signal_avg_dbm"] == -65.0
    assert d["signal_dbm"] == -66.0
    assert d["expected_mbps"] == 114.688
    assert d["connected_time_s"] == 3600
    assert d["sta_tx_bytes"] == 6279838
    assert d["tx_mcs"] == 5


IW_SURVEY_SAMPLE = """Survey data from wlx8876b9eae0ff
	frequency:			5200 MHz
	noise:				-90 dBm
Survey data from wlx8876b9eae0ff
	frequency:			5180 MHz [in use]
	noise:				-95 dBm
	channel active time:		10000 ms
	channel busy time:		2500 ms
	channel receive time:		1200 ms
	channel transmit time:		300 ms
"""


def test_iw_survey_in_use_channel(monkeypatch):
    monkeypatch.setattr(wp, "_run", lambda cmd: IW_SURVEY_SAMPLE)
    d = wp.collect_iw_survey("wlx8876b9eae0ff")
    # must pick the [in use] block, not the first block
    assert d["noise_dbm"] == -95.0
    assert d["noise_valid"] is True
    assert d["channel_active_ms"] == 10000.0
    assert d["channel_busy_ms"] == 2500.0


def test_sysfs_flags_hex_up_running(tmp_path, monkeypatch):
    base = tmp_path / "wlan0"
    (base / "statistics").mkdir(parents=True)
    monkeypatch.setattr(wp, "SYS_NET", str(tmp_path))
    # flags is hex in sysfs; 0x1043 = 4163 = UP|BROADCAST|RUNNING|MULTICAST
    # (matches the interface's own `ifconfig` output).
    (base / "flags").write_text("0x1043\n")
    (base / "address").write_text("88:76:b9:ea:e0:ff\n")

    d = wp.collect_sysfs("wlan0")
    assert d["up"] is True
    assert d["running"] is True
    assert d["mac_address"] == "88:76:b9:ea:e0:ff"

    # a down interface (no UP bit)
    (base / "flags").write_text("0x1002\n")
    assert wp.collect_sysfs("wlan0")["up"] is False


def test_link_up_from_carrier(tmp_path, monkeypatch):
    base = tmp_path / "wlan0"
    base.mkdir()
    monkeypatch.setattr(wp, "SYS_NET", str(tmp_path))

    (base / "carrier").write_text("1\n")
    assert wp.link_up("wlan0") is True

    (base / "carrier").write_text("0\n")
    assert wp.link_up("wlan0") is False


def test_link_up_operstate_fallback(tmp_path, monkeypatch):
    base = tmp_path / "wlan0"
    base.mkdir()
    monkeypatch.setattr(wp, "SYS_NET", str(tmp_path))
    # no carrier file -> fall back to operstate
    (base / "operstate").write_text("up\n")
    assert wp.link_up("wlan0") is True
    (base / "operstate").write_text("down\n")
    assert wp.link_up("wlan0") is False


def test_link_up_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(wp, "SYS_NET", str(tmp_path))
    assert wp.link_up("nope") is None


def test_iw_denied_returns_empty(monkeypatch):
    # Operation not permitted -> _run returns None (non-zero exit, no stdout)
    monkeypatch.setattr(wp, "_run", lambda cmd: None)
    assert wp.collect_iw_station("wlan0") == {}
    assert wp.collect_iw_survey("wlan0") == {}
    assert wp.collect_iw_link("wlan0") == {}
