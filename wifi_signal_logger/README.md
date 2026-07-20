# WiFi Signal Logger (SNR / RSSI / link metrics)

A small, dependency-free tool to **record WiFi communication metrics over time**
on Ubuntu 22.04 / NVIDIA Jetson and write them to CSV.

It samples the connected WiFi link at a fixed interval and logs, per sample:

| Column | Meaning |
|---|---|
| `timestamp_iso`, `timestamp_unix` | sample time (local ISO-8601 + Unix epoch) |
| `iface` | wireless interface (e.g. `wlan0`) |
| `ssid`, `bssid` | associated network name + AP MAC |
| `freq_mhz`, `channel` | operating frequency / channel |
| `signal_dbm` | **RSSI** — received signal power in dBm |
| `signal_avg_dbm` | driver's rolling-average signal |
| `noise_dbm` | noise floor of the in-use channel |
| `snr_db` | **SNR** = `signal_dbm − noise_dbm` |
| `link_quality` | driver link quality (e.g. `70/70`) |
| `tx_bitrate_mbps`, `rx_bitrate_mbps` | negotiated PHY rates |

### A note on "RSRP"

**RSRP is a cellular (LTE/5G) metric and has no WiFi equivalent.** For WiFi, the
received-power metric is the **RSSI / signal level in dBm**, recorded here as
`signal_dbm`. If you later add a cellular modem and want true RSRP/RSRQ/SINR,
that needs a separate modem-based reader (AT commands or ModemManager) — this
tool is WiFi-only.

## Requirements

- Python 3.8+ (stock on Ubuntu 22.04 / JetPack) — **no pip packages needed**.
- `iw` (preferred). Install with `sudo apt install iw`. Falls back to the legacy
  `iwconfig` (`wireless-tools`) and `/proc/net/wireless` when available.
- Run with `sudo` (or grant `CAP_NET_ADMIN`): some drivers only expose
  `signal` / survey `noise` to privileged callers. Without it the tool still
  runs and records whatever the driver exposes, leaving unknown fields blank.

## Usage

```bash
# Auto-detect the interface, sample every second, print a live table,
# and append to ./wifi_signal_log.csv:
sudo python3 wifi_signal_logger.py

# Explicit interface, 0.5 s interval, stop after 60 s, custom output file:
sudo python3 wifi_signal_logger.py -i wlan0 -t 0.5 -d 60 -o run1.csv

# CSV only, no console output (e.g. from a service/cron):
sudo python3 wifi_signal_logger.py --quiet
```

| Flag | Default | Description |
|---|---|---|
| `-i, --iface` | auto | wireless interface; auto-detected via `iw dev` / sysfs |
| `-t, --interval` | `1.0` | seconds between samples |
| `-d, --duration` | `0` | stop after N seconds (`0` = run until Ctrl+C) |
| `-o, --output` | `wifi_signal_log.csv` | CSV path (appends by default) |
| `--overwrite` | off | truncate the CSV instead of appending |
| `--quiet` | off | suppress the live table |

The CSV is flushed after every sample, so a `kill`/reboot never loses recorded
rows. Stop cleanly with `Ctrl+C` (or `SIGTERM`).

## Where the numbers come from

1. `iw dev <iface> station dump` — signal, signal-avg, tx/rx bitrate.
2. `iw dev <iface> link` — SSID, BSSID, frequency, and fills any gaps.
3. `iw dev <iface> survey dump` — **noise floor** of the `[in use]` channel,
   used to derive SNR. (Many drivers only report noise here, not in `link`.)
4. `/proc/net/wireless` — link quality, and signal/noise fallback (the driver's
   `-256` "invalid noise" sentinel is ignored).
5. `iwconfig <iface>` — last-resort fallback if `iw` is unavailable.

If `noise_dbm` comes out blank, your WiFi driver simply does not report a noise
floor (common on some `mt76`/`brcmfmac` chipsets); `signal_dbm` and the rest are
still logged, but `snr_db` will be empty for those samples.

## Quick analysis

```bash
# Mean/min signal and SNR from a run:
python3 - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open("wifi_signal_log.csv")))
sig = [int(r["signal_dbm"]) for r in rows if r["signal_dbm"]]
snr = [int(r["snr_db"])    for r in rows if r["snr_db"]]
print(f"samples={len(rows)}  signal: mean={st.mean(sig):.1f} min={min(sig)} dBm")
if snr: print(f"SNR: mean={st.mean(snr):.1f} min={min(snr)} dB")
PY
```
