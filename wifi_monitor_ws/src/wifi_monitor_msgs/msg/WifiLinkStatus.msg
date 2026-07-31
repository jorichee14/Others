# WifiLinkStatus.msg
# Snapshot of a wireless interface link + traffic statistics.
# Populated from /proc/net/wireless, `iw`/`iwconfig`, and
# /sys/class/net/<iface>/statistics. Fields that could not be read are
# reported as NaN (floats) or 0 (integers) with `valid` flags where noted.

std_msgs/Header header

# --- Identity ---------------------------------------------------------------
string   interface          # e.g. "wlx8876b9eae0ff"
string   mac_address        # interface hardware address (ether)
bool     up                 # interface is administratively UP and RUNNING
bool     associated         # currently associated with an access point

# --- Association ------------------------------------------------------------
string   essid              # network name, e.g. "BML"
string   bssid              # access point MAC, e.g. "82:2A:A8:CB:D4:34"
string   mode               # e.g. "Managed"
float64  frequency_ghz      # e.g. 5.18
int32    channel            # derived from frequency (-1 if unknown)
float64  bit_rate_mbps      # negotiated PHY rate, e.g. 270.0
float64  tx_power_dbm       # e.g. 12.0

# --- Signal quality ---------------------------------------------------------
int32    link_quality       # e.g. 70
int32    link_quality_max   # e.g. 70
float64  link_quality_ratio # link_quality / link_quality_max (NaN if unknown)
float64  signal_dbm         # RSSI, e.g. -37.0 (NaN if unknown)
float64  noise_dbm          # noise floor (NaN if the driver does not report it)
float64  snr_db             # signal_dbm - noise_dbm (NaN if noise unknown)
bool     noise_valid        # true when the driver reported a real noise floor

# --- Wireless error counters (iwconfig / proc) ------------------------------
int64    rx_invalid_nwid
int64    rx_invalid_crypt
int64    rx_invalid_frag
int64    tx_excessive_retries
int64    invalid_misc
int64    missed_beacon

# --- Interface traffic statistics (/sys/class/net/<iface>/statistics) -------
uint64   rx_packets
uint64   rx_bytes
uint64   rx_errors
uint64   rx_dropped
uint64   rx_overruns         # rx_fifo_errors
uint64   rx_frame_errors
uint64   tx_packets
uint64   tx_bytes
uint64   tx_errors
uint64   tx_dropped
uint64   tx_overruns         # tx_fifo_errors
uint64   tx_carrier_errors
uint64   collisions
