# IperfResult.msg
# One completed iperf3 measurement to a fixed server (the static wired
# laptop, or another robot running `iperf3 -s`). Published per test so
# results can be time-joined to robot pose. server_address/server_port
# identify which target a sample belongs to.

std_msgs/Header header

string   server_address     # iperf3 server IP/host
uint16   server_port        # default 5201
string   protocol           # "TCP" | "UDP"
bool     reverse            # true => downlink (server -> robot); false => uplink
float64  duration_s         # requested test duration

bool     success            # false if iperf3 failed / server unreachable
string   error              # error text when success is false, else ""

# --- Throughput -------------------------------------------------------------
float64  bitrate_mbps       # goodput at the receiver (Mbit/s), NaN if failed
uint64   bytes              # total bytes transferred
uint32   retransmits        # TCP retransmits (0 for UDP / when unreported)

# --- TCP round-trip time (from the sender stats, Linux client) --------------
# NaN when unavailable (UDP tests, or platform without RTT reporting).
float64  rtt_ms_mean
float64  rtt_ms_min
float64  rtt_ms_max

# --- UDP reliability (NaN / 0 for TCP tests) --------------------------------
float64  jitter_ms
uint64   lost_packets
uint64   total_packets
float64  lost_percent
