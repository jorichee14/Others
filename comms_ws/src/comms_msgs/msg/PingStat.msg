# PingStat.msg
# One ICMP ping outcome plus rolling-window latency/loss statistics.
# Produced by the ping_monitor node (robot pings a fixed host, e.g. the
# iperf server). Continuous, cheap (no link saturation), and works whenever
# the target replies to ICMP -- the complement to iperf throughput.

std_msgs/Header header

string   target             # host being pinged, e.g. 192.168.233.142
uint32   icmp_seq           # sequence number of this ping
bool     reply              # true = reply received; false = timeout / loss
float64  rtt_ms             # this reply's RTT (NaN when reply is false)

# --- Rolling window statistics ---------------------------------------------
uint32   window             # number of samples in the window
float64  loss_percent       # percent of the window that was lost
float64  rtt_ms_avg         # mean RTT over received replies (NaN if none)
float64  rtt_ms_min
float64  rtt_ms_max
