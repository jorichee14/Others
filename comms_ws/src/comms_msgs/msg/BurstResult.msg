# BurstResult.msg
# One fixed-size burst sent into an otherwise idle link and echoed back.
#
# This is the deadline question, not the capacity question. Saturated TCP
# goodput says how much the link moves over 10 s; this says whether a 40 kB
# feature map arrives within 50 ms. A link can average 184 Mbit/s and still
# have a 200 ms window where nothing gets through -- invisible in an
# aggregate, decisive for collaborative perception.
#
# Round-trip, because one-way delay would need synchronised clocks at both
# ends and a timestamping receiver. The echo server retransmits nothing, so
# a lost packet stays lost and shows up in packets_returned.

std_msgs/Header header

string   target              # echo server host
uint16   port                # echo server UDP port

# --- What was sent ----------------------------------------------------------
uint32   burst_bytes         # total payload bytes in this burst
uint32   packet_bytes        # per-datagram payload size
uint32   packets_sent

# --- What came back ---------------------------------------------------------
uint32   packets_returned
float64  loss_percent        # 100 * (1 - returned/sent)

# --- Timing (NaN when nothing returned) -------------------------------------
float64  completion_ms       # first byte sent -> last echo received
float64  first_reply_ms      # first byte sent -> first echo received
float64  spread_ms           # completion_ms - first_reply_ms (echo dispersion)

# --- The verdict ------------------------------------------------------------
float64  deadline_ms         # deadline this burst was judged against
bool     met_deadline        # complete AND within deadline_ms
bool     complete            # every packet came back

# --- Effective rate ---------------------------------------------------------
# burst_bytes over completion_ms. NOT comparable to iperf goodput: this is a
# short burst on an idle link including round-trip, so it will read lower
# than saturated capacity by design.
float64  effective_mbps
