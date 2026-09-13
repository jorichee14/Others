# Health of the CSI extractor, published at a low rate so a broken node is
# distinguishable from a node whose channel is simply idle.
std_msgs/Header header
string   interface
bool     monitor_mode
uint16   chanspec
uint16   channel
uint16   bandwidth_mhz
string   firmware            # dmesg banner; must contain "nexmon"
float32  frames_per_sec
uint64   frames_total
uint64   dropped_total
string   mac_filter          # empty when unfiltered
