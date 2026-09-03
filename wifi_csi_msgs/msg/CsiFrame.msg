# One Nexmon CSI measurement: the channel response for a single received 802.11 frame.
std_msgs/Header header

# --- from the 18-byte nexmon header ---------------------------------------
string     src_mac          # source MAC of the measured frame, "aa:bb:cc:dd:ee:ff"
int8       rssi             # dBm
uint8      frame_control    # 0x88 QoS Data, 0x94 Block Ack, 0x80 Beacon
uint16     seq
uint8      core
uint8      spatial_stream
uint16     chanspec         # raw, e.g. 0xe02a
uint16     chip_version

# --- decoded configuration ------------------------------------------------
uint16     channel          # e.g. 36
uint16     bandwidth_mhz    # 20 / 40 / 80

# --- the measurement ------------------------------------------------------
# csi_real[i] + j*csi_imag[i] is the response of subcarrier subcarrier_index[i].
# Indices are RAW firmware slot numbers, so they stay comparable across nodes
# and captures even after trimming.
int32[]    subcarrier_index
float32[]  csi_real
float32[]  csi_imag

# --- provenance -----------------------------------------------------------
bool       trimmed          # constant fields / DC null removed
uint32     raw_slots        # slots in the original payload (256 at 80MHz)
