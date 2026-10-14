#!/bin/bash
# CSI settings for the rig, one place.
#
#   sudo ./csi_params.sh rx                       # sniffer (mobile_1/sniffer): arm nexmon
#   sudo ./csi_params.sh rx aa:bb:cc:dd:ee:ff     # ...only frames from that transmitter
#   sudo ./csi_params.sh tx                       # transmitter (infra_1, mobile_2): VHT, 1 SS, then verify
#   sudo ./csi_params.sh check [secs]             # sniffer: are CSI frames arriving? (no ROS)
#   sudo ./csi_params.sh all [secs]               # sniffer: arm, then check
#
# `rx` takes one MAC or a comma-separated list; with none it uses $MACS (below),
# and MACS= (empty) means no filter at all.
#
# Override with env: IFACE=wlan0 CHAN=149/80 CORE=1 NSS=1 MCS=0-9
# MCS defaults to 0-9 (rate adaptation picks it); MCS=4 or MCS=3-5 fixes or limits it.
# rx: BCM43455c0, 1 antenna -> core 1, 1 spatial stream, channel 149 / 80 MHz.
# tx: send VHT, 1 spatial stream (MCS left to rate adaptation), so every frame the
#     sniffer sees is one 1x1 PPDU.
set -eo pipefail

IFACE=${IFACE:-wlan0}
CHAN=${CHAN:-149/80}
CORE=${CORE:-1}
NSS=${NSS:-1}
MCS=${MCS:-0-9}
# transmitters the sniffer listens to: mobile_2's dongle and infra_1's Pi
MACS=${MACS-88:76:b9:ea:e1:01,d8:3a:dd:19:66:3e}
PORT=${PORT:-5500}                              # nexmon_csi's UDP port
TARGET=${TARGET-192.168.51.11}                  # tx: short ping burst to this host so the link has traffic

arm() {
    local mac=${1-$MACS}
    local args=(-c "$CHAN" -C "$CORE" -N "$NSS")
    [ -n "$mac" ] && args+=(-m "$mac")
    ip link set "$IFACE" up
    nexutil -I"$IFACE" -s500 -b -l34 -v"$(makecsiparams "${args[@]}")"
    nexutil -I"$IFACE" -m1                      # monitor mode
    echo "armed: ${args[*]}"
    nexutil -I"$IFACE" -k                       # must print $CHAN
}

check() {
    local secs=${1:-5}
    command -v tcpdump >/dev/null || { echo "tcpdump missing: sudo apt install tcpdump"; exit 1; }
    echo "== interface =="
    iw dev "$IFACE" info | grep -E "type|channel" || true
    local mode
    mode=$(iw dev "$IFACE" info | awk '/type/ {print $2}')
    [ "$mode" = "monitor" ] || echo "WARNING: $IFACE is '$mode', not 'monitor' -- something undid the arming"
    local pcap
    pcap=$(mktemp /tmp/csi_check.XXXXXX.pcap)
    trap 'rm -f "$pcap"' EXIT
    echo "== listening ${secs}s for CSI packets (udp dst port $PORT) =="
    timeout "$secs" tcpdump -i "$IFACE" -nn -U -w "$pcap" "udp and dst port $PORT" 2>/dev/null || true
    python3 - "$pcap" "$secs" "$PORT" <<'PYEOF'
import collections, struct, sys

path, secs, port = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
NAMES = {"88:76:b9:ea:e1:01": "mobile_2", "d8:3a:dd:19:66:3e": "infra_1",
         "88:76:b9:ea:e0:ff": "mobile_1"}
data = open(path, "rb").read()
frames, rssi, ctrl = collections.Counter(), collections.defaultdict(list), collections.Counter()
if len(data) >= 24:
    magic = data[:4]
    end = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    link = struct.unpack(end + "I", data[20:24])[0]
    pos = 24
    while pos + 16 <= len(data):
        caplen = struct.unpack(end + "I", data[pos + 8:pos + 12])[0]
        pkt = data[pos + 16:pos + 16 + caplen]
        pos += 16 + caplen
        payload = None
        if link == 1 and len(pkt) > 42 and pkt[12:14] == b"\x08\x00":      # Ethernet / IPv4
            ihl = (pkt[14] & 0x0F) * 4
            if struct.unpack(">H", pkt[14 + ihl + 2:14 + ihl + 4])[0] == port:
                payload = pkt[14 + ihl + 8:]
        if payload is None:                                                # other link types: heuristic
            i = pkt.find(b"\x11\x11", 28, 100)
            payload = pkt[i:] if i >= 0 else None
        # nexmon_csi: magic(2) rssi(1) frame_control(1) src_mac(6) seq(2) ...
        if payload and len(payload) >= 10 and payload[:2] == b"\x11\x11":
            mac = ":".join("%02x" % b for b in payload[4:10])
            frames[mac] += 1
            rssi[mac].append(struct.unpack("b", payload[2:3])[0])
            ctrl[payload[3]] += 1
total = sum(frames.values())
print("frames: %d  (%.1f /s)" % (total, total / secs))
for mac, n in frames.most_common():
    r = rssi[mac]
    print("  %-18s %-9s %6d frames  %6.1f /s   rssi %d dBm" %
          (mac, NAMES.get(mac, "?"), n, n / secs, sum(r) / len(r)))
if ctrl:
    print("  frame_control:", ", ".join("%d x%d" % kv for kv in ctrl.most_common()))
if total == 0:
    print("NO CSI FRAMES. Most likely, in order:")
    print("  1. the MAC filter: re-arm with MACS= (empty) to hear everyone")
    print("  2. nobody is sending on channel 149/80: start a ping flood on the transmitter")
    print("  3. the interface left monitor mode (see WARNING above): nmcli dev set wlan0 managed no")
    sys.exit(1)
PYEOF
}

# Compare what the radio reports with what the sniffer is armed for.
verify_tx() {
    local ch=${CHAN%/*} bw=${CHAN#*/} want_freq fails=0
    want_freq=$((5000 + 5 * ch))
    local link info
    link=$(iw dev "$IFACE" link 2>/dev/null || true)
    info=$(iw dev "$IFACE" info 2>/dev/null || true)
    ok()   { printf '  ok    %s\n' "$1"; }
    bad()  { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

    echo "== verify: $IFACE (sniffer listens on $CHAN) =="
    echo "  mac   $(cat /sys/class/net/"$IFACE"/address 2>/dev/null)"
    if grep -q "Not connected" <<<"$link" || [ -z "$link" ]; then
        bad "not connected to an access point"
        echo "$fails problem(s)"; return 1
    fi
    ok "connected to $(awk '/SSID/ {print $2}' <<<"$link")"
    local freq; freq=$(awk '/freq:/ {print int($2)}' <<<"$link")
    [ "$freq" = "$want_freq" ] && ok "frequency $freq MHz (channel $ch)" \
                               || bad "frequency $freq MHz, the sniffer is on $want_freq (channel $ch)"
    local width; width=$(grep -o "width: [0-9]* MHz" <<<"$info" | head -1 | grep -o "[0-9]*")
    [ "$width" = "$bw" ] && ok "width $width MHz" || bad "width ${width:-?} MHz, the sniffer expects $bw"
    local rate; rate=$(grep "tx bitrate" <<<"$link" || true)
    if [ -z "$rate" ]; then
        bad "no tx bitrate yet (no traffic): send some and run again"
    else
        grep -q "VHT-NSS $NSS" <<<"$rate" && grep -q "VHT-MCS" <<<"$rate" \
            && ok "${rate#*: }" || bad "tx is not VHT with $NSS stream: ${rate#*: }"
    fi
    if [ "$fails" -eq 0 ]; then echo "tx setup OK"; else echo "$fails problem(s)"; fi
    [ "$fails" -eq 0 ]
}

case "${1:-}" in
rx)
    arm "${2-$MACS}"
    ;;
check)
    check "${2:-5}"
    ;;
all)
    arm "$MACS"
    sleep 1
    check "${2:-5}"
    ;;
tx)
    # VHT only, NSS $NSS, MCS $MCS (5 GHz; 0-9 = any). `iw ... bitrates` needs the interface
    # associated; re-run after reconnecting.
    # Some USB drivers refuse this ("Operation not supported"). That is fine for a
    # 1x1 dongle, which only sends one stream anyway: the verify step below checks
    # the rate the link really uses.
    iw dev "$IFACE" set bitrates legacy-5 ht-mcs-5 vht-mcs-5 "$NSS:$MCS" \
        || echo "note: this driver does not allow locking the rate set; checking what the link uses instead"
    if [ -n "$TARGET" ]; then
        echo "sending a short burst to $TARGET so the link reports a tx rate..."
        ping -i 0.01 -c 200 -q "$TARGET" >/dev/null 2>&1 || echo "  (no reply from $TARGET; the burst still went out)"
    fi
    verify_tx
    ;;
txcheck)
    verify_tx
    ;;
*)  cat <<USAGE
usage: sudo ./csi_params.sh <mode>
  rx [mac,...]   sniffer: arm nexmon (default MACs: \$MACS)
  check [secs]   sniffer: count CSI frames per transmitter, no ROS
  all [secs]     sniffer: arm, then check
  tx             transmitter (infra_1, mobile_2): VHT, 1 spatial stream, then verify
  txcheck        transmitter: verify only (after traffic has been running)
env: IFACE=$IFACE CHAN=$CHAN MACS=$MACS TARGET=$TARGET
USAGE
    exit 1 ;;
esac
