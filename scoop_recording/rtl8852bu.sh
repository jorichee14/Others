#!/bin/bash
# RTL8852BU (vendor driver 8852bu) settings for the rig's Wi-Fi dongles.
#
#   sudo ./rtl8852bu.sh status                 # what the driver runs with, and what the file says
#   sudo ./rtl8852bu.sh csi                    # CSI transmitter: no HE, tx 1 stream, rx 2 streams
#   sudo ./rtl8852bu.sh normal                 # driver defaults (HE on, all streams)
#   sudo ./rtl8852bu.sh set rtw_tx_nss=1 ...   # any 8852bu options
#   sudo ./rtl8852bu.sh reload                 # reload the driver with what the file says
#
# csi / normal / set write /etc/modprobe.d/8852bu.conf, which is read at every
# boot, and reload the driver now. NORELOAD=1 only writes the file, so it takes
# effect at the next boot.
#
# The file is always written whole: the rtw89 blacklist, the usb-storage quirk
# and USB3 mode are kept in every mode, so changing a setting cannot lose them.
#
# Reloading drops the Wi-Fi on every RTL8852BU dongle for ~20 s. Over ssh through
# one of those dongles, use NORELOAD=1 and reboot, or run it at the robot.
set -eo pipefail

CONF=${CONF:-/etc/modprobe.d/8852bu.conf}
PARAMS=${PARAMS:-/sys/module/8852bu/parameters}
NET=${NET:-/sys/class/net}
BASE="rtw_switch_usb_mode=1"                    # USB3 mode, kept in every mode
CSI="rtw_he_enable=0 rtw_tx_nss=1 rtw_rx_nss=2" # the sniffer cannot read HE, and has one antenna
SHOW="rtw_he_enable rtw_vht_enable rtw_tx_nss rtw_rx_nss rtw_switch_usb_mode"

# Interfaces driven by 8852bu.
ifaces() {
    local i
    for i in "$NET"/*; do
        [ "$(basename "$(readlink -f "$i/device/driver" 2>/dev/null)")" = "rtl8852bu" ] \
            && basename "$i"
    done
    return 0
}

write_conf() {                                  # $@ = 8852bu options
    cat > "$CONF" <<EOF
# $CONF -- written by rtl8852bu.sh, $(date '+%F %T')
# Keep the in-kernel rtw89 driver from taking the dongle.
blacklist rtw89_8852bu
blacklist rtw89_8852bu_git
# Disable the virtual CD-ROM mode, or the dongle can come up as a CD drive.
options usb-storage quirks=0bda:1a2b:i
options 8852bu $*
EOF
    echo "wrote $CONF: options 8852bu $*"
}

reload() {
    if [ "${NORELOAD:-0}" = "1" ]; then
        echo "NORELOAD=1: the file is set; it takes effect at the next boot (or: sudo $0 reload)"
        return
    fi
    echo "reloading 8852bu -- Wi-Fi on its dongles drops for a moment..."
    modprobe -r 8852bu
    modprobe 8852bu
    local i n=0 all
    while [ "$n" -lt 30 ]; do                   # wait until every dongle is back on its network
        sleep 1; n=$((n + 1)); all=1
        for i in $(ifaces); do
            iw dev "$i" link 2>/dev/null | grep -q "Connected" || all=0
        done
        [ -n "$(ifaces)" ] && [ "$all" = 1 ] && break
    done
    echo "back after ${n}s"
}

status() {
    echo "== file: $CONF =="
    grep -E "^(options 8852bu|blacklist|options usb-storage)" "$CONF" 2>/dev/null | sed 's/^/  /' \
        || echo "  (missing)"
    echo "== driver now =="
    local k want now differ=0
    for k in $SHOW; do
        now=$(cat "$PARAMS/$k" 2>/dev/null || echo "-")
        want=$(grep -E "^options 8852bu" "$CONF" 2>/dev/null | grep -o "$k=[^ ]*" | cut -d= -f2 || true)
        if [ -n "$want" ] && [ "$want" != "$now" ]; then
            printf '  %-22s %-4s  (file says %s)\n' "$k" "$now" "$want"; differ=1
        else
            printf '  %-22s %s\n' "$k" "$now"
        fi
    done
    [ "$differ" = 0 ] || echo "  the driver is not running what the file says: sudo $0 reload, or reboot"
    echo "== dongles =="
    local i
    for i in $(ifaces); do
        echo "  $i  $(cat "$NET/$i/address" 2>/dev/null)"
        iw dev "$i" link 2>/dev/null | grep -E "SSID|freq|tx bitrate|rx bitrate" | sed 's/^\s*/    /'
    done
    [ -n "$(ifaces)" ] || echo "  (no interface is using 8852bu)"
}

need_root() { [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }; }

case "${1:-}" in
status)
    status
    ;;
csi)
    need_root
    write_conf "$BASE $CSI"
    reload
    status
    ;;
normal)
    need_root
    write_conf "$BASE"
    reload
    status
    ;;
set)
    need_root
    shift
    [ $# -gt 0 ] || { echo "usage: sudo $0 set rtw_option=value ..."; exit 1; }
    for kv in "$@"; do
        [[ "$kv" =~ ^rtw_[a-z0-9_]+=[^[:space:]]+$ ]] || { echo "not an option: $kv"; exit 1; }
    done
    opts=$BASE
    for kv in "$@"; do                          # the user's value wins over BASE
        [ "${kv%%=*}" = "${BASE%%=*}" ] && opts=""
    done
    write_conf "$opts $*"
    reload
    status
    ;;
reload)
    need_root
    NORELOAD=0 reload
    status
    ;;
*)  cat <<USAGE
usage: sudo $0 <mode>
  status                 what the driver runs with now, the file, and each dongle's link
  csi                    CSI transmitter: $CSI
  normal                 driver defaults (HE on, all streams)
  set rtw_x=v ...        any 8852bu options (USB3 mode kept unless you set it)
  reload                 reload the driver with what the file says
NORELOAD=1 writes the file only; it takes effect at the next boot.
USAGE
    exit 1 ;;
esac
