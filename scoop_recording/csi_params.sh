#!/bin/bash
# CSI settings for the rig, one place.
#
#   sudo ./csi_params.sh rx                       # sniffer (mobile_1/sniffer): arm nexmon
#   sudo ./csi_params.sh rx aa:bb:cc:dd:ee:ff     # ...only frames from that transmitter
#   sudo ./csi_params.sh tx                       # transmitter (infra_1, mobile_2): VHT, 1 SS
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

case "${1:?usage: csi_params.sh rx [mac] | tx}" in
rx)
    MAC=${2:-}
    ARGS=(-c "$CHAN" -C "$CORE" -N "$NSS")
    [ -n "$MAC" ] && ARGS+=(-m "$MAC")
    ip link set "$IFACE" up
    nexutil -I"$IFACE" -s500 -b -l34 -v"$(makecsiparams "${ARGS[@]}")"
    nexutil -I"$IFACE" -m1                      # monitor mode
    echo "armed: ${ARGS[*]}"
    nexutil -I"$IFACE" -k                       # must print $CHAN
    ;;
tx)
    # VHT only, NSS $NSS, MCS $MCS (5 GHz; 0-9 = any). `iw ... bitrates` needs the interface
    # associated; re-run after reconnecting.
    iw dev "$IFACE" set bitrates legacy-5 ht-mcs-5 vht-mcs-5 "$NSS:$MCS"
    iw dev "$IFACE" info | grep -E "channel|txpower"
    iw dev "$IFACE" link | grep -E "tx bitrate" || echo "(not associated yet)"
    ;;
*)  echo "usage: csi_params.sh rx [mac] | tx"; exit 1 ;;
esac
