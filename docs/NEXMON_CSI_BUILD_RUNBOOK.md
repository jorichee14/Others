# Nexmon CSI Node — Complete Build Runbook

**Raspberry Pi 4B / Ubuntu 22.04 · verified end-to-end on hardware · 2026-08-12**

End state: node boots straight into CSI collection with no intervention. Built-in `wlan0` runs patched firmware in monitor mode; a USB dongle carries normal WiFi on a different band; Ethernet is the management path.

---

## Verified platform

| Property | Value |
|---|---|
| Board | Raspberry Pi 4 Model B Rev 1.5 |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 5.15.0-1105-raspi, aarch64 |
| CSI radio | BCM43455c0 (`BCM4345/6`), MAC `d8:3a:dd:19:66:3e` |
| Stock firmware | 7.45.241 (1a2f2fa CY) |
| Patched firmware | 7.45.189 (nexmon.org/csi: a975-1) |
| Driver module | `brcmfmac` (single module; `brcmfmac_wcc` is 6.6+ only) |
| Management dongle | Edimax N150 / RTL8188EU, 2.4 GHz only |
| Firmware path | `brcm/brcmfmac43455-sdio.bin` → `cypress/cyfmac43455-sdio.bin` → `/etc/alternatives/` |
| Network stack | netplan + systemd-networkd |
| CSI channel | 36/80 (chanspec `0xe02a`) |

Results achieved: 15,957 valid 80 MHz frames in 30 s (~530/s) from ambient traffic; 6,164 frames at 102.7 Hz from a controlled `ping -i 0.01` transmitter; DC-null to in-band ratio 30:1; per-subcarrier temporal variation 0.4% in a static room.

---

# Part 1 — Build

## 1.1 Ethernet first

CSI collection takes `wlan0` from the network stack **and** the patched firmware cannot sustain an association (Appendix A15). Work over Ethernet or a serial console.

```bash
echo $SSH_CONNECTION          # 3rd field = server IP
ip route get 1.1.1.1          # prints dev <iface>
```

A successful `ping` proves nothing about which interface carried it (Appendix A20).

Static address for a direct cable, in its own file — never edit `50-cloud-init.yaml` for this:

```bash
sudo tee /etc/netplan/99-wired.yaml >/dev/null <<'EOF'
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: no
      addresses: [192.168.25.2/24]
      optional: true
EOF
sudo chmod 600 /etc/netplan/99-wired.yaml
sudo netplan try
```

No gateway, no nameservers — the default route must stay on WiFi so `apt` works through Part 1. `netplan try` auto-reverts after 120 s if you lose the link.

Interface naming varies by image (`eth0`, `end0`, `enp1s0`). Confirm with `ip -br link` before writing.

**Do not hold the kernel.** Advice to pin 5.4 belongs to the old patched-driver path. The vendor-command path is kernel-agnostic.

## 1.2 Verify the platform

```bash
uname -m && uname -r
lsb_release -d
tr -d '\0' < /proc/device-tree/model; echo
df -h /home
sudo dmesg | grep -i brcmfmac | head -5
sudo dmesg | grep -o 'using [a-z/0-9._-]*43455[a-z/0-9._-]*'
```

`sudo` is required for dmesg — Ubuntu 22.04 sets `kernel.dmesg_restrict=1`.

Expect `aarch64`, a `-raspi` kernel, `BCM4345/6`, ≥3 GB free. **Record the firmware path** — on this node it was `brcm/brcmfmac43455-sdio`.

Inspect the full chain, because Ubuntu layers symlinks:

```bash
ls -l /lib/firmware/brcm/brcmfmac43455-sdio.bin
ls -l /lib/firmware/cypress/cyfmac43455-sdio.bin
update-alternatives --display cyfmac43455-sdio.bin
```

On this node the alternatives group already existed (`cyfmac43455-sdio-minimal.bin` at priority 10, `-standard.bin` at 50, auto mode). That is the good case — no file needs deleting. See Appendix A5 for the other case.

## 1.3 tmux and dependencies

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y tmux
tmux new -s nexmon
```

Detach `Ctrl-b d`, reattach `tmux attach -t nexmon`. If `full-upgrade` installs a new kernel, **reboot before the build**, not during.

```bash
sudo apt install -y git libgmp3-dev gawk qpdf bison flex make autoconf \
  libtool texinfo xxd libnl-3-dev libnl-genl-3-dev bc libssl-dev \
  tcpdump libcap2-bin
```

If apt fails with `Conflicting values set for option Signed-By`, see Appendix A2.

## 1.4 armhf multiarch

The cross-toolchain is a 32-bit ARM binary. Without these it fails with `arm-none-eabi-gcc: not found` **even though the file exists**. Required on every 64-bit image.

```bash
sudo dpkg --add-architecture armhf
sudo apt update
sudo apt install -y libc6:armhf libisl23:armhf libmpfr6:armhf \
  libmpc3:armhf libstdc++6:armhf
sudo ln -s /usr/lib/arm-linux-gnueabihf/libisl.so.23 \
           /usr/lib/arm-linux-gnueabihf/libisl.so.10
sudo ln -s /usr/lib/arm-linux-gnueabihf/libmpfr.so.6 \
           /usr/lib/arm-linux-gnueabihf/libmpfr.so.4
sudo apt install -y python2.7
```

python2.7 is needed by the `b43` beautifier; it is in universe on 22.04.

Verify before the hour-long build:

```bash
dpkg --print-foreign-architectures        # armhf
ls -l /usr/lib/arm-linux-gnueabihf/libisl.so.10 /usr/lib/arm-linux-gnueabihf/libmpfr.so.4
python2.7 --version                        # 2.7.18
```

## 1.5 Build nexmon

```bash
cd ~
git clone --depth=1 https://github.com/seemoo-lab/nexmon.git
cd nexmon
source setup_env.sh
echo "NEXMON_ROOT=$NEXMON_ROOT"
```

**`NEXMON_ROOT` must print a real path.** Empty means every later `cd $NEXMON_ROOT/...` becomes `cd /...` and files land in the wrong place (Appendix A3). **Re-source `setup_env.sh` in every new shell, tmux pane, and after every reboot.**

Confirm the toolchain runs — this validates 1.4:

```bash
$NEXMON_ROOT/buildtools/gcc-arm-none-eabi-5_4-2016q2-linux-armv7l/bin/arm-none-eabi-gcc --version
```

```bash
sed -i '1 s/$/2.7/' $NEXMON_ROOT/buildtools/b43-v3/debug/b43-beautifier
make 2>&1 | tee ~/nexmon-make.log
```

30–60 minutes; it walks every supported chip. Then:

```bash
ls -l $NEXMON_ROOT/buildtools/b43-v3/assembler/b43-asm.bin
ls -l $NEXMON_ROOT/firmwares/bcm43455c0/7_45_189/
grep -inE "error 1|error 2|\*\*\*" ~/nexmon-make.log | head
```

`b43-asm.bin` must exist. The firmware directory needs `ucode.bin`, `templateram.bin`, `flashpatches.bin`, `definitions.mk`. Grep should return nothing.

## 1.6 nexutil and makecsiparams

```bash
cd $NEXMON_ROOT/utilities/nexutil
sudo -E make install USE_VENDOR_CMD=1
sudo setcap cap_net_admin+ep /usr/bin/nexutil
nexutil -V
```

Two non-negotiable details:

- **`sudo -E`** preserves `NEXMON_ROOT`. Plain `sudo` drops it and the build fails.
- **`USE_VENDOR_CMD=1`** routes nexutil through the stock driver's Broadcom vendor-command interface. Without it every IOCTL is rejected. This flag is why no patched `brcmfmac` is needed and kernel version stops mattering.

`ld` warnings about `getaddrinfo` in static binaries are cosmetic. `nexutil -V` querying the live chip (`chipnum 0x4345`, `chiprev 0x6`) proves the vendor path works.

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189
pwd                                # MUST be the full path before cloning
git clone --depth=1 https://github.com/seemoo-lab/nexmon_csi.git
cd nexmon_csi
ls Makefile.rpi ../version.mk
ls $NEXMON_ROOT/firmwares/bcm43455c0/7_45_189/definitions.mk
cd utils/makecsiparams
make
sudo install -m 0755 makecsiparams /usr/local/bin/makecsiparams
makecsiparams -c 157/80 -C 1 -N 1
```

**Do not run `make install` in makecsiparams** — on aarch64 it falls into an `adb push` branch meant for Android (Appendix A4).

## 1.7 Build the firmware

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/nexmon_csi
make -f Makefile.rpi 2>&1 | tee ~/fw-build.log
ls -l *.bin
grep -iE "hunk|FAILED|Error [12]" ~/fw-build.log | head
```

Expect `brcmfmac43455-sdio.bin` at ~617 KB, no grep hits. The default `all:` target has no `is-rpi` dependency, so the guard that aborts install targets never fires (Appendix A18).

`Hunk #N FAILED` still produces a `.bin` — a broken one. Grepping the log matters more than seeing the file.

## 1.8 Install the firmware

```bash
sudo mkdir -p /lib/firmware/nexmon
sudo cp brcmfmac43455-sdio.bin /lib/firmware/nexmon/
sudo update-alternatives --install /lib/firmware/cypress/cyfmac43455-sdio.bin \
  cyfmac43455-sdio.bin /lib/firmware/nexmon/brcmfmac43455-sdio.bin 100
sudo update-alternatives --set cyfmac43455-sdio.bin \
  /lib/firmware/nexmon/brcmfmac43455-sdio.bin
```

Priority 100 beats the stock candidates. `--set` switches to **manual mode**, preventing a `linux-firmware-raspi` update from reasserting stock.

Verify the whole chain:

```bash
update-alternatives --display cyfmac43455-sdio.bin | head -4
cmp "$(readlink -f /lib/firmware/brcm/brcmfmac43455-sdio.bin)" \
    /lib/firmware/nexmon/brcmfmac43455-sdio.bin && echo "FIRMWARE OK"
```

## 1.9 Remove the duplicate supplicant

netplan generates a **per-interface** unit, `netplan-wpa-wlan0.service`. Ubuntu also ships a generic `wpa_supplicant.service`. Both running on one interface means neither wins (Appendix A7).

```bash
sudo systemctl stop wpa_supplicant.service
sudo systemctl disable wpa_supplicant.service
```

Do this once; it persists and belongs in the image.

---

# Part 2 — Management dongle

The patched firmware cannot hold an association, so a node in CSI mode is unreachable over its own WiFi. A second radio solves this permanently.

**Pick 2.4 GHz** so the dongle's own traffic never lands in your 5 GHz CSI band.

## 2.1 Driver

Ubuntu's staging `r8188eu` is broken on 5.15 — it registers `phy0` but fails `nl80211: Failed to set interface to mode 2: -19 (No such device)`, and `iw` also returns `-19`. Replace it:

```bash
sudo apt install -y dkms build-essential git bc
sudo apt install -y linux-headers-$(uname -r)
ls -d /lib/modules/$(uname -r)/build          # must resolve

cd ~
git clone https://github.com/aircrack-ng/rtl8188eus.git
cd rtl8188eus
sudo sh -c "echo 'blacklist r8188eu' > /etc/modprobe.d/realtek.conf"
make -j4
sudo make install
sudo modprobe -r r8188eu 2>/dev/null
sudo modprobe 8188eu
iw dev                                        # must now list the dongle
```

Note: `raspberrypi-kernel-headers` is a Raspberry Pi OS package and does not exist on Ubuntu.

## 2.2 Find what it can actually see

```bash
sudo ip link set wlx... up
sudo iw dev wlx... scan | grep -E "SSID:" | sort -u
```

RTL8188EU is **2.4 GHz only** — 5 GHz SSIDs can never appear. Configure only SSIDs present in this list; entries that can't match make the supplicant cycle forever.

## 2.3 Bypass netplan for WiFi

netplan on this version emits a malformed `ssid=P"ASUS_70"` — with or without YAML quotes — so the supplicant searches for a network that doesn't exist and scans forever (Appendix A21). Use a hand-written config and unit:

```bash
sudo mkdir -p /etc/wpa_supplicant
wpa_passphrase "YOUR_SSID" "yourpassword" | sudo tee /etc/wpa_supplicant/dongle.conf
sudo sed -i '1i ctrl_interface=/run/wpa_supplicant\ncountry=TW' /etc/wpa_supplicant/dongle.conf
sudo chmod 600 /etc/wpa_supplicant/dongle.conf
```

`wpa_passphrase` stores the hashed PSK, not the plaintext.

Substitute your own dongle's interface name (MAC-derived, different per dongle — `ip -br link | awk '/^wlx/{print $1}'`):

```bash
sudo tee /etc/systemd/system/dongle-wifi.service >/dev/null <<'EOF'
[Unit]
Description=WiFi for USB dongle
After=network.target
Wants=network.target

[Service]
Type=simple
ExecStartPre=/sbin/ip link set wlx08beac48300f up
ExecStart=/sbin/wpa_supplicant -i wlx08beac48300f -c /etc/wpa_supplicant/dongle.conf -Dnl80211
ExecStartPost=/bin/sh -c 'sleep 8; /sbin/dhclient wlx08beac48300f'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now dongle-wifi.service
sleep 20
ip -br addr show wlx08beac48300f
```

Then **remove the dongle's block from netplan entirely** so it stops generating a competing broken config.

## 2.4 Free wlan0

Comment out or delete the `wlan0:` entry under `wifis:` in `50-cloud-init.yaml`. It is dedicated to CSI now.

Since cloud-init warns its file is regenerated:

```bash
sudo sh -c 'echo "network: {config: disabled}" > /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg'
```

---

# Part 3 — Pin the interface name

With two radios, whichever enumerates first can claim `wlan0`. On this node the built-in came up as `wlan1` after a reboot, `csi.service` failed its device dependency, and `nexutil` reported `error getting interface index` (Appendix A22).

Both mechanisms, for redundancy — substitute your own built-in MAC:

```bash
sudo tee /etc/systemd/network/10-wlan0.link >/dev/null <<'EOF'
[Match]
MACAddress=d8:3a:dd:19:66:3e

[Link]
Name=wlan0
EOF

sudo tee /etc/udev/rules.d/70-wlan0.rules >/dev/null <<'EOF'
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="?*", ATTR{address}=="d8:3a:dd:19:66:3e", NAME="wlan0"
EOF

sudo udevadm control --reload
sudo update-initramfs -u -k all
```

To fix a live mis-naming without rebooting:

```bash
sudo ip link set wlan1 down
sudo ip link set wlan1 name wlan0
```

---

# Part 4 — Autonomous boot

The CSI extractor config (`-s500`) and monitor mode (`-m1`) live in **chip RAM**. Both reset on every driver reload and reboot. There is no NVRAM path — persistence means re-arming at boot.

## 4.1 The arming script

Includes a retry loop: on two separate reboots the service reported success with monitor mode 1 and the correct chanspec, yet emitted nothing until `-s500` was reissued (Appendix A23).

```bash
sudo tee /usr/local/bin/csi-start >/dev/null <<'EOF'
#!/bin/bash
set -e
CHAN="${1:-36/80}"
MAC="${2:-}"
ARGS=(-c "$CHAN" -C 1 -N 1)
[ -n "$MAC" ] && ARGS+=(-m "$MAC")

systemctl stop netplan-wpa-wlan0.service 2>/dev/null || true
printf '[Match]\nName=wlan0\n\n[Link]\nUnmanaged=yes\n' \
  > /etc/systemd/network/10-csi-unmanaged.network
networkctl reload

for i in $(seq 30); do ip link show wlan0 &>/dev/null && break; sleep 1; done
ip link set wlan0 up
sleep 2
iw dev wlan0 set power_save off || true

PARAMS="$(makecsiparams "${ARGS[@]}")"
for attempt in 1 2 3 4 5; do
  nexutil -Iwlan0 -s500 -b -l34 -v"$PARAMS"
  nexutil -Iwlan0 -m1
  sleep 2
  if timeout 4 tcpdump -i wlan0 dst port 5500 -c 1 >/dev/null 2>&1; then
    echo "armed on attempt $attempt: $(nexutil -Iwlan0 -m) $(nexutil -Iwlan0 -k)"
    exit 0
  fi
  sleep 3
done
echo "WARNING: armed but no CSI packets seen after 5 attempts"
echo "  $(nexutil -Iwlan0 -m)  $(nexutil -Iwlan0 -k)"
EOF
sudo chmod +x /usr/local/bin/csi-start
```

The `Unmanaged=yes` drop-in stops networkd touching `wlan0`. **Never stop `systemd-networkd` itself** — that drops `eth0` and your session with it (Appendix A24).

## 4.2 The unit

```bash
sudo tee /etc/systemd/system/csi.service >/dev/null <<'EOF'
[Unit]
Description=Arm nexmon CSI extractor on wlan0
Requires=sys-subsystem-net-devices-wlan0.device
After=sys-subsystem-net-devices-wlan0.device systemd-networkd.service
Conflicts=netplan-wpa-wlan0.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/csi-start 36/80
ExecStop=-/usr/bin/nexutil -Iwlan0 -m0
ExecStop=-/bin/rm -f /etc/systemd/network/10-csi-unmanaged.network
ExecStop=-/usr/bin/networkctl reload

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable csi.service
```

Add a MAC filter to `ExecStart` once you have a dedicated transmitter — but note a node with a filter looks dead whenever that transmitter is idle.

## 4.3 Verify across a cold boot

```bash
sudo reboot
```

```bash
ip -br link                                        # wlan0 on the built-in MAC
journalctl -u csi.service -b --no-pager | tail -3  # "armed on attempt N"
ip -br addr show wlx08beac48300f                   # dongle has an IP
sudo dmesg | grep -i "Firmware: BCM" | tail -1     # nexmon.org/csi
nexutil -Iwlan0 -m                                 # 1
sudo timeout 15 tcpdump -i wlan0 dst port 5500 -c 10
```

All six must pass before you trust the node.

---

# Part 5 — Mode switching

Switching requires a **firmware swap**, not just monitor mode.

```bash
# → CSI
sudo update-alternatives --set cyfmac43455-sdio.bin \
  /lib/firmware/nexmon/brcmfmac43455-sdio.bin
sudo modprobe -r brcmfmac && sudo modprobe brcmfmac
for i in $(seq 20); do ip link show wlan0 &>/dev/null && break; sleep 1; done
sleep 5
sudo dmesg | grep -i "Firmware: BCM" | tail -1     # nexmon.org/csi
sudo systemctl start csi.service
```

```bash
# → normal WiFi on wlan0
sudo systemctl stop csi.service
sudo update-alternatives --set cyfmac43455-sdio.bin \
  /lib/firmware/cypress/cyfmac43455-sdio-standard.bin
sudo modprobe -r brcmfmac && sudo modprobe brcmfmac
for i in $(seq 20); do ip link show wlan0 &>/dev/null && break; sleep 1; done
sleep 5
sudo dmesg | grep -i "Firmware: BCM" | tail -1     # 7.45.241
sudo systemctl restart netplan-wpa-wlan0.service
sleep 12
sudo dhclient -v wlan0
```

**Four rules:**

1. **`--set`, never `--auto`.** Auto picks by priority, and the patched blob at 100 outranks stock at 10 and 50 (Appendix A6).
2. **Verify from the dmesg banner, checking its timestamp is fresh.** `--display` reports intent; the banner reports what loaded (Appendix A9).
3. **Wait for `wlan0` after `modprobe`** — it is destroyed and recreated (Appendix A8).
4. **`nexutil -m` is the only truthful monitor state.** `iwconfig` says `Managed` either way (Appendix A12).

---

# Part 6 — Collect

## 6.1 Transmitter

CSI is produced **per received OFDM frame**. Observed on the same working config: 15,982 frames in 30 s on a busy channel, 3 frames in 30 s on an idle one.

- **5 GHz only.** 2.4 GHz beacons are DSSS and produce no channel estimate (Appendix A11).
- Pin the AP: fixed channel, fixed width, auto-channel off, avoid DFS (52–144).
- From a device associated on that channel:

```bash
ip route get 192.168.51.1        # confirm it goes out the WiFi iface
ping -i 0.01 192.168.51.1        # ~100 frames/s
```

## 6.2 Capture

```bash
nexutil -Iwlan0 -s500 -b -l34 \
  -v"$(makecsiparams -c 36/80 -C 1 -N 1 -m <tx-mac>)"
nexutil -Iwlan0 -m1
sudo timeout 60 tcpdump -i wlan0 dst port 5500 -w ~/run.pcap
```

Working output: `IP 10.10.10.10.5500 > 255.255.255.255.5500: UDP, length 1042`. The source is synthetic, stamped by the firmware.

| Bandwidth | Subcarriers | Payload |
|---|---|---|
| 20 MHz | 64 | 274 B |
| 40 MHz | 128 | 530 B |
| 80 MHz | 256 | 1042 B |

`[|llc]` lines in unfiltered captures are the raw 802.11 frames the CSI came from — tcpdump misparsing, not an error.

**The MAC filter is essential.** Without it you collect from dozens of transmitters at different bandwidths, and the averaged profile is meaningless (Appendix A25).

## 6.3 Parse

Header layout, corrected from observed data (Appendix A13):

| Offset | Size | Field |
|---|---|---|
| 0–1 | 2 | magic `0x1111` |
| 2 | 1 | RSSI (int8) |
| 3 | 1 | frame control |
| 4–9 | 6 | source MAC |
| 10–11 | 2 | sequence number |
| 12–13 | 2 | core / spatial stream |
| 14–15 | 2 | chanspec |
| 16–17 | 2 | chip version |
| 18+ | 4×N | subcarriers, int16 pairs (imag, real) |

```bash
cat > ~/csiview.py <<'PYEOF'
import struct, sys, numpy as np
from collections import Counter
S = " .:-=+*#%@"
path = sys.argv[1]
mac = sys.argv[2].lower() if len(sys.argv) > 2 else None
f = open(path, 'rb'); f.read(24)
fr, meta = [], []
while True:
    ph = f.read(16)
    if len(ph) < 16: break
    _, _, cl, _ = struct.unpack('<IIII', ph)
    pkt = f.read(cl)
    if len(pkt) < cl: break
    p = pkt[14 + (pkt[14] & 0xF) * 4 + 8:]
    if len(p) != 1042 or struct.unpack_from('<H', p, 0)[0] != 0x1111: continue
    src = p[4:10].hex(':')
    if mac and src != mac: continue
    r = np.frombuffer(p[18:], dtype='<i2').astype(np.float32)
    fr.append((r[1::2] + 1j * r[0::2]).astype(np.complex64))
    meta.append((struct.unpack_from('<b', p, 2)[0], p[3], src))
if not fr: sys.exit('no frames (wrong MAC? empty capture?)')
X = np.stack(fr)
prof = np.abs(X).mean(axis=0).copy(); prof[0:2] = 0
thr = prof.max() * 0.05
loud = np.where(prof > thr)[0]; lo, hi = int(loud.min()), int(loud.max())
nulls = [i for i in range(lo, hi + 1) if prof[i] <= thr]
nm = float(prof[nulls].mean()) if nulls else 0.0; im = float(prof[loud].mean())
print('frames        :', X.shape)
print('frame types   :', [(hex(k), v) for k, v in Counter(m[1] for m in meta).most_common(4)])
print('top sources   :', Counter(m[2] for m in meta).most_common(3))
print('RSSI          :', min(m[0] for m in meta), 'to', max(m[0] for m in meta), 'dBm')
print('occupied span :', lo, '..', hi, '(%d carriers)' % loud.size)
print('interior nulls:', nulls)
print('in-band / null: %.0f / %.1f = %s:1  %s' % (
    im, nm, '>1000' if nm < .5 else '%.1f' % (im / nm),
    'PARSE OK' if (nm < .5 or im / nm > 10) else 'SUSPECT'))
B = X[:, lo:hi + 1]; amp = np.abs(B); W = min(96, B.shape[1])
def line(v, a, b):
    v = np.asarray(v, float)[np.linspace(0, len(v) - 1, W).astype(int)]
    if b <= a: return S[0] * W
    q = ((v - a) / (b - a) * 9).clip(0, 9)
    return ''.join(S[int(round(x))] for x in q)
print('\nsubcarrier profile (mean |CSI|)')
print(' ', line(amp.mean(0), amp.mean(0).min(), amp.mean(0).max()))
print('\ntemporal std per subcarrier (motion shows here)')
print(' ', line(amp.std(0), amp.std(0).min(), amp.std(0).max()))
dev = np.abs(amp - amp.mean(0, keepdims=True))
a2, b2 = np.percentile(dev, 2), np.percentile(dev, 98)
print('\nmotion waterfall: 20 time slices (top = oldest) x %d subcarriers' % W)
ri = np.linspace(0, dev.shape[0] - 1, 21).astype(int)
for i in range(20):
    print('  %2d |%s|' % (i, line(dev[ri[i]:max(ri[i+1], ri[i]+1)].mean(0), a2, b2)))
np.save(path.replace('.pcap', '.npy'), B)
print('\nsaved', B.shape, '->', path.replace('.pcap', '.npy'))
PYEOF
python3 ~/csiview.py ~/run.pcap <tx-mac>
```

numpy is preinstalled on Ubuntu 22.04; no venv needed.

## 6.4 Acceptance test

The verdict is the **interior nulls** line: exactly one, at the centre of the occupied span. That is the DC null the 802.11 standard requires to be empty, and its position confirms alignment.

Verified good capture (run6, single transmitter):

```
occupied span : 132 .. 188 (57 carriers)
interior nulls: [160]
in-band / null: 1856 / 61 = 30.3:1   PARSE OK
```

160 sits exactly ±28 from both edges — the textbook 20 MHz allocation. Slots **127–131 are constant firmware fields**, not subcarriers: `std = 0.0` and one unique value across 6,164 frames, with magnitudes up to 11,498 against ~1,850 for real ones (Appendix A26). Drop them; drop the DC null too, leaving exactly 56 subcarriers.

```python
X = np.load('run.npy')[:, 5:]     # raw 132..188
X = np.delete(X, 28, axis=1)      # remove DC null at raw 160
```

## 6.5 Discriminating motion from gain

The test that matters for sensing:

```bash
python3 - <<'EOF'
import numpy as np
a = np.abs(np.load('run.npy'))[:, 5:]
fm = a.mean(1); z = (fm - fm.mean())/fm.std()
hits = np.where(z < -3)[0]
print('frame-mean swing %.3f%%' % (100*fm.std()/fm.mean()))
if hits.size:
    runs = np.split(hits, np.where(np.diff(hits) > 1)[0]+1)
    for r in runs[:8]:
        print('  %d frames, min z=%.1f' % (len(r), z[r].min()))
EOF
```

Then for any event found, check whether it was frequency-selective:

- **All subcarriers fall together, none rise** → broadband: rate adaptation, AGC, or power change.
- **Some rise while others fall** → frequency-selective: something moved in the propagation path.

Verified example from run6: a 0.7 s event at t≈19 s dropped **0 of 57** subcarriers upward, uniformly −1%. Gain, not motion. That is what a clean null case looks like.

---

# Part 7 — Replicate

Everything durable is on disk: firmware blob, alternatives in manual mode, `csi-start`, `csi.service`, `dongle-wifi.service`, the interface-naming rules, and the disabled generic supplicant.

```bash
sudo shutdown -h now
# card into laptop
lsblk                                    # IDENTIFY CAREFULLY
sudo dd if=/dev/sdX of=~/csi-node-ref.img bs=4M status=progress
sudo sync
```

> `dd` has no confirmation. Pointing `of=` at your own disk destroys it.

**Per-node edits after cloning**, in order of how badly they bite:

1. **Built-in MAC** in `/etc/systemd/network/10-wlan0.link` **and** `/etc/udev/rules.d/70-wlan0.rules`. Different on every Pi; missing this produces the `wlan1` failure.
2. **Dongle interface name** in `dongle-wifi.service` — MAC-derived, so different per dongle. Better: discover it, `ip -br link | awk '/^wlx/{print $1}'`, so the image works unmodified.
3. Hostname and the `eth0` static in `99-wired.yaml`.

All nodes must be the same Pi model. A 114 GB card yields a 114 GB image — gzip or shrink first.

**Time sync matters once nodes are compared.** The Pi has no RTC; without NTP the clock reverts to a stale on-disk value (observed drifting two months). Run NTP on the Orin and sync every node over Ethernet.

---

# Appendix — Issues encountered

Ordered by cost. Each records the symptom, the real cause, and the fix.

## A6. `update-alternatives --auto` does not mean "revert to stock" ★ most expensive

**Symptom:** after switching "back to stock", WiFi associated (`wpa_state=COMPLETED`, −32 dBm) but DHCP never completed — three `DHCPDISCOVER` with no `DHCPOFFER`, on **two different routers on two different bands**. A manual static address could not even ARP the gateway. Carrier dropped and re-gained every ~62 s.

**Wrongly pursued:** router MAC filtering, AP isolation, DHCP pool exhaustion, netplan hex-PSK handling, missing `Network File` in networkd.

**Actual cause:** `--auto` selects the **highest-priority** alternative, not the distribution default. The patched blob was priority 100 against stock at 10 and 50, so auto kept selecting nexmon. Every "stock firmware" test was still running CSI firmware.

**Detection:** the dmesg banner, and only the banner. `--display` reported "auto mode" reassuringly while pointing at the patched blob.

**Fix:** always `--set` with an explicit path, both directions. Confirmed immediately: `DHCPOFFER` → `DHCPACK` on the first attempt.

**Lesson:** the routers were never at fault. Verify the chip's actual state, not the configuration's intent.

## A15. Patched firmware cannot sustain an association

Association succeeds and authenticates, but carrier drops every ~62 s and DHCP never completes. The nexmon patch modifies the receive path; association survives the initial handshake but not ongoing keepalive/rekey.

**Consequence:** monitor mode alone is not a mode switch — moving between CSI and normal WiFi needs a firmware swap plus driver reload (~10 s each way). This is the entire reason for the management dongle in Part 2.

## A21. netplan emits a malformed SSID

**Symptom:** the dongle's supplicant stuck at `wpa_state=SCANNING` forever despite the network being visible in a scan.

**Cause:** the generated `/run/netplan/wpa-*.conf` contained `ssid=P"ASUS_70"` — a stray `P` prefix — with or without quotes in the YAML. The supplicant searched for a network literally named `P"ASUS_70"`.

**Fix:** bypass netplan for WiFi entirely; hand-written `wpa_supplicant.conf` plus your own systemd unit (§2.3). More reproducible across a fleet anyway.

## A22. Interface naming race with two radios

**Symptom:** after a reboot the built-in radio came up as `wlan1`, `csi.service` failed its `Requires=sys-subsystem-net-devices-wlan0.device`, and `nexutil` reported `error getting interface index` with a nonsense chanspec (`0x6863, 6g85/160`).

**Cause:** the USB dongle enumerated first and the naming order shifted.

**Fix:** pin by MAC using both a systemd `.link` file and a udev rule (Part 3), then `update-initramfs -u -k all`. Verified across a reboot.

## A23. Extractor arms but does not emit after boot

**Symptom:** twice after reboot, `csi.service` reported success, `nexutil -m` returned 1, chanspec was correct — and no CSI packets appeared. Reissuing `-s500` by hand fixed it immediately both times.

**Likely cause:** `-s500` landing before the interface has fully settled after the SDIO probe.

**Fix:** the retry loop in `csi-start` (§4.1) reissues up to five times and stops as soon as a packet is seen, logging which attempt succeeded.

## A7. Two wpa_supplicant instances competing

`netplan-wpa-wlan0.service` and the generic `wpa_supplicant.service` both on `wlan0`; neither wins. Introduced by following advice to start the generic unit — wrong for a netplan+networkd system. Stop and disable the generic one.

## A3. `$NEXMON_ROOT` empty in a new shell

`cd: /patches/bcm43455c0/7_45_189: No such file or directory`, then a clone that appeared to succeed but landed in the wrong place; repeating produced a nested second clone. `setup_env.sh` only exports into the shell that sources it. Always `echo "$NEXMON_ROOT"` and `pwd` before cloning. `Makefile.rpi` uses `include ../version.mk` and `$(FW_PATH)/definitions.mk` — only the correct location resolves them.

## A2. ROS 2 apt source declared twice

`E: Conflicting values set for option Signed-By`. Two declarations of the same repo — `ros2.list` (legacy keyring path) and `ros2.sources` (deb822, key inlined, a symlink into `/usr/share/ros-apt-source/`). `grep` alone missed it; `ls -la /etc/apt/sources.list.d/` revealed the symlink. Disable the legacy `.list`; keeping it would break again on the next `ros2-apt-source` update.

## A10. `-b 0x88` is a frame-control filter, not a MAC filter

`makecsiparams -h`: *"filter frames starting with byte"*. `0x88` is QoS Data, which correctly rejects beacons (`0x80`) — and, as it turned out, nearly everything on a channel dominated by Block Acks (`0x94`, 6,397 of 6,882 frames). Drop it while debugging; use the MAC filter instead, which doesn't care about frame type.

## A11. 2.4 GHz beacons produce no CSI

Channel 1 showed steady traffic with readable SSIDs in hex but zero CSI packets. 2.4 GHz beacons are typically 1 Mbps DSSS — no OFDM channel estimate exists. Moving to 5 GHz produced packets immediately.

## A26. Constant firmware fields at the start of the payload

Slots 127–131 read up to 11,498 against ~1,850 for real subcarriers, with `std = 0.0` and one unique value across 6,164 frames. A real subcarrier cannot do that — thermal noise alone guarantees variation. They are fixed fields, not measurements.

They also distorted every plot: the linear colour ramp fitted to a value that wasn't signal, crushing real structure into flat purple. **Detection is one command** — `std == 0` across a capture — and worth running on every new node.

## A25. Mixed transmitters smear the profile

An unfiltered capture gave 4,653 frames from dozens of sources (top three: 358, 323, 318) with 3,481 beacons. Each transmitter uses a different 20 MHz subchannel within the 80 MHz window, so the averaged amplitude profile smears and the interior-null test becomes meaningless. Filtering to one MAC restored a clean single null at 160.

## A5. `update-alternatives` silently no-ops on a real file

Not hit on this node (Ubuntu already had an alternatives group) but reproduced in testing. If the target is a **regular file** rather than a symlink, `--install` prints `not replacing ... with a link`, marks the group broken, and **still reports the new alternative as selected** while the stock blob stays on disk. Install appears to succeed; you get zero CSI with nothing in any log.

Fix: back up twice, `rm` the target so alternatives can own the path, then verify byte-for-byte with `cmp`.

## A13. The magic value is 2 bytes, not 4

Parser reported `magic 0x94ad1111`, `0x94de1111`, `0x94ce1111` — varying per packet. The header is `<H` magic `0x1111`, then RSSI (int8) and frame control as separate bytes. Header total is still 18 bytes, so subcarrier counts were unaffected; only a strict 4-byte magic check breaks, rejecting every genuine packet.

Bonus: per-frame RSSI (−87 to −28 dBm observed) lets you discard measurements at the noise floor.

## A14. Padded bandwidth and mismatched subcarrier counts

`ValueError: all input arrays must have the same shape` when stacking — transmitters on the channel used different bandwidths and the chip reports each frame's actual width. Group by subcarrier count before stacking.

At 80 MHz with a 20 MHz transmitter, only ~64 of 256 slots carry energy, positioned by which subchannel was used. Also seen: 25 frames of 1046 bytes among 15,957 of 1042 — a 4-byte trailer. Filter on exactly 1042.

## A16. `fftshift` may be wrong for your build

A null-to-in-band ratio of **19.3** — inverted, energy in the slots that should be empty — comes from applying `np.fft.fftshift` to data already in frequency order. Profile mean amplitude by raw index before shifting to check.

## A9. Stale dmesg banner

After `modprobe -r`/`modprobe`, the banner showed the previous firmware because `dmesg` ran before the reload finished writing. `sleep 5` first, and read the bracketed kernel timestamp to confirm the line is fresh.

## A8. `modprobe -r` destroys `wlan0`

`Cannot find device "wlan0"` and `receive_packet failed: Network is down` from `dhclient` immediately after a driver reload. Wait for the interface rather than guessing a sleep:

```bash
for i in $(seq 20); do ip link show wlan0 &>/dev/null && break; sleep 1; done
```

## A24. Never stop `systemd-networkd`

An early version of the mode-switch stopped `systemd-networkd` wholesale to release `wlan0`. That drops `eth0` — the management link and, on a multi-node rig, the CSI transport. Stop only the per-interface `netplan-wpa-wlan0.service` and use an `Unmanaged=yes` drop-in scoped by name.

## A1. `dmesg` requires sudo

Ubuntu 22.04 sets `kernel.dmesg_restrict=1`. Unprivileged `dmesg` returns `read kernel buffer failed: Operation not permitted`.

## A4. `make install` in makecsiparams targets Android

Only does a `cp` when `uname -m` is `armv6l`/`armv7l`; on aarch64 it falls into an `adb push` branch. Use `sudo install -m 0755 makecsiparams /usr/local/bin/`.

## A12. `iwconfig` always reports "Managed"

The driver is never informed of the chip's monitor state. Use `nexutil -Iwlan0 -m` — the only truthful source. This trips up nearly everyone.

## A17. Extractor config does not persist

`-s500` and `-m1` write to chip RAM; every driver reload and reboot clears them. The firmware selection and the `Unmanaged=yes` drop-in *do* persist. Re-arm via `csi.service`.

## A18. `is-rpi` guard — correction

Documentation says Ubuntu's arm64 kernel omits the `Model` line from `/proc/cpuinfo`, making `Makefile.rpi`'s `is-rpi` guard abort on real hardware. **On this system it was present** and the guard would have passed. The runbook uses the default `all:` target and a manual install regardless — works either way and adds verification the upstream target lacks. Don't patch the Makefile; the edit is lost on the next `git pull`.

## A19. Sample rate is traffic-bound

Observed on the same working config: 15,982 frames in 30 s, then 3 frames in 30 s, then 1 in 15 s. Few packets is **not** a fault by itself. Distinguish with an unfiltered capture — `[|llc]` frames present means the radio hears the channel and there is nothing measurable.

## A20. False connectivity checks

`ip -br addr show eth0` returned `Device "eth0" does not exist` while `ping archive.ubuntu.com` succeeded — over WiFi. Interface naming varies (`eth0`, `end0`, `enp1s0`). Use `ip route get 1.1.1.1` (prints `dev <iface>`) and `echo $SSH_CONNECTION`, never a bare ping.

## A27. Staging `r8188eu` is broken on 5.15

Registers `phy0` and advertises full nl80211 capability, then fails `Failed to set interface to mode 2: -19 (No such device)`; `iw ... set type managed` fails identically. Replace with out-of-tree `aircrack-ng/rtl8188eus` (§2.1). Also produces spurious `ctrl_iface exists and seems to be in use` errors from unclean exits — clear `/run/wpa_supplicant/<iface>` between attempts.

**For a fleet:** DKMS means rebuilding on every kernel update on every node. An MT7612U (dual-band) or MT7601U (2.4 GHz) has in-kernel support and needs no maintenance.
