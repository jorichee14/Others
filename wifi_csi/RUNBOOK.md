# Bring-up runbook

Three roles, and nearly every failure in practice comes from running a command
on the wrong one:

| Role | Example host | Interface | Purpose |
|---|---|---|---|
| **Sniffer** | `wicoms-robot1` | `wlan0` (nexmon) | runs `csi_publisher`; never associates |
| **Transmitter** | `wicomsrobot` | `wlp0s20f3` | associated station whose CSI you measure |
| **Transmitter** | `mic-713` | `wlx8876b9eae0ff` | second one, optional |

Reference values from the verified bench: AP `ASUS_70_5G` on `192.168.51.0/24`,
gateway `192.168.51.1`, channel 44 @ 20 MHz. **Verify the channel every session** —
the AP moves clients on its own.

---

## Part 1 — one-time setup

Skip once done, but re-check after any reimage. Without these, every session
starts by re-debugging a silent zero.

### 1a. Sniffer only — let CSI reach the socket

```bash
sudo tee /etc/sysctl.d/99-nexmon-csi.conf <<'EOF'
# The extractor injects dumps as UDP from 10.10.10.10, which has no route back,
# so reverse-path filtering drops them before any socket. Visible in tcpdump,
# invisible to the node. Effective value is max(all, <iface>), so 'all' must be
# 0 too; keep the wired link protected explicitly.
net.ipv4.conf.all.rp_filter = 0
net.ipv4.conf.wlan0.rp_filter = 0
net.ipv4.conf.eth0.rp_filter = 1
EOF
sudo sysctl --system
```

### 1b. Sniffer only — stop the radio being retuned

A scan hops channels and stomps the armed chanspec. **Do not run this on a
transmitter** — it would disable its Wi-Fi.

```bash
sudo tee /etc/NetworkManager/conf.d/99-nexmon.conf <<'EOF'
[keyfile]
unmanaged-devices=interface-name:wlan0
EOF
sudo systemctl restart NetworkManager
sudo systemctl disable --now wpa_supplicant
```

### 1c. Every machine — clock sync (NTP)

`header.stamp` is the kernel receive time on `CLOCK_REALTIME`, the same base as
ROS 2's clock. Cross-machine comparison is meaningless until the clocks agree,
and the Pi has no RTC — on boot it restores a stale on-disk time.

Pick one host as master (best clock, ideally with upstream internet):

```bash
# MASTER
sudo apt install -y chrony
sudo tee -a /etc/chrony/chrony.conf <<'EOF'
allow 192.168.51.0/24
local stratum 8
EOF
sudo systemctl restart chrony
```

```bash
# EVERY OTHER MACHINE (sniffer + transmitters)
sudo apt install -y chrony
sudo tee -a /etc/chrony/chrony.conf <<'EOF'
server 192.168.51.1 iburst minpoll 0 maxpoll 4
makestep 1.0 -1
EOF
sudo systemctl restart chrony
```

Replace `192.168.51.1` with the master's address. `makestep 1.0 -1` is the part
that matters without an RTC: it steps the clock at any time instead of slewing a
large offset over hours.

```bash
chronyc tracking      # 'System time' should settle well under 1 ms
chronyc sources -v
```

Is PTP worth it instead? Only with NIC hardware timestamping:

```bash
ethtool -T eth0       # hardware-transmit/hardware-receive => sub-microsecond
```

### 1d. Router — stop the channel moving

In the ASUS admin page, set the 5 GHz band to a **fixed** channel and **fixed**
control bandwidth, not Auto. Prefer non-DFS (36–48, 149–165 in most regions);
DFS channels can be forced to move by radar detection regardless.

### 1e. Sniffer only — keep ROS off the measured channel

DDS discovery is multicast and will use `wlan0`, putting your own traffic on the
channel you are measuring.

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General>
  <Interfaces><NetworkInterface name="eth0"/></Interfaces>
</General></Domain></CycloneDDS>'
```

Fast DDS needs the equivalent interface whitelist in its XML profile. Verify:
`sudo tcpdump -i wlan0 -n port 7400` should be silent.

---

## Part 2 — every session

### 2a. Transmitters — confirm band and channel

```bash
iw dev <iface> info
```

Every transmitter must report the **same** channel and width; one radio tunes to
one chanspec. If one is on 2.4 GHz, it cannot be captured at all — join the
5 GHz SSID first. Note the MAC (`addr`), ignoring any `P2P-device` entry: that
is a virtual interface carrying no normal traffic.

### 2b. Sniffer — match the config to what you just saw

```bash
cd ~/workspaces/src/wifi_csi
sed -i 's|^    channel: .*|    channel: "44/20"|' config/csi.yaml
sed -i 's|^    mac_filter: .*|    mac_filter: "<mac1>,<mac2>"|' config/csi.yaml
cd ~/workspaces
colcon build --packages-select wifi_csi_msgs wifi_csi
source install/setup.bash
```

Build both packages after any `.msg` change; `wifi_csi` alone is enough for
config or code edits.

### 2c. Transmitters — generate traffic

An associated station that is idle transmits almost nothing. One per
transmitter, left running for the whole capture:

```bash
sudo ping -I <iface> -i 0.01 192.168.51.1
```

`-i 0.01` is 10 ms, giving roughly 100 frames/s; `-i 0.002` gives ~500.
Sub-0.2 s intervals need root. Run it on **both** transmitters or their rates
will not be comparable.

### 2d. Sniffer — launch

```bash
ros2 launch wifi_csi csi.launch.py
```

Wanted: `extractor armed on attempt 1`, then the `/mobileN/csi <- <mac>` map, and
no `0 frames/s` warnings.

### 2e. Sniffer — verify

```bash
ros2 topic hz /mobile1/csi
timeout 5 ros2 topic echo /csi_publisher/csi --field src_mac \
  | grep -E '^[0-9a-f:]{17}$' | sort | uniq -c
ros2 topic echo /csi_publisher/status --once | grep -E 'stamp_lag_sec|dropped_total|channel'
```

The rate should roughly track your ping rate. `stamp_lag_sec` should be small
and flat — climbing means the socket is backing up.

### 2f. Record

The topics are best-effort and `ros2 bag record` defaults to reliable, so it
subscribes and captures **nothing** without an override:

```bash
cat > qos.yaml <<'EOF'
/mobile1/csi:
  reliability: best_effort
  history: keep_last
  depth: 100
/mobile2/csi:
  reliability: best_effort
  history: keep_last
  depth: 100
EOF

ros2 bag record /mobile1/csi /mobile2/csi /csi_publisher/status \
  --qos-profile-overrides-path qos.yaml
```

Always include `/csi_publisher/status`: it carries `stamp_lag_sec`,
`dropped_total` and the chanspec, so a bad session is diagnosable afterwards.

Recording with `trim: false` keeps every slot and is the safer default for a
dataset — the trim is reproducible offline with `usable_slots()`, whereas
discarded slots are not. Just never feed untrimmed frames into amplitude
analysis: the constant firmware slots reach ~11498 against ~1850 for real
subcarriers and will dominate any scaling.

---

## If frames stop

See the ordered bisection in `README.md` ("Debugging zero frames/s"). Do not
guess between causes — they all look identical from the node's logs.
