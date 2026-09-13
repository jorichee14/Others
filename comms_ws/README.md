# comms_ws

One ROS 2 (Humble) workspace for the robot's communications monitoring:
Wi-Fi link quality, throughput, latency, time sync and Nexmon CSI.

Previously six packages (`wifi_monitor`, `wifi_monitor_msgs`, `ntp_monitor`,
`ntp_monitor_msgs`, `wifi_csi`, `wifi_csi_msgs`); now two.

```
comms_ws/
└── src/
    ├── comms_msgs/          all interfaces (ament_cmake)
    │   └── msg/             WifiLinkStatus, IperfResult, PingStat,
    │                        NtpStatus, CsiFrame, CsiStatus
    └── comms/               all nodes (ament_python)
        ├── comms/wifi/      wifi_monitor_node, iperf_runner_node, ping_monitor_node
        ├── comms/ntp/       ntp_client_node, ntp_server_node
        ├── comms/csi/       csi_publisher, csi_monitor
        ├── launch/          one launch file per subsystem (unchanged)
        ├── config/          csi.yaml, ntp_monitor.yaml, rosbag_qos_override.yaml
        ├── scripts/         ntp_bag_postprocess.py (standalone, not installed)
        └── doc/csi.md       Nexmon CSI setup notes
```

## Build

```bash
cd ~/comms_ws
colcon build --symlink-install
source install/setup.bash
```

## Launch

The launch files are still separate — nothing was combined, only the package
name in front of them changed.

```bash
ros2 launch comms session.launch.py           # continuous route session (one bag)
ros2 launch comms survey.launch.py            # stationary point, all nodes
ros2 launch comms burst_monitor.launch.py     # deadline pass -> /wifi/burst
ros2 launch comms wifi_monitor.launch.py      # passive link monitor -> /wifi/status
ros2 launch comms iperf_runner.launch.py      # throughput + RTT     -> /wifi/iperf
ros2 launch comms ping_monitor.launch.py      # ICMP RTT + loss      -> /wifi/ping
ros2 launch comms survey.launch.py            # wifi + iperf + ntp client, one pass
ros2 launch comms ntp_monitor.launch.py       # NTP client + server  -> /ntp/...
ros2 launch comms csi.launch.py               # Nexmon CSI           -> /csi_publisher/csi
```

Examples:

```bash
ros2 launch comms survey.launch.py server_address:=192.168.233.142 run_ping:=true
ros2 launch comms survey.launch.py bidirectional:=true bidir_period_s:=10.0
ros2 launch comms survey.launch.py namespace:=robota
ros2 launch comms ntp_monitor.launch.py mode:=client
ros2 launch comms csi.launch.py config:=/path/to/csi_b.yaml namespace:=robot_b
```

### iperf direction

`reverse:=false` (default) measures uplink, `reverse:=true` downlink.
`bidirectional:=true` alternates the two from a single instance — per test in
periodic mode, or every `bidir_period_s` seconds in continuous mode. It is
sequential rather than `iperf3 --bidir` because Wi-Fi is half-duplex, so
simultaneous two-way traffic contends with itself and measures neither
direction cleanly. Each `IperfResult` records its own direction in `reverse`,
so filter on that field when plotting. Alternating halves the sample rate per
direction.

There is no slot scheduling and no robot-to-robot instance: one iperf client
per robot, against the wired server.

## Individual executables

```bash
ros2 run comms wifi_monitor_node
ros2 run comms burst_monitor_node
ros2 run comms iperf_runner_node
ros2 run comms ping_monitor_node
ros2 run comms ntp_client_node
ros2 run comms ntp_server_node
ros2 run comms csi_publisher
ros2 run comms csi_monitor
```

## Recording a survey

`/ntp/client/status` is TRANSIENT_LOCAL, so rosbag2 needs the QoS override:

```bash
ros2 bag record --qos-profile-overrides-path \
    src/comms/config/rosbag_qos_override.yaml \
    /wifi/status /wifi/iperf /ntp/client/status /tf /tf_static /odom
```

Then correct the timestamps offline:

```bash
python3 src/comms/scripts/ntp_bag_postprocess.py ./bag /wifi/iperf /ntp/client/status
```

## Tests

```bash
colcon test --packages-select comms
# or directly:
cd src/comms && python3 -m pytest test
```

## What changed from the old packages

| Old | New |
| --- | --- |
| `wifi_monitor_msgs/msg/X` | `comms_msgs/msg/X` |
| `ntp_monitor_msgs/msg/NtpStatus` | `comms_msgs/msg/NtpStatus` |
| `wifi_csi_msgs/msg/X` | `comms_msgs/msg/X` |
| `from wifi_monitor import wifi_parsers` | `from comms.wifi import wifi_parsers` |
| `from ntp_monitor.ntp_parser import ...` | `from comms.ntp.ntp_parser import ...` |
| `ros2 launch wifi_csi csi.launch.py` | `ros2 launch comms csi.launch.py` |

Node names, topic names, parameters and message fields are all unchanged, so
existing bags, YAML param files and RViz/plotting configs keep working — only
the message *package* prefix moved.

The merged package is MIT licensed; the NTP nodes were originally Apache-2.0.
