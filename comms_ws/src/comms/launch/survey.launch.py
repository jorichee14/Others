"""Full Wi-Fi survey stack in one launch.

Nodes on the robot:
  * wifi_monitor  -> /wifi/status        (5 Hz passive: RSSI, MCS/rate, retries)
  * iperf_runner  -> /wifi/iperf         (continuous throughput + RTT via ss) [run_iperf]
  * ping_monitor  -> /wifi/ping          (RTT + loss)                         [run_ping]
  * ntp_client    -> /ntp/client/status  (clock offset for timestamp fixup)   [run_ntp]
  * burst_monitor -> /wifi/burst         (deadline compliance)                [run_burst]

Continuous iperf now reports RTT itself (via ss), so ping is OFF by default
-- it would just duplicate the RTT. Enable run_ping only when you want RTT +
loss WITHOUT running iperf (i.e. run_iperf:=false), e.g. monitoring latency
during real operation without saturating the link.

The NTP client only reads the local chrony/ntpd daemon, so it costs nothing
on the air and runs alongside the survey. It needs chrony already syncing to
the time server -- check `chronyc tracking` first. The NTP *server* node runs
on the time server itself (ntp_monitor.launch.py mode:=server), not here.

burst_monitor answers a different question from iperf: not "how much does
this link move in 10 s" but "will a 40 kB feature map arrive within 50 ms".
It is OFF by default because it wants an IDLE link -- run it as its own pass
with run_iperf:=false, since a saturated link would only measure the queue
iperf built. It needs burst_echo_server.py running on the target:

    python3 burst_echo_server.py --port 5600

The server host is the iperf3 server, the ping target and the echo target;
pass it once.
Record with pose to map it offline (the NTP topic is TRANSIENT_LOCAL, hence
the QoS override):

    ros2 bag record --qos-profile-overrides-path \
        $(ros2 pkg prefix comms)/share/comms/config/rosbag_qos_override.yaml \
        /wifi/status /wifi/iperf /ntp/client/status /tf /tf_static /odom
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _str(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def _dbl(name: str) -> ParameterValue:
    """Force a DOUBLE parameter (integer-looking values would be rejected)."""
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "namespace", default_value="",
            description="Namespace for the nodes and topics ('' = none).",
        ),
        DeclareLaunchArgument(
            "server_address", default_value="192.168.233.142",
            description="iperf3 server + ping target (the static wired laptop).",
        ),
        DeclareLaunchArgument(
            "server_port", default_value="5201",
            description="iperf3 server port (must match `iperf3 -s -p N`).",
        ),
        DeclareLaunchArgument(
            "interface", default_value="",
            description="Wireless interface ('' = auto-detect).",
        ),
        DeclareLaunchArgument(
            "continuous", default_value="true",
            description="true = one long iperf3, a sample per "
            "continuous_interval_s (moving survey). false = periodic tests, "
            "ONE aggregate message per test with slow-start omitted "
            "(stationary spots).",
        ),
        DeclareLaunchArgument(
            "duration_s", default_value="10.0",
            description="Periodic mode: measured window per test.",
        ),
        DeclareLaunchArgument(
            "interval_s", default_value="0.0",
            description="Periodic mode: gap between tests. 0 = back-to-back.",
        ),
        DeclareLaunchArgument(
            "omit_s", default_value="1.0",
            description="Periodic mode: seconds of TCP slow-start discarded "
            "before the average is taken (iperf3 -O).",
        ),
        DeclareLaunchArgument(
            "reverse", default_value="false",
            description="Direction when not bidirectional: false = uplink, "
            "true = downlink.",
        ),
        DeclareLaunchArgument(
            "protocol", default_value="tcp",
            description="'tcp' gives capacity + RTT; 'udp' gives jitter, "
            "loss and packet counts instead (rtt_ms_* go NaN).",
        ),
        DeclareLaunchArgument(
            "udp_bitrate_mbps", default_value="-1.0",
            description="UDP target rate in Mbit/s. -1 = AUTO: probe TCP "
            "capacity at this location first and offer udp_rate_fraction of "
            "it. 0 = unlimited, which floods, so the loss only tells you it "
            "flooded. A fixed number is a rate measured somewhere else.",
        ),
        DeclareLaunchArgument(
            "udp_rate_fraction", default_value="0.9",
            description="Auto mode: fraction of probed capacity to offer.",
        ),
        DeclareLaunchArgument(
            "parallel", default_value="4",
            description="Parallel TCP streams for the throughput test.",
        ),
        DeclareLaunchArgument(
            "continuous_interval_s", default_value="1.0",
            description="Continuous throughput/RTT rate (s): 1.0=1Hz, 0.2=5Hz.",
        ),
        DeclareLaunchArgument(
            "wifi_rate_hz", default_value="5.0",
            description="Passive monitor sampling rate.",
        ),
        DeclareLaunchArgument(
            "bidirectional", default_value="false",
            description="Alternate the continuous iperf between uplink and "
            "downlink every bidir_period_s (each message's 'reverse' field "
            "records its direction).",
        ),
        DeclareLaunchArgument(
            "bidir_period_s", default_value="10.0",
            description="Seconds per direction when bidirectional.",
        ),
        DeclareLaunchArgument(
            "run_iperf", default_value="true",
            description="Continuous iperf throughput+RTT (via ss). Set false "
            "for non-saturating monitoring during real operation.",
        ),
        DeclareLaunchArgument(
            "ntp_rate_hz", default_value="10.0",
            description="NTP client publish rate. The daemon is polled at "
            "1 Hz internally; 10 Hz just makes interpolation onto sensor "
            "timestamps easy.",
        ),
        DeclareLaunchArgument(
            "run_ntp", default_value="true",
            description="Publish clock offset for offline timestamp "
            "correction. Set false if no NTP daemon is configured.",
        ),
        DeclareLaunchArgument(
            "burst_bytes", default_value="40960",
            description="Payload per burst (40960 = 40 kB, roughly one "
            "compressed collaborative-perception feature map).",
        ),
        DeclareLaunchArgument(
            "deadline_ms", default_value="50.0",
            description="Deadline each burst is judged against.",
        ),
        DeclareLaunchArgument(
            "burst_port", default_value="5600",
            description="Port of burst_echo_server.py on the target.",
        ),
        DeclareLaunchArgument(
            "run_burst", default_value="false",
            description="Deadline pass. Wants an idle link -- pair it with "
            "run_iperf:=false, or the bursts measure iperf's queue.",
        ),
        DeclareLaunchArgument(
            "run_ping", default_value="false",
            description="Also run the ping node. Redundant when run_iperf is "
            "true (iperf+ss already gives RTT); enable it for RTT+loss during "
            "operation when run_iperf is false.",
        ),
    ]

    server = _str("server_address")
    iface = _str("interface")
    ns = LaunchConfiguration("namespace")

    wifi = Node(
        package="comms", executable="wifi_monitor_node",
        name="wifi_monitor", namespace=ns, output="screen",
        parameters=[{
            "interface": iface,
            "publish_rate_hz": _dbl("wifi_rate_hz"),
        }],
    )

    iperf = Node(
        package="comms", executable="iperf_runner_node",
        name="iperf_runner", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_iperf")),
        parameters=[{
            "server_address": server,
            "server_port": LaunchConfiguration("server_port"),
            "interface": iface,
            "continuous": LaunchConfiguration("continuous"),
            "reverse": LaunchConfiguration("reverse"),
            "protocol": _str("protocol"),
            "udp_bitrate_mbps": _dbl("udp_bitrate_mbps"),
            "udp_rate_fraction": _dbl("udp_rate_fraction"),
            "duration_s": _dbl("duration_s"),
            "interval_s": _dbl("interval_s"),
            "omit_s": _dbl("omit_s"),
            "parallel": LaunchConfiguration("parallel"),
            "continuous_interval_s": _dbl("continuous_interval_s"),
            "bidirectional": LaunchConfiguration("bidirectional"),
            "bidir_period_s": _dbl("bidir_period_s"),
        }],
    )

    ping = Node(
        package="comms", executable="ping_monitor_node",
        name="ping_monitor", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_ping")),
        parameters=[{
            "target": server,
            "interface": iface,
        }],
    )

    ntp = Node(
        package="comms", executable="ntp_client_node",
        name="ntp_client_node", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_ntp")),
        parameters=[{
            "publish_rate": _dbl("ntp_rate_hz"),
        }],
    )

    burst = Node(
        package="comms", executable="burst_monitor_node",
        name="burst_monitor", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_burst")),
        parameters=[{
            "target": server,
            "port": LaunchConfiguration("burst_port"),
            "interface": iface,
            "burst_bytes": LaunchConfiguration("burst_bytes"),
            "deadline_ms": _dbl("deadline_ms"),
        }],
    )

    return LaunchDescription(args + [wifi, iperf, ping, ntp, burst])
