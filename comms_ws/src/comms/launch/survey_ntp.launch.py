"""Wi-Fi survey plus NTP clock logging, in one launch.

Each node has its own on/off switch, so the same file covers a dedicated
survey pass, a non-saturating monitoring run, or clock logging alone:

  * wifi_monitor  -> /wifi/status        (5 Hz passive: RSSI, MCS/rate, retries) [run_wifi]
  * iperf_runner  -> /wifi/iperf         (continuous throughput + RTT via ss)    [run_iperf]
  * ntp_client    -> /ntp/client/status  (10 Hz clock offset)                    [run_ntp]
  * ping_monitor  -> /wifi/ping          (ICMP RTT + rolling loss)               [run_ping]

With bidirectional:=true the iperf node publishes to 'wifi/iperf_up' and
'wifi/iperf_down' instead of the single 'wifi/iperf'.

Latency note: the RTT carried in the IperfResult messages is the bulk flow's
own tcpi_rtt, and on downlink segments it does not update (the robot only
sends ACKs). Use /wifi/ping as the latency source; it is direction-agnostic
and survives the segment swaps.

The NTP client only reads the local chrony/ntpd daemon, so it adds no traffic
to the link being measured -- but it needs chrony already syncing to the time
server, so set run_ntp:=false if none is configured. The NTP *server* node
belongs on the time server itself (ntp_monitor.launch.py mode:=server).

    # loaded pass: goodput both directions + latency under load
    ros2 launch comms survey_ntp.launch.py \\
        server_address:=192.168.51.11 interface:=wlan0 \\
        parallel:=4 bidirectional:=true run_ping:=true

    # idle pass: baseline latency, nothing saturating the link
    ros2 launch comms survey_ntp.launch.py \\
        server_address:=192.168.51.11 run_iperf:=false run_ping:=true

Record it (the NTP topic is TRANSIENT_LOCAL, hence the override):

    ros2 bag record --qos-profile-overrides-path \\
        $(ros2 pkg prefix comms)/share/comms/config/rosbag_qos_override.yaml \\
        /wifi/status /wifi/iperf_up /wifi/iperf_down /wifi/ping \\
        /ntp/client/status /tf /tf_static /odom
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _str(name: str) -> ParameterValue:
    """Force a STRING parameter (an all-digit value would become an int)."""
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def _dbl(name: str) -> ParameterValue:
    """Force a DOUBLE parameter (integer-looking values would be rejected)."""
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "namespace", default_value="",
            description="Namespace for the nodes and the /wifi topics. Note "
            "that /ntp/client/status is absolute and stays global.",
        ),
        DeclareLaunchArgument(
            "server_address", default_value="192.168.51.11",
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

        # --- what to run ---------------------------------------------------
        DeclareLaunchArgument(
            "run_wifi", default_value="true",
            description="Passive link monitor. Free (reads the driver only), "
            "so leave it on unless you specifically want just one stream.",
        ),
        DeclareLaunchArgument(
            "run_iperf", default_value="true",
            description="Continuous iperf throughput. Saturates the link -- "
            "set false for the idle baseline pass.",
        ),
        DeclareLaunchArgument(
            "run_ntp", default_value="true",
            description="Publish clock offset for offline timestamp "
            "correction. Set false if no NTP daemon is configured.",
        ),
        DeclareLaunchArgument(
            "run_ping", default_value="true",
            description="ICMP RTT + rolling loss. This is the latency source "
            "for the dataset; the iperf RTT is the bulk flow's own and is "
            "stale on downlink segments.",
        ),

        # --- per-node settings ---------------------------------------------
        DeclareLaunchArgument(
            "wifi_rate_hz", default_value="5.0",
            description="Passive monitor sampling rate.",
        ),
        DeclareLaunchArgument(
            "reverse", default_value="false",
            description="Direction when not bidirectional: false = uplink, "
            "true = downlink.",
        ),
        DeclareLaunchArgument(
            "parallel", default_value="4",
            description="Parallel TCP streams for the throughput test.",
        ),
        DeclareLaunchArgument(
            "continuous_interval_s", default_value="1.0",
            description="Continuous throughput rate (s): 1.0=1Hz, 0.2=5Hz.",
        ),
        DeclareLaunchArgument(
            "bidirectional", default_value="false",
            description="Alternate uplink/downlink every bidir_period_s, "
            "publishing to wifi/iperf_up and wifi/iperf_down.",
        ),
        DeclareLaunchArgument(
            "bidir_period_s", default_value="10.0",
            description="Seconds per direction when bidirectional.",
        ),
        DeclareLaunchArgument(
            "ping_interval_s", default_value="1.0",
            description="Seconds between pings.",
        ),
        DeclareLaunchArgument(
            "ntp_rate_hz", default_value="10.0",
            description="NTP client publish rate. The daemon is polled at "
            "1 Hz internally; 10 Hz just makes interpolation onto sensor "
            "timestamps easy.",
        ),
    ]

    server = _str("server_address")
    iface = _str("interface")
    ns = LaunchConfiguration("namespace")

    wifi = Node(
        package="comms", executable="wifi_monitor_node",
        name="wifi_monitor", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_wifi")),
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
            "continuous": True,
            "reverse": LaunchConfiguration("reverse"),
            "parallel": LaunchConfiguration("parallel"),
            "continuous_interval_s": _dbl("continuous_interval_s"),
            "bidirectional": LaunchConfiguration("bidirectional"),
            "bidir_period_s": _dbl("bidir_period_s"),
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

    ping = Node(
        package="comms", executable="ping_monitor_node",
        name="ping_monitor", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_ping")),
        parameters=[{
            "target": server,
            "interface": iface,
            "interval_s": _dbl("ping_interval_s"),
        }],
    )

    return LaunchDescription(args + [wifi, iperf, ntp, ping])
