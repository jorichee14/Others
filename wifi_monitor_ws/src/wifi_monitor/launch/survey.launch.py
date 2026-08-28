"""Full Wi-Fi survey stack in one launch.

Nodes on the robot:
  * wifi_monitor  -> /wifi/status  (5 Hz passive: RSSI, MCS/rate, retries)
  * iperf_runner  -> /wifi/iperf   (continuous throughput + RTT via ss)   [run_iperf]
  * ping_monitor  -> /wifi/ping    (RTT + loss)                           [run_ping]

Continuous iperf now reports RTT itself (via ss), so ping is OFF by default
-- it would just duplicate the RTT. Enable run_ping only when you want RTT +
loss WITHOUT running iperf (i.e. run_iperf:=false), e.g. monitoring latency
during real operation without saturating the link.

The server host is the iperf3 server and the ping target; pass it once.
Record with pose to map it offline:

    ros2 bag record /wifi/status /wifi/iperf /wifi/ping /tf /odom
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
            "interface", default_value="",
            description="Wireless interface ('' = auto-detect).",
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
        package="wifi_monitor", executable="wifi_monitor_node",
        name="wifi_monitor", namespace=ns, output="screen",
        parameters=[{
            "interface": iface,
            "publish_rate_hz": _dbl("wifi_rate_hz"),
        }],
    )

    iperf = Node(
        package="wifi_monitor", executable="iperf_runner_node",
        name="iperf_runner", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_iperf")),
        parameters=[{
            "server_address": server,
            "interface": iface,
            "continuous": True,
            "parallel": LaunchConfiguration("parallel"),
            "continuous_interval_s": _dbl("continuous_interval_s"),
            "bidirectional": LaunchConfiguration("bidirectional"),
            "bidir_period_s": _dbl("bidir_period_s"),
        }],
    )

    ping = Node(
        package="wifi_monitor", executable="ping_monitor_node",
        name="ping_monitor", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_ping")),
        parameters=[{
            "target": server,
            "interface": iface,
        }],
    )

    return LaunchDescription(args + [wifi, iperf, ping])
