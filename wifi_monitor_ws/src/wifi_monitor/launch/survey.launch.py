"""Full Wi-Fi survey stack in one launch.

Starts three nodes on the robot:
  * wifi_monitor  -> /wifi/status  (5 Hz passive: RSSI, MCS/rate, retries)
  * iperf_runner  -> /wifi/iperf   (continuous 1 Hz throughput to the server)
  * ping_monitor  -> /wifi/ping    (1 Hz RTT + rolling loss to the server)

The server host is used both as the iperf3 server and the ping target, so
pass it once. iperf saturates the link (dedicated survey pass); ping does
not. Record everything with pose to map it offline:

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


def generate_launch_description() -> LaunchDescription:
    args = [
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
            "run_iperf", default_value="true",
            description="Set false to survey with passive + ping only (no "
            "link saturation) during real operation.",
        ),
    ]

    server = _str("server_address")
    iface = _str("interface")

    wifi = Node(
        package="wifi_monitor", executable="wifi_monitor_node",
        name="wifi_monitor", output="screen",
        parameters=[{
            "interface": iface,
            "publish_rate_hz": LaunchConfiguration("wifi_rate_hz"),
        }],
    )

    iperf = Node(
        package="wifi_monitor", executable="iperf_runner_node",
        name="iperf_runner", output="screen",
        condition=IfCondition(LaunchConfiguration("run_iperf")),
        parameters=[{
            "server_address": server,
            "interface": iface,
            "continuous": True,
            "parallel": LaunchConfiguration("parallel"),
            "continuous_interval_s": LaunchConfiguration("continuous_interval_s"),
        }],
    )

    ping = Node(
        package="wifi_monitor", executable="ping_monitor_node",
        name="ping_monitor", output="screen",
        parameters=[{
            "target": server,
            "interface": iface,
        }],
    )

    return LaunchDescription(args + [wifi, iperf, ping])
