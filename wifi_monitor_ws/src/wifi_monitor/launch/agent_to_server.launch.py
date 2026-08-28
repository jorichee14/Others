"""Agent-to-server measurement only: passive monitor + one iperf client.

No robot B, no r2r instance, no slots -- just this robot against the wired
iperf3 server. The server side is only `iperf3 -s -p <server_port>`.

    # periodic monitoring (uplink, 2 s test every 30 s):
    ros2 launch wifi_monitor agent_to_server.launch.py \
        server_address:=192.168.233.142

    # dense continuous survey pass (~2 Hz samples, saturates the link):
    ros2 launch wifi_monitor agent_to_server.launch.py \
        server_address:=192.168.233.142 continuous:=true

Direction: uplink by default; reverse:=true for downlink; or
bidirectional:=true to alternate both (halves each direction's sampling
rate -- see the README).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
            "server_address", default_value="192.168.233.142",
            description="The wired iperf3 server.",
        ),
        DeclareLaunchArgument("server_port", default_value="5201"),
        DeclareLaunchArgument(
            "interface", default_value="",
            description="Wireless interface ('' = auto-detect).",
        ),
        DeclareLaunchArgument(
            "reverse", default_value="false",
            description="true => downlink (server -> robot).",
        ),
        DeclareLaunchArgument(
            "bidirectional", default_value="false",
            description="Alternate uplink/downlink (each message's 'reverse' "
            "field records its direction).",
        ),
        DeclareLaunchArgument(
            "continuous", default_value="false",
            description="Survey mode: one long iperf, dense samples, "
            "saturates the link. false = periodic bursts (monitoring).",
        ),
        DeclareLaunchArgument(
            "duration_s", default_value="2.0",
            description="Periodic mode: per-test duration.",
        ),
        DeclareLaunchArgument(
            "interval_s", default_value="30.0",
            description="Periodic mode: gap between tests.",
        ),
        DeclareLaunchArgument(
            "parallel", default_value="4",
            description="Parallel TCP streams (fills the link).",
        ),
        DeclareLaunchArgument(
            "continuous_interval_s", default_value="0.5",
            description="Continuous mode: reporting interval (0.5 = 2 Hz).",
        ),
        DeclareLaunchArgument(
            "bidir_period_s", default_value="10.0",
            description="Continuous+bidirectional: seconds per direction.",
        ),
        DeclareLaunchArgument(
            "wifi_rate_hz", default_value="5.0",
            description="Passive monitor sampling rate.",
        ),
    ]

    iface = _str("interface")

    wifi = Node(
        package="wifi_monitor", executable="wifi_monitor_node",
        name="wifi_monitor", output="screen",
        parameters=[{
            "interface": iface,
            "publish_rate_hz": _dbl("wifi_rate_hz"),
        }],
    )

    iperf = Node(
        package="wifi_monitor", executable="iperf_runner_node",
        name="iperf_runner", output="screen",
        parameters=[{
            "interface": iface,
            "server_address": _str("server_address"),
            "server_port": LaunchConfiguration("server_port"),
            "reverse": LaunchConfiguration("reverse"),
            "bidirectional": LaunchConfiguration("bidirectional"),
            "continuous": LaunchConfiguration("continuous"),
            "duration_s": _dbl("duration_s"),
            "interval_s": _dbl("interval_s"),
            "parallel": LaunchConfiguration("parallel"),
            "continuous_interval_s": _dbl("continuous_interval_s"),
            "bidir_period_s": _dbl("bidir_period_s"),
        }],
    )

    return LaunchDescription(args + [wifi, iperf])
