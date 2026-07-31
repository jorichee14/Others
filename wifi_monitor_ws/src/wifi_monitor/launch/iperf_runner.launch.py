"""Launch the iperf_runner node against a fixed (wired) iperf3 server."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _str(name: str) -> ParameterValue:
    """Force a launch arg to be a STRING parameter.

    Without this, launch infers the type from the text, so a value like
    "0" (udp_bitrate) or an all-digit interface name would be passed as an
    int and the node's string parameter declaration would reject it.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "server_address",
            default_value="192.168.233.142",
            description="iperf3 server IP/host (the static wired laptop).",
        ),
        DeclareLaunchArgument("server_port", default_value="5201"),
        DeclareLaunchArgument(
            "interface", default_value="",
            description="Wi-Fi iface for the link-down failsafe (''=auto).",
        ),
        DeclareLaunchArgument(
            "protocol", default_value="tcp",
            description="'tcp' (capacity + RTT) or 'udp' (loss + jitter).",
        ),
        DeclareLaunchArgument(
            "duration_s", default_value="2.0",
            description="Per-test duration in seconds.",
        ),
        DeclareLaunchArgument(
            "interval_s", default_value="30.0",
            description="Gap between tests (0 => back-to-back survey mode).",
        ),
        DeclareLaunchArgument(
            "reverse", default_value="false",
            description="true => downlink (server->robot).",
        ),
        DeclareLaunchArgument(
            "udp_bitrate_mbps", default_value="0.0",
            description="UDP target rate in Mbit/s, e.g. 300 (0 = unlimited; "
            "only for protocol:=udp).",
        ),
        DeclareLaunchArgument(
            "parallel", default_value="1",
            description="Parallel TCP streams (iperf3 -P). Try 4 for capacity.",
        ),
        DeclareLaunchArgument("omit_s", default_value="1.0"),
        DeclareLaunchArgument(
            "connect_timeout_ms", default_value="2000",
            description="iperf3 --connect-timeout so a dead server fails fast.",
        ),
        DeclareLaunchArgument(
            "reconnect_poll_s", default_value="3.0",
            description="Poll/backoff period while link is down or a test "
            "fails.",
        ),
    ]

    node = Node(
        package="wifi_monitor",
        executable="iperf_runner_node",
        name="iperf_runner",
        output="screen",
        parameters=[{
            "server_address": _str("server_address"),
            "server_port": LaunchConfiguration("server_port"),
            "interface": _str("interface"),
            "protocol": _str("protocol"),
            "duration_s": LaunchConfiguration("duration_s"),
            "interval_s": LaunchConfiguration("interval_s"),
            "reverse": LaunchConfiguration("reverse"),
            "udp_bitrate_mbps": LaunchConfiguration("udp_bitrate_mbps"),
            "parallel": LaunchConfiguration("parallel"),
            "omit_s": LaunchConfiguration("omit_s"),
            "connect_timeout_ms": LaunchConfiguration("connect_timeout_ms"),
            "reconnect_poll_s": LaunchConfiguration("reconnect_poll_s"),
        }],
    )

    return LaunchDescription(args + [node])
