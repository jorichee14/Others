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


def _dbl(name: str) -> ParameterValue:
    """Force a launch arg to be a DOUBLE parameter.

    Same reason as _str: a value typed without a decimal point (e.g.
    start_delay_s:=10) would be inferred as INTEGER and rejected by the
    node's double parameter declaration.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=float)


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
            "bidirectional", default_value="false",
            description="Alternate uplink/downlink from this one instance "
            "(per test, or per bidir_period_s segment in continuous mode). "
            "Each message's 'reverse' field records its direction.",
        ),
        DeclareLaunchArgument(
            "bidir_period_s", default_value="10.0",
            description="Continuous+bidirectional: seconds per direction "
            "before swapping.",
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
            "continuous", default_value="false",
            description="Survey mode: one long iperf3, ~1 Hz results, no "
            "per-test overhead. Saturates the link (dedicated survey pass).",
        ),
        DeclareLaunchArgument(
            "rtt_via_ss", default_value="true",
            description="In continuous mode, fill RTT from the live socket via "
            "`ss` (needs iproute2) -- dense RTT without ping.",
        ),
        DeclareLaunchArgument(
            "continuous_interval_s", default_value="1.0",
            description="Continuous-mode report interval (s): 1.0=1Hz, "
            "0.2-0.5=2-5Hz. Lower = denser but noisier.",
        ),
        DeclareLaunchArgument(
            "connect_timeout_ms", default_value="2000",
            description="iperf3 --connect-timeout so a dead server fails fast.",
        ),
        DeclareLaunchArgument(
            "reconnect_poll_s", default_value="3.0",
            description="Poll/backoff period while link is down or a test "
            "fails.",
        ),
        DeclareLaunchArgument(
            "name", default_value="iperf_runner",
            description="Node name. Set a distinct name (e.g. "
            "iperf_runner_r2r) to run a second instance alongside the first.",
        ),
        DeclareLaunchArgument(
            "topic", default_value="wifi/iperf",
            description="Output topic. Give a second instance its own topic "
            "(e.g. wifi/iperf_r2r) so the streams stay separable.",
        ),
        DeclareLaunchArgument(
            "start_delay_s", default_value="0.0",
            description="Delay before the first test. Offset a second "
            "instance by interval_s/2 so the two never test at once.",
        ),
    ]

    node = Node(
        package="wifi_monitor",
        executable="iperf_runner_node",
        name=LaunchConfiguration("name"),
        output="screen",
        remappings=[("wifi/iperf", LaunchConfiguration("topic"))],
        parameters=[{
            "server_address": _str("server_address"),
            "server_port": LaunchConfiguration("server_port"),
            "interface": _str("interface"),
            "protocol": _str("protocol"),
            "duration_s": _dbl("duration_s"),
            "interval_s": _dbl("interval_s"),
            "reverse": LaunchConfiguration("reverse"),
            "bidirectional": LaunchConfiguration("bidirectional"),
            "bidir_period_s": _dbl("bidir_period_s"),
            "udp_bitrate_mbps": _dbl("udp_bitrate_mbps"),
            "parallel": LaunchConfiguration("parallel"),
            "omit_s": _dbl("omit_s"),
            "connect_timeout_ms": LaunchConfiguration("connect_timeout_ms"),
            "reconnect_poll_s": _dbl("reconnect_poll_s"),
            "continuous": LaunchConfiguration("continuous"),
            "rtt_via_ss": LaunchConfiguration("rtt_via_ss"),
            "continuous_interval_s": _dbl("continuous_interval_s"),
            "start_delay_s": _dbl("start_delay_s"),
        }],
    )

    return LaunchDescription(args + [node])
