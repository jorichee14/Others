"""Deadline pass: fixed-size bursts into an idle link.

Answers "will a 40 kB feature map arrive within 50 ms", which saturated TCP
goodput cannot. Run this as its own pass with iperf OFF -- the point is to
measure what a real burst meets, and a saturated link would only measure the
queue iperf built.

The laptop needs the echo server (stdlib Python, no ROS):

    python3 burst_echo_server.py --port 5600

Typical:

    ros2 launch comms burst_monitor.launch.py target:=192.168.51.11

Sweep the size to find where the link starts missing deadlines:

    ros2 launch comms burst_monitor.launch.py burst_bytes:=20480
    ros2 launch comms burst_monitor.launch.py burst_bytes:=81920
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
            "namespace", default_value="",
            description="Namespace for the node and topic ('' = none).",
        ),
        DeclareLaunchArgument(
            "target", default_value="192.168.51.11",
            description="Host running burst_echo_server.py.",
        ),
        DeclareLaunchArgument("port", default_value="5600"),
        DeclareLaunchArgument(
            "burst_bytes", default_value="40960",
            description="Total payload per burst. 40960 = 40 kB, the rough "
            "size of one compressed collaborative-perception feature map.",
        ),
        DeclareLaunchArgument(
            "packet_bytes", default_value="1400",
            description="Per-datagram payload. Keep under a 1500-byte MTU so "
            "IP does not fragment.",
        ),
        DeclareLaunchArgument(
            "interval_s", default_value="1.0",
            description="Seconds between bursts.",
        ),
        DeclareLaunchArgument(
            "deadline_ms", default_value="50.0",
            description="Deadline each burst is judged against.",
        ),
        DeclareLaunchArgument(
            "timeout_ms", default_value="500.0",
            description="Stop waiting for stragglers. Keep well above "
            "deadline_ms so late-but-arrived is distinguishable from lost.",
        ),
        DeclareLaunchArgument(
            "interface", default_value="",
            description="Wireless iface for the link-down failsafe.",
        ),
    ]

    node = Node(
        package="comms",
        executable="burst_monitor_node",
        name="burst_monitor",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[{
            "target": _str("target"),
            "port": LaunchConfiguration("port"),
            "burst_bytes": LaunchConfiguration("burst_bytes"),
            "packet_bytes": LaunchConfiguration("packet_bytes"),
            "interval_s": _dbl("interval_s"),
            "deadline_ms": _dbl("deadline_ms"),
            "timeout_ms": _dbl("timeout_ms"),
            "interface": _str("interface"),
        }],
    )

    return LaunchDescription(args + [node])
