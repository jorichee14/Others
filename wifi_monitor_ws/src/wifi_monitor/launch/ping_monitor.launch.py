"""Launch the ping_monitor node against a fixed host."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _str(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "target", default_value="192.168.233.142",
            description="Host to ping (usually the iperf3 server).",
        ),
        DeclareLaunchArgument("interval_s", default_value="1.0"),
        DeclareLaunchArgument("timeout_s", default_value="1.0"),
        DeclareLaunchArgument("window", default_value="20"),
        DeclareLaunchArgument(
            "interface", default_value="",
            description="Bind ping to this interface (''=default route).",
        ),
    ]

    node = Node(
        package="wifi_monitor",
        executable="ping_monitor_node",
        name="ping_monitor",
        output="screen",
        parameters=[{
            "target": _str("target"),
            "interval_s": LaunchConfiguration("interval_s"),
            "timeout_s": LaunchConfiguration("timeout_s"),
            "window": LaunchConfiguration("window"),
            "interface": _str("interface"),
        }],
    )

    return LaunchDescription(args + [node])
