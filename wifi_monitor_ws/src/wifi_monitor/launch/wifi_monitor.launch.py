"""Launch the wifi_monitor node with declared, overridable parameters."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "interface",
            default_value="",
            description="Wireless interface (empty = auto-detect first).",
        ),
        DeclareLaunchArgument(
            "publish_rate_hz",
            default_value="2.0",
            description="Sampling / publishing rate in Hz (2-5 Hz suits a "
            "slowly moving robot).",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="wifi",
            description="header.frame_id stamped on each message.",
        ),
        DeclareLaunchArgument(
            "warn_signal_dbm",
            default_value="-70.0",
            description="RSSI at/below which diagnostics warn.",
        ),
        DeclareLaunchArgument(
            "error_signal_dbm",
            default_value="-80.0",
            description="RSSI at/below which diagnostics error.",
        ),
    ]

    node = Node(
        package="wifi_monitor",
        executable="wifi_monitor_node",
        name="wifi_monitor",
        output="screen",
        parameters=[
            {
                "interface": LaunchConfiguration("interface"),
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                "frame_id": LaunchConfiguration("frame_id"),
                "warn_signal_dbm": LaunchConfiguration("warn_signal_dbm"),
                "error_signal_dbm": LaunchConfiguration("error_signal_dbm"),
            }
        ],
    )

    return LaunchDescription(args + [node])
