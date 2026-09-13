"""
ntp_monitor.launch.py
─────────────────────
Launches both the NTP client and server monitor nodes.

Usage
  ros2 launch comms ntp_monitor.launch.py
  ros2 launch comms ntp_monitor.launch.py mode:=client
  ros2 launch comms ntp_monitor.launch.py mode:=server
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="both",
        description="Which nodes to launch: 'client', 'server', or 'both'",
    )
    rate_arg = DeclareLaunchArgument(
        "publish_rate",
        default_value="1.0",
        description="Publish rate in Hz",
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Namespace for this agent's NTP topics, e.g. mobile_1, "
        "mobile_2, infra_1. The node's topics are relative, so this gives "
        "/mobile_2/ntp/client/status. Empty = /ntp/client/status.",
    )
    count_clients_arg = DeclareLaunchArgument(
        "count_clients",
        default_value="true",
        description="[server] Whether to count connected NTP clients",
    )

    mode = LaunchConfiguration("mode")
    rate = LaunchConfiguration("publish_rate")
    count_clients = LaunchConfiguration("count_clients")

    client_node = Node(
        package="comms",
        executable="ntp_client_node",
        name="ntp_client_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {"publish_rate": rate},
        ],
        output="screen",
    )

    server_node = Node(
        package="comms",
        executable="ntp_server_node",
        name="ntp_server_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {"publish_rate": rate},
            {"count_clients": count_clients},
        ],
        output="screen",
    )

    def select_nodes(context):
        m = context.launch_configurations.get("mode", "both")
        if m == "client":
            return [client_node]
        elif m == "server":
            return [server_node]
        return [client_node, server_node]

    return LaunchDescription(
        [
            mode_arg,
            rate_arg,
            count_clients_arg,
            namespace_arg,
            OpaqueFunction(function=select_nodes),
        ]
    )
