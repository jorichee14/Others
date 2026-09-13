from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=PathJoinSubstitution(
            [FindPackageShare('comms'), 'config', 'csi.yaml'])),
        DeclareLaunchArgument('node_name', default_value='csi_publisher'),
        # Vary this, not node_name: the params file keys on /**/csi_publisher,
        # which matches any namespace but only that literal node name.
        DeclareLaunchArgument('namespace', default_value=''),
        Node(
            package='comms',
            executable='csi_publisher',
            name=LaunchConfiguration('node_name'),
            namespace=LaunchConfiguration('namespace'),
            parameters=[LaunchConfiguration('config')],
            output='screen',
        ),
    ])
