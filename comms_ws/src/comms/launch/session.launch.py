"""Continuous session: one launch, one bag, for a whole route.

The alternative is stopping and restarting the stack at every measurement
point, which costs a launch cycle per stop and leaves you with one bag per
point to reassemble later. Here everything runs for the duration of the
route and one bag covers the whole thing; it is cut into per-point segments
offline using LiDAR pose, since every sample is timestamped on the same
clock and a stop is just where velocity sits near zero.

    ros2 launch comms session.launch.py \\
        server_address:=192.168.51.11 server_port:=5201 \\
        interface:=wlx8876b9eae0ff namespace:=mobile_1

Segmentation is done offline from LiDAR pose: stops are where velocity sits
near zero, and every sample already carries a timestamp on the same clock, so
no manual labelling is needed during the run. Record the whole session as one
bag in a second terminal:

    ros2 bag record --qos-profile-overrides-path \\
        $(ros2 pkg prefix comms)/share/comms/config/rosbag_qos_override.yaml \\
        -o session_$(date +%Y%m%d_%H%M) \\
        /mobile_1/wifi/status /mobile_1/wifi/iperf_up \\
        /mobile_1/wifi/iperf_down /mobile_1/wifi/ping \\
        /ntp/client/status /mobile1/csi /csi_publisher/status \\
        /tf /tf_static /odom /scan

WHICH MEASUREMENT RUNS
======================
Continuous collection forces a choice that per-point collection does not.
The capacity measurement saturates the link and fills the AP's queue; the
burst measurement needs an idle link. They cannot both be true at once, and
there is no gap to settle in between when the stack never stops.

So pick one per session and traverse the route twice:

  mode:=capacity   (default) continuous iperf at 1 Hz + passive monitor.
                   Throughput and loaded RTT everywhere, moving and stopped.
  mode:=deadline   bursts + ping on an idle link. Deadline compliance and
                   unloaded latency everywhere.

Two traverses back to back keeps the environment comparable between them.
Running both at once would make the burst numbers measure iperf's queue.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _str(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def _dbl(name: str) -> ParameterValue:
    """Force a DOUBLE parameter (integer-looking values would be rejected)."""
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _mode_is(mode: str) -> IfCondition:
    return IfCondition(PythonExpression(
        ["'", LaunchConfiguration("mode"), "' == '", mode, "'"]
    ))


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "mode", default_value="capacity",
            description="'capacity' = continuous iperf (saturates the link). "
            "'deadline' = bursts + ping (needs an idle link). One per "
            "traverse; they cannot share a session.",
        ),
        DeclareLaunchArgument(
            "namespace", default_value="",
            description="Namespace for the nodes and /wifi topics.",
        ),
        DeclareLaunchArgument(
            "server_address", default_value="192.168.51.11",
            description="iperf3 server, ping target and echo host.",
        ),
        DeclareLaunchArgument("server_port", default_value="5201"),
        DeclareLaunchArgument("burst_port", default_value="5600"),
        DeclareLaunchArgument(
            "interface", default_value="",
            description="Wireless interface ('' = auto-detect).",
        ),

        # --- capacity mode --------------------------------------------------
        DeclareLaunchArgument(
            "parallel", default_value="2",
            description="Parallel TCP streams. Set from the measured sweep.",
        ),
        DeclareLaunchArgument(
            "continuous_interval_s", default_value="1.0",
            description="Sample rate (s). 1.0 = 1 m per sample at 1 m/s.",
        ),
        DeclareLaunchArgument(
            "reverse", default_value="false",
            description="false = uplink, true = downlink. Used when not "
            "bidirectional.",
        ),
        DeclareLaunchArgument(
            "bidirectional", default_value="false",
            description="Alternate directions in one traverse, publishing to "
            "wifi/iperf_up and wifi/iperf_down. Off by default: each swap "
            "costs a ~2s reconnect, which on a moving robot is a ~2m hole in "
            "BOTH topics, and per-direction density halves to one sample "
            "every 2m. Two unidirectional traverses cost the same driving "
            "and have neither problem.",
        ),
        DeclareLaunchArgument(
            "bidir_period_s", default_value="30.0",
            description="Seconds per direction when bidirectional. Longer "
            "than the 10s stationary default so a moving traverse pays the "
            "reconnect hole every ~30m instead of every ~10m.",
        ),

        # --- deadline mode --------------------------------------------------
        DeclareLaunchArgument(
            "burst_bytes", default_value="40960",
            description="Payload per burst (40 kB ~ one compressed feature "
            "map).",
        ),
        DeclareLaunchArgument(
            "deadline_ms", default_value="50.0",
            description="Deadline each burst is judged against.",
        ),
        DeclareLaunchArgument(
            "burst_interval_s", default_value="1.0",
            description="Seconds between bursts.",
        ),

        # --- always on ------------------------------------------------------
        DeclareLaunchArgument("wifi_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("ntp_rate_hz", default_value="10.0"),
        DeclareLaunchArgument(
            "run_ntp", default_value="true",
            description="Set false if no NTP daemon is configured.",
        ),
    ]

    server = _str("server_address")
    iface = _str("interface")
    ns = LaunchConfiguration("namespace")

    wifi = Node(
        package="comms", executable="wifi_monitor_node",
        name="wifi_monitor", namespace=ns, output="screen",
        parameters=[{
            "interface": iface,
            "publish_rate_hz": _dbl("wifi_rate_hz"),
        }],
    )

    iperf = Node(
        package="comms", executable="iperf_runner_node",
        name="iperf_runner", namespace=ns, output="screen",
        condition=_mode_is("capacity"),
        parameters=[{
            "server_address": server,
            "server_port": LaunchConfiguration("server_port"),
            "interface": iface,
            "continuous": True,
            "bidirectional": LaunchConfiguration("bidirectional"),
            "bidir_period_s": _dbl("bidir_period_s"),
            "reverse": LaunchConfiguration("reverse"),
            "parallel": LaunchConfiguration("parallel"),
            "continuous_interval_s": _dbl("continuous_interval_s"),
        }],
    )

    burst = Node(
        package="comms", executable="burst_monitor_node",
        name="burst_monitor", namespace=ns, output="screen",
        condition=_mode_is("deadline"),
        parameters=[{
            "target": server,
            "port": LaunchConfiguration("burst_port"),
            "interface": iface,
            "burst_bytes": LaunchConfiguration("burst_bytes"),
            "deadline_ms": _dbl("deadline_ms"),
            "interval_s": _dbl("burst_interval_s"),
        }],
    )

    # Ping only in deadline mode: under a saturating iperf it would measure
    # the queue iperf built, not the link.
    ping = Node(
        package="comms", executable="ping_monitor_node",
        name="ping_monitor", namespace=ns, output="screen",
        condition=_mode_is("deadline"),
        parameters=[{
            "target": server,
            "interface": iface,
        }],
    )

    ntp = Node(
        package="comms", executable="ntp_client_node",
        name="ntp_client_node", namespace=ns, output="screen",
        condition=IfCondition(LaunchConfiguration("run_ntp")),
        parameters=[{"publish_rate": _dbl("ntp_rate_hz")}],
    )

    return LaunchDescription(args + [wifi, iperf, burst, ping, ntp])
