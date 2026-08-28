"""One-command bringup for the two-robot deployment (agent<->server + r2r).

Starts everything a robot needs for the interleaved measurement scheme,
selected by ``role``:

  role:=a  (the r2r *client* robot)
    * wifi_monitor                          -> /wifi/status
    * iperf_runner      A->server, slot t=0 -> /wifi/iperf
    * iperf_runner_r2r  A<->B bidirectional, slot t=10 -> /wifi/iperf_r2r
      (needs robot_b_address)

  role:=b  (the r2r *server* robot)
    * wifi_monitor                          -> /wifi/status
    * iperf3 -s on r2r_port (respawned if it dies), serving robot A
    * iperf_runner      B->server, slot t=20 -> /wifi/iperf

The laptop still runs its own servers (no ROS needed there):

    iperf3 -s -p 5201 &     # serves robot A
    iperf3 -s -p 5211 &     # serves robot B

Agent->server tests are one-directional (uplink; reverse:=true for
downlink); the r2r instance alternates direction each test so one client
covers A->B and B->A. Slots assume the default interval_s of 30 with three
10 s slots -- start both robots' launches within a few seconds of each
other so the slots stay aligned.

Examples:

    # robot A:
    ros2 launch wifi_monitor agent.launch.py role:=a \
        robot_b_address:=192.168.233.7

    # robot B:
    ros2 launch wifi_monitor agent.launch.py role:=b
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _str(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def _dbl(name: str) -> ParameterValue:
    """Force a launch arg to be a DOUBLE parameter.

    Without this, a value typed without a decimal point (start_delay_s:=0)
    is inferred as INTEGER and rejected by the node's double declaration.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _role_is(role: str) -> IfCondition:
    return IfCondition(
        PythonExpression(["'", LaunchConfiguration("role"), "' == '", role, "'"])
    )


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "role", choices=["a", "b"],
            description="'a' = r2r client robot (runs the A<->B tests); "
            "'b' = r2r server robot (runs iperf3 -s for robot A).",
        ),
        DeclareLaunchArgument(
            "namespace", default_value="",
            description="Namespace for the nodes and topics, e.g. robota -> "
            "/robota/wifi/status, /robota/wifi/iperf. '' = no namespace.",
        ),
        DeclareLaunchArgument(
            "laptop_address", default_value="192.168.233.142",
            description="The wired iperf3 server (agent->server tests).",
        ),
        DeclareLaunchArgument(
            "robot_b_address", default_value="",
            description="Robot B's IP (role:=a). Empty = no robot B yet: the "
            "r2r instance is skipped and only A->server runs.",
        ),
        DeclareLaunchArgument(
            "laptop_port_a", default_value="5201",
            description="Laptop server port used by robot A.",
        ),
        DeclareLaunchArgument(
            "laptop_port_b", default_value="5211",
            description="Laptop server port used by robot B.",
        ),
        DeclareLaunchArgument(
            "r2r_port", default_value="5202",
            description="Port robot B serves / robot A tests for the r2r link.",
        ),
        DeclareLaunchArgument(
            "interface", default_value="",
            description="Wireless interface ('' = auto-detect).",
        ),
        DeclareLaunchArgument(
            "duration_s", default_value="2.0",
            description="Per-test duration; keep well under the 10 s slots.",
        ),
        DeclareLaunchArgument(
            "interval_s", default_value="30.0",
            description="Cycle length shared by all instances (3 slots).",
        ),
        DeclareLaunchArgument(
            "reverse", default_value="false",
            description="Direction of the agent->server tests: false = "
            "uplink, true = downlink. (The r2r test always alternates.)",
        ),
        DeclareLaunchArgument(
            "parallel", default_value="1",
            description="Parallel TCP streams (iperf3 -P).",
        ),
        DeclareLaunchArgument(
            "wifi_rate_hz", default_value="5.0",
            description="Passive monitor sampling rate.",
        ),
        # Slot offsets within the cycle; defaults assume interval_s=30.
        DeclareLaunchArgument("slot_a_server_s", default_value="0.0"),
        DeclareLaunchArgument("slot_a_r2r_s", default_value="10.0"),
        DeclareLaunchArgument("slot_b_server_s", default_value="20.0"),
    ]

    iface = _str("interface")
    ns = LaunchConfiguration("namespace")

    wifi = Node(
        package="wifi_monitor", executable="wifi_monitor_node",
        name="wifi_monitor", namespace=ns, output="screen",
        parameters=[{
            "interface": iface,
            "publish_rate_hz": _dbl("wifi_rate_hz"),
        }],
    )

    common = {
        "interface": iface,
        "duration_s": _dbl("duration_s"),
        "interval_s": _dbl("interval_s"),
        "parallel": LaunchConfiguration("parallel"),
    }

    # --- role a ------------------------------------------------------------
    a_server = Node(
        package="wifi_monitor", executable="iperf_runner_node",
        name="iperf_runner", namespace=ns, output="screen",
        condition=_role_is("a"),
        parameters=[{
            **common,
            "server_address": _str("laptop_address"),
            "server_port": LaunchConfiguration("laptop_port_a"),
            "reverse": LaunchConfiguration("reverse"),
            "start_delay_s": _dbl("slot_a_server_s"),
        }],
    )

    # Only started when robot B exists (robot_b_address non-empty).
    a_r2r = Node(
        package="wifi_monitor", executable="iperf_runner_node",
        name="iperf_runner_r2r", namespace=ns, output="screen",
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("role"), "' == 'a' and '",
            LaunchConfiguration("robot_b_address"), "' != ''",
        ])),
        remappings=[("wifi/iperf", "wifi/iperf_r2r")],
        parameters=[{
            **common,
            "server_address": _str("robot_b_address"),
            "server_port": LaunchConfiguration("r2r_port"),
            "bidirectional": True,
            "start_delay_s": _dbl("slot_a_r2r_s"),
        }],
    )

    # --- role b ------------------------------------------------------------
    b_iperf_server = ExecuteProcess(
        cmd=["iperf3", "-s", "-p", LaunchConfiguration("r2r_port")],
        name="iperf3_server_r2r", output="screen",
        condition=_role_is("b"),
        respawn=True, respawn_delay=3.0,
    )

    b_server = Node(
        package="wifi_monitor", executable="iperf_runner_node",
        name="iperf_runner", namespace=ns, output="screen",
        condition=_role_is("b"),
        parameters=[{
            **common,
            "server_address": _str("laptop_address"),
            "server_port": LaunchConfiguration("laptop_port_b"),
            "reverse": LaunchConfiguration("reverse"),
            "start_delay_s": _dbl("slot_b_server_s"),
        }],
    )

    return LaunchDescription(
        args + [wifi, a_server, a_r2r, b_iperf_server, b_server]
    )
