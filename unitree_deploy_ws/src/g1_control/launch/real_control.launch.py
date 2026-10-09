"""Launch RENet and the deliberate START-gated real LowCmd bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    enable_voice = LaunchConfiguration("enable_voice")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("g1_control"),
                        "config",
                        "control_first_real.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("enable_voice", default_value="true"),
            Node(
                package="g1_control",
                executable="renet_policy",
                name="g1_renet_policy",
                output="screen",
                emulate_tty=True,
                parameters=[params_file],
            ),
            Node(
                package="g1_control",
                executable="lowcmd_bridge",
                name="g1_lowcmd_bridge",
                output="screen",
                emulate_tty=True,
                parameters=[params_file],
            ),
            Node(
                package="g1_control",
                executable="voice_announcer",
                name="g1_voice_announcer",
                output="screen",
                emulate_tty=True,
                parameters=[params_file],
                condition=IfCondition(enable_voice),
            ),
        ]
    )
