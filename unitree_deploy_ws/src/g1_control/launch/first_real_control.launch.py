"""Bring up G1 perception and the START-gated first real-control stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    control_params = LaunchConfiguration("control_params")
    depth_params = LaunchConfiguration("depth_params")
    realsense_params = LaunchConfiguration("realsense_params")
    elevation_params = LaunchConfiguration("elevation_params")
    livox_config = LaunchConfiguration("livox_config")
    fastlio_config = LaunchConfiguration("fastlio_config")
    enable_voice = LaunchConfiguration("enable_voice")

    # Foxy's realsense2_camera 4.51.1 launch file evaluates config_file as a
    # Python expression and cannot safely receive a composed package path.
    # Start the same official node directly and give it the filter YAML.
    camera = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="camera",
        name="camera",
        output="screen",
        emulate_tty=True,
        parameters=[
            realsense_params,
            {
                "depth_module.profile": "848x480x30",
                "enable_depth": True,
                "enable_color": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_gyro": False,
                "enable_accel": False,
                # Recover D4xx devices left in a stale streaming/recovery
                # state by a previously interrupted camera process.
                "initial_reset": True,
                "pointcloud.enable": False,
                "align_depth.enable": False,
            },
        ],
    )

    depth = Node(
        package="g1_depth",
        executable="depth_policy_preprocessor",
        name="g1_depth_policy_preprocessor",
        output="screen",
        emulate_tty=True,
        parameters=[depth_params],
    )

    livox = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_lidar_publisher",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "user_config_path": livox_config,
                "xfer_format": 1,
                "multi_topic": 0,
                "frame_id": "livox_frame",
            }
        ],
    )

    fastlio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping",
        output="screen",
        emulate_tty=True,
        parameters=[fastlio_config],
    )

    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="g1_world_to_camera_init",
        output="screen",
        arguments=[
            "0", "0", "0",
            "0.9996734", "0", "-0.0255576", "0",
            "g1_world", "camera_init",
        ],
    )

    lidar_mount_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="g1_torso_to_livox",
        output="screen",
        # Previously validated G1 torso -> head-mounted MID-360 transform.
        # Quaternion arguments avoid Foxy's positional Euler ambiguity.
        arguments=[
            "0.0002835", "0.00003", "0.428434",
            "0.999673352", "0", "-0.025557564", "0",
            "torso_link", "livox_frame",
        ],
    )

    elevation = Node(
        package="lidar",
        executable="persistent_elevation_map",
        name="g1_persistent_elevation_map",
        output="screen",
        emulate_tty=True,
        parameters=[elevation_params],
    )

    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_control"), "launch", "real_control.launch.py"]
            )
        ),
        launch_arguments={
            "params_file": control_params,
            "enable_voice": enable_voice,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "control_params",
                default_value=(
                    "/home/unitree/cjh/src/g1_control/config/"
                    "control_first_real.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "depth_params",
                default_value="/home/unitree/cjh/src/g1_depth/config/depth_policy.yaml",
            ),
            DeclareLaunchArgument(
                "realsense_params",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("g1_depth"),
                        "config",
                        "realsense_d435i.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "elevation_params",
                default_value=(
                    "/home/unitree/cjh/src/g1_lidar/config/phase3_persistent.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "livox_config",
                default_value=(
                    "/home/unitree/Livox-SDK2-ros2/src/livox_ros_driver2/"
                    "config/MID360_config.json"
                ),
            ),
            DeclareLaunchArgument(
                "fastlio_config",
                default_value=(
                    "/home/unitree/fastlio2_ws/src/fast_lio/config/g1_mid360.yaml"
                ),
            ),
            DeclareLaunchArgument("enable_voice", default_value="true"),
            camera,
            depth,
            livox,
            fastlio,
            world_tf,
            lidar_mount_tf,
            elevation,
            control,
        ]
    )
