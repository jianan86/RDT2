from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "data/piper_urdf/agx_arm_description/urdf/piper_pika.urdf"
DEFAULT_RVIZ = REPO_ROOT / "data/piper_urdf/rviz/pika_ik.rviz"
DEFAULT_TRAJECTORY = Path("/home/workspace/pika_ik_traj/episode0_right.npz")


def generate_launch_description() -> LaunchDescription:
    trajectory = LaunchConfiguration("trajectory")
    urdf = LaunchConfiguration("urdf")
    rviz_config = LaunchConfiguration("rviz_config")
    fps = LaunchConfiguration("fps")
    loop = LaunchConfiguration("loop")
    failed_policy = LaunchConfiguration("failed_policy")
    publish_target_tcp = LaunchConfiguration("publish_target_tcp")
    target_tcp_frame = LaunchConfiguration("target_tcp_frame")
    target_tcp_parent = LaunchConfiguration("target_tcp_parent")
    use_rviz = LaunchConfiguration("use_rviz")

    player_cmd = [
        "python3",
        str(REPO_ROOT / "data/play_pika_ik_rviz.py"),
        "--trajectory",
        trajectory,
        "--fps",
        fps,
        "--failed-policy",
        failed_policy,
        "--target-tcp-frame",
        target_tcp_frame,
        "--target-tcp-parent",
        target_tcp_parent,
    ]
    player_cmd_with_tcp = player_cmd + ["--publish-target-tcp"]
    player_cmd_without_tcp = player_cmd + ["--no-publish-target-tcp"]

    return LaunchDescription(
        [
            DeclareLaunchArgument("trajectory", default_value=str(DEFAULT_TRAJECTORY)),
            DeclareLaunchArgument("urdf", default_value=str(DEFAULT_URDF)),
            DeclareLaunchArgument("rviz_config", default_value=str(DEFAULT_RVIZ)),
            DeclareLaunchArgument("fps", default_value="30"),
            DeclareLaunchArgument("loop", default_value="true"),
            DeclareLaunchArgument("failed_policy", default_value="hold"),
            DeclareLaunchArgument("publish_target_tcp", default_value="true"),
            DeclareLaunchArgument("target_tcp_frame", default_value="target_tcp"),
            DeclareLaunchArgument("target_tcp_parent", default_value="world"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                arguments=[urdf],
            ),
            ExecuteProcess(
                cmd=player_cmd_with_tcp + ["--loop"],
                output="screen",
                condition=IfCondition(PythonExpression(["'", loop, "' == 'true' and '", publish_target_tcp, "' == 'true'"])),
            ),
            ExecuteProcess(
                cmd=player_cmd_with_tcp,
                output="screen",
                condition=IfCondition(PythonExpression(["'", loop, "' != 'true' and '", publish_target_tcp, "' == 'true'"])),
            ),
            ExecuteProcess(
                cmd=player_cmd_without_tcp + ["--loop"],
                output="screen",
                condition=IfCondition(PythonExpression(["'", loop, "' == 'true' and '", publish_target_tcp, "' != 'true'"])),
            ),
            ExecuteProcess(
                cmd=player_cmd_without_tcp,
                output="screen",
                condition=IfCondition(PythonExpression(["'", loop, "' != 'true' and '", publish_target_tcp, "' != 'true'"])),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
