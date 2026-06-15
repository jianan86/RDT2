from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "data/piper_urdf/agx_arm_description/urdf/piper_pika.urdf"
DEFAULT_RVIZ = REPO_ROOT / "data/piper_urdf/rviz/pika_ik.rviz"
DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0601_dex")


def generate_launch_description() -> LaunchDescription:
    input_root = LaunchConfiguration("input_root")
    urdf = LaunchConfiguration("urdf")
    rviz_config = LaunchConfiguration("rviz_config")
    arm_mode = LaunchConfiguration("arm_mode")
    side = LaunchConfiguration("side")
    fps = LaunchConfiguration("fps")
    max_episodes = LaunchConfiguration("max_episodes")
    max_frames = LaunchConfiguration("max_frames")
    loop = LaunchConfiguration("loop")
    failed_policy = LaunchConfiguration("failed_policy")
    init_hold_sec = LaunchConfiguration("init_hold_sec")
    publish_target_tcp = LaunchConfiguration("publish_target_tcp")
    use_rviz = LaunchConfiguration("use_rviz")

    replay_cmd = [
        "python3",
        str(REPO_ROOT / "data/replay_pika.py"),
        "--mode",
        "sim",
        "--input-root",
        input_root,
        "--urdf",
        urdf,
        "--arm-mode",
        arm_mode,
        "--side",
        side,
        "--fps",
        fps,
        "--max-episodes",
        max_episodes,
        "--max-frames",
        max_frames,
        "--failed-policy",
        failed_policy,
        "--init-hold-sec",
        init_hold_sec,
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("input_root", default_value=str(DEFAULT_INPUT_ROOT)),
            DeclareLaunchArgument("urdf", default_value=str(DEFAULT_URDF)),
            DeclareLaunchArgument("rviz_config", default_value=str(DEFAULT_RVIZ)),
            DeclareLaunchArgument("arm_mode", default_value="dual"),
            DeclareLaunchArgument("side", default_value="both"),
            DeclareLaunchArgument("fps", default_value="30"),
            DeclareLaunchArgument("max_episodes", default_value="1"),
            DeclareLaunchArgument("max_frames", default_value="0"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("failed_policy", default_value="hold"),
            DeclareLaunchArgument("init_hold_sec", default_value="1.0"),
            DeclareLaunchArgument("publish_target_tcp", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                arguments=[urdf],
            ),
            ExecuteProcess(
                cmd=replay_cmd + ["--loop", "--publish-target-tcp"],
                output="screen",
                condition=IfCondition(PythonExpression(["'", loop, "' == 'true' and '", publish_target_tcp, "' == 'true'"])),
            ),
            ExecuteProcess(
                cmd=replay_cmd + ["--publish-target-tcp"],
                output="screen",
                condition=IfCondition(PythonExpression(["'", loop, "' != 'true' and '", publish_target_tcp, "' == 'true'"])),
            ),
            ExecuteProcess(
                cmd=replay_cmd + ["--loop", "--no-publish-target-tcp"],
                output="screen",
                condition=IfCondition(PythonExpression(["'", loop, "' == 'true' and '", publish_target_tcp, "' != 'true'"])),
            ),
            ExecuteProcess(
                cmd=replay_cmd + ["--no-publish-target-tcp"],
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
