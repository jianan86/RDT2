from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT_URDF = REPO_ROOT / "data/piper_urdf/agx_arm_description/urdf/piper_pika.urdf"
DEFAULT_URDF = Path("/tmp/rdt2_piper_pika_resolved.urdf")
DEFAULT_RVIZ = REPO_ROOT / "data/piper_urdf/rviz/pika_ik.rviz"
DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0601_dex")
AGX_PACKAGE_ROOT = REPO_ROOT / "data/piper_urdf/agx_arm_description"


def prepare_default_urdf() -> Path:
    text = RAW_DEFAULT_URDF.read_text()
    text = text.replace("package://agx_arm_description/", AGX_PACKAGE_ROOT.as_uri() + "/")
    DEFAULT_URDF.write_text(text)
    return DEFAULT_URDF



def launch_value(context: object, name: str) -> str:
    return LaunchConfiguration(name).perform(context).strip()


def launch_bool(context: object, name: str) -> bool:
    value = launch_value(context, name).rstrip(",").lower()
    return value in ("1", "true", "yes", "on")


def make_replay_process(context: object) -> list[ExecuteProcess]:
    episodes = launch_value(context, "episodes").rstrip(",")
    cmd = [
        "python3",
        str(REPO_ROOT / "data/replay_pika.py"),
        "--mode",
        "sim",
        "--input-root",
        launch_value(context, "input_root"),
        "--urdf",
        launch_value(context, "urdf"),
        "--arm-mode",
        launch_value(context, "arm_mode"),
        "--side",
        launch_value(context, "side"),
        "--fps",
        launch_value(context, "fps"),
        "--max-episodes",
        launch_value(context, "max_episodes"),
        "--max-frames",
        launch_value(context, "max_frames"),
        "--failed-policy",
        launch_value(context, "failed_policy"),
        "--init-hold-sec",
        launch_value(context, "init_hold_sec"),
    ]
    if episodes:
        cmd.extend(["--episodes", *episodes.split()])
    if launch_bool(context, "loop"):
        cmd.append("--loop")
    if launch_bool(context, "publish_target_tcp"):
        cmd.append("--publish-target-tcp")
    else:
        cmd.append("--no-publish-target-tcp")
    return [ExecuteProcess(cmd=cmd, output="screen")]


def generate_launch_description() -> LaunchDescription:
    urdf = LaunchConfiguration("urdf")
    rviz_config = LaunchConfiguration("rviz_config")
    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("input_root", default_value=str(DEFAULT_INPUT_ROOT)),
            DeclareLaunchArgument("urdf", default_value=str(prepare_default_urdf())),
            DeclareLaunchArgument("rviz_config", default_value=str(DEFAULT_RVIZ)),
            DeclareLaunchArgument("arm_mode", default_value="dual"),
            DeclareLaunchArgument("side", default_value="both"),
            DeclareLaunchArgument("episodes", default_value=""),
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
            OpaqueFunction(function=make_replay_process),
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
