from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.report_pika_replay_anomalies import (
    EpisodeAnomalyResult,
    SideAnomalyResult,
    build_report,
    format_episode_line,
    normalize_args,
)


def test_side_result_counts_problem_frames():
    result = SideAnomalyResult(total_frames=10, problem_indices=[1, 4, 9])

    assert result.problem_frames == 3
    assert result.problem_ratio == 0.3


def test_episode_result_sums_side_frames():
    result = EpisodeAnomalyResult(
        episode="episode0",
        sides={
            "right": SideAnomalyResult(total_frames=10, problem_indices=[1]),
            "left": SideAnomalyResult(total_frames=12, problem_indices=[2, 3]),
        },
    )

    assert result.problem_frames == 3
    assert result.total_frames == 22
    assert result.problem_ratio == 3 / 22


def test_format_episode_line_reports_total_and_each_side():
    result = EpisodeAnomalyResult(
        episode="episode0",
        sides={
            "right": SideAnomalyResult(total_frames=10, problem_indices=[1]),
            "left": SideAnomalyResult(total_frames=12, problem_indices=[2, 3]),
        },
    )

    assert format_episode_line(result) == "episode0 total 3/22 right 1/10 left 2/12"


def test_build_report_keeps_machine_readable_counts():
    args = argparse.Namespace(
        input_root=Path("/data"),
        urdf=Path("/robot.urdf"),
        arm_mode="dual",
        side="both",
        episodes=None,
        max_sync_delta_sec=0.02,
        gripper_input_max=0.1,
        gripper_output_max=0.088,
        include_problem_indices=True,
    )
    result = EpisodeAnomalyResult(
        episode="episode0",
        sides={"right": SideAnomalyResult(total_frames=2, problem_indices=[1])},
    )

    report = build_report(args, [result])

    assert report["problem_frames"] == 1
    assert report["total_frames"] == 2
    assert report["episodes"][0]["sides"]["right"]["problem_indices"] == [1]


def test_normalize_args_treats_non_positive_limits_as_unbounded():
    args = argparse.Namespace(max_frames=0, max_episodes=-1)

    normalized = normalize_args(args)

    assert normalized.max_frames is None
    assert normalized.max_episodes is None
