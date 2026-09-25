#!/usr/bin/env python3
"""Process AnyScaleDexLearn samples across all benchmark grasp types."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STAGES = ("format", "eval")

HAND_CONFIGS = {
    "shadow": {
        "single_hand": "shadow",
        "dual_hand": "dual_dummy_arm_shadow",
    },
    "leap_sp": {
        "single_hand": "leap_sp",
        "dual_hand": "dual_dummy_arm_leap_sp",
    },
}

GRASP_TYPES = [
    ("right_two", "single"),
    ("right_three", "single"),
    ("right_full", "single"),
    ("both_three", "dual"),
    ("both_full", "dual"),
]
GRASP_TYPE_NAMES = [name for name, _ in GRASP_TYPES]

DEFAULT_PREGRASP_EVAL_HYDRA_ARGS = [
    "task.pose_adjustment.pregrasp_extrapolate_ratio=2.0",
]

DEFAULT_ADDITIONAL_EVAL_HYDRA_ARGS = {
    "right_two": [
        "task.pose_adjustment.squeeze_extrapolate_ratio=0.4",
    ],
    "right_three": [
        "task.pose_adjustment.squeeze_extrapolate_ratio=0.6",
    ],
    "right_full": [
        "task.pose_adjustment.squeeze_extrapolate_ratio=1.0",
    ],
    "both_three": [
        "task.pose_adjustment.squeeze_extrapolate_ratio=2.0",
    ],
    "both_full": [
        "task.pose_adjustment.squeeze_extrapolate_ratio=2.0",
    ],
}

ADDITIONAL_EVAL_HYDRA_ARGS = {
    "default": DEFAULT_ADDITIONAL_EVAL_HYDRA_ARGS,
    "leap_sp": {
        "right_two": [
            "task.pose_adjustment.squeeze_extrapolate_ratio=0.6",
        ],
        "right_three": [
            "task.pose_adjustment.squeeze_extrapolate_ratio=0.8",
        ],
    },
    # "shadow": {
    #     "right_two": [
    #         "task.pose_adjustment.squeeze_extrapolate_ratio=0.6",
    #     ],
    #     "right_three": [
    #         "task.pose_adjustment.squeeze_extrapolate_ratio=0.8",
    #     ],
    # },
}


@dataclass(frozen=True)
class LearningGraspJob:
    """One AnyScaleDexLearn format/eval job for one grasp type."""

    suffix: str
    exp_name: str
    hand: str
    data_path: Path
    output_path: Path
    additional_eval_hydra_args: list[str]

    @property
    def stage_output_paths(self) -> dict[str, list[Path]]:
        """Return output paths written by each selected stage.

        Args:
            None.

        Returns:
            Mapping from stage name to output paths that may be removed before rerun.
        """

        return {
            "format": [self.output_path],
            "eval": [
                self.output_path / "evaluation",
                self.output_path / "succgrasp",
                self.output_path / "debug",
            ],
        }


def shell_text(command: list[str]) -> str:
    """Format a command for readable dry-run and failure output.

    Args:
        command: Command tokens passed to subprocess.

    Returns:
        Shell-escaped command string.
    """

    return " ".join(shlex.quote(part) for part in command)


def parse_stage_list(stages: list[str] | None) -> list[str]:
    """Normalize selected stages to the canonical Learning pipeline order.

    Args:
        stages: Optional stage names from the command line.

    Returns:
        Selected stages ordered as format then eval.
    """

    selected = set(stages or DEFAULT_STAGES)
    return [stage for stage in DEFAULT_STAGES if stage in selected]


def resolve_additional_eval_hydra_args(hand_name: str, grasp_type: str) -> list[str]:
    """Resolve eval-only Hydra overrides for one hand family and grasp type.

    Args:
        hand_name: Hand family selected by CLI, such as `shadow` or `leap_sp`.
        grasp_type: Grasp type suffix, such as `right_two`.

    Returns:
        Hydra override tokens for the evaluation command.
    """

    hand_overrides = ADDITIONAL_EVAL_HYDRA_ARGS.get(hand_name, {})
    additional_args = list(DEFAULT_PREGRASP_EVAL_HYDRA_ARGS)
    if grasp_type in hand_overrides:
        additional_args.extend(hand_overrides[grasp_type])
        return additional_args

    default_overrides = ADDITIONAL_EVAL_HYDRA_ARGS.get("default", {})
    if grasp_type in default_overrides:
        additional_args.extend(default_overrides[grasp_type])
        return additional_args

    raise KeyError(f"missing eval Hydra args for hand={hand_name!r}, grasp_type={grasp_type!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Args:
        None.

    Returns:
        Configured ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=sorted(HAND_CONFIGS), required=True)
    parser.add_argument("--run-name", "--run_name", dest="run_name", required=True)
    parser.add_argument(
        "--learning-path",
        "--learning_path",
        dest="learning_path",
        required=True,
        help="Root folder containing AnyScaleDexLearn sampled .npy files.",
    )
    parser.add_argument("--max-num", "--max_num", dest="max_num", type=int, default=-1)
    parser.add_argument("--n-worker", "--n_worker", dest="n_worker", type=int, default=96)
    parser.add_argument(
        "--grasp-type",
        "--grasp_type",
        dest="grasp_types",
        choices=GRASP_TYPE_NAMES,
        action="append",
        help="Run only one grasp type. Repeat to select multiple. Default: all five grasp types.",
    )
    parser.add_argument(
        "--stage",
        choices=DEFAULT_STAGES,
        action="append",
        help="Run only one pipeline stage. Repeat to select multiple. Default: format and eval.",
    )
    parser.add_argument("--eval-start", "--eval_start", dest="eval_start", type=int, default=None)
    parser.add_argument("--eval-end", "--eval_end", dest="eval_end", type=int, default=None)
    parser.add_argument(
        "--format-arg",
        "--format_arg",
        dest="format_args",
        action="append",
        default=[],
        help="Extra Hydra override appended only to format commands. Repeat as needed.",
    )
    parser.add_argument(
        "--eval-arg",
        "--eval_arg",
        dest="eval_args",
        action="append",
        default=[],
        help="Extra Hydra override appended only to eval commands. Repeat as needed.",
    )
    parser.add_argument("--yes", action="store_true", help="Delete existing selected outputs without prompting.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def build_jobs(args: argparse.Namespace) -> list[LearningGraspJob]:
    """Create Learning jobs for the selected grasp types.

    Args:
        args: Parsed command line arguments.

    Returns:
        Ordered list of jobs.
    """

    hand_config = HAND_CONFIGS[args.hand]
    selected_grasp_types = set(args.grasp_types or GRASP_TYPE_NAMES)
    jobs = []
    for suffix, hand_kind in GRASP_TYPES:
        if suffix not in selected_grasp_types:
            continue
        hand = hand_config[f"{hand_kind}_hand"]
        exp_name = f"{args.run_name}_{suffix}"
        jobs.append(
            LearningGraspJob(
                suffix=suffix,
                exp_name=exp_name,
                hand=hand,
                data_path=Path(args.learning_path),
                output_path=Path("output") / f"{exp_name}_{hand}",
                additional_eval_hydra_args=resolve_additional_eval_hydra_args(args.hand, suffix),
            )
        )
    if not jobs:
        raise SystemExit("no grasp jobs selected")
    return jobs


def should_delete(paths: list[Path], assume_yes: bool) -> bool:
    """Return whether existing output paths should be deleted.

    Args:
        paths: Existing files or folders that would be overwritten.
        assume_yes: Whether deletion is pre-approved by the CLI.

    Returns:
        True when cleanup should proceed.
    """

    if assume_yes:
        return True

    print("Existing output files/folders found:")
    for path in paths:
        print(f"  {path}")
    reply = input("Delete these existing files/folders before processing? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


def cleanup_existing_outputs(jobs: list[LearningGraspJob], stages: list[str], dry_run: bool, assume_yes: bool) -> None:
    """Optionally delete outputs for the selected stages.

    Args:
        jobs: Jobs whose outputs should be checked.
        stages: Pipeline stages selected for this run.
        dry_run: Whether commands are only being printed.
        assume_yes: Whether existing outputs should be deleted without prompt.

    Returns:
        None.
    """

    cleanup_paths = []
    for job in jobs:
        if "format" in stages:
            cleanup_paths.extend(job.stage_output_paths["format"])
            continue
        for stage in stages:
            cleanup_paths.extend(job.stage_output_paths[stage])

    existing_paths = []
    seen_paths = set()
    for path in cleanup_paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        existing_paths.append(path)

    if not existing_paths:
        return
    if dry_run:
        print("Dry run: existing output paths would be checked before processing:")
        for path in existing_paths:
            print(f"  {path}")
        return
    if not should_delete(existing_paths, assume_yes):
        print("Keeping existing output files/folders.")
        return

    for path in existing_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print("Deleted existing output files/folders.")


def format_command(job: LearningGraspJob, max_num: int, extra_args: list[str]) -> list[str]:
    """Build the Learning format command for one grasp type.

    Args:
        job: Learning job metadata.
        max_num: Maximum raw file count passed through Hydra.
        extra_args: Additional Hydra override tokens for format.

    Returns:
        Command tokens for subprocess.
    """

    return [
        sys.executable,
        "src/main.py",
        "task=format",
        f"exp_name={job.exp_name}",
        f"hand={job.hand}",
        "task.data_name=Learning",
        f"task.max_num={max_num}",
        f"task.data_path={job.data_path}",
        *extra_args,
    ]


def eval_command(job: LearningGraspJob, args: argparse.Namespace) -> list[str]:
    """Build the eval command for one grasp type.

    Args:
        job: Learning job metadata.
        args: Parsed command line arguments.

    Returns:
        Command tokens for subprocess.
    """

    command = [
        sys.executable,
        "src/main.py",
        "task=eval",
        f"exp_name={job.exp_name}",
        f"hand={job.hand}",
        "task.debug_viewer=False",
        f"task.max_num={args.max_num}",
        f"n_worker={args.n_worker}",
        *job.additional_eval_hydra_args,
    ]
    if args.eval_start is not None:
        command.append(f"task.start={args.eval_start}")
    if args.eval_end is not None:
        command.append(f"task.end={args.eval_end}")
    command.extend(args.eval_args)
    return command


def run_command(command: list[str], dry_run: bool) -> None:
    """Run or print one subprocess command.

    Args:
        command: Command tokens to execute.
        dry_run: Whether to print instead of executing.

    Returns:
        None. Raises CalledProcessError if the command fails.
    """

    print(shell_text(command))
    if not dry_run:
        subprocess.run(command, check=True)


def process_job(job: LearningGraspJob, index: int, total: int, stages: list[str], args: argparse.Namespace) -> None:
    """Run selected stages for one Learning grasp type.

    Args:
        job: Learning job metadata.
        index: One-based job index for progress output.
        total: Total number of selected jobs.
        stages: Pipeline stages selected for this run.
        args: Parsed command line arguments.

    Returns:
        None.
    """

    print(f"\n[{index}/{total}] Processing {job.suffix}...")
    if "format" in stages:
        run_command(format_command(job, args.max_num, args.format_args), args.dry_run)
    if "eval" in stages:
        run_command(eval_command(job, args), args.dry_run)


def main(argv: list[str] | None = None) -> int:
    """Run the AnyScaleDexLearn all-grasp-type processing pipeline.

    Args:
        argv: Optional argument list for tests; None uses sys.argv.

    Returns:
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    stages = parse_stage_list(args.stage)

    try:
        jobs = build_jobs(args)
        cleanup_existing_outputs(jobs, stages, args.dry_run, args.yes)
        print(f"Processing AnyScaleDexLearn samples for hand: {args.hand}, run: {args.run_name}")
        print(f"Learning path: {args.learning_path}")
        print(f"Selected grasp types: {', '.join(job.suffix for job in jobs)}")
        print(f"Selected stages: {', '.join(stages)}")
        print(f"max_num: {args.max_num}")
        print("================================================")
        for job_index, job in enumerate(jobs, start=1):
            process_job(job, job_index, len(jobs), stages, args)
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}: {shell_text(exc.cmd)}", file=sys.stderr)
        return exc.returncode
    except SystemExit as exc:
        parser.error(str(exc))

    print("\n================================================")
    print("All selected AnyScaleDexLearn grasp types processed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
