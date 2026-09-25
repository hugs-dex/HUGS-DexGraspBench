#!/usr/bin/env python3
"""Process all tabletop grasp types exported by BimanBODex."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BODEX_PATH = os.environ.get("HUGS_BODEX_OUTPUT_ROOT")
DEFAULT_STAGES = ("format", "eval", "collect")

HAND_CONFIGS = {
    "shadow": {
        "single_hand": "shadow",
        "dual_hand": "dual_dummy_arm_shadow",
        "single_dataset": "sim_shadow",
        "dual_dataset": "sim_dual_dummy_arm_shadow",
    },
    "leap_sp": {
        "single_hand": "leap_sp",
        "dual_hand": "dual_dummy_arm_leap_sp",
        "single_dataset": "sim_leap_sp",
        "dual_dataset": "sim_dual_dummy_arm_leap_sp",
    },
}

GRASP_TYPES = [
    ("right_two", "single", "tabletop_two"),
    ("right_three", "single", "tabletop_three"),
    ("right_full", "single", "tabletop_full"),
    ("both_three", "dual", "tabletop_three"),
    ("both_full", "dual", "tabletop_full"),
]

GRASP_TYPE_NAMES = [name for name, _, _ in GRASP_TYPES]

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
    # Add per-hand overrides here. Missing grasp types fall back to "default".
    "leap_sp": {
        "right_two": [
            "task.pose_adjustment.squeeze_extrapolate_ratio=0.6",
        ],
        "right_three": [
            "task.pose_adjustment.squeeze_extrapolate_ratio=0.8",
        ],
    },
}


@dataclass(frozen=True)
class GraspJob:
    """One format/eval/collect job for a single grasp type."""

    suffix: str
    exp_name: str
    hand: str
    dataset: str
    tabletop_split: str
    data_path: Path
    output_path: Path
    additional_eval_hydra_args: list[str]

    @property
    def stage_output_paths(self) -> dict[str, list[Path]]:
        """Return output paths written by each pipeline stage.

        Args:
            None.

        Returns:
            Mapping from stage name to folders that should be cleared before rerunning that stage.
        """

        return {
            "format": [self.output_path],
            "eval": [
                self.output_path / "evaluation",
                self.output_path / "succgrasp",
                self.output_path / "debug",
                self.output_path / "succ_collect",
            ],
            "collect": [self.output_path / "succ_collect"],
        }


def shell_text(command: list[str]) -> str:
    """Format a command for readable dry-run and error output.

    Args:
        command: Command tokens that will be passed to subprocess.

    Returns:
        Shell-escaped command string.
    """

    return " ".join(shlex.quote(part) for part in command)


def parse_stage_list(stages: list[str] | None) -> list[str]:
    """Normalize selected stages to the canonical pipeline order.

    Args:
        stages: Optional stage names from the command line.

    Returns:
        Selected stages ordered as format, eval, collect.
    """

    selected = set(stages or DEFAULT_STAGES)
    return [stage for stage in DEFAULT_STAGES if stage in selected]


def parse_run_name_list(run_name_args: list[str]) -> list[str]:
    """Normalize one or more --run-name values into an ordered unique list.

    Args:
        run_name_args: Raw command line values collected from --run-name.

    Returns:
        Ordered run names with comma-separated items expanded and duplicates removed.
    """

    normalized_run_names = []
    seen_run_names = set()
    for raw_value in run_name_args:
        for run_name in raw_value.split(","):
            normalized_name = run_name.strip()
            if not normalized_name or normalized_name in seen_run_names:
                continue
            seen_run_names.add(normalized_name)
            normalized_run_names.append(normalized_name)
    return normalized_run_names


def resolve_additional_eval_hydra_args(hand_name: str, grasp_type: str) -> list[str]:
    """Resolve default eval-only Hydra overrides for one hand and grasp type.

    Args:
        hand_name: Hand family selected by --hand, such as shadow or leap_sp.
        grasp_type: Grasp type suffix, such as right_full or both_full.

    Returns:
        A copy of Hydra override tokens. Hand-specific entries override the default table.
    """

    hand_overrides = ADDITIONAL_EVAL_HYDRA_ARGS.get(hand_name, {})
    if grasp_type in hand_overrides:
        return list(hand_overrides[grasp_type])

    default_overrides = ADDITIONAL_EVAL_HYDRA_ARGS.get("default", {})
    if grasp_type in default_overrides:
        return list(default_overrides[grasp_type])

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
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_names",
        nargs="+",
        required=True,
        help="One or more run names. Multiple values can be passed as space-separated or comma-separated items.",
    )
    parser.add_argument("--bodex-path", "--bodex_path", dest="bodex_path", default=DEFAULT_BODEX_PATH)
    parser.add_argument("--max-num", "--max_num", dest="max_num", type=int, default=-1)
    parser.add_argument(
        "--grasp-type",
        "--grasp_type",
        dest="grasp_types",
        choices=GRASP_TYPE_NAMES,
        action="append",
        help="Run only one grasp type. Repeat to select multiple. Default: all five grasp types.",
    )
    both_three_group = parser.add_mutually_exclusive_group()
    both_three_group.add_argument(
        "--include-both-three",
        "--include_both_three",
        dest="include_both_three",
        action="store_true",
        default=True,
        help="Include both_three in processing. This is the default.",
    )
    both_three_group.add_argument(
        "--exclude-both-three",
        "--exclude_both_three",
        dest="include_both_three",
        action="store_false",
        help="Exclude both_three before building format/eval/collect jobs.",
    )
    parser.add_argument(
        "--stage",
        choices=DEFAULT_STAGES,
        action="append",
        help="Run only one pipeline stage. Repeat to select multiple. Default: format, eval, collect.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def build_jobs(args: argparse.Namespace) -> list[GraspJob]:
    """Create the five grasp processing jobs for the selected hand family.

    Args:
        args: Parsed command line arguments, including grasp-type filters.

    Returns:
        Ordered list of grasp jobs.
    """

    return build_jobs_for_run_name(args, args.run_name)


def build_jobs_for_run_name(args: argparse.Namespace, run_name: str) -> list[GraspJob]:
    """Create the grasp processing jobs for one selected run name.

    Args:
        args: Parsed command line arguments, including grasp-type filters.
        run_name: Run name whose exported graspdata should be processed.

    Returns:
        Ordered list of grasp jobs for the provided run name.
    """

    if not args.bodex_path:
        raise SystemExit("set HUGS_BODEX_OUTPUT_ROOT or pass --bodex-path")
    hand_config = HAND_CONFIGS[args.hand]
    jobs = []
    selected_grasp_types = set(args.grasp_types or GRASP_TYPE_NAMES)
    if not args.include_both_three:
        selected_grasp_types.discard("both_three")
    for suffix, hand_kind, tabletop_split in GRASP_TYPES:
        if suffix not in selected_grasp_types:
            continue
        hand = hand_config[f"{hand_kind}_hand"]
        dataset = hand_config[f"{hand_kind}_dataset"]
        exp_name = f"{run_name}_{suffix}"
        jobs.append(
            GraspJob(
                suffix=suffix,
                exp_name=exp_name,
                hand=hand,
                dataset=dataset,
                tabletop_split=tabletop_split,
                data_path=Path(args.bodex_path) / dataset / tabletop_split / run_name / "graspdata",
                output_path=Path("output") / f"{exp_name}_{hand}",
                additional_eval_hydra_args=resolve_additional_eval_hydra_args(args.hand, suffix),
            )
        )
    return jobs


def build_jobs_for_run_names(args: argparse.Namespace, run_names: list[str]) -> list[tuple[str, list[GraspJob]]]:
    """Create grasp jobs for every selected run name before processing starts.

    Args:
        args: Parsed command line arguments, including hand and grasp-type filters.
        run_names: Ordered run names normalized from the command line.

    Returns:
        Ordered pairs of run name and its grasp jobs.
    """

    jobs_by_run_name = []
    for run_name in run_names:
        jobs = build_jobs_for_run_name(args, run_name)
        if not jobs:
            raise SystemExit("no grasp jobs selected")
        jobs_by_run_name.append((run_name, jobs))
    return jobs_by_run_name


def should_delete(paths: list[Path]) -> bool:
    """Ask whether existing output paths should be deleted.

    Args:
        paths: Existing files or folders that would be overwritten by this run.

    Returns:
        True if the user confirms deletion; otherwise False.
    """

    print("Existing output files/folders found:")
    for path in paths:
        print(f"  {path}")
    reply = input("Delete these existing files/folders before processing? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


def cleanup_existing_outputs(jobs: list[GraspJob], stages: list[str], dry_run: bool) -> None:
    """Optionally delete existing output paths for the selected stages.

    Args:
        jobs: Jobs whose output folders should be checked.
        stages: Pipeline stages selected for this run.
        dry_run: Whether commands are only being printed.

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
    if not should_delete(existing_paths):
        print("Keeping existing output files/folders.")
        return

    for path in existing_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print("Deleted existing output files/folders.")


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


def format_command(job: GraspJob, max_num: int) -> list[str]:
    """Build the format command for one grasp job.

    Args:
        job: Grasp job metadata.
        max_num: Maximum raw item count passed through Hydra.

    Returns:
        Command tokens for subprocess.
    """

    return [
        sys.executable,
        "src/main.py",
        "task=format",
        f"exp_name={job.exp_name}",
        f"hand={job.hand}",
        "task.data_name=BimanBODex",
        f"task.max_num={max_num}",
        f"task.data_path={job.data_path}",
    ]


def eval_command(job: GraspJob, max_num: int) -> list[str]:
    """Build the eval command for one grasp job.

    Args:
        job: Grasp job metadata.
        max_num: Maximum eval item count passed through Hydra.

    Returns:
        Command tokens for subprocess.
    """

    return [
        sys.executable,
        "src/main.py",
        "task=eval",
        f"exp_name={job.exp_name}",
        f"hand={job.hand}",
        "task.debug_viewer=False",
        f"task.max_num={max_num}",
        *job.additional_eval_hydra_args,
    ]


def collect_command(job: GraspJob) -> list[str]:
    """Build the collect command for one grasp job.

    Args:
        job: Grasp job metadata.

    Returns:
        Command tokens for subprocess.
    """

    return [sys.executable, "src/main.py", "task=collect", f"exp_name={job.exp_name}", f"hand={job.hand}"]


def process_job(job: GraspJob, index: int, total: int, stages: list[str], max_num: int, dry_run: bool) -> None:
    """Run format, eval, and collect for one grasp type.

    Args:
        job: Grasp job metadata.
        index: One-based job index for progress output.
        total: Total number of jobs in this run.
        stages: Pipeline stages selected for this run.
        max_num: Maximum raw/eval item count passed through Hydra.
        dry_run: Whether to print commands instead of executing them.

    Returns:
        None. Raises CalledProcessError if any command fails.
    """

    print(f"\n[{index}/{total}] Processing {job.suffix}...")
    if "format" in stages:
        run_command(format_command(job, max_num), dry_run)
    if "eval" in stages:
        run_command(eval_command(job, max_num), dry_run)
    if "collect" in stages:
        run_command(collect_command(job), dry_run)


def process_run_name(
    args: argparse.Namespace,
    run_name: str,
    jobs: list[GraspJob],
    run_index: int,
    run_total: int,
    stages: list[str],
) -> None:
    """Run the selected stages for one run name across all chosen grasp types.

    Args:
        args: Parsed command line arguments.
        run_name: Run name to process.
        jobs: Grasp jobs already built for this run name.
        run_index: One-based run index for progress output.
        run_total: Total number of run names in this invocation.
        stages: Pipeline stages selected for this run.

    Returns:
        None. Raises SystemExit or CalledProcessError on failure.
    """

    print(f"Processing all grasp types for hand: {args.hand}, run: {run_name} ({run_index}/{run_total})")
    print(f"Selected grasp types: {', '.join(job.suffix for job in jobs)}")
    print(f"Selected stages: {', '.join(stages)}")
    print(f"BimanBODex path: {args.bodex_path}")
    print(f"max_num: {args.max_num}")
    print("================================================")

    for job_index, job in enumerate(jobs, start=1):
        process_job(job, job_index, len(jobs), stages, args.max_num, args.dry_run)


def main(argv: list[str] | None = None) -> int:
    """Run the all-grasp-type processing pipeline.

    Args:
        argv: Optional argument list for tests; None uses sys.argv.

    Returns:
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_name_list = parse_run_name_list(args.run_names)
    if not args.run_name_list:
        parser.error("no run names selected")
    stages = parse_stage_list(args.stage)

    try:
        jobs_by_run_name = build_jobs_for_run_names(args, args.run_name_list)
        all_jobs = [job for _, jobs in jobs_by_run_name for job in jobs]
        cleanup_existing_outputs(all_jobs, stages, args.dry_run)
        for run_index, (run_name, jobs) in enumerate(jobs_by_run_name, start=1):
            args.run_name = run_name
            process_run_name(args, run_name, jobs, run_index, len(jobs_by_run_name), stages)
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}: {shell_text(exc.cmd)}", file=sys.stderr)
        return exc.returncode
    except SystemExit as exc:
        parser.error(str(exc))

    print("\n================================================")
    print("All selected run names and grasp types processed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
