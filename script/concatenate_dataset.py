# In this script, get the succ_collecct in <run_name>_<grasp_type>.
# grasp_type include right_two, right_three, right_full, both_three, both_full.
# The resulted data should be with a structure like:
# <grasp_type>/object_name/xxx/xxx.npy
# The save path of the final dataset should be a argument.

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def _to_jsonable_list(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _count_grasps(grasp_data):
    for key, value in grasp_data.items():
        if "obj" in key or key == "scene_path":
            continue
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, list):
            return len(value)
    raise ValueError("Cannot infer grasp count from grasp data.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Run name prefix")
    parser.add_argument("--dataset_name", type=str, default="BimanBODex", help="Dataset subdirectory name")
    parser.add_argument("--output_root", type=str, default=None, help="Output root")
    parser.add_argument("--data_root", type=str, default="output", help="Data root")
    parser.add_argument(
        "--hand_name",
        type=str,
        default="shadow",
        choices=("shadow", "leap_sp"),
        help="Hand name (e.g., shadow)",
    )
    parser.add_argument("--yes", action="store_true", help="Replace an existing target dataset without prompting.")
    args = parser.parse_args()

    if args.output_root is None:
        dataset_root = os.environ.get("ANYSCALEGRASP_DATA_ROOT") or os.environ.get("AnyScaleGraspDataset")
        if dataset_root is None:
            raise ValueError("Set ANYSCALEGRASP_DATA_ROOT or pass --output_root explicitly.")
        args.output_root = str(Path(dataset_root) / args.dataset_name)

    bimanual_hand_name = f"dual_dummy_arm_{args.hand_name}"
    grasp_types = ["right_two", "right_three", "right_full", "both_three", "both_full"]
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_hand_dir = output_dir / args.hand_name
    if target_hand_dir.exists():
        if not args.yes:
            reply = input(f"Delete existing dataset directory {target_hand_dir}? [y/N] ").strip().lower()
            if reply not in {"y", "yes"}:
                raise SystemExit("refusing to replace existing dataset")
        shutil.rmtree(target_hand_dir)
    target_hand_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_grasps = 0
    stats = {}
    grasp_stats = {}
    metadata_saved = False

    for grasp_type in grasp_types:
        exp_name = f"{args.run_name}_{grasp_type}"
        hand = args.hand_name if "right" in grasp_type else bimanual_hand_name
        succ_collect_dir = Path(args.data_root) / f"{exp_name}_{hand}" / "succ_collect"

        if not succ_collect_dir.exists():
            print(f"Warning: {succ_collect_dir} does not exist, skipping {grasp_type}")
            stats[grasp_type] = 0
            grasp_stats[grasp_type] = 0
            continue

        grasp_type_dir = target_hand_dir / grasp_type
        grasp_type_dir.mkdir(parents=True, exist_ok=True)

        sample_npy = next(succ_collect_dir.rglob("*.npy"), None)
        if sample_npy is not None:
            grasp_data = np.load(sample_npy, allow_pickle=True).item()
            joint_names = grasp_data.get("joint_names")[0]
            wrist_body_names = grasp_data.get("wrist_body_names")[0]
            if joint_names is None:
                print(f"Warning: joint_names not found in {sample_npy}")
            else:
                metadata_path = grasp_type_dir / "metadata.json"
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "bench_contract_version": "1.0",
                            "path_root": "ANYSCALEGRASP_DATA_ROOT",
                            "hand_family": args.hand_name,
                            "grasp_type": grasp_type,
                            "joint_names": _to_jsonable_list(joint_names),
                            "wrist_body_names": _to_jsonable_list(wrist_body_names),
                        },
                        f,
                        indent=2,
                    )
                metadata_saved = True
                print(f"Saved metadata to {metadata_path}")

        count = 0
        grasp_count = 0
        for npy_file in succ_collect_dir.rglob("*.npy"):
            grasp_data = np.load(npy_file, allow_pickle=True).item()
            rel_path = npy_file.relative_to(succ_collect_dir)
            dest_path = grasp_type_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(npy_file, dest_path)
            count += 1
            grasp_count += _count_grasps(grasp_data)

        stats[grasp_type] = count
        grasp_stats[grasp_type] = grasp_count
        total_files += count
        total_grasps += grasp_count
        print(f"{grasp_type}: {count} files, {grasp_count} grasps")

    print(f"\n{'=' * 50}")
    print("Dataset concatenation complete")
    print(f"Output directory: {output_dir}")
    print("\nStatistics:")
    for grasp_type, count in stats.items():
        print(f"  {grasp_type}: {count} files, {grasp_stats[grasp_type]} grasps")
    print(f"  Total: {total_files} files")
    print(f"  Total: {total_grasps} grasps")
    if not metadata_saved:
        print(f"  Warning: metadata.json was not created for any grasp type of {args.hand_name}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
