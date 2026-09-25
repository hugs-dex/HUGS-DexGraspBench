"""
Check Data Consistency Across Grasp Directories

This script verifies data consistency between graspdata, evaluation, and succgrasp directories.

Usage:
    python test/check_data_consistency.py <exp_name>

Example:
    python test/check_data_consistency.py minitest_right_full_shadow

The script will:
1. Find corresponding files across all three directories
2. Display keys and data types for each directory
3. Check consistency between graspdata and evaluation
4. Verify that succgrasp correctly links to evaluation files
"""

import os
import sys
from glob import glob
import numpy as np


def test_data_keys(exp_name):
    output_dir = f"./output/{exp_name}"

    # Find sample files
    grasp_files = glob(os.path.join(output_dir, "graspdata/**/*.npy"), recursive=True)
    eval_files = glob(os.path.join(output_dir, "evaluation/**/*.npy"), recursive=True)
    succ_files = glob(os.path.join(output_dir, "succgrasp/**/*.npy"), recursive=True)

    print(f"Testing data keys for experiment: {exp_name}")
    print("=" * 60)

    # Find corresponding files by matching relative paths
    grasp_data = None
    eval_data = None
    succ_data = None
    grasp_file = None
    eval_file = None
    succ_file = None

    # Find a file that exists in all three directories
    if grasp_files:
        for gf in grasp_files:
            rel_path = os.path.relpath(gf, os.path.join(output_dir, "graspdata"))
            ef = os.path.join(output_dir, "evaluation", rel_path)
            sf = os.path.join(output_dir, "succgrasp", rel_path)

            if os.path.exists(ef) and os.path.exists(sf):
                grasp_file = gf
                eval_file = ef
                succ_file = sf
                print("\nUsing corresponding files:")
                print(f"  Relative path: {rel_path}")
                print("  All three files exist: ✅")
                break

        if not grasp_file:
            # Fallback: use first grasp file even if others don't exist
            grasp_file = grasp_files[0]
            rel_path = os.path.relpath(grasp_file, os.path.join(output_dir, "graspdata"))
            eval_file = os.path.join(output_dir, "evaluation", rel_path)
            succ_file = os.path.join(output_dir, "succgrasp", rel_path)
            print("\nUsing files (not all exist):")
            print(f"  Relative path: {rel_path}")
            print("  Grasp: exists")
            print(f"  Eval: {'exists' if os.path.exists(eval_file) else 'NOT FOUND'}")
            print(f"  Succ: {'exists' if os.path.exists(succ_file) else 'NOT FOUND'}")

    # Test graspdata
    if grasp_files:
        print(f"\nGRASPDATA ({len(grasp_files)} files)")
        print("-" * 60)
        grasp_data = np.load(grasp_file, allow_pickle=True).item()
        print(f"Sample file: {os.path.basename(grasp_file)}")
        print(f"Keys: {sorted(grasp_data.keys())}")
        for key in sorted(grasp_data.keys()):
            value = grasp_data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__}")

    # Test evaluation
    if os.path.exists(eval_file):
        print(f"\nEVALUATION ({len(eval_files)} files)")
        print("-" * 60)
        eval_data = np.load(eval_file, allow_pickle=True).item()
        print(f"Sample file: {os.path.basename(eval_file)}")
        print(f"Keys: {sorted(eval_data.keys())}")
        for key in sorted(eval_data.keys()):
            value = eval_data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")

    # Test succgrasp
    if os.path.exists(succ_file):
        print(f"\nSUCCGRASP ({len(succ_files)} files)")
        print("-" * 60)
        # Follow symlink to actual file
        actual_file = os.path.realpath(succ_file)
        succ_data = np.load(actual_file, allow_pickle=True).item()
        print(f"Sample file: {os.path.basename(succ_file)} -> {os.path.basename(actual_file)}")
        print(f"Keys: {sorted(succ_data.keys())}")
        for key in sorted(succ_data.keys()):
            value = succ_data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__}")

    # Check consistency between graspdata and evaluation
    if grasp_data and eval_data:
        print("\nCONSISTENCY CHECK: GRASPDATA vs EVALUATION")
        print("-" * 60)
        common_keys = set(grasp_data.keys()) & set(eval_data.keys())
        print(f"Common keys: {sorted(common_keys)}")

        inconsistent = []
        for key in sorted(common_keys):
            grasp_val = grasp_data[key]
            eval_val = eval_data[key]
            if isinstance(grasp_val, np.ndarray) and isinstance(eval_val, np.ndarray):
                if not np.array_equal(grasp_val, eval_val):
                    inconsistent.append(key)
                    print(f"  ❌ {key}: VALUES DIFFER")
                else:
                    print(f"  ✅ {key}: consistent")
            elif grasp_val != eval_val:
                inconsistent.append(key)
                print(f"  ❌ {key}: VALUES DIFFER")
            else:
                print(f"  ✅ {key}: consistent")

        if inconsistent:
            print(f"\n⚠️  Found {len(inconsistent)} inconsistent keys: {inconsistent}")
        else:
            print("\n✅ All common keys are consistent!")

    # Check consistency between evaluation and succgrasp
    if eval_data and succ_data:
        print("\nCONSISTENCY CHECK: EVALUATION vs SUCCGRASP")
        print("-" * 60)

        # Check if they are identical (since succ should be symlink to eval)
        if set(eval_data.keys()) == set(succ_data.keys()):
            print("✅ Same keys in both files")

            all_match = True
            for key in eval_data.keys():
                eval_val = eval_data[key]
                succ_val = succ_data[key]
                if isinstance(eval_val, np.ndarray) and isinstance(succ_val, np.ndarray):
                    if not np.array_equal(eval_val, succ_val):
                        print(f"  ❌ {key}: VALUES DIFFER")
                        all_match = False
                elif eval_val != succ_val:
                    print(f"  ❌ {key}: VALUES DIFFER")
                    all_match = False

            if all_match:
                print("✅ All values match - succgrasp correctly links to evaluation")
        else:
            print("❌ Different keys in evaluation and succgrasp")
            eval_only = set(eval_data.keys()) - set(succ_data.keys())
            succ_only = set(succ_data.keys()) - set(eval_data.keys())
            if eval_only:
                print(f"  Keys only in evaluation: {sorted(eval_only)}")
            if succ_only:
                print(f"  Keys only in succgrasp: {sorted(succ_only)}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Graspdata files: {len(grasp_files)}")
    print(f"  Evaluation files: {len(eval_files)}")
    print(f"  Successful files: {len(succ_files)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_data_keys.py <exp_name>")
        sys.exit(1)
    test_data_keys(sys.argv[1])
