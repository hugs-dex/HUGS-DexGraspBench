#!/usr/bin/env bash
set -euo pipefail

# Set HUGS_BENCH_EXP to an existing formatted run before executing this example.
EXP_NAME="${HUGS_BENCH_EXP:?set HUGS_BENCH_EXP to an existing formatted run}"
python src/main.py task=eval exp_name="$EXP_NAME" hand=shadow task.debug_viewer=False
python src/main.py task=stat exp_name="$EXP_NAME" hand=shadow
