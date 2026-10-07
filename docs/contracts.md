# HUGS-DexGraspBench contract

This document defines the boundary between a grasp producer and the Bench. The
producer owns candidate generation and source data; the Bench owns conversion,
MuJoCo filtering, result records, and grouped successful samples.

## Input

BimanBODex records contain `robot_pose`, `joint_names`, and `scene_path`. Learning
records additionally contain `pred_grasp_type_id`, metadata, and stage qpos. The
five IDs are stable: `1:right_two`, `2:right_three`, `3:right_full`,
`4:both_three`, `5:both_full`. IDs outside this set are rejected.

`scene_path` is resolved against `HUGS_DATASET_ROOT` and must identify a
scene config whose target object contains `file_path`, `pose`, and `scale`.
Absolute paths are accepted for legacy inputs. New records never persist an
absolute dataset root.

## Units and order

Translations and distances are metres; object mass is kilograms; density is
kg/m³. Quaternions use `[w, x, y, z]`. `joint_names` is the authoritative qpos
order. Dual records keep right and left wrist and hand fields distinct; missing
metadata or a mismatched qpos length is an error.

## Formatted record

Every output `.npy` includes `obj_path` and `scene_path` as POSIX paths relative
to the dataset root, `obj_scale`, `obj_pose`, `joint_names`, stage qpos fields,
`bench_contract_version: "1.0"`, and `path_root:
"HUGS_DATASET_ROOT"`. Learning records retain `pred_grasp_type_id` and
`pred_grasp_type`.

If Learning wrist IK fails, the formatter writes `format_ik_failed: true`,
`format_ik_failed_side`, `format_ik_failed_stage`, and NaN stage qpos. Evaluation
records classify this as `eval_failure_reason: "format_ik_failed"`.

## Evaluation and collection

Evaluation preserves the formatted fields and adds `succ_flag`,
`eval_failure_reason` when applicable, simulation deltas, analytic force-closure
metrics, penetration/contact metrics, resolved object-physics diagnostics, and
robot poses extracted from the evaluated MuJoCo state. `succgrasp` is a link or
copy of records whose configured simulation criterion passed. It is not a
physical-robot success claim. `succ_collect` groups records by their relative
object/scene folder and writes `metadata.json` with joint order, hand family,
grasp type, and contract version.

The reproducibility tuple for any reported run is: producer commit, Bench commit,
input data/asset revision, resolved evaluation config, seed, worker count,
Python/MuJoCo versions, and output manifest. Metric threshold or pose-adjustment
changes require a new configuration version.
