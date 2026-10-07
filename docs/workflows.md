# Workflow Guide

Install the [environment](installation.md) and prepare the
[producer inputs](../README.md#inputs). Commands run from HUGS-DexGraspBench unless
stated otherwise. The examples assume sibling HUGS repositories.

## Supported Hands and Modes

| Family | Single hand | Dual dummy arm |
| --- | --- | --- |
| Shadow | `shadow` | `dual_dummy_arm_shadow` |
| Leap-SP | `leap_sp` | `dual_dummy_arm_leap_sp` |

IDs are fixed: `1:right_two`, `2:right_three`, `3:right_full`, `4:both_three`,
and `5:both_full`. The first three use the single-hand config; the last two use
the dual config. Batch entry points reject other hands and robot arms.

## BODex Inputs

The batch wrapper reads the producer output root, then appends the manipulation
path, run name, and `graspdata`. With BODex's default output root:

```bash
export HUGS_BODEX_OUTPUT_ROOT="$(realpath ../HUGS-BODex/src/curobo/content/assets/output)"
uv run python script/process_all_grasp_types.py \
  --hand shadow --run-name surface_demo --max-num 20 --dry-run
```

For example, `right_full` reads
`$HUGS_BODEX_OUTPUT_ROOT/sim_shadow/tabletop_full/surface_demo/graspdata/`
and writes to `output/surface_demo_right_full_shadow/`.
Bimanual types read from `sim_dual_dummy_arm_shadow/` and use the matching dual hand.
If BODex used a custom `HUGS_OUTPUT_ROOT`, pass that root instead.
`--bodex-path` can replace the environment variable.

Remove `--dry-run` to run `format`, `eval`, and `collect` in order. Use
`--stage format`, `--stage eval`, or `--stage collect` to run individual stages;
repeat the flag for a subset. A later stage needs the earlier stage's saved outputs.
Use `--grasp-type right_full` to process one mode; repeat it for a subset.
All five modes, including `both_three`, are included by default.

## Learned Robot Inputs

Sample the robot model in [DexLearn](https://github.com/hugs-dex/HUGS-DexLearn#robot-grasp)
before running this workflow:

```bash
uv run python script/process_learning_grasp_types.py \
  --hand shadow --run-name learned_shadow \
  --learning-path ../HUGS-DexLearn/output/shadowMulti_robotMultiHierar_shadow/tests/step_050000/shadowMulti \
  --max-num 20 --n-worker 4 --dry-run
```

Remove `--dry-run` to run `format` then `eval`. This wrapper does not collect
training data automatically. It recursively reads sample NPY files and filters
on `pred_grasp_type_id`; Human Prior exports do not use this interface.

The formatter needs metadata containing the complete source joint order,
including both hands for dual-layout samples. It searches the dataset root
inferred from scene paths and `HUGS_DATASET_ROOT` for:

- `human_DGN2k_full/<hand>/both_full/metadata.json`
- `BimanBODex*/<hand>/both_full/metadata.json`

The metadata must match the model's training data. A sample directory alone
is insufficient. `+task.learning_metadata_group=...` can select another group
when passed as a `--format-arg`, with `both_full` retained as a fallback.
If wrist IK fails, conversion saves NaN qpos and `format_ik_failed=true`;
evaluation counts this as a failure rather than silently omitting it.

## Execution and Reruns

Both wrappers print their commands with `--dry-run`. This does not format,
simulate, or collect records. `--max-num` limits raw files at conversion and
records at evaluation; a raw file may contain multiple candidates. It is not
a fixed count of simulated grasps, and small subsets can leave some types empty.

The base configuration uses `n_worker=96`. The Learning wrapper exposes
`--n-worker` for evaluation and accepts `--format-arg n_worker=4` for conversion.
The BODex wrapper has no worker-count flag. For a small manual run, use explicit
stage commands instead of that wrapper:

```bash
uv run python src/main.py task=format hand=shadow exp_name=surface_demo_right_full \
  task.data_name=BimanBODex \
  "task.data_path=$HUGS_BODEX_OUTPUT_ROOT/sim_shadow/tabletop_full/surface_demo/graspdata" \
  task.max_num=20 n_worker=4

uv run python src/main.py task=eval hand=shadow exp_name=surface_demo_right_full \
  task.max_num=20 task.pose_adjustment.squeeze_extrapolate_ratio=1.0 n_worker=4

uv run python src/main.py task=collect hand=shadow exp_name=surface_demo_right_full n_worker=4
```

This right-full example matches that mode's BODex wrapper override. Other modes
use different overrides; read the wrapper's printed commands before substituting
manual stage execution.

Wrappers prompt before deleting existing selected outputs. Reformatting can remove
the entire experiment directory; reevaluation clears the selected evaluation and
derived success outputs. Use distinct run names to preserve separate experiments.
The Learning wrapper's `--yes` bypasses prompts; it is unnecessary for a first run.

## Metrics and Outputs

Each experiment writes under `output/<run>_<type>_<hand>/`:

`<hand>` is the resolved hand configuration: `shadow` or `leap_sp` for single-hand
modes, and `dual_dummy_arm_shadow` or `dual_dummy_arm_leap_sp` for bimanual modes.

| Directory | Contents |
| --- | --- |
| `graspdata/` | Formatted grasps with metadata and stage qpos |
| `evaluation/` | Evaluated records, success flags, metrics, and failure reasons |
| `succgrasp/` | Links or copies of records passing the simulation criterion |
| `succ_collect/` | Grouped arrays and per-type metadata after `collect` |

See [contracts](contracts.md) for units, WXYZ quaternion order, joint ordering,
fields, and success semantics. The base metric settings live in
[`config/task/eval.yaml`](../config/task/eval.yaml). The wrappers also add
hand/type-specific squeeze adjustments; the Learning wrapper adds a pregrasp
adjustment. Record the final resolved config, rather than only the base YAML,
when comparing results. Include producer and Bench commits, asset revision,
seed, worker count, and MuJoCo version. Simulation success is not physical-robot success.

Direct Hydra tasks support `save_root`; the batch wrappers construct their cleanup
paths under `output/`. Keep default output paths when using those wrappers.

## Viewer

Replay one formatted grasp:

```bash
uv run python src/main.py task=eval hand=shadow exp_name=surface_demo_right_full \
  task.debug_viewer=True task.start=0 task.end=1 n_worker=1
```

This reruns simulation and writes evaluation output. Use a formatted run that has
at least one grasp. The default `mjviser` browser backend binds to
`127.0.0.1:8080`, waits for a client, and holds the final frame. Forward the port
over SSH for remote use. Set `task.viewer.port=8081` for another port or
`task.viewer.backend=mujoco` for the native GUI.

## Prepare a Robot Training Dataset

First run BODex synthesis and Bench `format → eval → collect` on the desired
training collection. A ten-scene quick start only demonstrates the interface;
it may yield empty types and is not a complete training dataset.

The assembly script copies grouped success records and creates per-type metadata:

```bash
uv run python script/concatenate_dataset.py \
  --run_name human_demo --hand_name shadow --data_root output \
  --dataset_name BimanBODex_human_demo
```

This writes under
`$HUGS_DATASET_ROOT/BimanBODex_human_demo/shadow/<grasp_type>/`.
Use the run name of the collection you actually evaluated; `human_demo` is an
example name. The script prompts before replacing an existing target hand
directory. Missing source types are skipped, so inspect its counts and metadata
before training. In particular, the default robot loader needs `both_full`
metadata with the complete joint order. Retain the matching object splits and
partial point clouds.

From DexLearn, point the Robot workflow at the assembled data:

```bash
python -m dexlearn.main task=train algo=robotMultiHierar data=shadowMulti \
  "data.paths.grasp_path=$HUGS_DATASET_ROOT/BimanBODex_human_demo/shadow" exp_name=shadow

python -m dexlearn.main task=sample algo=robotMultiHierar \
  data=shadowMulti test_data=shadowMulti exp_name=shadow ckpt=050000 \
  "data.paths.grasp_path=$HUGS_DATASET_ROOT/BimanBODex_human_demo/shadow" \
  "test_data.grasp_path=$HUGS_DATASET_ROOT/BimanBODex_human_demo/shadow"
```

Apply the same two path overrides to DexLearn's visualization command. Its default
path is `human_DGN2k_full/shadow`, which differs from this example. Naming the
assembled dataset `BimanBODex_human_demo` also lets the Bench Learning formatter
find its metadata on the return evaluation path. Avoid multiple conflicting
metadata sets for the same hand: the formatter uses the first matching candidate.
Robot URDF/mesh assets and the remaining DexLearn prerequisites are described in
[its README](https://github.com/hugs-dex/HUGS-DexLearn#data-and-assets).
