# HUGS-DexGraspBench

MuJoCo-based format conversion, simulation filtering, and success-sample collection for HUGS grasp producers. The Bench consumes producer artifacts; it does not redistribute producer datasets, checkpoints, or object meshes.

This candidate was extracted from `BimanDexGraspBench` at `76378fc4ac4c7cb472fa63760af482afcb1c8412`. It has a new repository history. The project-wide license decision is still pending, so this repository does not make an OSI-license claim.

## Supported matrix

The first release supports these four hand configurations:

| Family | Single hand | Dual dummy arm |
| --- | --- | --- |
| Shadow | `shadow` | `dual_dummy_arm_shadow` |
| Leap-SP | `leap_sp` | `dual_dummy_arm_leap_sp` |

All five tabletop grasp types are fixed by the contract: `1:right_two`, `2:right_three`, `3:right_full`, `4:both_three`, and `5:both_full`. The first three use the single-hand configuration; the last two use its dual dummy-arm configuration. Other hands and robot arms are outside this release and are rejected by the batch entry points.

## Installation

Use Python 3.10 with the versions validated by the source project:

```bash
conda create -n hugs-dexgraspbench python=3.10
conda activate hugs-dexgraspbench
pip install numpy==1.26.4 mujoco==3.6.0 mjviser==0.0.14 viser==1.0.27 \
  pillow==12.2.0 trimesh==4.11.5 hydra-core transforms3d matplotlib \
  scikit-learn imageio tqdm 'qpsolvers[clarabel]'
git submodule update --init --recursive
pip install -e ./third_party/pytorch_kinematics -e ./third_party/utils_python
```

MuJoCo Menagerie is required by the Shadow MJCF. No USD or `pxr` dependency is part of this release.

Set the external dataset root before evaluating records. It must contain the producer's `object/` tree (including `info/simplified.json` for each object):

```bash
export ANYSCALEGRASP_DATA_ROOT=/path/to/AnyScaleGrasp
```

New converted records store `scene_path` and `obj_path` relative to this root. The legacy `AnyScaleGraspDataset` variable is accepted only when reading older records.

## Producer workflows

Format a BimanBODex sample directory supplied by the producer:

```bash
python src/main.py task=format hand=shadow \
  exp_name=my_run_right_full task.data_name=BimanBODex \
  task.data_path=/path/to/producer/graspdata task.max_num=10
```

Run headless MuJoCo evaluation and collect successful samples:

```bash
python src/main.py task=eval hand=shadow exp_name=my_run_right_full \
  task.debug_viewer=False n_worker=1 task.start=0 task.end=10
python src/main.py task=collect hand=shadow exp_name=my_run_right_full n_worker=1
```

The five-type batch wrapper accepts a producer output root through `HUGS_BODEX_OUTPUT_ROOT` or `--bodex-path`. It prompts before deleting selected outputs; `--dry-run` prints the exact commands without running them:

```bash
export HUGS_BODEX_OUTPUT_ROOT=/path/to/BimanBODex/output
python script/process_all_grasp_types.py --hand shadow --run-name my_run --dry-run
```

AnyScaleDexLearn samples use the corresponding wrapper. It filters on `pred_grasp_type_id`, requires metadata with the complete joint order, and saves NaN qpos plus an explicit `format_ik_failed` sentinel when IK cannot solve a sample:

```bash
python script/process_learning_grasp_types.py --hand leap_sp \
  --run-name learned_run --learning-path /path/to/learning/samples --dry-run
```

## Paths, units, and output contract

The format contract uses object and scene identifiers from the producer, metre translations, `[w, x, y, z]` quaternions, kilogram mass, kg/m³ density, and metre contact-distance thresholds. Joint names and qpos are always saved in the metadata order; dual records contain right and left wrist/hand fields. The grasp-type IDs above are never renumbered.

Each formatted `.npy` contains at least `obj_path`, `scene_path`, `obj_scale`, `obj_pose`, the stage qpos fields, `joint_names`, `bench_contract_version`, and `path_root`. Evaluation records preserve those fields and add `succ_flag`, `eval_failure_reason` when applicable, simulation/analytic/contact metrics, resolved object physics diagnostics, and the evaluated robot poses. `succgrasp` contains only records passing the configured simulation criterion; it is not a claim of physical-robot success. `succ_collect` contains grouped arrays and a per-grasp-type `metadata.json`.

The default metric configuration is in `config/task/eval.yaml`. It fixes the simulation thresholds, friction, density policy, pose-adjustment semantics, MuJoCo arena capacity, and viewer defaults. Changing a threshold or default is a new configuration version and must be recorded with the producer commit, Bench commit, asset revision, seed, worker count, and MuJoCo version.

## Viewer

Headless evaluation is the default. For one grasp, enable the `mjviser` browser viewer and forward its localhost port over SSH:

```bash
python src/main.py task=eval hand=shadow exp_name=my_run_right_full \
  task.debug_viewer=True task.start=0 task.end=1
```

Use `task.viewer.backend=mujoco` for the native MuJoCo GUI. The browser backend binds to `127.0.0.1:8080` by default and must not be exposed on a shared host.

## Assets and validation

Hand MJCF/mesh assets are kept under `assets/hand/`; object assets remain an external producer/data-repository responsibility. `script/check_urdf_mjcf.py` checks an externally supplied URDF/MJCF pair. Contract and viewer tests can be run without a dataset:

```bash
python -m compileall -q src script test
python -m unittest discover -s test -p 'test_*.py'
```

The repository intentionally contains no generated evaluation output, cache, checkpoint, or sample object dataset.
