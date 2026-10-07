<h1 align="center">HUGS-DexGraspBench</h1>

<p align="center">Evaluate and filter synthesized and learned robot grasps in MuJoCo.</p>

<p align="center">
  <a href="https://github.com/hugs-dex/HUGS-Main">HUGS Project</a> ·
  <a href="#producer-workflows">Quick Start</a> ·
  <a href="#documentation">Documentation</a>
</p>

Convert outputs from **HUGS-BODex** or **HUGS-DexLearn**, evaluate them in simulation,
and collect successful synthesis samples for robot learning. Supports Shadow and
Leap-SP with single-hand and dual dummy-arm configurations across five contact modes.

## Installation

Use **Linux x86_64, Python 3.10, and [uv](https://docs.astral.sh/uv/)**. CUDA format
conversion requires a compatible NVIDIA driver. Run from the repository root:

```bash
git submodule update --init
uv sync --locked
```

`uv` manages Python and the local `.venv`; run commands with `uv run`.
See [installation and checks](docs/installation.md) for driver requirements,
CPU fallback, and dependencies.

## Inputs

Prepare producer grasp records and their matching object scenes/meshes, including
per-object `info/simplified.json`. Learned robot samples also need the original
training metadata with the complete joint order.

```bash
export HUGS_DATASET_ROOT=/path/to/hugs-dataset
```

See [HUGS data preparation](https://github.com/hugs-dex/HUGS-Main#data) and the
[input contracts](docs/contracts.md). Producer data, checkpoints, and object
meshes are obtained separately; hand assets are under `assets/hand/`.

## Producer Workflows

Examples assume sibling repository checkouts and default producer output roots.
Run these commands from HUGS-DexGraspBench after generating the corresponding inputs.

### BODex Grasps

Use the `surface_demo` run from the [BODex quick start](https://github.com/hugs-dex/HUGS-BODex#quick-start):

```bash
export HUGS_BODEX_OUTPUT_ROOT="$(realpath ../HUGS-BODex/src/curobo/content/assets/output)"
uv run python script/process_all_grasp_types.py \
  --hand shadow --run-name surface_demo --max-num 20 --dry-run
```

Remove `--dry-run` to **format → evaluate → collect**. The root contains the
suite/manipulation/run directories; pass the root, not an individual `graspdata/`
folder. Use `--run-name human_demo` for the corresponding human-initialized run.

### Learned Robot Grasps

Use the saved `shadow` robot samples from [DexLearn](https://github.com/hugs-dex/HUGS-DexLearn#robot-grasp):

```bash
uv run python script/process_learning_grasp_types.py \
  --hand shadow --run-name learned_shadow \
  --learning-path ../HUGS-DexLearn/output/shadowMulti_robotMultiHierar_shadow/tests/step_050000/shadowMulti \
  --max-num 20 --n-worker 4 --dry-run
```

Remove `--dry-run` to **format → evaluate**. This consumes robot samples;
Human Prior exports are inputs to BODex. For Leap-SP, use `--hand leap_sp` with
matching Leap-SP records and metadata.

`--max-num` limits input records, not necessarily the total number of simulated
grasps. Review [worker settings and reruns](docs/workflows.md#execution-and-reruns)
before executing; the wrappers prompt before deleting existing selected outputs.

## View Results

After formatting, replay one right-full grasp from the same BODex run:

```bash
uv run python src/main.py task=eval hand=shadow exp_name=surface_demo_right_full \
  task.debug_viewer=True task.start=0 task.end=1 n_worker=1
```

This runs simulation again and opens the browser viewer at `http://127.0.0.1:8080`.
For remote use, forward the port over SSH. See [viewer options](docs/workflows.md#viewer).

## Outputs and Next Steps

Under `output/<run>_<type>_<hand>/`, `evaluation/` contains metrics and `succgrasp/`
contains records passing the configured simulation criterion. The BODex workflow
also writes grouped samples to `succ_collect/`.

Follow [training dataset preparation](docs/workflows.md#prepare-a-robot-training-dataset)
to assemble successful samples for DexLearn. Simulation success does not establish
physical-robot success.

## Documentation

- [Installation and environment checks](docs/installation.md)
- [Producer workflows, outputs, and dataset preparation](docs/workflows.md)
- [Formats, units, and evaluation contracts](docs/contracts.md)
- [Hand assets](assets/hand/README.md)

## Usage Terms

A project-wide license has not yet been selected; this repository makes no OSI-license
claim. See [release metadata](manifest.json), the [HUGS usage terms](https://github.com/hugs-dex/HUGS-Main#acknowledgements-and-usage-terms),
and the [HUGS citation](https://github.com/hugs-dex/HUGS-Main#citation).
