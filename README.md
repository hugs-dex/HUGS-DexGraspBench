<h1 align="center">HUGS-DexGraspBench</h1>

<p align="center">Evaluate and filter synthesized and learned robot grasps in MuJoCo.</p>

<p align="center">
  <a href="https://github.com/hugs-dex/HUGS-Main">HUGS Project</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#headless-evaluation">Headless Evaluation</a> ·
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

Prepare grasp records from BODex or DexLearn and their matching object assets.
`HUGS_DATASET_ROOT` is the local directory containing the shared HUGS dataset.
For DGN_2k, point it to the parent of `object/`:

```bash
export HUGS_DATASET_ROOT=/path/to/hugs-dataset
```

```text
hugs-dataset/
└── object/
    └── DGN_2k/
        ├── scene_cfg/
        └── processed_data/    # Object meshes and per-object info/simplified.json
```

Bench resolves saved scene and object paths relative to this directory. Learned
robot samples also need the matching training metadata with the complete joint
order, such as `human_DGN2k_full/<hand>/both_full/metadata.json` under this root.

See [HUGS data preparation](https://github.com/hugs-dex/HUGS-Main#data) and the
[input contracts](docs/contracts.md). Producer data, checkpoints, and object
meshes are obtained separately; hand assets are under `assets/hand/`.

## Usage

Examples assume sibling repository checkouts and default producer output roots.
Run them from the HUGS-DexGraspBench root after generating the corresponding inputs.

For BODex inputs, set the directory containing its generated runs:

```bash
export HUGS_BODEX_OUTPUT_ROOT="$(realpath ../HUGS-BODex/src/curobo/content/assets/output)"
```

This root contains `<suite>/<manipulation>/<run>/graspdata/`. If BODex used a
custom `HUGS_OUTPUT_ROOT`, use that directory instead.

### Quick Start

Use the `surface_demo` run from the [BODex quick start](https://github.com/hugs-dex/HUGS-BODex#quick-start).
Convert up to 20 right-full input records, then simulate one grasp with the
Viser-based browser viewer:

```bash
uv run python src/main.py task=format hand=shadow exp_name=surface_demo_right_full \
  task.data_name=BimanBODex \
  "task.data_path=$HUGS_BODEX_OUTPUT_ROOT/sim_shadow/tabletop_full/surface_demo/graspdata" \
  task.max_num=20 n_worker=4

uv run python src/main.py task=eval hand=shadow exp_name=surface_demo_right_full \
  task.debug_viewer=True task.viewer.backend=mjviser \
  task.start=0 task.end=1 n_worker=1
```

Open `http://127.0.0.1:8080` to watch the simulation. It waits for the browser to
connect and holds the final frame; close the tab to finish saving results.
For remote use, forward port 8080 over SSH. The input run must contain at least
one right-full grasp. See [viewer options](docs/workflows.md#viewer).

### Headless Evaluation

These commands process all available inputs across all five grasp modes without
a viewer or a record-count limit.

#### BODex Grasps

After [full dataset synthesis](https://github.com/hugs-dex/HUGS-BODex#full-dataset-synthesis),
run **format → evaluate → collect**:

```bash
uv run python script/process_all_grasp_types.py \
  --hand shadow --run-name surface_full
```

Use the name of your BODex run: `surface_full` for the full surface run,
`human_full` for the full human-initialized run, or `surface_demo` for the quick-start inputs.

#### Learned Robot Grasps

Run **format → evaluate** on saved `shadow` robot samples from
[DexLearn](https://github.com/hugs-dex/HUGS-DexLearn#robot-grasp):

```bash
uv run python script/process_learning_grasp_types.py \
  --hand shadow --run-name learned_shadow \
  --learning-path ../HUGS-DexLearn/output/shadowMulti_robotMultiHierar_shadow/tests/step_050000/shadowMulti \
  --n-worker 4 --format-arg n_worker=4
```

This consumes robot samples; Human Prior exports are inputs to BODex.
For Leap-SP, use `--hand leap_sp` with matching records and metadata in either workflow.

The BODex wrapper uses 96 workers by default; the DexLearn command above uses 4.
See [worker settings and reruns](docs/workflows.md#execution-and-reruns) to adjust
parallelism or run individual stages. Both wrappers prompt before deleting existing
selected outputs.

## Outputs and Next Steps

Under `output/<run>_<type>_<hand>/`, `evaluation/` contains metrics and `succgrasp/`
contains records passing the configured simulation criterion. The BODex workflow
also writes grouped samples to `succ_collect/`.

For bimanual modes, `<hand>` is `dual_dummy_arm_shadow` or `dual_dummy_arm_leap_sp`.
MuJoCo skips very small convex object parts and logs their counts; the grasps are
still evaluated.

Follow [training dataset preparation](docs/workflows.md#prepare-a-robot-training-dataset)
to assemble successful samples for DexLearn. Simulation success does not establish
physical-robot success.

## Documentation

- [Installation and environment checks](docs/installation.md)
- [Usage, outputs, and dataset preparation](docs/workflows.md)
- [Formats, units, and evaluation contracts](docs/contracts.md)
- [Hand assets](assets/hand/README.md)

## Usage Terms

A project-wide license has not yet been selected; this repository makes no OSI-license
claim. See [release metadata](manifest.json), the [HUGS usage terms](https://github.com/hugs-dex/HUGS-Main#acknowledgements-and-usage-terms),
and the [HUGS citation](https://github.com/hugs-dex/HUGS-Main#citation).
