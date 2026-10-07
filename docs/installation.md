# Installation

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) from the repository
root. Python 3.10 is selected by `.python-version`; `pyproject.toml` declares
dependencies and `uv.lock` fixes their resolved versions.

```bash
# Keep recursive initialization anonymous and HTTPS-only, including nested submodules.
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
uv sync --locked
```

`uv` downloads Python 3.10 if needed, creates `.venv`, and installs
`pytorch_kinematics` and `utils_python` from the submodules in editable mode.
Use `uv run` for subsequent commands; no activation is required.

## GPU and Simulation Dependencies

On Linux x86_64, the locked environment installs PyTorch 2.8.0 from the official
CUDA 12.8 wheel index. The host needs a compatible NVIDIA driver; a separate
CUDA toolkit is not required for these prebuilt wheels.

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must be `True` for CUDA-based format conversion.
Without a compatible driver, conversion falls back to CPU; this does not verify
the CUDA workflow. MuJoCo Menagerie is required by the Shadow MJCF. No USD or
`pxr` dependency is part of this release.

## Inputs and Assets

Set `HUGS_DATASET_ROOT` to a dataset root containing the producer's `object/`
tree, including `info/simplified.json` for each object. New converted records
store `scene_path` and `obj_path` relative to this root. Set it when reading or
writing those records; see the [input contract](contracts.md).

Hand MJCF/mesh assets live under `assets/hand/`. Object assets, producer grasp
records, and checkpoints are external inputs. `script/check_urdf_mjcf.py`
checks an externally supplied URDF/MJCF pair.

## Development Checks

Contract and viewer tests can run without a dataset:

```bash
uv run python -m compileall -q src script test
uv run python -m unittest discover -s test -p 'test_*.py'
```

These checks do not establish success on real producer data. See the
[producer workflows](workflows.md) to prepare and evaluate actual inputs.
Source provenance and release scope are recorded in [manifest.json](../manifest.json).
