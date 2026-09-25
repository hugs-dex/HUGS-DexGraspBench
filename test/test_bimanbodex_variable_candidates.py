import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
fake_util = types.ModuleType("util")
fake_util.__path__ = []
fake_rot_util = types.ModuleType("util.rot_util")
fake_rot_util.torch_quaternion_to_matrix = lambda value: value
fake_rot_util.torch_matrix_to_quaternion = lambda value: value
fake_portable_paths = types.ModuleType("util.portable_paths")
fake_portable_paths.portable_reference = lambda value: str(value)
fake_portable_paths.resolve_path = lambda value, required=False: str(value)
fake_transforms3d = types.ModuleType("transforms3d")
fake_transforms3d.quaternions = types.ModuleType("transforms3d.quaternions")

with mock.patch.dict(
    sys.modules,
    {
        "util": fake_util,
        "util.rot_util": fake_rot_util,
        "util.portable_paths": fake_portable_paths,
        "transforms3d": fake_transforms3d,
        "transforms3d.quaternions": fake_transforms3d.quaternions,
    },
):
    spec = importlib.util.spec_from_file_location(
        "convert_format_under_test", REPO_ROOT / "src/task/convert_format.py"
    )
    convert_format = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert_format)


class BimanBODexVariableCandidateTest(unittest.TestCase):
    def _run_conversion(self, candidate_count):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            data_file = source_dir / "scene_grasp.npy"

            robot_pose = np.arange(candidate_count * 3 * 29, dtype=np.float32).reshape(
                1, candidate_count, 3, 29
            )
            raw_data = {
                "robot_pose": robot_pose,
                "joint_names": [f"joint_{index}" for index in range(29)],
                "scene_path": np.array(["/tmp/src/curobo/content/scenes/scene.npy"]),
            }
            np.save(data_file, raw_data)

            configs = types.SimpleNamespace(
                task=types.SimpleNamespace(data_path=str(source_dir)),
                grasp_dir=str(output_dir),
            )
            scene_cfg = {
                "task": {"obj_name": "target"},
                "scene": {
                    "target": {
                        "scale": [1.0],
                        "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "file_path": "/tmp/assets/target/coacd/object.urdf",
                    }
                },
            }

            with mock.patch.object(convert_format, "load_scene_cfg", return_value=scene_cfg):
                convert_format.BimanBODex((str(data_file), configs))

            output_files = sorted(output_dir.rglob("*_grasp.npy"))
            converted = [np.load(path, allow_pickle=True).item() for path in output_files]
            return output_files, converted, robot_pose

    def test_partial_artifact_creates_one_file_per_returned_candidate(self):
        output_files, converted, robot_pose = self._run_conversion(candidate_count=7)

        self.assertEqual(len(output_files), 7)
        self.assertEqual([path.name for path in output_files], [f"{index}_grasp.npy" for index in range(7)])
        for index, grasp in enumerate(converted):
            np.testing.assert_array_equal(grasp["pregrasp_qpos"], robot_pose[0, index, 0])
            np.testing.assert_array_equal(grasp["grasp_qpos"], robot_pose[0, index, 1])
            np.testing.assert_array_equal(grasp["squeeze_qpos"], robot_pose[0, index, 2])

    def test_empty_artifact_creates_no_files_without_error(self):
        output_files, converted, _ = self._run_conversion(candidate_count=0)

        self.assertEqual(output_files, [])
        self.assertEqual(converted, [])


if __name__ == "__main__":
    unittest.main()
