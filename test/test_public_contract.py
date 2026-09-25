import json
import os
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "script"))

from process_all_grasp_types import GRASP_TYPES, HAND_CONFIGS  # noqa: E402
from process_learning_grasp_types import HAND_CONFIGS as LEARNING_HAND_CONFIGS  # noqa: E402
from util.portable_paths import portable_reference  # noqa: E402


class PublicContractTest(unittest.TestCase):
    def test_manifest_and_supported_matrix(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["source_commit"], "76378fc4ac4c7cb472fa63760af482afcb1c8412")
        self.assertEqual(set(manifest["supported_hand_families"]), {"shadow", "leap_sp"})
        self.assertEqual(HAND_CONFIGS, {
            "shadow": {
                "single_hand": "shadow",
                "dual_hand": "dual_dummy_arm_shadow",
                "single_dataset": "sim_shadow",
                "dual_dataset": "sim_dual_dummy_arm_shadow",
            },
            "leap_sp": {
                "single_hand": "leap_sp",
                "dual_hand": "dual_dummy_arm_leap_sp",
                "single_dataset": "sim_leap_sp",
                "dual_dataset": "sim_dual_dummy_arm_leap_sp",
            },
        })
        self.assertEqual(LEARNING_HAND_CONFIGS, {
            "shadow": {"single_hand": "shadow", "dual_hand": "dual_dummy_arm_shadow"},
            "leap_sp": {"single_hand": "leap_sp", "dual_hand": "dual_dummy_arm_leap_sp"},
        })

    def test_grasp_type_mapping_is_stable(self):
        self.assertEqual(
            [name for name, _, _ in GRASP_TYPES],
            ["right_two", "right_three", "right_full", "both_three", "both_full"],
        )

    def test_hand_config_paths_and_no_removed_configs(self):
        expected = {
            "shadow": "assets/hand/shadow/right_hand_v2.xml",
            "dual_dummy_arm_shadow": "assets/hand/dual_dummy_arm_shadow/dual_dummy_arm_shadow.xml",
            "leap_sp": "assets/hand/leap_sp/leap_sp.xml",
            "dual_dummy_arm_leap_sp": "assets/hand/dual_dummy_arm_leap_sp/dual_dummy_arm_leap_sp.xml",
        }
        for config_name, xml_path in expected.items():
            config = yaml.safe_load((ROOT / "config/hand" / f"{config_name}.yaml").read_text())
            self.assertEqual(config["xml_path"].strip(), xml_path)
            self.assertTrue((ROOT / xml_path).exists())
        for removed in ("leap", "allegro", "ur10e_shadow", "dual_ur5_shadow", "dual_dummy_arm_leap"):
            self.assertFalse((ROOT / "config/hand" / f"{removed}.yaml").exists())

    def test_portable_path_contract(self):
        old_root = os.environ.get("ANYSCALEGRASP_DATA_ROOT")
        try:
            os.environ["ANYSCALEGRASP_DATA_ROOT"] = "/tmp/example-dataset"
            self.assertEqual(
                portable_reference("/srv/BimanBODex/src/curobo/content/assets/object/DGN_2k/scene_cfg/a.npy"),
                "object/DGN_2k/scene_cfg/a.npy",
            )
            self.assertEqual(
                portable_reference("/tmp/example-dataset/object/foo"),
                "object/foo",
            )
        finally:
            if old_root is None:
                os.environ.pop("ANYSCALEGRASP_DATA_ROOT", None)
            else:
                os.environ["ANYSCALEGRASP_DATA_ROOT"] = old_root

    def test_no_generated_or_usd_release_content(self):
        self.assertFalse((ROOT / "assets/example_object").exists())
        self.assertFalse((ROOT / "src/task/vis_usd.py").exists())
        self.assertFalse((ROOT / "src/util/usd_helper.py").exists())
        tracked_text = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "third_party" in path.parts or "test" in path.parts:
                continue
            if path.suffix in {".py", ".yaml", ".yml", ".sh", ".toml"}:
                tracked_text.append(path.read_text(errors="ignore"))
        joined = "\n".join(tracked_text)
        forbidden_strings = (
            "task=vusd", "task=vobj", "from pxr", "import pxr", "usd-core",
            "/mnt/disk1/", "../BimanBODex", "../AnyScaleDexLearn",
        )
        for forbidden in forbidden_strings:
            self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main()
