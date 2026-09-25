import os
import json
from glob import glob
import logging
import multiprocessing
from pathlib import Path

import numpy as np
from transforms3d import quaternions as tq
import torch
from tqdm import tqdm

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_quaternion
from util.portable_paths import portable_reference, resolve_path


def load_scene_cfg(scene_path):
    """Load a scene config npy and resolve relative `*_path` entries.

    Args:
        scene_path: Path to the scene config `.npy` file.

    Returns:
        A scene config dictionary with normalized absolute-like paths.
    """
    scene_path = resolve_path(scene_path, required=True)
    scene_cfg = np.load(scene_path, allow_pickle=True).item()

    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.join(os.path.dirname(scene_path), v)
        return

    update_relative_path(scene_cfg["scene"])

    return scene_cfg


LEARNING_GRASP_TYPE_ID_TO_NAME = {
    1: "right_two",
    2: "right_three",
    3: "right_full",
    4: "both_three",
    5: "both_full",
}
LEARNING_RIGHT_TYPE_IDS = {1, 2, 3}
LEARNING_BOTH_TYPE_IDS = {4, 5}
LEARNING_IK_CACHE = {}
LEARNING_METADATA_CACHE = {}
LEARNING_STAGE_NAMES = ("pregrasp", "grasp", "squeeze")
LEARNING_OPTIONAL_KEYS = ("grasp_error", "grasp_type_id", "pred_grasp_type_prob")
LEARNING_HAND_FAMILY_SUFFIXES = ("leap_sp", "shadow")


def _resolve_scene_path(scene_path):
    """Resolve scene path with `AnyScaleGraspDataset` fallback.

    Args:
        scene_path: Raw scene path stored in learning data.

    Returns:
        Existing scene path after fallback mapping, or original path.
    """
    return resolve_path(scene_path)


def _get_target_grasp_type_from_exp_name(exp_name):
    """Extract optional target grasp type suffix from experiment name.

    Args:
        exp_name: Experiment name string.

    Returns:
        A grasp type string like `right_two`/`both_three`, or `None`.
    """
    for grasp_type in LEARNING_GRASP_TYPE_ID_TO_NAME.values():
        if exp_name.endswith(f"_{grasp_type}"):
            return grasp_type
    return None


def _get_learning_source_family(hand_name):
    """Resolve the source hand family used by Learning metadata.

    Args:
        hand_name: Benchmark hand config name, such as `leap_sp` or
            `dual_dummy_arm_leap_sp`.

    Returns:
        Source family folder name used by dataset metadata.
    """
    for suffix in LEARNING_HAND_FAMILY_SUFFIXES:
        if hand_name == suffix or hand_name.endswith(f"_{suffix}"):
            return suffix
    return hand_name.split("_")[-1]


def _get_learning_metadata_group(configs):
    """Resolve the metadata group used to split Learning source qpos.

    Args:
        configs: Runtime config object.

    Returns:
        Metadata group name. Defaults to `both_full` because Learning samples store
        both wrist poses and both hands even when evaluating single-hand grasp types.
    """
    return str(getattr(configs.task, "learning_metadata_group", "both_full"))


def _metadata_candidate_paths(dataset_root, source_family, metadata_group):
    """Build ordered metadata path candidates for Learning conversion.

    Args:
        dataset_root: AnyScaleGrasp dataset root inferred from scene paths.
        source_family: Source hand family, e.g. `leap_sp`.
        metadata_group: Grasp type group, e.g. `right_two` or `both_full`.

    Returns:
        Ordered list of candidate metadata paths.
    """
    if dataset_root is None:
        return []

    candidate_groups = [metadata_group, "both_full"] if metadata_group != "both_full" else ["both_full"]
    candidate_paths = []
    for group in candidate_groups:
        candidate_paths.append(os.path.join(dataset_root, "human_DGN2k_full", source_family, group, "metadata.json"))
    for group in candidate_groups:
        candidate_paths.extend(
            sorted(glob(os.path.join(dataset_root, "BimanBODex*", source_family, group, "metadata.json")))
        )
    return candidate_paths


def _load_learning_metadata(scene_path, source_family, metadata_group="both_full"):
    """Load and cache learning metadata for a source hand family.

    Args:
        scene_path: Resolved scene path used to infer dataset root.
        source_family: Source hand family, e.g. `leap_sp` or `shadow`.
        metadata_group: Grasp type group whose metadata should be preferred.

    Returns:
        Metadata dict with joint names and grouped joint-name lists.
    """
    marker = f"{os.sep}object{os.sep}"
    dataset_root = scene_path.split(marker, 1)[0] if marker in scene_path else None
    cache_key = (dataset_root, source_family, metadata_group)
    if cache_key in LEARNING_METADATA_CACHE:
        return LEARNING_METADATA_CACHE[cache_key]

    candidate_paths = []
    candidate_paths.extend(_metadata_candidate_paths(dataset_root, source_family, metadata_group))
    env_dataset_root = os.environ.get("ANYSCALEGRASP_DATA_ROOT") or os.environ.get("AnyScaleGraspDataset")
    if env_dataset_root:
        candidate_paths.extend(_metadata_candidate_paths(env_dataset_root, source_family, metadata_group))
    metadata_path = next((path for path in dict.fromkeys(candidate_paths) if os.path.exists(path)), None)
    if metadata_path is None:
        raise FileNotFoundError(
            f"Cannot find metadata.json for source_family={source_family}, metadata_group={metadata_group}. "
            f"Checked roots from scene_path={scene_path} and AnyScaleGraspDataset={env_dataset_root}."
        )
    with open(metadata_path, "r") as f:
        raw_metadata = json.load(f)

    joint_names = list(raw_metadata["joint_names"])
    metadata = {
        "metadata_path": metadata_path,
        "joint_names": joint_names,
        "wrist_body_names": list(raw_metadata.get("wrist_body_names", [])),
        "right_arm_joint_names": [name for name in joint_names if name.startswith("ra_")],
        "left_arm_joint_names": [name for name in joint_names if name.startswith("la_")],
        "right_hand_joint_names": [name for name in joint_names if name.startswith("rh_")],
        "left_hand_joint_names": [name for name in joint_names if name.startswith("lh_")],
    }

    if len(metadata["right_hand_joint_names"]) == 0:
        raise ValueError(f"No right hand joints found in {metadata_path}.")
    if len(metadata["left_hand_joint_names"]) == 0:
        raise ValueError(f"No left hand joints found in {metadata_path}.")
    if len(metadata["right_arm_joint_names"]) == 0 or len(metadata["left_arm_joint_names"]) == 0:
        raise ValueError(f"No dual-arm joints found in {metadata_path}.")

    LEARNING_METADATA_CACHE[cache_key] = metadata
    return metadata


def _split_learning_robot_qpos(stage_qpos, metadata):
    """Split source stage qpos into wrist poses and finger joints.

    Args:
        stage_qpos: Source stage qpos vector.
        metadata: Learning metadata dict containing joint-name partitions.

    Returns:
        Tuple `(right_pose, right_joint, left_pose, left_joint)`.
    """
    stage_qpos = np.asarray(stage_qpos, dtype=np.float32).reshape(-1)
    if stage_qpos.shape[0] < 14:
        raise ValueError(f"Invalid qpos size {stage_qpos.shape[0]}: should include two 7D wrist poses at minimum.")

    right_joint_num = len(metadata["right_hand_joint_names"])
    left_joint_num = len(metadata["left_hand_joint_names"])
    expected_dim = 14 + right_joint_num + left_joint_num
    if stage_qpos.shape[0] != expected_dim:
        raise ValueError(
            f"Unexpected qpos size {stage_qpos.shape[0]} for metadata={metadata['metadata_path']}. "
            f"Expected {expected_dim} (14 + {right_joint_num} + {left_joint_num})."
        )

    right_pose = stage_qpos[:7]
    right_joint = stage_qpos[7 : 7 + right_joint_num]
    left_pose = stage_qpos[7 + right_joint_num : 14 + right_joint_num]
    left_joint = stage_qpos[14 + right_joint_num :]
    return right_pose, right_joint, left_pose, left_joint


def _quat_normalize(quat):
    """Normalize quaternion safely.

    Args:
        quat: Quaternion array-like in `[w, x, y, z]` order.

    Returns:
        Unit quaternion; identity quaternion when norm is near zero.
    """
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _pose_to_matrix_torch(pose, device):
    """Convert 7D pose `[x,y,z,w,x,y,z]` to a batched 4x4 tensor.

    Args:
        pose: Pose array-like with 3D position and quaternion.
        device: Torch device for output tensor.

    Returns:
        Tensor of shape `(1, 4, 4)` on `device` with dtype `float64`.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = tq.quat2mat(_quat_normalize(pose[3:7]))
    mat[:3, 3] = pose[:3]
    return torch.tensor(mat, dtype=torch.float64, device=device).unsqueeze(0)


def _get_learning_ik_solver(configs, metadata, source_family):
    """Create or reuse cached PK IK solver for learning dual-hand conversion.

    Args:
        configs: Runtime config object.
        metadata: Learning metadata dict.
        source_family: Source hand family string.

    Returns:
        Solver dict containing robot helper, indices, and cache state.
    """
    import pytorch_kinematics as pk
    from mr_utils.robot.pk_helper import PytorchKinematicsHelper

    if hasattr(configs.hand, "wrist_body_names") and configs.hand.wrist_body_names is not None:
        wrist_body_names = list(configs.hand.wrist_body_names)
        if len(wrist_body_names) >= 2:
            wrist_body_names = wrist_body_names[:2]
        else:
            wrist_body_names = []
    else:
        wrist_body_names = []
    if len(wrist_body_names) < 2:
        wrist_body_names = list(metadata.get("wrist_body_names", []))[:2]
    if len(wrist_body_names) < 2:
        raise ValueError(
            f"Cannot determine dual wrist_body_names. "
            f"hand={configs.hand_name}, config_wrist={getattr(configs.hand, 'wrist_body_names', None)}, "
            f"metadata={metadata['metadata_path']}"
        )

    raw_mjcf_path = str(getattr(configs.hand, "xml_path", "")).strip()
    if raw_mjcf_path == "":
        raise ValueError(f"configs.hand.xml_path is empty for hand={configs.hand_name}")
    mjcf_candidates = [raw_mjcf_path]
    if not os.path.isabs(raw_mjcf_path):
        mjcf_candidates.append(os.path.join(os.path.dirname(__file__), "..", "..", raw_mjcf_path))
    mjcf_path = next((path for path in mjcf_candidates if os.path.exists(path)), None)
    if mjcf_path is None:
        raise FileNotFoundError(
            f"Cannot find MJCF for PK IK solver from configs.hand.xml_path={raw_mjcf_path}, "
            f"hand={configs.hand_name}, source_family={source_family}"
        )

    gpu_id = int(getattr(configs, "gpu_id", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        logging.warning("CUDA is unavailable, fallback to CPU for Learning IK solver.")
        device = torch.device("cpu")

    cache_key = (
        mjcf_path,
        tuple(metadata["right_arm_joint_names"]),
        tuple(metadata["left_arm_joint_names"]),
        tuple(wrist_body_names),
        str(device),
    )
    if cache_key in LEARNING_IK_CACHE:
        return LEARNING_IK_CACHE[cache_key]

    chain = pk.build_chain_from_mjcf(mjcf_path).to(device=device, dtype=torch.float64)
    robot_helper = PytorchKinematicsHelper(chain, base_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device)
    joint_names = list(robot_helper.robot_joint_names)

    for wrist_name in wrist_body_names:
        robot_helper.create_serial_chain(wrist_name)
        robot_helper.create_ik_solver(
            wrist_name,
            pos_tolerance=1e-3,
            rot_tolerance=1e-2,
            max_iterations=50,
            retry_configs=None,
            num_retries=20,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="any",
            debug=False,
            lr=0.2,
            regularlization=1e-3,
        )

    right_arm_indices = [joint_names.index(name) for name in metadata["right_arm_joint_names"]]
    left_arm_indices = [joint_names.index(name) for name in metadata["left_arm_joint_names"]]

    solver = {
        "robot_helper": robot_helper,
        "wrist_body_names": wrist_body_names,
        "device": device,
        "neutral_full_q": torch.zeros((len(joint_names),), dtype=torch.float64, device=device),
        "right_arm_indices": right_arm_indices,
        "left_arm_indices": left_arm_indices,
    }
    LEARNING_IK_CACHE[cache_key] = solver
    return solver


def _solve_dummy_arm_qpos(solver, side, target_pose):
    """Solve one-side dummy-arm IK for a target wrist pose.

    Args:
        solver: IK solver dict from `_get_learning_ik_solver`.
        side: `"right"` or `"left"`.
        target_pose: Target wrist pose in 7D format.

    Returns:
        Arm joint vector for the selected side.
    """
    ee_idx = 0 if side == "right" else 1
    ee_name = solver["wrist_body_names"][ee_idx]
    matrix = _pose_to_matrix_torch(target_pose, solver["device"])
    ref_configs = solver["neutral_full_q"].unsqueeze(0).clone()
    ik_result = solver["robot_helper"].solve_ik_batch(ee_name, matrix, ref_configs=ref_configs, use_ref_as_init=True)
    if not bool(ik_result["success"][0].item()):
        raise RuntimeError(f"PK IK failed for side={side}, ee={ee_name}")

    full_q = ik_result["q"][0]
    arm_indices = solver["right_arm_indices"] if side == "right" else solver["left_arm_indices"]
    return full_q[arm_indices].detach().cpu().numpy().astype(np.float32)


def _solve_learning_batch_ik_with_retry(
    solver,
    ee_name,
    matrix,
    ref_configs,
    side_name,
    max_retry_rounds=3,
    allow_failure=False,
):
    """Solve batch IK and retry failed items with new random seeds.

    Args:
        solver: IK solver dict from `_get_learning_ik_solver`.
        ee_name: End-effector body name for IK.
        matrix: Target pose matrices of shape `(B, 4, 4)`.
        ref_configs: Reference full joint configs for first-pass initialization.
        side_name: Side label used in error messages (`right`/`left`).
        max_retry_rounds: Number of retry rounds for failed items.
        allow_failure: If True, return success flags instead of raising when
            some items remain unsolved.

    Returns:
        Dict with `q`, `success`, and `failed_indices`. Raises on failure unless
        `allow_failure` is True.
    """
    robot_helper = solver["robot_helper"]
    result = robot_helper.solve_ik_batch(ee_name, matrix, ref_configs=ref_configs.clone(), use_ref_as_init=True)
    success = result["success"].detach().clone()
    q = result["q"].detach().clone()

    remaining = ~success
    for _ in range(max_retry_rounds):
        if not bool(remaining.any().item()):
            break

        retry_matrix = matrix[remaining]
        # Retry with ref_configs=None to force new random IK seeds.
        retry_result = robot_helper.solve_ik_batch(ee_name, retry_matrix, ref_configs=None, use_ref_as_init=False)
        retry_success = retry_result["success"].detach()
        if not bool(retry_success.any().item()):
            continue

        remaining_idx = torch.nonzero(remaining, as_tuple=False).squeeze(1)
        solved_local_idx = torch.nonzero(retry_success, as_tuple=False).squeeze(1)
        solved_global_idx = remaining_idx[solved_local_idx]
        q[solved_global_idx] = retry_result["q"][solved_local_idx].detach()
        success[solved_global_idx] = True
        remaining = ~success

    if bool((~success).any().item()):
        fail_idx = int(torch.nonzero(~success, as_tuple=False)[0].item())
        if not allow_failure:
            raise RuntimeError(f"PK IK failed in batched Learning conversion ({side_name}), item={fail_idx}")

    failed_indices = torch.nonzero(~success, as_tuple=False).squeeze(1).detach().cpu().numpy().astype(np.int64)
    return {"q": q, "success": success, "failed_indices": failed_indices}


def _mark_learning_ik_failed_record(record, metadata, failures):
    """Mark one Learning record as an IK-conversion failure.

    Args:
        record: Batched Learning conversion record with `new_data` and stages.
        metadata: Learning metadata dict with target joint order.
        failures: List of failure dictionaries containing side, stage, and target pose.

    Returns:
        None. The record's output data is updated in-place with NaN qpos fields.
    """
    nan_qpos = np.full((len(metadata["joint_names"]),), np.nan, dtype=np.float32)
    record["new_data"]["format_ik_failed"] = True
    record["new_data"]["format_ik_failed_reason"] = "dummy_arm_ik_failed"
    record["new_data"]["format_ik_failed_side"] = sorted({failure["side"] for failure in failures})
    record["new_data"]["format_ik_failed_stage"] = sorted({failure["stage"] for failure in failures})
    record["new_data"]["format_ik_failed_targets"] = failures
    for stage in LEARNING_STAGE_NAMES:
        record["new_data"][f"{stage}_qpos"] = nan_qpos.copy()


def _convert_learning_stage_single(stage_qpos, metadata):
    """Convert one source stage to single-hand qpos.

    Args:
        stage_qpos: Source stage qpos.
        metadata: Learning metadata dict.

    Returns:
        Concatenated right wrist pose and right-hand joints.
    """
    right_pose, right_joint, _, _ = _split_learning_robot_qpos(stage_qpos, metadata)
    return np.concatenate([right_pose, right_joint], axis=0).astype(np.float32)


def _assemble_learning_dual_qpos(metadata, right_arm_qpos, right_joint, left_arm_qpos, left_joint):
    """Assemble dual-hand qpos in metadata joint-name order.

    Args:
        metadata: Learning metadata dict with full `joint_names` order.
        right_arm_qpos: Solved right-arm dummy joints.
        right_joint: Source right-hand finger joints.
        left_arm_qpos: Solved left-arm dummy joints.
        left_joint: Source left-hand finger joints.

    Returns:
        Full dual-hand qpos array aligned to `metadata["joint_names"]`.
    """
    if len(right_arm_qpos) != len(metadata["right_arm_joint_names"]):
        raise ValueError("Right arm qpos dim mismatch when converting Learning both-hand sample.")
    if len(left_arm_qpos) != len(metadata["left_arm_joint_names"]):
        raise ValueError("Left arm qpos dim mismatch when converting Learning both-hand sample.")
    if len(right_joint) != len(metadata["right_hand_joint_names"]):
        raise ValueError("Right hand joint dim mismatch when converting Learning both-hand sample.")
    if len(left_joint) != len(metadata["left_hand_joint_names"]):
        raise ValueError("Left hand joint dim mismatch when converting Learning both-hand sample.")

    joint_value_map = {}
    joint_value_map.update({name: value for name, value in zip(metadata["right_arm_joint_names"], right_arm_qpos)})
    joint_value_map.update({name: value for name, value in zip(metadata["left_arm_joint_names"], left_arm_qpos)})
    joint_value_map.update({name: value for name, value in zip(metadata["right_hand_joint_names"], right_joint)})
    joint_value_map.update({name: value for name, value in zip(metadata["left_hand_joint_names"], left_joint)})
    return np.array([joint_value_map[name] for name in metadata["joint_names"]], dtype=np.float32)


def _convert_learning_stage_dual(stage_qpos, metadata, solver):
    """Convert one source stage to dual-hand qpos with IK.

    Args:
        stage_qpos: Source stage qpos.
        metadata: Learning metadata dict.
        solver: IK solver dict.

    Returns:
        Dual-hand qpos array in target joint order.
    """
    right_pose, right_joint, left_pose, left_joint = _split_learning_robot_qpos(stage_qpos, metadata)
    right_arm_qpos = _solve_dummy_arm_qpos(solver, "right", right_pose)
    left_arm_qpos = _solve_dummy_arm_qpos(solver, "left", left_pose)
    return _assemble_learning_dual_qpos(metadata, right_arm_qpos, right_joint, left_arm_qpos, left_joint)


def BimanBODex(params):
    """Convert bimanual BODex grasp samples to benchmark format.

    Args:
        params: Tuple `(data_file, configs)`.

    Returns:
        None. Converted files are saved to `configs.grasp_dir`.
    """
    data_file, configs = params[0], params[1]

    raw_data = np.load(data_file, allow_pickle=True).item()
    robot_pose = raw_data["robot_pose"][0]
    joint_names = raw_data["joint_names"]
    scene_reference = portable_reference(raw_data["scene_path"][0])
    scene_path = resolve_path(scene_reference, required=True)
    scene_cfg = load_scene_cfg(scene_path)
    obj_name = scene_cfg["task"]["obj_name"]

    new_data = {}
    new_data["obj_scale"] = scene_cfg["scene"][obj_name]["scale"][0]
    new_data["obj_pose"] = scene_cfg["scene"][obj_name]["pose"]
    new_data["obj_path"] = portable_reference(os.path.dirname(os.path.dirname(scene_cfg["scene"][obj_name]["file_path"])))
    new_data["scene_path"] = scene_reference
    new_data["bench_contract_version"] = "1.0"
    new_data["path_root"] = "ANYSCALEGRASP_DATA_ROOT"

    for i in range(len(robot_pose)):
        new_data["pregrasp_qpos"] = np.array(robot_pose[i, 0])
        new_data["grasp_qpos"] = np.array(robot_pose[i, 1])
        new_data["squeeze_qpos"] = np.array(robot_pose[i, 2])
        new_data["joint_names"] = list(joint_names)

        save_path = (
            data_file.replace(configs.task.data_path, configs.grasp_dir)
            .replace("_grasp.npy", f"/{i}_grasp.npy")
            .replace("_mogen.npy", f"/{i}_mogen.npy")
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, new_data)
    return



def _build_learning_new_data(raw_data, scene_cfg, target_obj, scene_path, pred_grasp_type_id, pred_grasp_type, joint_names):
    """Build shared output metadata dict for one learning sample.

    Args:
        raw_data: Source sample dict.
        scene_cfg: Loaded scene config dict.
        target_obj: Target object key in scene config.
        scene_path: Resolved scene path.
        pred_grasp_type_id: Predicted grasp-type id.
        pred_grasp_type: Predicted grasp-type name.
        joint_names: Output joint-name order.

    Returns:
        Initialized output dict with optional source metrics copied.
    """
    new_data = {
        "obj_path": portable_reference(os.path.dirname(os.path.dirname(scene_cfg["scene"][target_obj]["file_path"]))),
        "obj_pose": scene_cfg["scene"][target_obj]["pose"],
        "obj_scale": scene_cfg["scene"][target_obj]["scale"][0],
        "scene_path": portable_reference(scene_path),
        "bench_contract_version": "1.0",
        "path_root": "ANYSCALEGRASP_DATA_ROOT",
        "pred_grasp_type_id": pred_grasp_type_id,
        "pred_grasp_type": pred_grasp_type,
        "joint_names": joint_names,
    }
    for key in LEARNING_OPTIONAL_KEYS:
        if key in raw_data:
            new_data[key] = raw_data[key]
    return new_data

def Learning(params):
    """Convert one Learning sample file to benchmark grasp format.

    Args:
        params: Tuple `(data_file, configs)`.

    Returns:
        None. Invalid/incompatible samples are skipped; valid sample is saved.
    """
    data_file, configs = params[0], params[1]
    raw_data = np.load(data_file, allow_pickle=True).item()

    pred_grasp_type_id = int(np.asarray(raw_data.get("pred_grasp_type_id", 0)).item())
    pred_grasp_type = LEARNING_GRASP_TYPE_ID_TO_NAME.get(pred_grasp_type_id)
    if pred_grasp_type is None:
        return

    target_grasp_type = _get_target_grasp_type_from_exp_name(str(configs.exp_name))
    if target_grasp_type is not None and pred_grasp_type != target_grasp_type:
        return

    hand_name = str(configs.hand_name)
    is_dual_hand = not configs.hand.mocap

    # Route grasp types to single-hand or dual-hand setup.
    if is_dual_hand:
        if pred_grasp_type_id in LEARNING_RIGHT_TYPE_IDS:
            return
        if "dummy_arm" not in hand_name:
            raise NotImplementedError(f"Learning conversion of both-hand grasps requires dummy_arm setup: {hand_name}")
    elif pred_grasp_type_id in LEARNING_BOTH_TYPE_IDS:
        return

    # Build one output record with shared scene/object metadata.
    scene_reference = portable_reference(raw_data["scene_path"])
    scene_path = _resolve_scene_path(scene_reference)
    source_family = _get_learning_source_family(hand_name)
    metadata = _load_learning_metadata(scene_path, source_family, _get_learning_metadata_group(configs))
    scene_cfg = load_scene_cfg(scene_path)
    target_obj = scene_cfg["task"]["obj_name"]
    joint_names = metadata["joint_names"] if is_dual_hand else metadata["right_hand_joint_names"]
    new_data = _build_learning_new_data(
        raw_data, scene_cfg, target_obj, scene_reference, pred_grasp_type_id, pred_grasp_type, joint_names
    )

    if is_dual_hand:
        ik_solver = _get_learning_ik_solver(configs, metadata, source_family)
        stage_converter = lambda stage_qpos: _convert_learning_stage_dual(stage_qpos, metadata, ik_solver)
    else:
        stage_converter = lambda stage_qpos: _convert_learning_stage_single(stage_qpos, metadata)

    # Convert all stages with the same converter.
    for stage in LEARNING_STAGE_NAMES:
        new_data[f"{stage}_qpos"] = stage_converter(raw_data[f"{stage}_qpos"])

    save_path = data_file.replace(configs.task.data_path, configs.grasp_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, new_data)
    return

def LearningBatch(params):
    """Batch-convert Learning files with grouped batched IK.

    Args:
        params: Tuple `(data_file_list, configs)`.

    Returns:
        None. Converted files are saved to `configs.grasp_dir`.
    """
    data_file_lst, configs = params[0], params[1]
    if len(data_file_lst) == 0:
        return

    hand_name = str(configs.hand_name)
    if configs.hand.mocap:
        for data_file in data_file_lst:
            Learning((data_file, configs))
        return
    if "dummy_arm" not in hand_name:
        raise NotImplementedError(f"Learning conversion of both-hand grasps requires dummy_arm setup: {hand_name}")

    source_family = _get_learning_source_family(hand_name)
    target_grasp_type = _get_target_grasp_type_from_exp_name(str(configs.exp_name))
    metadata_group = _get_learning_metadata_group(configs)

    # Group by metadata signature so one IK solver can be reused for many files.
    grouped_records = {}
    for data_file in data_file_lst:
        raw_data = np.load(data_file, allow_pickle=True).item()
        pred_grasp_type_id = int(np.asarray(raw_data.get("pred_grasp_type_id", 0)).item())
        pred_grasp_type = LEARNING_GRASP_TYPE_ID_TO_NAME.get(pred_grasp_type_id)
        if pred_grasp_type is None or pred_grasp_type_id in LEARNING_RIGHT_TYPE_IDS:
            continue
        if target_grasp_type is not None and pred_grasp_type != target_grasp_type:
            continue

        scene_reference = portable_reference(raw_data["scene_path"])
        scene_path = _resolve_scene_path(scene_reference)
        metadata = _load_learning_metadata(scene_path, source_family, metadata_group)
        scene_cfg = load_scene_cfg(scene_path)
        target_obj = scene_cfg["task"]["obj_name"]
        new_data = _build_learning_new_data(
            raw_data,
            scene_cfg,
            target_obj,
            scene_reference,
            pred_grasp_type_id,
            pred_grasp_type,
            metadata["joint_names"],
        )

        stages = {}
        for stage in LEARNING_STAGE_NAMES:
            right_pose, right_joint, left_pose, left_joint = _split_learning_robot_qpos(raw_data[f"{stage}_qpos"], metadata)
            stages[stage] = {
                "right_pose": right_pose,
                "right_joint": right_joint,
                "left_pose": left_pose,
                "left_joint": left_joint,
            }

        group_key = metadata["metadata_path"]
        grouped_records.setdefault(group_key, {"metadata": metadata, "records": []})
        grouped_records[group_key]["records"].append(
            {
                "new_data": new_data,
                "save_path": data_file.replace(configs.task.data_path, configs.grasp_dir),
                "stages": stages,
            }
        )

    # Batch all stage wrist IK targets per metadata group.
    for group in grouped_records.values():
        metadata = group["metadata"]
        records = group["records"]
        if len(records) == 0:
            continue

        solver = _get_learning_ik_solver(configs, metadata, source_family)
        right_poses = []
        left_poses = []
        for record in records:
            for stage in LEARNING_STAGE_NAMES:
                right_poses.append(record["stages"][stage]["right_pose"])
                left_poses.append(record["stages"][stage]["left_pose"])

        right_matrix = torch.cat([_pose_to_matrix_torch(pose, solver["device"]) for pose in right_poses], dim=0)
        left_matrix = torch.cat([_pose_to_matrix_torch(pose, solver["device"]) for pose in left_poses], dim=0)
        ref_configs = solver["neutral_full_q"].unsqueeze(0).repeat(len(right_poses), 1)

        ik_retry_rounds = int(getattr(configs.task, "learning_ik_retry_rounds", 3))
        right_result = _solve_learning_batch_ik_with_retry(
            solver,
            solver["wrist_body_names"][0],
            right_matrix,
            ref_configs,
            side_name="right",
            max_retry_rounds=ik_retry_rounds,
            allow_failure=True,
        )
        left_result = _solve_learning_batch_ik_with_retry(
            solver,
            solver["wrist_body_names"][1],
            left_matrix,
            ref_configs,
            side_name="left",
            max_retry_rounds=ik_retry_rounds,
            allow_failure=True,
        )

        right_arm_batch = right_result["q"][:, solver["right_arm_indices"]].detach().cpu().numpy().astype(np.float32)
        left_arm_batch = left_result["q"][:, solver["left_arm_indices"]].detach().cpu().numpy().astype(np.float32)
        right_success = right_result["success"].detach().cpu().numpy().astype(bool)
        left_success = left_result["success"].detach().cpu().numpy().astype(bool)

        cursor = 0
        for record in records:
            ik_failures = []
            for stage in LEARNING_STAGE_NAMES:
                stage_data = record["stages"][stage]
                if not right_success[cursor]:
                    ik_failures.append(
                        {
                            "side": "right",
                            "stage": stage,
                            "target_pose": np.asarray(stage_data["right_pose"], dtype=np.float32),
                        }
                    )
                if not left_success[cursor]:
                    ik_failures.append(
                        {
                            "side": "left",
                            "stage": stage,
                            "target_pose": np.asarray(stage_data["left_pose"], dtype=np.float32),
                        }
                    )
                cursor += 1

            if len(ik_failures) > 0:
                _mark_learning_ik_failed_record(record, metadata, ik_failures)
                logging.warning(
                    "Learning IK failed for %s; saving NaN qpos sentinel so evaluation counts it as failure. "
                    "failures=%s",
                    record["save_path"],
                    [(failure["side"], failure["stage"]) for failure in ik_failures],
                )
            else:
                stage_cursor = cursor - len(LEARNING_STAGE_NAMES)
                for stage in LEARNING_STAGE_NAMES:
                    stage_data = record["stages"][stage]
                    record["new_data"][f"{stage}_qpos"] = _assemble_learning_dual_qpos(
                        metadata,
                        right_arm_batch[stage_cursor],
                        stage_data["right_joint"],
                        left_arm_batch[stage_cursor],
                        stage_data["left_joint"],
                    )
                    stage_cursor += 1
                record["new_data"]["format_ik_failed"] = False

            os.makedirs(os.path.dirname(record["save_path"]), exist_ok=True)
            np.save(record["save_path"], record["new_data"])

    return

# def Batched(params):
#     data_file, configs = params[0], params[1]
#     raw_data = np.load(data_file, allow_pickle=True).item()
#     scene_cfg = load_scene_cfg(raw_data["scene_path"])
#     target_obj = scene_cfg["task"]["obj_name"]
#     new_data = {}
#     new_data["obj_path"] = os.path.dirname(os.path.dirname(scene_cfg["scene"][target_obj]["file_path"]))
#     new_data["obj_pose"] = scene_cfg["scene"][target_obj]["pose"]
#     obj_scale_in_scene = scene_cfg["scene"][target_obj]["scale"][0]
#     save_path = data_file.replace(configs.task.data_path, configs.grasp_dir)
#     for i in range(raw_data["grasp_qpos"].shape[0]):
#         save_path = os.path.join(save_path.split(".npy")[0], f"{i}.npy")
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         new_data["obj_scale"] = obj_scale_in_scene * raw_data["scene_scale"][i]
#         new_data["grasp_qpos"] = raw_data["grasp_qpos"][i]
#         new_data["pregrasp_qpos"] = raw_data["pregrasp_qpos"][i]
#         new_data["squeeze_qpos"] = raw_data["squeeze_qpos"][i]
#         np.save(save_path, new_data)
#     return


def task_format(configs):
    """Run format-conversion task over the selected dataset source.

    Args:
        configs: Runtime config object.

    Returns:
        None. Converted grasp files are written under `configs.grasp_dir`.
    """
    if configs.task.data_name == "Learning":
        raw_data_struct = ["**", "*.npy"]
    elif configs.task.data_name == "BimanBODex":
        raw_data_struct = ["**", "*_grasp.npy"]
    else:
        raise ValueError("task.data_name must be BimanBODex or Learning in HUGS-DexGraspBench.")

    if not configs.task.data_path:
        raise ValueError("task.data_path is required; pass a producer sample directory outside the repository.")

    raw_data_path_lst = glob(os.path.join(configs.task.data_path, *raw_data_struct), recursive=True)
    raw_file_num = len(raw_data_path_lst)
    raw_data_path_lst = sorted(raw_data_path_lst)
    if configs.task.max_num > 0:
        raw_data_path_lst = np.random.permutation(raw_data_path_lst)[: configs.task.max_num]
    logging.info(
        f"Find {raw_file_num} raw files for {os.path.join(configs.task.data_path, *raw_data_struct)}, "
        f"and use {len(raw_data_path_lst)}"
    )

    if len(raw_data_path_lst) == 0:
        return

    n_pool_worker = int(configs.n_worker)
    if configs.task.data_name == "Learning" and not configs.hand.mocap:
        learning_batch_size = int(getattr(configs.task, "learning_batch_size", 512))
        learning_batch_size = max(1, learning_batch_size)
        data_batches = [
            raw_data_path_lst[i : i + learning_batch_size] for i in range(0, len(raw_data_path_lst), learning_batch_size)
        ]
        progress_total = len(data_batches)
        progress_unit = "batch"
        iterable_params = zip(data_batches, [configs] * len(data_batches))
        worker_fn = LearningBatch
        # Limit concurrent GPU-heavy workers to reduce CUDA OOM risk.
        max_cuda_workers = int(getattr(configs.task, "max_cuda_workers", 1))
        max_cuda_workers = max(1, max_cuda_workers)
        n_pool_worker = min(n_pool_worker, len(data_batches), max_cuda_workers)
        logging.info(f"Learning dual-hand batching enabled: batch_size={learning_batch_size}, num_batches={len(data_batches)}")
        logging.info(
            f"Learning dual-hand worker restriction: requested={configs.n_worker}, used={n_pool_worker}, "
            f"max_cuda_workers={max_cuda_workers}"
        )
    else:
        progress_total = len(raw_data_path_lst)
        progress_unit = "file"
        iterable_params = zip(raw_data_path_lst, [configs] * len(raw_data_path_lst))
        worker_fn = eval(configs.task.data_name)
        n_pool_worker = min(n_pool_worker, len(raw_data_path_lst))

    progress_desc = f"Converting {configs.task.data_name} {progress_unit}s"
    if configs.task.debug:
        results = []
        # Debug mode is serial, so progress updates after each converted file or learning batch.
        for params in tqdm(iterable_params, total=progress_total, desc=progress_desc):
            results.append(worker_fn(params))
    else:
        # CPU parallel
        with multiprocessing.Pool(processes=n_pool_worker) as pool:
            result_iter = pool.imap_unordered(worker_fn, iterable_params)
            # Multiprocessing progress advances when conversion jobs finish.
            results = list(tqdm(result_iter, total=progress_total, desc=progress_desc))

    grasp_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    logging.info(f"Get {len(grasp_lst)} grasp data in {configs.save_dir}")
    logging.info("Finish format conversion")
    return
