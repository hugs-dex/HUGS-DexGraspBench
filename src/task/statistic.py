import numpy as np
import os
from glob import glob
import multiprocessing
import logging

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_axis_angle


DEFAULT_TABLETOP_APPROACH_PLANE_ANGLE_BIN_NUM = 6
DEFAULT_TABLETOP_APPROACH_AZIMUTH_BIN_NUM = 12
DEFAULT_APPROACH_ANGLE_BIN_NUM = 18
DEFAULT_RELATIVE_APPROACH_POLAR_BIN_NUM = 6
DEFAULT_RELATIVE_APPROACH_AZIMUTH_BIN_NUM = 12
DEFAULT_WRIST_APPROACH_AXIS = (1.0, 0.0, 0.0)
DEFAULT_PAIRWISE_DISTANCE_SAMPLE_PAIR_NUM = 1_000_000


def compute_ROC_data(data_lst, metric_name):
    sim_results = np.array([d["succ_flag"] for d in data_lst])
    analytic_metrics = np.array([d[metric_name] for d in data_lst])

    sort_index = np.argsort(analytic_metrics)
    sim_results = sim_results[sort_index]
    analytic_metrics = analytic_metrics[sort_index]

    lens = len(sim_results)
    spilt_num = min(100, lens)
    spilt_lens = lens // spilt_num
    spilt_left = lens % spilt_num
    steps = [0]
    for i in range(spilt_left):
        steps.append(steps[-1] + spilt_lens + 1)
    for i in range(spilt_left, spilt_num):
        steps.append(steps[-1] + spilt_lens)
    assert steps[-1] == lens

    tp, fp, tn, fn = (
        np.zeros(spilt_num + 1),
        np.zeros(spilt_num + 1),
        np.zeros(spilt_num + 1),
        np.zeros(spilt_num + 1),
    )
    tp[0], fp[0] = 0, 0
    tn[0], fn[0] = np.sum(sim_results == 0), np.sum(sim_results == 1)
    for i in range(spilt_num):
        tp[i + 1] = tp[i] + np.sum(sim_results[steps[i] : steps[i + 1]] == 1)
        fp[i + 1] = fp[i] + np.sum(sim_results[steps[i] : steps[i + 1]] == 0)
        tn[i + 1] = tn[i] - np.sum(sim_results[steps[i] : steps[i + 1]] == 0)
        fn[i + 1] = fn[i] - np.sum(sim_results[steps[i] : steps[i + 1]] == 1)
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)  # shape=(spilt_num+1,)
    steps[-1] -= 1
    threshold = analytic_metrics[steps]
    return tpr, fpr, threshold


def draw_ROC_curve(data_lst, save_path):
    metric_name_lst = [
        "dfc_metric",
        "tdg_metric",
        "q1_metric",
        "qp_dfc_metric",
        "qp_metric",
    ]
    tpr_lst, fpr_lst, threshold_lst = [], [], []
    for metric_name in metric_name_lst:
        tpr, fpr, threshold = compute_ROC_data(data_lst, metric_name)
        tpr_lst.append(tpr)
        fpr_lst.append(fpr)
        threshold_lst.append(threshold)

    color_dict = {
        "qp_metric": "red",
        "qp_dfc_metric": "red",
        "dfc_metric": "blue",
        "tdg_metric": "green",
        "q1_metric": "cyan",
    }
    line_type_dict = {
        "qp_metric": "-",
        "qp_dfc_metric": "--",
        "dfc_metric": "-",
        "tdg_metric": "-",
        "q1_metric": "-",
    }
    name_dict = {
        "qp_metric": "QP_ours",
        "qp_dfc_metric": "QP_base",
        "dfc_metric": "DFC",
        "tdg_metric": "TDG",
        "q1_metric": "Q1",
    }
    plt.figure(figsize=(6, 6), dpi=600)
    plt.rcParams["font.size"] = 20
    plt.rcParams["lines.linewidth"] = 2
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Energy as Metric (ROC)")
    plt.plot([0, 1], [0, 1], color="black", label="Random", linestyle="--")

    for i in range(len(metric_name_lst)):
        tpr, fpr, threshold = tpr_lst[i], fpr_lst[i], threshold_lst[i]
        distance = tpr - fpr
        max_index = np.argmax(distance)
        auc_score = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=color_dict[metric_name_lst[i]],
            label=name_dict[metric_name_lst[i]],
            linestyle=line_type_dict[metric_name_lst[i]],
        )
        # print(
        #     "max distance is ",
        #     distance[max_index],
        #     "threshold is ",
        #     threshold[max_index],
        # )
        # print("tpr is ", tpr[max_index], "fpr is ", fpr[max_index])
        logging.info(f"AUC of {metric_name_lst[i]}: {auc_score}")
    plt.legend(fontsize=15)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    return


def draw_obj_scale_fig(data_lst, save_path):
    obj_scale_lst = [float(d["obj_scale"]) for d in data_lst]

    bins = np.linspace(0.05, 0.2, 11)

    # Create the histogram
    plt.hist(
        np.array(obj_scale_lst),
        bins=bins,
        color="skyblue",
        edgecolor="black",
        rwidth=0.8,
    )

    # Add labels and title
    plt.xlabel("Scale")
    plt.ylabel("Frequency")
    plt.title("Distribution of Object Scales")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return


def read_data(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    data["_source_path"] = npy_path
    return data


def is_success_eval_data(data, eval_dir=None, succ_dir=None):
    """Check whether one evaluation dictionary corresponds to a successful grasp.

    Args:
        data: Evaluation dictionary loaded from an eval `.npy` file.
        eval_dir: Optional evaluation root directory used for legacy filesystem fallback.
        succ_dir: Optional successful grasp root directory used for legacy filesystem fallback.

    Returns:
        True if the evaluated grasp succeeded, otherwise False.
    """

    if "succ_flag" in data:
        return bool(np.asarray(data["succ_flag"]).item())
    source_path = data.get("_source_path")
    if source_path is None or eval_dir is None or succ_dir is None:
        return False
    rel_path = os.path.relpath(source_path, eval_dir)
    return os.path.exists(os.path.join(succ_dir, rel_path))


def select_diversity_data(data_lst, configs):
    """Select evaluation samples used by diversity metrics.

    Args:
        data_lst: All loaded evaluation dictionaries.
        configs: Hydra/OmegaConf runtime config containing `task.diversity_success_only`.

    Returns:
        Evaluation dictionaries used by diversity metrics. By default this is the
        successful subset, because failed grasps can contain unstable or invalid
        approach directions.
    """

    if not bool(getattr(configs.task, "diversity_success_only", True)):
        return data_lst
    return [data for data in data_lst if is_success_eval_data(data, configs.eval_dir, configs.succ_dir)]


def _format_source_examples(data_lst, sample_mask, max_example_num=3):
    """Format source paths for invalid samples.

    Args:
        data_lst: Evaluation result dictionaries. Each dictionary may contain the
            internal `_source_path` key added by `read_data`.
        sample_mask: Boolean array where False marks invalid samples.
        max_example_num: Maximum number of invalid sample paths to include.

    Returns:
        A short comma-separated string of invalid sample source paths.
    """

    invalid_indices = np.flatnonzero(~sample_mask)
    examples = []
    for invalid_index in invalid_indices[:max_example_num]:
        examples.append(data_lst[invalid_index].get("_source_path", f"sample_index={invalid_index}"))
    return ", ".join(examples)


def _filter_finite_feature_rows(feature_array, data_lst, metric_name):
    """Drop PCA feature rows containing non-finite values.

    Args:
        feature_array: Two-dimensional PCA feature array.
        data_lst: Evaluation result dictionaries aligned with `feature_array` rows.
        metric_name: Human-readable metric name used in warning logs.

    Returns:
        Filtered two-dimensional feature array containing only finite rows.
    """

    finite_mask = np.isfinite(feature_array).all(axis=1)
    invalid_num = int((~finite_mask).sum())
    if invalid_num > 0:
        logging.warning(
            f"Skip {invalid_num} samples with non-finite {metric_name} PCA features. "
            f"Examples: {_format_source_examples(data_lst, finite_mask)}"
        )
    return feature_array[finite_mask]


def _get_valid_wrist_pose_array(data_lst, metric_name):
    """Read wrist poses and remove samples with invalid wrist quaternions.

    Args:
        data_lst: Evaluation result dictionaries containing `grasp_global_pose`.
        metric_name: Human-readable metric name used in warning logs.

    Returns:
        Wrist pose array with shape `(valid_sample_num, wrist_num, 7)`.
    """

    global_pose_array = np.stack([np.asarray(d["grasp_global_pose"], dtype=np.float32) for d in data_lst], axis=0)
    if global_pose_array.shape[1] % 7 != 0:
        raise ValueError(
            "`grasp_global_pose` should contain flattened 7D wrist poses, "
            f"but got feature dimension {global_pose_array.shape[1]}."
        )

    wrist_pose_array = global_pose_array.reshape(global_pose_array.shape[0], -1, 7)
    wrist_quat = wrist_pose_array[:, :, 3:7]
    quat_norm = np.linalg.norm(wrist_quat, axis=-1)
    valid_mask = np.isfinite(wrist_quat).all(axis=(1, 2)) & np.isfinite(quat_norm).all(axis=1) & (quat_norm > 1e-8).all(axis=1)
    invalid_num = int((~valid_mask).sum())
    if invalid_num > 0:
        logging.warning(
            f"Skip {invalid_num} samples with invalid {metric_name} wrist quaternion. "
            f"Examples: {_format_source_examples(data_lst, valid_mask)}"
        )
    return wrist_pose_array[valid_mask]


def _normalize_wrist_approach_axes(local_axes, wrist_num):
    """Normalize configured local approach axes for all wrist poses.

    Args:
        local_axes: Either one 3D local axis shared by all wrists or a list of
            per-wrist 3D local axes.
        wrist_num: Number of wrist poses stored in each evaluation record.

    Returns:
        Array with shape `(wrist_num, 3)` containing unit local approach axes.
    """

    axis_array = np.asarray(local_axes, dtype=np.float32)
    if axis_array.shape == (3,):
        axis_array = np.repeat(axis_array.reshape(1, 3), wrist_num, axis=0)
    elif axis_array.shape != (wrist_num, 3):
        raise ValueError(
            "Wrist approach axes should be one 3D axis or one 3D axis per wrist, "
            f"but got shape {axis_array.shape} for {wrist_num} wrist pose(s)."
        )

    axis_norm = np.linalg.norm(axis_array, axis=-1, keepdims=True)
    if not np.isfinite(axis_array).all() or (axis_norm[:, 0] <= 1e-8).any():
        raise ValueError("Wrist approach axes must be finite non-zero 3D vectors.")
    return axis_array / axis_norm


def _build_axis_aligned_tangent_basis(axis):
    """Build a deterministic orthonormal tangent basis around one axis.

    Args:
        axis: Three-dimensional unit vector used as the positive hemisphere pole.

    Returns:
        Tuple `(pole, tangent_x, tangent_y)` forming a right-handed orthonormal
        frame for polar/azimuth coordinates around `pole`.
    """

    pole = np.asarray(axis, dtype=np.float32).reshape(3)
    pole_norm = np.linalg.norm(pole)
    if (not np.isfinite(pole).all()) or pole_norm <= 1e-8:
        raise ValueError("Hemisphere pole axis must be a finite non-zero 3D vector.")
    pole = pole / pole_norm
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(pole, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tangent_x = np.cross(reference, pole)
    tangent_x = tangent_x / np.linalg.norm(tangent_x)
    tangent_y = np.cross(pole, tangent_x)
    tangent_y = tangent_y / np.linalg.norm(tangent_y)
    return pole, tangent_x, tangent_y


def _get_wrist_approach_vector_array(data_lst, metric_name, local_axes=DEFAULT_WRIST_APPROACH_AXIS):
    """Read world-frame wrist approaching vectors from evaluated wrist poses.

    Args:
        data_lst: Evaluation result dictionaries containing `grasp_global_pose`.
        metric_name: Human-readable metric name used in warning logs.
        local_axes: Wrist-frame axis or per-wrist axes that point along each hand's
            approaching direction.

    Returns:
        Array with shape `(valid_sample_num, wrist_num, 3)` containing normalized
        world-frame approaching vectors.
    """

    wrist_pose_array = _get_valid_wrist_pose_array(data_lst, metric_name)
    if wrist_pose_array.shape[0] == 0:
        raise ValueError(f"{metric_name} requires at least one valid wrist pose.")

    wrist_quat = torch.tensor(wrist_pose_array[:, :, 3:7]).float()
    wrist_rot = torch_quaternion_to_matrix(wrist_quat).numpy()
    local_axis_array = _normalize_wrist_approach_axes(local_axes, wrist_pose_array.shape[1])
    approach_vector_array = np.einsum("nwij,wj->nwi", wrist_rot, local_axis_array)
    vector_norm = np.linalg.norm(approach_vector_array, axis=-1, keepdims=True)
    valid_mask = np.isfinite(approach_vector_array).all(axis=(1, 2)) & (vector_norm[..., 0] > 1e-8).all(axis=1)
    invalid_num = int((~valid_mask).sum())
    if invalid_num > 0:
        logging.warning(
            f"Skip {invalid_num} samples with invalid {metric_name} approaching vector. "
            f"Examples: {_format_source_examples(data_lst, valid_mask)}"
        )
    return approach_vector_array[valid_mask] / vector_norm[valid_mask]


def get_configured_wrist_approach_axes(configs):
    """Read wrist approach axes from hand config with a legacy fallback.

    Args:
        configs: Hydra/OmegaConf runtime config containing `configs.hand`.

    Returns:
        Configured wrist approach axes, or the legacy default axis if the hand
        config does not define `wrist_approach_axes`.
    """

    return getattr(configs.hand, "wrist_approach_axes", DEFAULT_WRIST_APPROACH_AXIS)


def _compute_tabletop_approach_coverage(
    approach_vectors,
    plane_angle_bin_num=DEFAULT_TABLETOP_APPROACH_PLANE_ANGLE_BIN_NUM,
    azimuth_bin_num=DEFAULT_TABLETOP_APPROACH_AZIMUTH_BIN_NUM,
):
    """Compute upper-tabletop approach coverage by elevation-from-table and azimuth.

    Args:
        approach_vectors: Array with shape `(sample_num, 3)` containing unit vectors.
        plane_angle_bin_num: Number of bins for the signed angle from the tabletop
            plane. Only upper-tabletop directions are counted, so the range is
            `[0, 90]` degrees and the default 6 bins use 15-degree intervals.
        azimuth_bin_num: Number of bins around world vertical Z. The range is
            `[0, 360)` degrees; with 12 bins the actual interval is 30 degrees.

    Returns:
        Dictionary with coverage ratio, occupied bin count, total bin count, and
        sample/bin metadata.
    """

    approach_vectors = np.asarray(approach_vectors, dtype=np.float32)
    if approach_vectors.ndim != 2 or approach_vectors.shape[1] != 3:
        raise ValueError(f"Tabletop approach coverage expects vectors with shape (N, 3), but got {approach_vectors.shape}.")
    if approach_vectors.shape[0] == 0:
        raise ValueError("Tabletop approach coverage requires at least one valid approaching vector.")
    if plane_angle_bin_num <= 0 or azimuth_bin_num <= 0:
        raise ValueError("Tabletop approach coverage bin numbers must be positive.")

    vector_norm = np.linalg.norm(approach_vectors, axis=-1, keepdims=True)
    finite_mask = np.isfinite(approach_vectors).all(axis=1) & (vector_norm[:, 0] > 1e-8)
    vectors = approach_vectors[finite_mask] / vector_norm[finite_mask]
    if vectors.shape[0] == 0:
        raise ValueError("Tabletop approach coverage requires at least one finite approaching vector.")

    horizontal_norm = np.linalg.norm(vectors[:, :2], axis=-1)
    plane_angle_deg = np.degrees(np.arctan2(vectors[:, 2], horizontal_norm))

    # Tabletop diversity should not be inflated by directions pointing below the
    # table plane, so coverage is computed only over the upper half-space.
    upper_mask = plane_angle_deg >= 0.0
    upper_vectors = vectors[upper_mask]
    upper_plane_angle_deg = plane_angle_deg[upper_mask]
    if upper_vectors.shape[0] == 0:
        raise ValueError("Tabletop approach coverage requires at least one vector above the table plane.")

    azimuth = (np.arctan2(upper_vectors[:, 1], upper_vectors[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    azimuth_index = np.floor(azimuth / (2.0 * np.pi) * azimuth_bin_num).astype(np.int64)
    plane_angle_index = np.floor(upper_plane_angle_deg / 90.0 * plane_angle_bin_num).astype(np.int64)
    azimuth_index = np.clip(azimuth_index, 0, azimuth_bin_num - 1)
    plane_angle_index = np.clip(plane_angle_index, 0, plane_angle_bin_num - 1)
    occupied_grid_bins = set(zip(plane_angle_index.tolist(), azimuth_index.tolist()))
    occupied_plane_angle_bins = set(plane_angle_index.tolist())
    occupied_azimuth_bins = set(azimuth_index.tolist())
    total_grid_bin_num = int(plane_angle_bin_num * azimuth_bin_num)
    occupied_grid_bin_num = len(occupied_grid_bins)
    occupied_plane_angle_bin_num = len(occupied_plane_angle_bins)
    occupied_azimuth_bin_num = len(occupied_azimuth_bins)
    return {
        "coverage": float(occupied_grid_bin_num / total_grid_bin_num),
        "occupied_bins": int(occupied_grid_bin_num),
        "total_bins": total_grid_bin_num,
        "plane_angle_coverage": float(occupied_plane_angle_bin_num / plane_angle_bin_num),
        "occupied_plane_angle_bins": int(occupied_plane_angle_bin_num),
        "total_plane_angle_bins": int(plane_angle_bin_num),
        "azimuth_coverage": float(occupied_azimuth_bin_num / azimuth_bin_num),
        "occupied_azimuth_bins": int(occupied_azimuth_bin_num),
        "total_azimuth_bins": int(azimuth_bin_num),
        "sample_num": int(upper_vectors.shape[0]),
        "input_sample_num": int(vectors.shape[0]),
        "plane_angle_bins": int(plane_angle_bin_num),
        "plane_angle_bin_degrees": float(90.0 / plane_angle_bin_num),
        "azimuth_bins": int(azimuth_bin_num),
        "azimuth_bin_degrees": float(360.0 / azimuth_bin_num),
        "plane_angle_min_deg": float(np.min(upper_plane_angle_deg)),
        "plane_angle_max_deg": float(np.max(upper_plane_angle_deg)),
        "plane_angle_mean_deg": float(np.mean(upper_plane_angle_deg)),
        "below_table_percent": float(np.mean(plane_angle_deg < 0.0)),
    }


def get_wrist_tabletop_approach_coverage(data_lst, wrist_index=0, local_axes=DEFAULT_WRIST_APPROACH_AXIS):
    """Measure wrist approach-direction diversity with tabletop angle/azimuth bins.

    Args:
        data_lst: Evaluation result dictionaries containing `grasp_global_pose`.
        wrist_index: Wrist index to analyze; index 0 is the right hand in bimanual
            eval output and the only wrist for single-hand data.
        local_axes: Wrist-frame axis or per-wrist axes that point along each hand's
            approaching direction.

    Returns:
        Dictionary describing occupied tabletop angle/azimuth bins for the selected
        wrist's world-frame approaching vectors.
    """

    approach_vector_array = _get_wrist_approach_vector_array(
        data_lst,
        "wrist tabletop approach coverage",
        local_axes=local_axes,
    )
    if wrist_index < 0 or wrist_index >= approach_vector_array.shape[1]:
        raise ValueError(
            f"Wrist tabletop approach coverage requires wrist_index={wrist_index}, "
            f"but only {approach_vector_array.shape[1]} wrist pose(s) are available."
        )
    return _compute_tabletop_approach_coverage(approach_vector_array[:, wrist_index, :])


def _get_pairwise_sample_indices(sample_num, max_pair_num=DEFAULT_PAIRWISE_DISTANCE_SAMPLE_PAIR_NUM, rng_seed=0):
    """Return exact or sampled pair indices for pair-wise distance statistics.

    Args:
        sample_num: Number of samples to pair.
        max_pair_num: Maximum number of pairs to materialize before switching from
            exact enumeration to deterministic random sampling.
        rng_seed: Random seed used when sampling is required.

    Returns:
        Tuple `(row_index, col_index, total_pair_num, estimated)` where `estimated`
        is True when the returned pairs are a deterministic sample.
    """

    if sample_num < 2:
        raise ValueError("Pair-wise distance requires at least two valid samples.")
    total_pair_num = sample_num * (sample_num - 1) // 2
    if total_pair_num <= max_pair_num:
        row_index, col_index = np.triu_indices(sample_num, k=1)
        return row_index, col_index, int(total_pair_num), False

    rng = np.random.default_rng(rng_seed)
    row_index = rng.integers(0, sample_num, size=max_pair_num, endpoint=False)
    col_index = rng.integers(0, sample_num - 1, size=max_pair_num, endpoint=False)
    # Shift columns so every sampled pair has different endpoints without needing
    # rejection sampling on large arrays.
    col_index = col_index + (col_index >= row_index)
    return row_index, col_index, int(total_pair_num), True


def _summarize_pairwise_distances(pairwise_distances, sample_num, total_pair_num, estimated, distance_unit):
    """Summarize already computed pair-wise distance values.

    Args:
        pairwise_distances: One-dimensional pair-wise distance array.
        sample_num: Number of samples used to construct pair distances.
        total_pair_num: Total possible pair count before optional sampling.
        estimated: Whether `pairwise_distances` is a sampled subset.
        distance_unit: Unit string for the returned pair-wise distances.

    Returns:
        Dictionary with pair count and basic statistics of pair-wise distances.
    """

    pairwise_distances = np.asarray(pairwise_distances, dtype=np.float32).reshape(-1)
    if pairwise_distances.shape[0] == 0 or not np.isfinite(pairwise_distances).all():
        raise ValueError("Pair-wise distance got empty or non-finite distance values.")
    return {
        "sample_num": int(sample_num),
        "pair_num": int(pairwise_distances.shape[0]),
        "sampled_pair_num": int(pairwise_distances.shape[0]),
        "total_pair_num": int(total_pair_num),
        "estimated": bool(estimated),
        "distance_unit": distance_unit,
        "mean": float(np.mean(pairwise_distances)),
        "std": float(np.std(pairwise_distances)),
        "min": float(np.min(pairwise_distances)),
        "max": float(np.max(pairwise_distances)),
        "median": float(np.median(pairwise_distances)),
    }


def _compute_pairwise_scalar_distance_stats(values, distance_unit):
    """Summarize pair-wise distances between scalar metric values.

    Args:
        values: One-dimensional scalar array.
        distance_unit: Unit string for the returned pair-wise distances.

    Returns:
        Dictionary with pair count and basic statistics of pair-wise absolute
        distances.
    """

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_values = values[np.isfinite(values)]
    row_index, col_index, total_pair_num, estimated = _get_pairwise_sample_indices(finite_values.shape[0])
    pairwise_distances = np.abs(finite_values[row_index] - finite_values[col_index])
    return _summarize_pairwise_distances(
        pairwise_distances,
        finite_values.shape[0],
        total_pair_num,
        estimated,
        distance_unit,
    )


def get_wrist_approach_pairwise_distance(data_lst, wrist_index=0, local_axes=DEFAULT_WRIST_APPROACH_AXIS):
    """Measure pair-wise angular distance between wrist approach directions.

    Args:
        data_lst: Evaluation result dictionaries containing `grasp_global_pose`.
        wrist_index: Wrist index to analyze; index 0 is the right hand in bimanual
            eval output and the only wrist for single-hand data.
        local_axes: Wrist-frame axis or per-wrist axes that point along each hand's
            approaching direction.

    Returns:
        Dictionary with pair-wise angular distance statistics in degrees.
    """

    approach_vector_array = _get_wrist_approach_vector_array(
        data_lst,
        "wrist approach pair-wise distance",
        local_axes=local_axes,
    )
    if wrist_index < 0 or wrist_index >= approach_vector_array.shape[1]:
        raise ValueError(
            f"Wrist approach pair-wise distance requires wrist_index={wrist_index}, "
            f"but only {approach_vector_array.shape[1]} wrist pose(s) are available."
        )

    wrist_vectors = approach_vector_array[:, wrist_index, :]
    if wrist_vectors.shape[0] < 2:
        raise ValueError("Wrist approach pair-wise distance requires at least two valid wrist poses.")
    row_index, col_index, total_pair_num, estimated = _get_pairwise_sample_indices(wrist_vectors.shape[0])
    cos_distance = np.clip(np.sum(wrist_vectors[row_index] * wrist_vectors[col_index], axis=-1), -1.0, 1.0)
    pairwise_angles = np.degrees(np.arccos(cos_distance))
    return _summarize_pairwise_distances(
        pairwise_angles,
        wrist_vectors.shape[0],
        total_pair_num,
        estimated,
        "degree",
    )


def _get_left_approach_vectors_in_right_frame(data_lst, local_axes=DEFAULT_WRIST_APPROACH_AXIS):
    """Compute left wrist approach direction expressed in the right wrist frame.

    Args:
        data_lst: Evaluation result dictionaries containing bimanual
            `grasp_global_pose`.
        local_axes: Wrist-frame axis or per-wrist axes that point along each hand's
            approaching direction.

    Returns:
        Tuple `(relative_vectors, right_palm_axis)` where `relative_vectors` has
        shape `(sample_num, 3)` and contains normalized left approach vectors in
        the right wrist local frame. `right_palm_axis` is the right hand's local
        approach axis and defines the positive palm-side hemisphere.
    """

    wrist_pose_array = _get_valid_wrist_pose_array(data_lst, "bimanual relative approach hemisphere")
    if wrist_pose_array.shape[1] < 2:
        raise ValueError("Bimanual relative approach hemisphere requires at least two wrist poses.")

    wrist_quat = torch.tensor(wrist_pose_array[:, :, 3:7]).float()
    wrist_rot = torch_quaternion_to_matrix(wrist_quat).numpy()
    local_axis_array = _normalize_wrist_approach_axes(local_axes, wrist_pose_array.shape[1])
    left_approach_world = np.einsum("nij,j->ni", wrist_rot[:, 1, :, :], local_axis_array[1])
    left_approach_in_right_frame = np.einsum("nij,nj->ni", np.swapaxes(wrist_rot[:, 0, :, :], 1, 2), left_approach_world)
    vector_norm = np.linalg.norm(left_approach_in_right_frame, axis=-1, keepdims=True)
    valid_mask = np.isfinite(left_approach_in_right_frame).all(axis=1) & (vector_norm[:, 0] > 1e-8)
    invalid_num = int((~valid_mask).sum())
    if invalid_num > 0:
        logging.warning(
            f"Skip {invalid_num} samples with invalid bimanual relative approach vectors. "
            f"Examples: {_format_source_examples(data_lst, valid_mask)}"
        )
    if int(valid_mask.sum()) == 0:
        raise ValueError("Bimanual relative approach hemisphere requires at least one finite relative vector.")
    return left_approach_in_right_frame[valid_mask] / vector_norm[valid_mask], local_axis_array[0]


def _compute_relative_approach_hemisphere_coverage(
    relative_vectors,
    positive_axis,
    polar_bin_num=DEFAULT_RELATIVE_APPROACH_POLAR_BIN_NUM,
    azimuth_bin_num=DEFAULT_RELATIVE_APPROACH_AZIMUTH_BIN_NUM,
):
    """Compute hemisphere-grid coverage for left approach in the right hand frame.

    Args:
        relative_vectors: Array with shape `(sample_num, 3)` containing left
            approach directions expressed in the right wrist local frame.
        positive_axis: Right-hand local palm-side axis defining the positive
            hemisphere pole.
        polar_bin_num: Number of bins over polar angle `[0, 90]` degrees from
            the positive palm-side axis.
        azimuth_bin_num: Number of bins around the positive palm-side axis.

    Returns:
        Dictionary describing occupied polar/azimuth grid bins and diagnostics for
        samples outside the positive palm-side hemisphere.
    """

    if polar_bin_num <= 0 or azimuth_bin_num <= 0:
        raise ValueError("Relative approach hemisphere bin numbers must be positive.")
    relative_vectors = np.asarray(relative_vectors, dtype=np.float32)
    if relative_vectors.ndim != 2 or relative_vectors.shape[1] != 3:
        raise ValueError(f"Relative approach hemisphere expects vectors with shape (N, 3), but got {relative_vectors.shape}.")
    vector_norm = np.linalg.norm(relative_vectors, axis=-1, keepdims=True)
    finite_mask = np.isfinite(relative_vectors).all(axis=1) & (vector_norm[:, 0] > 1e-8)
    vectors = relative_vectors[finite_mask] / vector_norm[finite_mask]
    if vectors.shape[0] == 0:
        raise ValueError("Relative approach hemisphere requires at least one finite vector.")

    pole, tangent_x, tangent_y = _build_axis_aligned_tangent_basis(positive_axis)
    pole_projection = vectors @ pole
    hemisphere_mask = pole_projection >= 0.0
    hemisphere_vectors = vectors[hemisphere_mask]
    hemisphere_projection = pole_projection[hemisphere_mask]
    if hemisphere_vectors.shape[0] == 0:
        raise ValueError("Relative approach hemisphere requires at least one vector in the positive palm-side hemisphere.")

    polar_angle_deg = np.degrees(np.arccos(np.clip(hemisphere_projection, -1.0, 1.0)))
    azimuth = (np.arctan2(hemisphere_vectors @ tangent_y, hemisphere_vectors @ tangent_x) + 2.0 * np.pi) % (2.0 * np.pi)
    polar_index = np.floor(polar_angle_deg / 90.0 * polar_bin_num).astype(np.int64)
    azimuth_index = np.floor(azimuth / (2.0 * np.pi) * azimuth_bin_num).astype(np.int64)
    polar_index = np.clip(polar_index, 0, polar_bin_num - 1)
    azimuth_index = np.clip(azimuth_index, 0, azimuth_bin_num - 1)
    occupied_grid_bins = set(zip(polar_index.tolist(), azimuth_index.tolist()))
    occupied_polar_bins = set(polar_index.tolist())
    occupied_azimuth_bins = set(azimuth_index.tolist())
    total_grid_bin_num = int(polar_bin_num * azimuth_bin_num)
    return {
        "coverage": float(len(occupied_grid_bins) / total_grid_bin_num),
        "occupied_bins": int(len(occupied_grid_bins)),
        "total_bins": total_grid_bin_num,
        "polar_coverage": float(len(occupied_polar_bins) / polar_bin_num),
        "occupied_polar_bins": int(len(occupied_polar_bins)),
        "total_polar_bins": int(polar_bin_num),
        "azimuth_coverage": float(len(occupied_azimuth_bins) / azimuth_bin_num),
        "occupied_azimuth_bins": int(len(occupied_azimuth_bins)),
        "total_azimuth_bins": int(azimuth_bin_num),
        "sample_num": int(hemisphere_vectors.shape[0]),
        "input_sample_num": int(vectors.shape[0]),
        "polar_bins": int(polar_bin_num),
        "polar_bin_degrees": float(90.0 / polar_bin_num),
        "azimuth_bins": int(azimuth_bin_num),
        "azimuth_bin_degrees": float(360.0 / azimuth_bin_num),
        "polar_angle_min_deg": float(np.min(polar_angle_deg)),
        "polar_angle_max_deg": float(np.max(polar_angle_deg)),
        "polar_angle_mean_deg": float(np.mean(polar_angle_deg)),
        "negative_hemisphere_percent": float(np.mean(pole_projection < 0.0)),
        "positive_axis": [float(v) for v in pole.tolist()],
    }


def get_bimanual_relative_approach_geometry(
    data_lst,
    polar_bin_num=DEFAULT_RELATIVE_APPROACH_POLAR_BIN_NUM,
    azimuth_bin_num=DEFAULT_RELATIVE_APPROACH_AZIMUTH_BIN_NUM,
    local_axes=DEFAULT_WRIST_APPROACH_AXIS,
):
    """Measure bimanual relative approach as a right-frame hemisphere grid.

    Args:
        data_lst: Evaluation result dictionaries containing bimanual
            `grasp_global_pose`.
        polar_bin_num: Number of bins over polar angle `[0, 90]` degrees from
            the right palm-side positive axis.
        azimuth_bin_num: Number of bins around the right palm-side positive axis.
        local_axes: Wrist-frame axis or per-wrist axes that point along each hand's
            approaching direction.

    Returns:
        Dictionary containing 2D hemisphere-grid coverage for the left approach
        direction expressed in the right wrist local frame.
    """

    relative_vectors, positive_axis = _get_left_approach_vectors_in_right_frame(data_lst, local_axes=local_axes)
    return _compute_relative_approach_hemisphere_coverage(
        relative_vectors,
        positive_axis,
        polar_bin_num=polar_bin_num,
        azimuth_bin_num=azimuth_bin_num,
    )


def get_bimanual_relative_approach_pairwise_distance(data_lst, local_axes=DEFAULT_WRIST_APPROACH_AXIS):
    """Measure pair-wise distance between right-frame relative approach directions.

    Args:
        data_lst: Evaluation result dictionaries containing bimanual
            `grasp_global_pose`.
        local_axes: Wrist-frame axis or per-wrist axes that point along each hand's
            approaching direction.

    Returns:
        Dictionary with pair-wise angular distance statistics in degrees for left
        approach directions expressed in the right wrist local frame. Only vectors
        in the right palm-side positive hemisphere are compared.
    """

    relative_vectors, positive_axis = _get_left_approach_vectors_in_right_frame(data_lst, local_axes=local_axes)
    pole = _build_axis_aligned_tangent_basis(positive_axis)[0]
    hemisphere_vectors = relative_vectors[(relative_vectors @ pole) >= 0.0]
    if hemisphere_vectors.shape[0] < 2:
        raise ValueError("Bimanual relative approach pair-wise distance requires at least two positive-hemisphere vectors.")
    row_index, col_index, total_pair_num, estimated = _get_pairwise_sample_indices(hemisphere_vectors.shape[0])
    cos_distance = np.clip(np.sum(hemisphere_vectors[row_index] * hemisphere_vectors[col_index], axis=-1), -1.0, 1.0)
    pairwise_angles = np.degrees(np.arccos(cos_distance))
    return _summarize_pairwise_distances(
        pairwise_angles,
        hemisphere_vectors.shape[0],
        total_pair_num,
        estimated,
        "degree",
    )


def get_bimanual_relative_approach_angle_geometry(data_lst, angle_bin_num=DEFAULT_APPROACH_ANGLE_BIN_NUM, local_axes=DEFAULT_WRIST_APPROACH_AXIS):
    """Measure the legacy one-dimensional angle between two approach vectors.

    Args:
        data_lst: Evaluation result dictionaries containing bimanual
            `grasp_global_pose`.
        angle_bin_num: Number of equal-width bins over theta in `[0, 180]` degrees.
        local_axes: Wrist-frame axis or per-wrist axes that point along each hand's
            approaching direction.

    Returns:
        Dictionary containing legacy one-dimensional relative angle statistics.
    """

    approach_vector_array = _get_wrist_approach_vector_array(
        data_lst,
        "legacy bimanual relative approach angle",
        local_axes=local_axes,
    )
    if approach_vector_array.shape[1] < 2:
        raise ValueError("Legacy bimanual relative approach angle requires at least two wrist poses.")
    if angle_bin_num <= 0:
        raise ValueError("Approach-angle bin number must be positive.")

    right_approach = approach_vector_array[:, 0, :]
    left_approach = approach_vector_array[:, 1, :]
    cos_theta = np.clip(np.sum(right_approach * left_approach, axis=-1), -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    bin_index = np.floor(theta_deg / 180.0 * angle_bin_num).astype(np.int64)
    bin_index = np.clip(bin_index, 0, angle_bin_num - 1)
    occupied_bin_num = len(set(bin_index.tolist()))
    return {
        "theta_mean_deg": float(np.mean(theta_deg)),
        "theta_std_deg": float(np.std(theta_deg)),
        "theta_min_deg": float(np.min(theta_deg)),
        "theta_max_deg": float(np.max(theta_deg)),
        "theta_coverage": float(occupied_bin_num / angle_bin_num),
        "occupied_bins": int(occupied_bin_num),
        "total_bins": int(angle_bin_num),
        "theta_bin_degrees": float(180.0 / angle_bin_num),
        "sample_num": int(theta_deg.shape[0]),
    }


def get_diversity(data_lst):
    from sklearn.decomposition import PCA

    hand_poses = torch.tensor(
        np.stack([d["grasp_qpos"][:7] for d in data_lst], axis=0)
    ).float()
    hand_qpos = torch.tensor(
        np.stack([d["grasp_qpos"][7:] for d in data_lst], axis=0)
    ).float()
    obj_poses = torch.tensor(
        np.stack([d["obj_pose"] for d in data_lst], axis=0)
    ).float()

    obj_rot = torch_quaternion_to_matrix(obj_poses[:, 3:])
    hand_rot = torch_quaternion_to_matrix(hand_poses[:, 3:])
    hand_real_trans = (
        obj_rot.transpose(-1, -2) @ (hand_poses[:, :3] - obj_poses[:, :3]).unsqueeze(-1)
    ).squeeze(-1)
    hand_real_rot = obj_rot.transpose(-1, -2) @ hand_rot
    hand_final_pose = torch.cat(
        [
            hand_real_trans,
            torch_matrix_to_axis_angle(hand_real_rot),
            hand_qpos,
        ],
        dim=-1,
    )

    pca = PCA()
    pca.fit(hand_final_pose.numpy())
    explained_variance = []
    for i in range(5):
        explained_variance.append(np.sum(pca.explained_variance_ratio_[: i + 1]))
    return explained_variance


def compute_pca_cumulative_explained_variance(feature_array, max_component_num=5, data_lst=None, metric_name="diversity"):
    """Compute PCA cumulative explained variance over feature vectors.

    Args:
        feature_array: Two-dimensional array with shape `(sample_num, feature_dim)`.
        max_component_num: Maximum number of leading PCA components to summarize.
        data_lst: Optional evaluation result dictionaries aligned with feature rows.
        metric_name: Human-readable metric name used in warning logs.

    Returns:
        List of cumulative explained variance ratios for the first 1..N principal
        components, where `N` is capped by available PCA components and `max_component_num`.
    """

    from sklearn.decomposition import PCA

    feature_array = np.asarray(feature_array, dtype=np.float32)
    if feature_array.ndim != 2:
        raise ValueError(f"PCA diversity expects a 2D feature array, but got shape {feature_array.shape}.")
    if data_lst is not None:
        feature_array = _filter_finite_feature_rows(feature_array, data_lst, metric_name)
    elif not np.isfinite(feature_array).all():
        raise ValueError(f"PCA diversity got non-finite values in {metric_name} features.")
    if feature_array.shape[0] < 2:
        raise ValueError(f"PCA diversity requires at least two valid grasps for {metric_name}.")

    pca = PCA()
    pca.fit(feature_array)
    component_num = min(max_component_num, len(pca.explained_variance_ratio_))
    explained_variance = []
    for i in range(component_num):
        explained_variance.append(float(np.sum(pca.explained_variance_ratio_[: i + 1])))
    return explained_variance


def get_grasp_joint_pos_diversity(data_lst):
    """Measure grasp joint-position diversity with PCA.

    Args:
        data_lst: Evaluation result dictionaries containing `grasp_joint_pos`.

    Returns:
        List of cumulative explained variance ratios from PCA over grasp joint-position
        vectors.
    """

    joint_pos_array = np.stack([np.asarray(d["grasp_joint_pos"], dtype=np.float32) for d in data_lst], axis=0)
    return compute_pca_cumulative_explained_variance(
        joint_pos_array,
        data_lst=data_lst,
        metric_name="joint position diversity",
    )


def task_stat(configs):
    grasp_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    succ_lst = glob(os.path.join(configs.succ_dir, "**/*.npy"), recursive=True)
    eval_lst = glob(os.path.join(configs.eval_dir, "**/*.npy"), recursive=True)
    logging.info(
        f"Find {len(grasp_lst)} grasp data in {configs.grasp_dir}, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )

    # Grasp success rate
    logging.info(f"Grasp success rate: {len(succ_lst)/len(eval_lst)}")

    # Object success rate
    obj_eval_lst = set(
        [
            os.path.dirname(f)
            for f in glob(
                os.path.join(configs.eval_dir, "**/*.npy"),
                recursive=True,
            )
        ]
    )
    obj_succ_lst = set(
        [
            os.path.dirname(f)
            for f in glob(
                os.path.join(configs.succ_dir, "**/*.npy"),
                recursive=True,
            )
        ]
    )
    logging.info(f"Object success rate: {len(obj_succ_lst)/len(obj_eval_lst)}")

    if len(eval_lst) == 0:
        logging.error("No evaluated grasp!")

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(read_data, eval_lst)
        data_lst = list(result_iter)

    if configs.task.scale_fig:
        save_path = os.path.join(configs.log_dir, "objscale_distribution.png")
        draw_obj_scale_fig(data_lst, save_path)

    if configs.task.roc_fig:
        if "qp_metric" not in data_lst[0]:
            logging.warning(
                "Please set 'eval_analytic' and 'eval_simulate' to True while evaluation"
            )
        else:
            save_path = os.path.join(configs.log_dir, "analytic_metric_ROC.png")
            draw_ROC_curve(data_lst, save_path)

    if configs.task.diversity:
        diversity_data_lst = select_diversity_data(data_lst, configs)
        logging.info(
            f"Use {len(diversity_data_lst)} / {len(data_lst)} evaluated grasps for diversity "
            f"(success_only={bool(getattr(configs.task, 'diversity_success_only', True))})."
        )
        if len(diversity_data_lst) == 0:
            logging.warning("Skip diversity metrics: no grasps selected for diversity.")
        elif "grasp_global_pose" not in diversity_data_lst[0] or "grasp_joint_pos" not in diversity_data_lst[0]:
            logging.warning(
                "Please rerun evaluation before calculating the new diversity metrics; "
                "`grasp_global_pose` and `grasp_joint_pos` are required."
            )
        else:
            wrist_approach_axes = get_configured_wrist_approach_axes(configs)
            wrist_approach_coverage = get_wrist_tabletop_approach_coverage(
                diversity_data_lst,
                wrist_index=0,
                local_axes=wrist_approach_axes,
            )
            logging.info(f"Wrist tabletop approach coverage: {wrist_approach_coverage}")
            try:
                relative_approach_geometry = get_bimanual_relative_approach_geometry(
                    diversity_data_lst,
                    local_axes=wrist_approach_axes,
                )
                logging.info(f"Bimanual relative approach geometry: {relative_approach_geometry}")
            except ValueError as error:
                logging.warning(f"Skip bimanual relative approach geometry: {error}")
            try:
                joint_pos_diversity = get_grasp_joint_pos_diversity(diversity_data_lst)
                logging.info(f"Grasp joint position diversity: {joint_pos_diversity}")
            except ValueError as error:
                logging.warning(f"Skip grasp joint position diversity: {error}")

    if getattr(configs.task, "legacy_diversity", False):
        legacy_data_lst = select_diversity_data(data_lst, configs)
        if len(legacy_data_lst) == 0:
            logging.warning("Skip legacy diversity: no grasps selected for diversity.")
        else:
            pca_eigenvalue = get_diversity(legacy_data_lst)
            logging.info(f"Legacy diversity: {pca_eigenvalue}")

    if "ho_pene" not in data_lst[0]:
        logging.warning("Please set 'eval_dist' to True while evaluation")
    else:
        average_penetration_depth = np.mean([d["ho_pene"] for d in data_lst])
        logging.info(f"Penetration depth: {average_penetration_depth}")
        average_self_penetration_depth = np.mean([d["self_pene"] for d in data_lst])
        logging.info(f"Self-penetration depth: {average_self_penetration_depth}")
        average_contact_distance = np.mean([d["contact_dist"] for d in data_lst])
        logging.info(f"Contact distance: {average_contact_distance}")
        average_contact_number = np.mean([d["contact_num"] for d in data_lst])
        logging.info(f"Contact number: {average_contact_number}")
        average_contact_consis = np.mean([d["contact_consis"] for d in data_lst])
        logging.info(f"Contact consistency: {average_contact_consis}")

    logging.info(f"Finish statistics")
