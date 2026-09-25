import os
import multiprocessing
import logging
from glob import glob
import traceback

import numpy as np
from tqdm import tqdm

from .eval_func import *
from util.viewer_util import MjviserDebugViewerSession, normalize_debug_viewer_config


def normalize_eval_index_range(input_items, start_index, end_index):
    """Slice evaluation inputs by configured start and end indices.

    Args:
        input_items: Evaluation items after optional random sampling and max_num truncation.
        start_index: Inclusive start index. Negative values are clamped to 0.
        end_index: Exclusive end index. Negative values mean the end of the list.

    Returns:
        Tuple `(sliced_items, normalized_start, normalized_end)` used by evaluation and logs.
    """

    total_num = len(input_items)
    normalized_start = max(int(start_index), 0)
    normalized_end = total_num if int(end_index) < 0 else min(int(end_index), total_num)
    normalized_start = min(normalized_start, total_num)
    normalized_end = max(normalized_start, normalized_end)
    return input_items[normalized_start:normalized_end], normalized_start, normalized_end


def filter_input_paths_by_obj_scale(input_path_lst, target_obj_scale):
    """Keep only grasp files whose saved object scale matches the requested value."""

    if target_obj_scale is None:
        return input_path_lst, 0

    target_obj_scale = float(target_obj_scale)
    filtered_input_path_lst = []
    failed_num = 0
    for input_path in input_path_lst:
        try:
            grasp_data = np.load(input_path, allow_pickle=True).item()
            obj_scale = float(grasp_data["obj_scale"])
        except Exception as exc:
            failed_num += 1
            logging.warning(f"Failed to read obj_scale from {input_path}: {exc}")
            continue
        if np.isclose(obj_scale, target_obj_scale):
            filtered_input_path_lst.append(input_path)
    return filtered_input_path_lst, failed_num


def safe_eval_one(params):
    input_npy_path, configs = params[0], params[1]
    viewer_session = params[2] if len(params) > 2 else None
    eval_runner = None
    try:
        if configs.hand.mocap:
            eval_func_name = f"{configs.setting}MocapEval"
        else:
            eval_func_name = f"{configs.setting}ArmEval"

        eval_runner = eval(eval_func_name)(input_npy_path, configs, viewer_session=viewer_session)
        eval_runner.run()
        skipped_tiny_meshes = []
        if getattr(eval_runner, "mj_ho", None) is not None:
            skipped_tiny_meshes = list(getattr(eval_runner.mj_ho, "skipped_tiny_object_meshes", []))
        return {
            "tiny_convex_skipped_case": len(skipped_tiny_meshes) > 0,
            "tiny_convex_skipped_mesh_count": len(skipped_tiny_meshes),
            "tiny_convex_skipped_meshes": skipped_tiny_meshes,
        }
    except Exception:
        error_traceback = traceback.format_exc()
        logging.warning(f"Failed to evaluate {input_npy_path}\n{error_traceback}")
        skipped_tiny_meshes = []
        if eval_runner is not None and getattr(eval_runner, "mj_ho", None) is not None:
            skipped_tiny_meshes = list(getattr(eval_runner.mj_ho, "skipped_tiny_object_meshes", []))
        return {
            "tiny_convex_skipped_case": len(skipped_tiny_meshes) > 0,
            "tiny_convex_skipped_mesh_count": len(skipped_tiny_meshes),
            "tiny_convex_skipped_meshes": skipped_tiny_meshes,
        }


def summarize_tiny_convex_mesh_skips(results):
    """Summarize tiny convex mesh partial skips reported by eval workers.

    Args:
        results: Iterable of worker return dictionaries from `safe_eval_one`.

    Returns:
        Tuple `(case_count, mesh_instance_count, unique_mesh_count)` where case count
        is the number of evaluated grasps with at least one partial convex mesh skip.
    """

    case_count = 0
    mesh_instance_count = 0
    unique_mesh_keys = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("tiny_convex_skipped_case", False):
            case_count += 1
        mesh_instance_count += int(result.get("tiny_convex_skipped_mesh_count", 0))
        for mesh_record in result.get("tiny_convex_skipped_meshes", []):
            obj_path = mesh_record.get("obj_path", "")
            mesh_name = mesh_record.get("mesh", "")
            unique_mesh_keys.add((obj_path, mesh_name))
    return case_count, mesh_instance_count, len(unique_mesh_keys)


def task_eval(configs):
    assert (
        configs.task.simulation_metrics is not None
        or configs.task.analytic_fc_metrics is not None
        or configs.task.pene_contact_metrics is not None
    ), "You should at least evaluate one kind of metrics"
    viewer_config = None
    if configs.task.debug_viewer:
        viewer_config = normalize_debug_viewer_config(getattr(configs.task, "viewer", None))

    input_path_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    init_num = len(input_path_lst)

    if configs.skip:
        eval_path_lst = glob(os.path.join(configs.eval_dir, "**/*.npy"), recursive=True)
        eval_path_lst = [p.replace(configs.eval_dir, configs.grasp_dir) for p in eval_path_lst]
        input_path_lst = list(set(input_path_lst).difference(set(eval_path_lst)))
    skip_num = init_num - len(input_path_lst)
    input_path_lst = sorted(input_path_lst)
    scale_filter = getattr(configs.task, "obj_scale", None)
    scale_filter_input_num = len(input_path_lst)
    input_path_lst, scale_filter_failed_num = filter_input_paths_by_obj_scale(input_path_lst, scale_filter)
    start_index = int(getattr(configs.task, "start", 0))
    end_index = int(getattr(configs.task, "end", -1))
    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num].tolist()
    sampled_input_num = len(input_path_lst)
    indexed_input_path_lst, normalized_start, normalized_end = normalize_eval_index_range(
        list(enumerate(input_path_lst)),
        start_index,
        end_index,
    )
    input_index_lst = [sampled_index for sampled_index, _ in indexed_input_path_lst]
    input_path_lst = [input_path for _, input_path in indexed_input_path_lst]

    logging.info(
        f"Find {init_num} grasp data in {configs.grasp_dir}, skip {skip_num} already evaluated, "
        f"filter obj_scale={scale_filter} from {scale_filter_input_num} to {len(input_path_lst)} "
        f"({scale_filter_failed_num} read failures), "
        f"apply max_num to get {sampled_input_num}, select index range "
        f"[{normalized_start}, {normalized_end}), and use {len(input_path_lst)}."
    )

    if len(input_path_lst) == 0:
        return

    playlist_enabled = viewer_config is not None and viewer_config.playlist_enabled
    if configs.task.debug_viewer and not playlist_enabled and len(input_path_lst) != 1:
        raise ValueError(
            "Debug viewer mode supports exactly one grasp per run. "
            "Use task.start=<INDEX> task.end=<INDEX+1> to select one grasp, or enable "
            "task.viewer.playlist.enabled=true for continuous mjviser playback."
        )
    if viewer_config is not None:
        logging.info(
            "Use debug viewer backend=%s host=%s port=%d playlist=%s.",
            viewer_config.backend,
            viewer_config.host,
            viewer_config.port,
            viewer_config.playlist_enabled,
        )

    enable_tqdm = bool(getattr(configs.task, "tqdm", True))
    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    progress_desc = "Evaluating grasps"
    viewer_session = None
    try:
        if playlist_enabled:
            viewer_session = MjviserDebugViewerSession(viewer_config)

        if configs.task.debug_viewer or configs.task.debug_render:
            enable_tqdm = enable_tqdm and not bool(configs.task.debug_viewer)
            # Debug rendering runs serially. Viewer mode prints each grasp explicitly because tqdm is disabled.
            iterator = tqdm(iterable_params, total=len(input_path_lst), desc=progress_desc, disable=not enable_tqdm)
            results = []
            for selected_position, ip in enumerate(iterator):
                if configs.task.debug_viewer:
                    print(f"Evaluate grasp index {input_index_lst[selected_position]}: {ip[0]}", flush=True)
                eval_params = ip
                if viewer_session is not None:
                    viewer_session.begin_grasp(
                        sequence_position=selected_position,
                        sequence_total=len(input_path_lst),
                        input_index=input_index_lst[selected_position],
                        input_path=ip[0],
                    )
                    eval_params = (ip[0], ip[1], viewer_session)
                results.append(safe_eval_one(eval_params))
        else:
            with multiprocessing.Pool(processes=configs.n_worker) as pool:
                result_iter = pool.imap_unordered(safe_eval_one, iterable_params)
                # Multiprocessing progress advances when worker jobs finish.
                results = list(tqdm(result_iter, total=len(input_path_lst), desc=progress_desc, disable=not enable_tqdm))
    finally:
        if viewer_session is not None:
            viewer_session.close()

    grasp_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    succ_lst = glob(os.path.join(configs.succ_dir, "**/*.npy"), recursive=True)
    eval_lst = glob(os.path.join(configs.eval_dir, "**/*.npy"), recursive=True)
    logging.info(
        f"Get {len(grasp_lst)} grasp data, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )
    tiny_skip_case_count, tiny_skip_mesh_count, tiny_skip_unique_mesh_count = summarize_tiny_convex_mesh_skips(results)
    logging.info(
        "Tiny convex mesh filtering summary: %d evaluated cases had at least one tiny convex mesh skipped; "
        "%d partial convex mesh instances were skipped across %d unique object mesh pieces. "
        "Only those tiny convex mesh pieces were skipped; the grasp cases themselves were still evaluated.",
        tiny_skip_case_count,
        tiny_skip_mesh_count,
        tiny_skip_unique_mesh_count,
    )
    logging.info("Finish evaluation")

    return
