import os
import uuid
from datetime import datetime

import numpy as np

from tools.utils.TrackerEvaluator import TrackerEvaluator, get_exp_by_file


# ======================== Constants & mappings ========================

DATASET_BASE_CONFIG = {
    "SDSObject": {
        "data_dir": "datasets/sds/images/Object",
        "eval_name": "SDS",
        "env_range_c": np.array([[0, 3840], [0, 2160]]),
    },
    "SMDVIS": {
        "data_dir": "datasets/smd/images/VIS",
        "eval_name": "SMD",
        "env_range_c": np.array([[0, 1920], [0, 1080]]),
    },
    "SMDNIR": {
        "data_dir": "datasets/smd/images/NIR",
        "eval_name": "SMD",
        "env_range_c": np.array([[0, 1920], [0, 1080]]),
    },
}

TRACKER_CONFIG_MAP = {
    "GNN_Prob_StoneSoup": "trackers/gnn_tracker/configs/prob_config.py",
    "GNN_NonProb_StoneSoup": "trackers/gnn_tracker/configs/non_prob_config.py",
    "JPDAwithNBest": "trackers/jpda_tracker/configs/JPDAwithNBest_config.py",
    "JPDAwithLBP": "trackers/jpda_tracker/configs/JPDAwithLBP_config.py",
    "JPDAwithEHM": "trackers/jpda_tracker/configs/JPDAwithEHM_config.py",
    "JPDAwithEHM2": "trackers/jpda_tracker/configs/JPDAwithEHM2_config.py",
    "MHTStoneSoup": "trackers/mht_tracker/configs/filter_config.py",
    "sort": "trackers/sort_tracker/configs/filter_config.py",
    "deepsort": "trackers/deepsort_tracker/configs/filter_config.py",
    "bytetrack": "trackers/byte_tracker/configs/filter_config.py",
    "botsort": "trackers/botsort_tracker/configs/filter_config.py",
    "strongsort": "trackers/strongsort_tracker/configs/filter_config.py",
    "modtreid": "trackers/motdt_tracker/configs/filter_config.py",
    "hybridsortreid": "trackers/hybrid_sort_tracker/configs/filter_config_reid.py",
    "hybridsort": "trackers/hybrid_sort_tracker/configs/filter_config.py",
    "ocsort": "trackers/ocsort_tracker/configs/filter_config.py",
}


def _get_tracker_config(tracker_name: str) -> str:
    """Return tracker config path given tracker name."""
    try:
        return TRACKER_CONFIG_MAP[tracker_name]
    except KeyError:
        raise ValueError(f"Unknown tracker name: {tracker_name}")


def tracking(
    dataset_name,
    data_cache,
    data_cache_type,
    data_pin_memory,
    data_workers,
    data_batch_size,
    data_split_name,
    detector_name,
    conf,
    nms,
    tsize,
    detector_device,
    detected_cache_path,
    detector_single_cls,
    detector_joint_nir_vis_train,
    detector_cache,
    detector_update_params,
    eval_detector,
    reid_device,
    reid_batch_size,
    reid_cache,
    gmc_cache,
    tracker_name,
    tracker_cache,
    tracker_update_params,
    eval_tracker,
    eval_tracker_metrics,
    vis_folder,
    save_cache,
    test_eval,
    logger,
):
    """
    Run a complete maritime MOT pipeline:
    1) Build dataset- and detector-related paths/configs
    2) Load detector_exp / tracker_exp and update with runtime params
    3) Run detection, ReID, GMC, and tracking via TrackerEvaluator
    4) Optionally run evaluation and return metrics
    """

    # ======================== Experiment name & JSON file ========================
    experiment_name = "track"

    if detector_single_cls:
        experiment_name = os.path.join(experiment_name, f"{dataset_name}_CAG")
        data_json_file = f"{data_split_name}_cag.json"
    else:
        experiment_name = os.path.join(experiment_name, dataset_name)
        data_json_file = f"{data_split_name}.json"

    if "SMD" in dataset_name:
        train_mode = "joint_vis_nir_train" if detector_joint_nir_vis_train else "single_train"
        experiment_name = os.path.join(experiment_name, train_mode)

    # ======================== Dataset configuration ========================
    base_cfg = DATASET_BASE_CONFIG.get(dataset_name)
    if base_cfg is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    data_dir = base_cfg["data_dir"]
    eval_tracker_dataset_name = base_cfg["eval_name"]
    env_range_c = base_cfg["env_range_c"]

    # ======================== ReID configuration ========================
    if "SMD" in dataset_name and detector_joint_nir_vis_train:
        dataset_name_tmp = "SMDVISNIR"
    else:
        dataset_name_tmp = dataset_name

    reid_config = f"exps/fastreid/configs/{dataset_name}/sbs_S50.yml"
    reid_weights = f"pretrained/Maritime/reid/{dataset_name_tmp}/sbs_S50/train/weights/model_best.pth"

    # ======================== Detector configuration ========================
    # Attach dataset name suffix for YOLOX-based detectors
    if "yolox" in detector_name.split("_"):
        detector_name_tmp = f"{detector_name}_{dataset_name}"
    else:
        detector_name_tmp = detector_name

    # Single-class (class-agnostic) mode suffix
    if detector_single_cls:
        detector_name_tmp = f"{detector_name_tmp}_CAG"
        dataset_name_tmp = f"{dataset_name_tmp}_CAG"

    # YOLOX vs other YOLO variants
    if "yolox" in detector_name.split("_"):
        detector_config = f"exps/yolox/mot/{detector_name_tmp}.py"
        detector_ckpt = (
            f"pretrained/Maritime/{detector_name}/{dataset_name_tmp}/train/weights/best_ckpt.pth"
        )
    else:
        detector_config = f"exps/yolo/mot/{detector_name_tmp}.py"
        detector_ckpt = f"pretrained/Maritime/{detector_name}/{dataset_name_tmp}/train/weights/best.pt"

    # ======================== CustomDataset overrides ========================
    # These are hard-coded overrides kept for compatibility with existing experiments
    if dataset_name == "CustomDataset_trainval":
        reid_config = "exps/fastreid/configs/SMDVISNIR/sbs_S50.yml"
        reid_weights = (
            "pretrained/Maritime/reid/SMDVISNIR/sbs_S50/train/weights/model_best.pth"
        )
        detector_config = "exps/yolox/mot/yolox_x_SMD_MVDD13_SeaShips_trainval.py"
        detector_ckpt = (
            "pretrained/Maritime/yolox_x/SMD_MVDD13_SeaShips_trainval/best_ckpt.pth"
        )
    elif dataset_name == "CustomDataset":
        reid_config = "exps/fastreid/configs/SMDVISNIR/sbs_S50.yml"
        reid_weights = (
            "pretrained/Maritime/reid/SMDVISNIR/sbs_S50/train/weights/model_best.pth"
        )
        detector_config = "exps/yolox/mot/yolox_x_SMD_MVDD13_SeaShips.py"
        detector_ckpt = "pretrained/Maritime/yolox_x/SMD_MVDD13_SeaShips/best_ckpt.pth"
    elif dataset_name == "CustomDataset_960":
        reid_config = "exps/fastreid/configs/SMDVISNIR/sbs_S50.yml"
        reid_weights = (
            "pretrained/Maritime/reid/SMDVISNIR/sbs_S50/train/weights/model_best.pth"
        )
        detector_config = "exps/yolox/mot/yolox_x_SMD_MVDD13_SeaShips_960.py"
        detector_ckpt = "pretrained/Maritime/yolox_x/SMD_MVDD13_SeaShips_960/best_ckpt.pth"

    # Attach environment range to tracker params
    tracker_update_params["env_range_c"] = env_range_c

    # ======================== Tracker configuration ========================
    tracker_config = _get_tracker_config(tracker_name)

    # ======================== Basic logging ========================
    logger.info(f"dataset_name: {dataset_name}")
    logger.info(f"data_dir: {data_dir}")
    logger.info(f"reid_config: {reid_config}")
    logger.info(f"reid_weights: {reid_weights}")
    logger.info(f"eval_tracker_dataset_name: {eval_tracker_dataset_name}")
    logger.info(f"env_range_c: {env_range_c}")
    logger.info(f"detector_name: {detector_name}")
    logger.info(f"detector_config: {detector_config}")
    logger.info(f"detector_ckpt: {detector_ckpt}")
    logger.info(f"tracker_name: {tracker_name}")
    logger.info(f"tracker_config: {tracker_config}")

    # ======================== Load experiment configs ========================
    detector_exp = get_exp_by_file(detector_config)
    tracker_exp = get_exp_by_file(tracker_config)

    logger.info("==============Update parameters.================================")
    for detector_param_name, detector_param_value in detector_update_params.items():
        setattr(detector_exp, detector_param_name, detector_param_value)
        logger.info(f"Set {detector_param_name} to {detector_param_value} in detector_exp.")
        if hasattr(detector_exp, "update_params"):
            detector_exp.update_params(detector_exp, detector_param_name, detector_param_value, logger)

    for tracker_param_name, tracker_param_value in tracker_update_params.items():
        setattr(tracker_exp, tracker_param_name, tracker_param_value)
        logger.info(f"Set {tracker_param_name} to {tracker_param_value} in tracker_exp.")
        if hasattr(tracker_exp, "update_params"):
            tracker_exp.update_params(tracker_exp, tracker_param_name, tracker_param_value, logger)
    logger.info("=================================================================")

    # ======================== Experiment directories & log file ========================
    cache_dir = os.path.join(
        os.path.dirname(detector_exp.output_dir),
        "cache",
        os.path.basename(detector_exp.output_dir),
        experiment_name,
    )
    experiment_dir = os.path.join(detector_exp.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_filename = os.path.join(
        experiment_dir, "logs", data_split_name, tracker_name, f"{current_time}.log"
    )
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    log_handler_id = logger.add(log_filename, level="INFO")

    # ======================== Cache paths ========================
    try:
        if detected_cache_path is None:
            detected_cache_path = os.path.join(
                cache_dir, "detected_results", os.path.basename(data_split_name) + ".pkl"
            )
            os.makedirs(os.path.dirname(detected_cache_path), exist_ok=True)
            logger.info(f"Set detected_cache_path to {detected_cache_path}.")
        else:
            logger.info(f"Set detected_cache_path to {detected_cache_path}.")

        reid_cache_path = os.path.join(
            cache_dir, "reid_results", os.path.basename(data_split_name) + ".pkl"
        )
        os.makedirs(os.path.dirname(reid_cache_path), exist_ok=True)
        logger.info(f"Set reid_cache_path to {reid_cache_path}.")

        if hasattr(tracker_exp, "with_gmc") and tracker_exp.with_gmc:
            gmc_cache_path = os.path.join(
                cache_dir,
                "gmc_results",
                os.path.basename(data_split_name),
                tracker_exp.gmc_method + str(tracker_exp.gmc_downscale) + ".pkl",
            )
            os.makedirs(os.path.dirname(gmc_cache_path), exist_ok=True)
            logger.info(f"Set gmc_cache_path to {gmc_cache_path}.")
        else:
            gmc_cache_path = None
            logger.info("Set gmc_cache_path to None.")

        tracker_to_eval = f"{detector_exp.detector_name}_{tracker_exp.tracker_name}_{tracker_exp.__module__}"

        tracker_cache_path = os.path.join(
            str(experiment_dir),
            "track_results",
            os.path.basename(data_split_name),
            tracker_to_eval + ".pkl",
        )
        os.makedirs(os.path.dirname(tracker_cache_path), exist_ok=True)
        logger.info(f"Set tracker_cache_path to {tracker_cache_path}.")

        tracker_result_path = os.path.join(
            str(experiment_dir),
            "track_results",
            f"{current_time}_{uuid.uuid4().hex[:8]}",
            os.path.basename(data_split_name),
            tracker_to_eval,
        )
        os.makedirs(tracker_result_path, exist_ok=True)
        logger.info(f"Set tracker_result_path to {tracker_result_path}.")

        # ======================== Override detector test-time params ========================
        if conf is not None:
            detector_exp.test_conf = conf
        if nms is not None:
            detector_exp.nmsthre = nms
        if tsize is not None:
            detector_exp.test_size = (tsize, tsize)

        # ======================== Build evaluator and run tracking ========================
        tracker_eval = TrackerEvaluator(
            data_dir=data_dir,
            data_json_file=data_json_file,
            data_split_name=data_split_name,
            data_cache=data_cache,
            data_cache_type=data_cache_type,
            data_batch_size=data_batch_size,
            data_workers=data_workers,
            data_pin_memory=data_pin_memory,
            detector_exp=detector_exp,
            detector_ckp=detector_ckpt,
            detector_device=detector_device,
            detector_cache=detector_cache,
            detector_cache_path=detected_cache_path,
            reid_config=reid_config,
            reid_weights=reid_weights,
            reid_device=reid_device,
            reid_batch_size=reid_batch_size,
            reid_cache=reid_cache,
            reid_cache_path=reid_cache_path,
            gmc_cache=gmc_cache,
            gmc_cache_path=gmc_cache_path,
            tracker_exp=tracker_exp,
            tracker_cache=tracker_cache,
            tracker_cache_path=tracker_cache_path,
            save_cache=save_cache,
        )

        eval_exp = {
            "BENCHMARK": eval_tracker_dataset_name,
            "SPLIT_TO_EVAL": data_split_name,
            "TRACKERS_TO_EVAL": [tracker_to_eval],
            "METRICS": eval_tracker_metrics,
            "TIME_PROGRESS": False,
            "USE_PARALLEL": False,
            "NUM_PARALLEL_CORES": 8,
            "GT_FOLDER": data_dir,
            "TRACKERS_FOLDER": os.path.dirname(os.path.dirname(tracker_result_path)),
            "TRACKER_SUB_FOLDER": "",
        }

        # Keep the original eval_types setting (class-agnostic evaluation)
        eval_types = ["CAG"]

        ap_res, tracker_results, tracker_results_class_agnostic = tracker_eval.tracking(
            eval_tracker_dataset_name,
            tracker_result_path,
            vis_folder,
            eval_exp,
            eval_detector,
            eval_tracker,
            eval_types,
            test_eval,
        )
    finally:
        logger.remove(log_handler_id)
        try:
            if 'tracker_eval' in locals():
                if getattr(tracker_eval, 'reid', None) is not None:
                    tracker_eval.reid.features_cache.clear()
                if getattr(tracker_eval, 'gmc', None) is not None:
                    tracker_eval.gmc.gmc_cache.clear()
                if getattr(tracker_eval, 'detector', None) is not None and hasattr(tracker_eval.detector, 'detector_cache'):
                    tracker_eval.detector.detector_cache.clear()
        except Exception:
            pass
    return ap_res, tracker_results, tracker_results_class_agnostic
