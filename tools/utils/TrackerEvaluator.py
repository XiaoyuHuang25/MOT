import importlib
import json
import os
import sys
import time
from multiprocessing import freeze_support

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

import tools.utils.Detector as Detectors
import tools.utils.Trackers as Trackers
import trackeval
from tools.utils.CustomDataset import DataPipeline, list_to_table
from tools.utils.ReID import ReID
from tools.utils.gmc import GMC

logger.remove()
logger.add(sys.stderr, level="INFO")

_EXP_CACHE = {}

def get_exp_by_file(exp_file):
    key = os.path.abspath(exp_file)
    if key in _EXP_CACHE:
        ExpCls = _EXP_CACHE[key]
        return ExpCls()

    module_name = os.path.basename(exp_file).split(".")[0]
    original_path = os.path.dirname(exp_file)
    try:
        sys.path.append(original_path)
        current_exp = importlib.import_module(module_name)
        ExpCls = current_exp.Exp
        _EXP_CACHE[key] = ExpCls
        return ExpCls()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    finally:
        sys.path.remove(original_path)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,{category},-1,-1\n'
    with open(filename, 'w') as f:
        for result in results:
            if len(result) == 5:
                frame_id, tlwhs, track_ids, category_ids, _ = result
            elif len(result) == 4:
                frame_id, tlwhs, track_ids, category_ids = result
            else:
                frame_id, tlwhs, track_ids = result
                category_ids = [-1] * len(track_ids)
            for tlwh, track_id, category_id in zip(tlwhs, track_ids, category_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id,
                                          x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1),
                                          category=category_id)
                f.write(line)
    logger.info('save results to {}'.format(filename))

class TrackerEvaluator(object):
    """
    Wrapper for running detection + ReID + GMC + tracking + evaluation
    on a given dataset split.
    """

    def __init__(
        self,
        data_dir="./datasets/smd/Data/NIR",
        data_json_file="val.json",
        data_split_name="val",
        data_cache=True,
        data_cache_type="disk",
        data_batch_size=1,
        data_workers=0,
        data_pin_memory=False,
        detector_exp=None,
        detector_ckp=None,
        detector_device=0,
        detector_cache=True,
        detector_cache_path=None,
        reid_config="exps/fastreid/SMDNIR/sbs_S50.yml",
        reid_weights="pretrained/Maritime/reid_smd_nir/model_best.pth",
        reid_device=0,
        reid_batch_size=1,
        reid_cache=True,
        reid_cache_path="YOLOX_outputs/yolox_x_smd_nir/reid_results/val.pkl",
        gmc_cache=False,
        gmc_cache_path=None,
        tracker_exp=None,
        tracker_cache=True,
        tracker_cache_path=None,
        save_cache=False,
    ):
        """
        Build detector, ReID, GMC, dataset/dataloader, and tracker objects.

        Parameters are mostly configuration and cache settings for each component.
        """

        # Tracker identity
        self.tracker_name = tracker_exp.tracker_name

        # -------------------- Detector --------------------
        detector_cls = getattr(Detectors, detector_exp.detector_name)
        self.detector = detector_cls(
            exp=detector_exp,
            ckpt=detector_ckp,
            device=detector_device,
            cache=detector_cache,
            cache_path=detector_cache_path,
        )

        # -------------------- ReID (optional) --------------------
        self.use_reid = bool(getattr(tracker_exp, "with_reid", False))
        if self.use_reid:
            logger.info("Using ReID in tracker")
            self.reid = ReID(
                reid_config,
                reid_weights,
                reid_device,
                reid_batch_size,
                reid_cache,
                reid_cache_path,
            )
        else:
            logger.info("Not using ReID in tracker")
            self.reid = None

        # -------------------- GMC (optional) --------------------
        self.use_gmc = bool(getattr(tracker_exp, "with_gmc", False))
        if self.use_gmc:
            logger.info("Using GMC in tracker")
            self.gmc = GMC(
                downscale=tracker_exp.gmc_downscale,
                cache=gmc_cache,
                cache_path=gmc_cache_path,
            )
        else:
            logger.info("Not using GMC in tracker")
            self.gmc = None

        # -------------------- Decide whether to load raw images --------------------
        # If detector/ReID/GMC are loading from cache, we do not need to pre-load data
        load_flags = [getattr(self.detector, "load", False)]
        if self.reid is not None:
            load_flags.append(getattr(self.reid, "load", False))
        if self.gmc is not None:
            load_flags.append(getattr(self.gmc, "load", False))

        load_data = any(load_flags)
        if not load_data:
            data_pin_memory = False
            logger.info("No need to load data; set pin_memory to False")

        # -------------------- Dataset & dataloader --------------------
        data_pipeline = DataPipeline(
            data_dir=data_dir,
            json_file=data_json_file,
            name=data_split_name,
            cache=data_cache,
            cache_type=data_cache_type,
            load_data=load_data,
            batch_size=data_batch_size,
            num_workers=data_workers,
            pin_memory=data_pin_memory,
        )
        self.dataset, self.dataloader = data_pipeline()

        # -------------------- Tracker --------------------
        tracker_cls = getattr(Trackers, self.tracker_name)
        self.tracker = tracker_cls(
            exp=tracker_exp,
        )

        # -------------------- Cache control --------------------
        self.save_cache = save_cache

    # ----------------------------------------------------------------------
    # Evaluation utilities
    # ----------------------------------------------------------------------
    def evaluation(self, eval_tracker_dataset_name, eval_types, params):
        """
        Run tracker evaluation using trackeval for the specified benchmark.

        eval_types:
            - contains 'CA'  -> run class-aware evaluation
            - contains 'CAG' -> run class-agnostic evaluation
        """
        if eval_tracker_dataset_name == "SMD":
            from trackeval.datasets import SMD2DBox as Box
            from trackeval.datasets.smd_2d_box_class_agnostic import (
                SMD2DBoxClassAgnostic as BoxClassAgnostic,
            )
        elif eval_tracker_dataset_name == "SDS":
            from trackeval.datasets import SDS2DBox as Box
            from trackeval.datasets.sds_2d_box_class_agnostic import (
                SDS2DBoxClassAgnostic as BoxClassAgnostic,
            )
        else:
            raise Exception("Unknown dataset name for evaluation")

        # Class-aware evaluation
        if "CA" in eval_types:
            info = self.box_eval(Box, params)
            for title, data in info.items():
                if "Simple" in title:
                    logger.info(list_to_table(data, title, separator=" "))
        else:
            info = None

        # Class-agnostic evaluation
        if "CAG" in eval_types:
            info_class_agnostic = self.box_eval(BoxClassAgnostic, params)
            for title, data in info_class_agnostic.items():
                if "Simple" in title:
                    logger.info(list_to_table(data, title, separator=" "))
        else:
            info_class_agnostic = None

        return info, info_class_agnostic

    @staticmethod
    def box_eval(Box, params):
        """
        Generic evaluation helper with trackeval.

        - Builds default eval/dataset/metrics configs
        - Overrides with user-provided params
        - Runs trackeval and reshapes the output into a table-like dict
        """
        freeze_support()

        # ---- Load defaults ----
        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config["DISPLAY_LESS_PROGRESS"] = False

        default_dataset_config = Box.get_default_dataset_config()
        default_metrics_config = {
            "METRICS": ["LP", "HOTA", "CLEAR", "Identity"],
            "THRESHOLD": 0.5,
        }

        # Merge default configs
        config = {
            **default_eval_config,
            **default_dataset_config,
            **default_metrics_config,
        }

        # Override with user params
        for key, value in params.items():
            assert key in config.keys(), f"Unknown option: {key}"
            config[key] = value
            logger.info(f"Set {key} to {value} in config.")

        # Split back into eval/dataset/metrics configs
        eval_config = {k: v for k, v in config.items() if k in default_eval_config}
        dataset_config = {
            k: v for k, v in config.items() if k in default_dataset_config
        }
        metrics_config = {
            k: v for k, v in config.items() if k in default_metrics_config
        }

        # ---- Build evaluator/dataset/metrics ----
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [Box(dataset_config)]

        metrics_list = []
        for metric in [
            trackeval.metrics.HOTA,
            trackeval.metrics.CLEAR,
            trackeval.metrics.Identity,
            trackeval.metrics.VACE,
        ]:
            if metric.get_name() in metrics_config["METRICS"]:
                metrics_list.append(metric(metrics_config))

        if len(metrics_list) == 0:
            raise Exception("No metrics selected for evaluation")

        # ---- Run evaluation ----
        output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

        # ---- Post-process results into human-readable tables ----
        info = {}
        for eval_dataset, eval_trackers in output_res.items():
            for eval_tracker, eval_sequences in eval_trackers.items():
                HOTA_title = f"{eval_dataset}-{eval_tracker}-HOTA"
                IDF1_title = f"{eval_dataset}-{eval_tracker}-IDF1"
                MOTA_title = f"{eval_dataset}-{eval_tracker}-MOTA"
                Count_title = f"{eval_dataset}-{eval_tracker}-Count"
                Simple_title = f"{eval_dataset}-{eval_tracker}-Simple"

                info.update(
                    {
                        HOTA_title: [["Sequence"]],
                        IDF1_title: [["Sequence"]],
                        MOTA_title: [["Sequence"]],
                        Count_title: [["Sequence"]],
                        Simple_title: [["Sequence"]],
                    }
                )

                for eval_sequence, eval_categories in eval_sequences.items():
                    # For combined sequence, create simple summary rows
                    if eval_sequence == "COMBINED_SEQ":
                        info[Simple_title].extend([["HOTA"], ["IDF1"], ["MOTA"]])

                    info[HOTA_title].append([eval_sequence])
                    info[IDF1_title].append([eval_sequence])
                    info[MOTA_title].append([eval_sequence])
                    info[Count_title].append([eval_sequence])

                    for eval_category, eval_metrics in eval_categories.items():
                        # Column header for metric tables (e.g., "cls_comb", "all")
                        header_list = [eval_category]

                        # Simple summary (only for COMBINED_SEQ)
                        if eval_sequence == "COMBINED_SEQ":
                            if not all(elem in info[Simple_title][0] for elem in header_list):
                                info[Simple_title][0].extend(header_list)

                            info[Simple_title][1].append(
                                f"{np.mean(eval_metrics['HOTA']['HOTA']) * 100:1.5g}"
                            )
                            info[Simple_title][2].append(
                                f"{np.mean(eval_metrics['Identity']['IDF1']) * 100:1.5g}"
                            )
                            info[Simple_title][3].append(
                                f"{np.mean(eval_metrics['CLEAR']['MOTA']) * 100:1.5g}"
                            )

                        # HOTA table
                        if not all(elem in info[HOTA_title][0] for elem in header_list):
                            info[HOTA_title][0].extend(header_list)
                        info[HOTA_title][-1].append(
                            f"{np.mean(eval_metrics['HOTA']['HOTA']) * 100:1.5g}"
                        )

                        # IDF1 table
                        if not all(elem in info[IDF1_title][0] for elem in header_list):
                            info[IDF1_title][0].extend(header_list)
                        info[IDF1_title][-1].append(
                            f"{float(eval_metrics['Identity']['IDF1']) * 100:1.5g}"
                        )

                        # MOTA table
                        if not all(elem in info[MOTA_title][0] for elem in header_list):
                            info[MOTA_title][0].extend(header_list)
                        info[MOTA_title][-1].append(
                            f"{float(eval_metrics['CLEAR']['MOTA']) * 100:1.5g}"
                        )

                        # Count table
                        frames = int(eval_metrics["Count"]["Frames"]) if "Frames" in eval_metrics["Count"] else 0
                        header_list = [
                            f"{eval_category}-Dets",
                            f"{eval_category}-GT_Dets",
                            f"{eval_category}-IDs",
                            f"{eval_category}-GT_IDs",
                            f"{eval_category}-Frames",
                        ]
                        if not all(elem in info[Count_title][0] for elem in header_list):
                            info[Count_title][0].extend(header_list)
                        info[Count_title][-1].extend(
                            [
                                f"{int(eval_metrics['Count']['Dets']):}",
                                f"{int(eval_metrics['Count']['GT_Dets']):d}",
                                f"{int(eval_metrics['Count']['IDs']):d}",
                                f"{int(eval_metrics['Count']['GT_IDs']):d}",
                                f"{frames:d}",
                            ]
                        )

                    # If there is no combined class metrics, fill with '/'
                    if "cls_comb_det_av" not in eval_categories and "cls_comb_cls_av" not in eval_categories:
                        for title in info:
                            if title != Simple_title:
                                info[title][-1].extend(["/", "/"])

        # Log as tables
        for title, data in info.items():
            logger.info(list_to_table(data, title, separator=" "))

        return info

    # ----------------------------------------------------------------------
    # Main tracking loop
    # ----------------------------------------------------------------------
    def tracking(
            self,
            eval_tracker_dataset_name,
            tracker_result_path,
            vis_folder,
            eval_exp,
            eval_detector,
            eval_tracker,
            eval_types,
            test_eval=False,
    ):
        """
        Run online tracking over the entire dataloader.

        - Produces txt results for each video
        - Optionally: visualizations (mp4), detector metrics, tracker metrics
        """
        n_samples = max(len(self.dataloader), 1)

        # -------------------- Speed guard configuration --------------------
        # After processing `fps_check_frames` frames, compute average FPS.
        # If the average FPS is lower than `min_fps_threshold`, abort tracking
        # and return zero results.
        min_fps_threshold = 10.0  # e.g., require at least 10 frame per second
        fps_check_frames = 30  # number of frames before checking FPS
        processed_frames = 0
        total_frames = 0

        # Accumulators for detector and tracker outputs
        detector_result_list = []
        detector_result_list_gt = []
        tracker_to_detectors_result_list = []

        results = []  # per-video in-memory buffer
        results_all_videos = {}  # for SDS test export

        inference_time = 0.0
        track_time = 0.0  # kept for compatibility with evaluate_prediction

        # Video visualization writer
        out = None
        fourcc = None
        fps = None
        if vis_folder is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 15

        last_video_name = None
        total_batches = len(self.dataloader)

        start_time = time.time()

        # -------------------- Main loop over batches --------------------
        for batch_idx, data in tqdm(
                enumerate(self.dataloader),
                total=total_batches,
                desc="Inferencing",
        ):
            # Each batch: list of (image, image_info)
            imgs_tmp = [data_i[0] for data_i in data]
            imgs_info_tmp = [data_i[1] for data_i in data]

            # Detector forward
            outputs, imgs_info = self.detector.run(imgs_tmp, imgs_info_tmp)
            inference_time += sum(
                val for img_info in imgs_info for val in img_info["detector_speed"].values()
            )

            # -------------------- Loop over frames in batch --------------------
            for img, img_info, detection in zip(imgs_tmp, imgs_info, outputs):
                video_name = img_info["video_name"]
                image_id = img_info["image_id"]
                file_name = img_info["file_name"]
                frame_id = img_info["frame_id"]
                target = img_info["target"]

                # -------------------- Handle video boundary --------------------
                if last_video_name != video_name:
                    # New video starts
                    new_video = True
                    if last_video_name is not None:
                        # Finalize previous video: flush txt and reset tracker state
                        logger.info(f"Finish tracking video: {last_video_name}")

                        self.tracker.reset_model()

                        result_filename = os.path.join(
                            tracker_result_path, f"{last_video_name}.txt"
                        )
                        results_all_videos[last_video_name] = results
                        write_results_no_score(result_filename, results)
                        results = []
                else:
                    new_video = False

                # -------------------- ReID (optional) --------------------
                if self.use_reid:
                    features, img_info = self.reid.run(img, detection["xyxy"], img_info)
                else:
                    # Dummy features and zeroed timings when ReID is disabled
                    features = np.array([None] * len(detection["xyxy"]))
                    img_info["reid_speed"] = {
                        "preprocess": 0,
                        "inference": 0,
                        "postprocess": 0,
                    }
                inference_time += sum(img_info["reid_speed"].values())

                # -------------------- GMC (optional) --------------------
                if self.use_gmc:
                    gmc_inference_start_time = time.time()
                    warp_matrix = self.gmc.apply(img, detection["xyxy"], img_info)
                    img_info["gmc_speed"] = {
                        "inference": time.time() - gmc_inference_start_time
                    }
                    inference_time += sum(img_info["gmc_speed"].values())
                else:
                    warp_matrix = None

                # -------------------- Tracking / test-eval mode --------------------
                if test_eval:
                    # Use GT boxes to test detector/tracker pipeline
                    detection_gt = self.detector.get_det_gt_in_raw_image(target)
                    online_targets_gt = [
                        np.vstack(
                            [
                                bbox.reshape(-1, 1),
                                np.array([[track_id]]),
                                np.array([[category + 1]]),
                            ]
                        )
                        for bbox, track_id, category in zip(
                            detection_gt["xyxy"],
                            detection_gt["track_ids"],
                            detection_gt["cls"],
                        )
                        if track_id != -1
                    ]
                    online_targets = online_targets_gt
                else:
                    detection_gt = None
                    online_targets = self.tracker.run(
                        detection, features, warp_matrix, img_info
                    )
                    inference_time += sum(img_info["tracker_speed"].values())

                # -------------------- Convert online targets to tlwh/ids/categories/scores --------------------
                online_tlwhs = []
                online_ids = []
                online_categories = []
                online_scores = []

                for t in online_targets:
                    t = t.squeeze()
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]

                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                    if len(t) > 5:
                        online_categories.append(t[5])
                    else:
                        online_categories.append(1)

                    if len(t) > 6:
                        online_scores.append(t[6])
                    else:
                        online_scores.append(1)

                # Save per-frame tracking result
                results.append(
                    (frame_id, online_tlwhs, online_ids, online_categories, online_scores)
                )

                # -------------------- Visualization --------------------
                if vis_folder is not None:
                    if img.shape[:2] == (0, 0):
                        img = self.dataset.read_img_from_id(image_id)

                    img_info["image_size"] = img.shape[:2]

                    if new_video:
                        # Close previous writer and open a new one
                        if out is not None:
                            out.release()
                        output_video_file = os.path.join(vis_folder, video_name + ".mp4")
                        out = cv2.VideoWriter(
                            output_video_file,
                            fourcc,
                            fps,
                            (img_info["image_size"][1], img_info["image_size"][0]),
                        )

                    result_image = self.detector.visual_without_scores(
                        online_targets, img, img_info
                    )
                    out.write(result_image)

                # -------------------- Prepare detection results for evaluation --------------------
                if test_eval:
                    # Detector eval with GT boxes
                    output_results_gt = self.detector.convert_to_coco_format(
                        detection_gt, img_info
                    )
                    detector_result_list_gt.extend(output_results_gt)

                output_results = self.detector.convert_to_coco_format(
                    detection, img_info
                )
                detector_result_list.extend(output_results)

                # Tracker output mapped back into detector-style results
                detection_from_tracker = {"tlwh": [], "xyxy": [], "scores": [], "cls": []}
                for online_tlwh, online_id, online_category in zip(
                        online_tlwhs, online_ids, online_categories
                ):
                    x, y, w, h = online_tlwh
                    detection_from_tracker["tlwh"].append([x, y, w, h])
                    detection_from_tracker["xyxy"].append([x, y, x + w, y + h])
                    detection_from_tracker["scores"].append(1)
                    detection_from_tracker["cls"].append(online_category - 1)

                output_results_from_tracker = self.detector.convert_to_coco_format(
                    detection_from_tracker, img_info
                )
                tracker_to_detectors_result_list.extend(output_results_from_tracker)

                last_video_name = video_name

        # -------------------- Finalize last video after loop --------------------
        if last_video_name is not None:
            logger.info("Finish tracking video: %s", last_video_name)
            result_filename = os.path.join(
                tracker_result_path, f"{last_video_name}.txt"
            )
            results_all_videos[last_video_name] = results
            write_results_no_score(result_filename, results)
            logger.info("Finish tracking all videos")

            # Sanity check on IDs and categories (uses last video's results)
            track_ids = [
                track_id_tmp
                for result in results
                if result[2]
                for track_id_tmp in result[2]
            ]
            logger.info(
                f"Max track id: {max(track_ids)}, min track id: {min(track_ids)}"
            )
            assert min(track_ids) >= 1, "Track id should start from 1"

            categories = [
                category_tmp
                for result in results
                if result[3]
                for category_tmp in result[3]
            ]
            logger.info(
                f"Max category: {max(categories)}, min category: {min(categories)}"
            )
            assert min(categories) >= 1, "Category should start from 1"

        # Release video writer if still open
        if vis_folder is not None and out is not None:
            out.release()

        logger.info(f"Total time: {time.time() - start_time}")

        # -------------------- Save caches (optional) --------------------
        if self.save_cache:
            logger.info("Save detector, reid and gmc cache")
            self.detector.save_cache(final=True)
            if self.use_reid:
                self.reid.save_cache(final=True)
            if self.use_gmc:
                self.gmc.save_cache(final=True)
        else:
            logger.info("Not saving detector, reid and gmc cache")

        # -------------------- Detector evaluation --------------------
        ap_res = [0, 0, 0]
        if eval_detector:
            if test_eval:
                logger.info("Using GT to test detector evaluation")
                *_, summary, _ = self.dataset.evaluate_prediction(
                    detector_result_list_gt, [inference_time, track_time, n_samples]
                )
                logger.info(summary)

            logger.info("Evaluating detector")
            *_, summary, ap_res = self.dataset.evaluate_prediction(
                detector_result_list, [inference_time, track_time, n_samples]
            )
            logger.info(summary)

            logger.info("Evaluating detector from tracker")
            *_, summary, _ = self.dataset.evaluate_prediction(
                tracker_to_detectors_result_list,
                [inference_time, track_time, n_samples],
            )
            logger.info(summary)

        # -------------------- Tracker evaluation --------------------
        if eval_tracker and not (
                eval_exp["BENCHMARK"] == "SDS"
                and eval_exp["SPLIT_TO_EVAL"] == "test"
        ):
            logger.info("Evaluating tracker")
            tracker_results, tracker_results_class_agnostic = self.evaluation(
                eval_tracker_dataset_name, eval_types, eval_exp
            )

        elif eval_tracker and (
                eval_exp["BENCHMARK"] == "SDS"
                and eval_exp["SPLIT_TO_EVAL"] == "test"
        ):
            # For SDS test split we only export JSON results
            tracker_results, tracker_results_class_agnostic = None, None
            logger.info("Skip evaluating tracker (SDS test split)")

            video_name_list = [
                "DJI_0001.mov",
                "DJI_0051.MP4",
                "DJI_0065.MP4",
                "DJI_0001_d3.mov",
                "DJI_0039.MP4",
                "DJI_0003.mov",
                "DJI_0064.MP4",
                "DJI_0069.MP4",
                "DJI_0011_d3.mov",
                "DJI_0057.MP4",
                "DJI_0032.MP4",
                "DJI_0001.MOV",
                "DJI_0010_d3.mov",
                "DJI_0063.MP4",
                "DJI_0059.MP4",
                "DJI_0006_d3.mov",
                "DJI_0055.MP4",
                "DJI_0041.MP4",
                "DJI_0038.MP4",
            ]
            assert len(results_all_videos) == len(video_name_list)

            json_results = []
            for video_name in video_name_list:
                # Keys in results_all_videos use '.' replaced by '_' in video_name
                results = results_all_videos[video_name.replace(".", "_")]
                for (
                        frame_id,
                        tlwhs,
                        track_ids,
                        categories,
                        confidences,
                ) in results:
                    json_result = []
                    for tlwh, track_id, category, confidence in zip(
                            tlwhs, track_ids, categories, confidences
                    ):
                        object_id = track_id
                        bbox_left = tlwh[0]
                        bbox_top = tlwh[1]
                        bbox_right = tlwh[0] + tlwh[2]
                        bbox_bottom = tlwh[1] + tlwh[3]
                        json_result.append(
                            [
                                object_id,
                                bbox_left,
                                bbox_top,
                                bbox_right,
                                bbox_bottom,
                                confidence,
                            ]
                        )
                    json_results.append([json_result])

            assert len(json_results) == 18253
            json_path = os.path.join(
                os.path.dirname(tracker_result_path),
                f"{os.path.basename(tracker_result_path)}.json",
            )
            logger.info(f"Save tracker results to {json_path}")
            json.dump(json_results, open(json_path, "w"), cls=NumpyEncoder)

        else:
            tracker_results, tracker_results_class_agnostic = None, None
            logger.info("Skip evaluating tracker")

        return ap_res, tracker_results, tracker_results_class_agnostic
