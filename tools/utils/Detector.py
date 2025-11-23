import json
import time
from copy import deepcopy
from functools import wraps
from pathlib import Path

import numpy as np
from loguru import logger
import torch
import os

from tqdm import tqdm

# from models.experimental import attempt_load
from tools.utils.CustomDataset import DataPipeline
from ultralytics import YOLO
# from ultralytics import YOLOv10 as BaseYOLOV10
# from utils.augmentations import letterbox
# from utils.general import check_suffix, check_img_size, non_max_suppression, scale_coords
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, xyxy2xywh
from tools.utils.visualize import vis
from torch.utils.data.dataloader import default_collate


def cache_inference():
    def decorator(func):
        @wraps(func)
        def wrapper(self, imgs, imgs_info):

            if self.cache and self.detector_cache_type == 'pickle':
                cache_outputs, cache_imgs_info = [], []
                for img_info in imgs_info:
                    video_name = img_info["video_name"]
                    image_id = img_info["image_id"]
                    output, img_info = self.detector_cache[video_name][image_id]
                    cache_outputs.append(output)
                    cache_imgs_info.append(img_info)
                outputs, imgs_info = cache_outputs, cache_imgs_info
            elif self.cache and self.detector_cache_type == 'json':
                cache_outputs, cache_imgs_info = [], []
                for img_info in imgs_info:
                    output = [x for x in self.detector_cache if x['image_id'] == img_info['frame_id']]
                    if len(output) == 0:
                        output = None
                    cache_outputs.append(output)
                    cache_imgs_info.append(img_info)
                outputs, imgs_info = cache_outputs, cache_imgs_info
            else:
                outputs, imgs_info = func(self, imgs, imgs_info)
                assert len(outputs) == len(imgs_info), "The number of outputs and imgs_info should be the same"
                for output, img_info in zip(outputs, imgs_info):
                    video_name = img_info["video_name"]
                    image_id = img_info["image_id"]
                    if video_name not in self.detector_cache:
                        self.detector_cache[video_name] = {}
                    self.detector_cache[video_name][image_id] = deepcopy(output), deepcopy(img_info)
            return outputs, imgs_info

        return wrapper

    return decorator


class Detector(object):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        self.exp = exp
        logger.info(f"Detector config: {self.exp.__dict__}")

        self.ckpt = ckpt

        self.timestamp = 0

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device}")
            else:
                self.device = torch.device("cpu")
                logger.warning("CUDA is not available, using CPU")

        self.model = None

        if os.path.exists(cache_path) and cache:
            self.cache = True
            logger.info("No need to load detector model, using cache")
        else:
            self.cache = False
            logger.info("Need to load detector model, cache file not found or not using cache")

        self.cache_path = cache_path

        if self.cache:
            logger.info("Loading detector cache in {}".format(self.cache_path))
            _, ext = os.path.splitext(self.cache_path)
            if ext == '.json':
                self.detector_cache_type = 'json'
                with open(self.cache_path) as f:
                    self.detector_cache = json.load(f)
            elif ext in ['.pkl', '.pickle']:
                self.detector_cache_type = 'pickle'
                self.detector_cache = torch.load(self.cache_path)
            else:
                raise ValueError("Unknown cache type")

        else:
            self.detector_cache_type = None
            self.detector_cache = {}

    @property
    def load(self):
        if self.model is None:
            return False
        else:
            return True

    @cache_inference()
    def inference(self, imgs, imgs_info):
        raise NotImplementedError

    def before_inference(self, imgs, imgs_info):
        raise NotImplementedError

    def after_inference(self, outputs, imgs_info):
        detections = []
        for output, img_info in zip(outputs, imgs_info):
            if self.detector_cache_type == 'json':
                detection = self.get_det_in_json(output, img_info)
            else:
                detection = self.get_det_in_raw_image(output, img_info)
            detections.append(detection)
        return detections, imgs_info

    def run(self, imgs, imgs_info):
        for img_info in imgs_info:
            img_info["detector_speed"] = {"preprocess": 0, "inference": 0, "postprocess": 0}

        if not self.cache:
            imgs, imgs_info = self.before_inference(imgs, imgs_info)

        outputs, imgs_info = self.inference(imgs, imgs_info)
        outputs, imgs_info = self.after_inference(outputs, imgs_info)
        for img_info in imgs_info:
            if "detector_speed" not in img_info:
                img_info["detector_speed"] = {"preprocess": 0, "inference": 0, "postprocess": 0}
        self.timestamp += 1
        return outputs, imgs_info

    def visual(self, detection, img, img_info, cls_names=None):
        if detection is None:
            return img
        file_name = os.path.join(img_info["split_name"], img_info["file_name"])
        cls = detection['cls']
        scores = detection['scores']
        bboxes = detection['xyxy']
        track_ids = detection['track_ids']

        vis_res = vis(img, bboxes, scores, cls, track_ids, self.exp.test_conf, cls_names, file_name)
        return vis_res

    def visual_without_scores(self, online_targets, img, img_info, cls_names=None):
        online_targets = np.array(online_targets)
        if np.array(online_targets).shape[0] == 0:
            return img
        online_targets = online_targets.reshape(-1, online_targets[0].shape[0])
        file_name = os.path.join(img_info["split_name"], img_info["file_name"])
        cls = online_targets[:, 5]
        bboxes = online_targets[:, 0:4]
        track_ids = online_targets[:, 4]
        scores = online_targets[:, 6] if online_targets.shape[1] > 6 else None

        vis_res = vis(img, bboxes, scores, cls, track_ids, self.exp.test_conf, cls_names, file_name)
        return vis_res

    @staticmethod
    def convert_to_coco_format(detection, img_info):
        data_list = []
        if np.array(detection['tlwh']).shape[0] == 0:
            return data_list
        image_id = img_info["image_id"]

        cls = np.array(detection['cls'])
        scores = np.array(detection['scores'])
        bboxes = np.array(detection['tlwh'])

        for ind in range(bboxes.shape[0]):
            label = cls[ind].item() + 1
            pred_data = {
                "image_id": image_id,
                "category_id": label,
                "bbox": bboxes[ind].tolist(),
                "score": scores[ind].item(),
                "segmentation": [],
            }  # COCO json format, need to add image_id
            data_list.append(pred_data)
        return data_list

    @staticmethod
    def get_det_in_raw_image(output, img_info):
        raise NotImplementedError

    @staticmethod
    def get_det_in_json(output, img_info):
        if output is None:
            detection = {'xyxy': np.zeros((0, 4)),
                         'tlwh': np.zeros((0, 4)),
                         'cls': np.zeros((0, 1)),
                         'scores': np.zeros((0, 1))}
        else:
            tlwh = np.array([tmp_['bbox'] for tmp_ in output])
            xyxy = deepcopy(tlwh)
            xyxy[:, 2] = xyxy[:, 2] + xyxy[:, 0]
            xyxy[:, 3] = xyxy[:, 3] + xyxy[:, 1]
            cls = np.array([tmp_['category_id'] for tmp_ in output])
            scores = np.array([tmp_['score'] for tmp_ in output])
            detection = {'xyxy': xyxy,
                         'tlwh': tlwh,
                         'cls': cls,
                         'scores': scores}
        return detection

    @staticmethod
    def get_det_gt_in_raw_image(target):
        target = deepcopy(target)
        bboxes = target[:, 0:4]
        cls = target[:, 4]
        scores = np.array([1.0] * len(cls))
        track_ids = target[:, 5]
        detection = {'xyxy': bboxes,
                     'tlwh': xyxy2xywh(deepcopy(bboxes)),
                     'cls': cls,
                     'scores': scores,
                     'track_ids': track_ids}
        return detection

    def load_model(self, trt_file=None):
        raise NotImplementedError

    def save_cache(self, final=False):
        # if self.timestamp % 100 == 0:
        #     os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        #     torch.save(self.detector_cache, self.cache_path)
        if final and not os.path.exists(self.cache_path):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(self.detector_cache, self.cache_path)
            logger.info("Save detector cache in {}".format(self.cache_path))
        else:
            logger.info("No need to save detector cache")


class YOLOX(Detector):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        super(YOLOX, self).__init__(
            exp,
            ckpt,
            device,
            cache,
            cache_path,
        )
        self.preproc = ValTransform(legacy=self.exp.test_legacy)

        if self.cache:
            self.model, self.trt_file, self.decoder = None, None, None
        else:
            self.model, self.trt_file, self.decoder = self.load_model()

        if self.trt_file is not None:
            assert torch.cuda.is_available(), "CUDA is not available!"

        if self.trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(self.trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).to(self.device)
            self.model(x)
            self.model = model_trt

    @cache_inference()
    def inference(self, imgs, imgs_info):
        inference_start_time = time.time()
        with torch.no_grad():
            outputs = self.model(imgs)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            inference_time = time.time() - inference_start_time
            postprocess_start_time = time.time()
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf,
                self.exp.nmsthre, class_agnostic=self.exp.class_agnostic,
            )
            postprocess_time = time.time() - postprocess_start_time

        outputs_res = []
        for output, img_info in zip(outputs, imgs_info):
            if output is not None:
                output_array = output.cpu().numpy()
                if self.exp.single_cls:
                    if not np.all(output_array[:, 6] == 0):
                        logger.warning("Single class model, but the output is not all zero")
                        output_array[:, 6] = 0
                outputs_res.append(output_array)
                img_info["detector_speed"]["inference"] += inference_time
                img_info["detector_speed"]["postprocess"] += postprocess_time
            else:
                outputs_res.append(None)
                img_info["detector_speed"]["inference"] += 0
                img_info["detector_speed"]["postprocess"] += 0

        return outputs_res, imgs_info

    def before_inference(self, imgs, imgs_info):

        img_list = []
        for img, img_info in zip(imgs, imgs_info):
            preprocess_start_time = time.time()
            ratio = min(self.exp.test_size[0] / img.shape[0], self.exp.test_size[1] / img.shape[1])
            img_info["ratio"] = ratio
            new_img, _ = self.preproc(img, None, self.exp.test_size)
            img_info["new_img_size"] = new_img.shape[:2]
            img_list.append(new_img)
            preprocess_time = time.time() - preprocess_start_time
            img_info["detector_speed"]["preprocess"] += preprocess_time

        padded_imgs = default_collate(img_list)

        padded_imgs = padded_imgs.to(self.device)
        if self.exp.test_fp16:
            padded_imgs = padded_imgs.half()  # to FP16
        return padded_imgs, imgs_info

    def load_model(self, trt_file=None):
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))

        model.to(self.device)
        if self.exp.test_fp16:
            model.half()  # to FP16
        model.eval()

        if trt_file is None:
            ckpt_file = self.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if self.exp.test_fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if trt_file is not None:
            assert not self.exp.test_fuse, "TensorRT model is not support model fusing!"
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None
        return model, trt_file, decoder

    @staticmethod
    def get_det_in_raw_image(output, img_info):
        ratio = img_info["ratio"]
        if output is None:
            return {'xyxy': np.zeros((0, 4)),
                    'tlwh': np.zeros((0, 4)),
                    'cls': np.zeros((0, 1)),
                    'scores': np.zeros((0, 1))}
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        output = deepcopy(output)
        bboxes = output[:, 0:4]
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        detection = {'xyxy': bboxes,
                     'tlwh': xyxy2xywh(deepcopy(bboxes)),
                     'cls': cls,
                     'scores': scores}
        return detection


class YOLOV8(Detector):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        super(YOLOV8, self).__init__(
            exp,
            ckpt,
            device,
            cache,
            cache_path,
        )

        if self.cache:
            self.model = None
        else:
            self.model = self.load_model()

    @cache_inference()
    def inference(self, imgs, imgs_info):
        outputs = self.model.predict(source=imgs,
                                     conf=self.exp.test_conf,
                                     iou=self.exp.nmsthre,
                                     imgsz=self.exp.test_size,
                                     half=self.exp.half,
                                     device=self.device,
                                     max_det=self.exp.max_det,
                                     vid_stride=self.exp.vid_stride,
                                     stream_buffer=self.exp.stream_buffer,
                                     visualize=self.exp.visualize,
                                     augment=self.exp.augment,
                                     agnostic_nms=self.exp.class_agnostic,
                                     classes=self.exp.classes,
                                     retina_masks=self.exp.retina_masks,
                                     embed=self.exp.embed)
        outputs_res = []
        for output, img_info in zip(outputs, imgs_info):
            if output is not None:
                output_array = output.boxes.data.cpu().numpy()
                if self.exp.single_cls:
                    if not np.all(output_array[:, 5] == 0):
                        logger.warning("Single class model, but the output is not all zero")
                        output_array[:, 5] = 0
                outputs_res.append(output_array)
                img_info["detector_speed"]["preprocess"] += output.speed["preprocess"]
                img_info["detector_speed"]["inference"] += output.speed["inference"]
                img_info["detector_speed"]["postprocess"] += output.speed["postprocess"]
            else:
                outputs_res.append(None)
                img_info["detector_speed"]["preprocess"] += 0
                img_info["detector_speed"]["inference"] += 0
                img_info["detector_speed"]["postprocess"] += 0

        return outputs_res, imgs_info

    def before_inference(self, imgs, imgs_info):
        return imgs, imgs_info

    def load_model(self, trt_file=None):
        model = YOLO(self.ckpt)
        return model

    @staticmethod
    def get_det_in_raw_image(output, img_info):
        if output is None:
            return {'xyxy': np.zeros((0, 4)),
                    'tlwh': np.zeros((0, 4)),
                    'cls': np.zeros((0, 1)),
                    'scores': np.zeros((0, 1))}
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        output = deepcopy(output)
        bboxes = output[:, 0:4]

        cls = output[:, 5]
        scores = output[:, 4]
        detection = {'xyxy': bboxes,
                     'tlwh': xyxy2xywh(deepcopy(bboxes)),
                     'cls': cls,
                     'scores': scores}
        return detection


class YOLOV10(YOLOV8):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        super(YOLOV10, self).__init__(
            exp,
            ckpt,
            device,
            cache,
            cache_path,
        )


class YOLOV5(YOLOV8):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        super(YOLOV5, self).__init__(
            exp,
            ckpt,
            device,
            cache,
            cache_path,
        )


class YOLOV6(YOLOV8):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        super(YOLOV6, self).__init__(
            exp,
            ckpt,
            device,
            cache,
            cache_path,
        )


class YOLOV9(YOLOV8):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        super(YOLOV9, self).__init__(
            exp,
            ckpt,
            device,
            cache,
            cache_path,
        )


class YOLOV11(YOLOV8):
    def __init__(
            self,
            exp,
            ckpt: str,
            device: int,
            cache: bool,
            cache_path: str = None,
    ):
        super(YOLOV11, self).__init__(
            exp,
            ckpt,
            device,
            cache,
            cache_path,
        )
