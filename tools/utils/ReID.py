import argparse
import ast
import copy
import importlib
import os
import sys
import time

from functools import wraps

from loguru import logger

import torch

import os

from tools.fastreid.fast_reid_interfece import FastReIDInterface


def cache_inference():
    def decorator(func):
        @wraps(func)
        def wrapper(self, img, bboxes, img_info):
            video_name = img_info["video_name"]
            image_id = img_info["image_id"]
            cache_exist = video_name in self.features_cache and image_id in self.features_cache[video_name]
            if self.cache and cache_exist:
                _cache = self.features_cache[video_name][image_id]
                if isinstance(_cache, tuple):
                    feature, img_info = _cache
                else:
                    feature = _cache
            else:
                feature, img_info = func(self, img, bboxes, img_info)
                if video_name not in self.features_cache:
                    self.features_cache[video_name] = {}
                self.features_cache[video_name][image_id] = copy.deepcopy(feature), copy.deepcopy(img_info)
            return feature, img_info

        return wrapper

    return decorator


class ReID(object):
    def __init__(
            self,
            config,
            weights: str,
            device: int,
            batch_size: int,
            cache: bool,
            cache_path: str = None
    ):
        self.config = config
        self.weights = weights
        self.batch_size = batch_size

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device}")
            else:
                self.device = torch.device("cpu")
                logger.warning("CUDA is not available, using CPU")

        if os.path.exists(cache_path) and cache:
            self.model = None
            self.cache = True
            logger.info("No need to load reid model, using cache")
        else:
            self.model = self.model_init()
            self.cache = False
            logger.info("Need to load reid model, cache file not found or not using cache")

        self.cache_path = cache_path
        if self.cache:
            logger.info("Loading reid cache in {}".format(self.cache_path))
            self.features_cache = torch.load(self.cache_path)
        else:
            self.features_cache = {}

        self.timestamp = 0

    @property
    def load(self):
        if self.model is None:
            return False
        else:
            return True

    @cache_inference()
    def inference(self, img, bboxes, img_info):
        inference_start_time = time.time()
        features = self.model.inference(img, bboxes)
        inference_time = time.time() - inference_start_time
        img_info["reid_speed"]["inference"] += inference_time
        return features, img_info

    def before_inference(self, img, bboxes, img_info):
        return img, bboxes, img_info

    def after_inference(self, features, img_info):
        return features, img_info

    def run(self, img, bboxes, img_info):
        img_info["reid_speed"] = {"preprocess": 0, "inference": 0, "postprocess": 0}
        if not self.cache:
            img, bboxes, img_info = self.before_inference(img, bboxes, img_info)
        features, img_info = self.inference(img, bboxes, img_info)
        features, img_info = self.after_inference(features, img_info)
        if "reid_speed" not in img_info:
            img_info["reid_speed"] = {"preprocess": 0, "inference": 0, "postprocess": 0}
        self.timestamp += 1
        return features, img_info

    def model_init(self):
        return FastReIDInterface(self.config, self.weights, self.device, self.batch_size)

    def save_cache(self, final=False):
        # if self.timestamp % 100 == 0:
        #     os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        #     torch.save(self.features_cache, self.cache_path)
        if final and not os.path.exists(self.cache_path):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(self.features_cache, self.cache_path)
            logger.info("Save reid cache in {}".format(self.cache_path))
        else:
            logger.info("No need to save reid cache")
