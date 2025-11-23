import os
from functools import wraps

import torch
from loguru import logger
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

def cache_gmc():
    def decorator(func):
        @wraps(func)
        def wrapper(self, dets, features, img_info):
            video_name = img_info["video_name"]
            image_id = img_info["image_id"]
            cache_exist = video_name in self.gmc_cache and image_id in self.gmc_cache[video_name]
            if self.cache and cache_exist:
                result = self.gmc_cache[video_name][image_id]
            else:
                result = func(self, dets, features, img_info)
                if video_name not in self.gmc_cache:
                    self.gmc_cache[video_name] = {}
                self.gmc_cache[video_name][image_id] = copy.deepcopy(result)
            return result

        return wrapper

    return decorator



class GMC:
    def __init__(self,
                 downscale=2,
                 cache: bool = False,
                 cache_path: str = None
                 ):

        self.downscale = max(1, int(downscale))

        if os.path.exists(cache_path) and cache:
            self.model = None
            self.cache = True
            logger.info("No need to load gmc model, using cache")
        else:
            number_of_iterations = 100
            termination_eps = 1e-5
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
            
            self.cache = False
            logger.info("Need to load gmc model, cache file not found or not using cache")

        self.cache_path = cache_path
        if self.cache:
            logger.info("Loading gmc cache in {}".format(self.cache_path))
            self.gmc_cache = torch.load(self.cache_path)
        else:
            self.gmc_cache = {}

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    @property
    def load(self):
        return not self.cache

    @cache_gmc()
    def apply(self, raw_frame, detections, img_info):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1.0:

            frame = cv2.resize(frame, (0, 0), fx= 1 / self.downscale, fy= 1 / self.downscale, interpolation=cv2.INTER_LINEAR)
            width = width // self.downscale
            height = height // self.downscale
            scale = [1 / self.downscale, 1 / self.downscale]


        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            # Initialization done
            self.initializedFirstFrame = True
            if H.shape[0] == 2:
                H = np.vstack((H, np.array([[0, 0, 1]])))           # warp_matrix from [2, 3] to [3, 3]
            return H

        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
            if self.downscale > 1.0:
                H[0, 2] = H[0, 2] / scale[0]
                H[1, 2] = H[1, 2] / scale[1]
            if H.shape[0] == 2:
                H = np.vstack((H, np.array([[0, 0, 1]])))           # warp_matrix from [2, 3] to [3, 3]
            self.prevFrame = frame.copy()
        except:
            print('Warning: find transform failed. Set warp as identity')
            if H.shape[0] == 2:
                H = np.vstack((H, np.array([[0, 0, 1]])))

        assert H.shape[0] == 3, "ECC warp matrix should be 3x3"
        return H
    