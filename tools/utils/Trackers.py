import gc
import os
import time
from copy import deepcopy
from datetime import timedelta
from functools import wraps

import numpy as np
import torch
from loguru import logger

from trackers.botsort_tracker.bot_sort import BoTSORT as BoTSORTTracker
from trackers.byte_tracker.byte_tracker import BYTETracker
from trackers.deepsort_tracker.deepsort import DeepSort
from trackers.utils.state import VisualDetection
from trackers.gnn_tracker.gnn_tracker import GNNTracker
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
from trackers.hybrid_sort_tracker.hybrid_sort_reid import Hybrid_Sort_ReID
from trackers.jpda_tracker.jpda_tracker import JPDATracker
from trackers.mht_tracker.mht_tracker import MHTTracker

from trackers.ocsort_tracker.ocsort import OCSort
from trackers.sort_tracker.sort import Sort
from trackers.strongsort_tracker.strong_sort import StrongSORT as StrongSORTTracker

class Tracker(object):
    def __init__(
            self,
            exp,
    ):
        self.exp = exp
        logger.info(f"Tracker config: {self.exp.__dict__}")

        self.timestamp = 0
        self.new_track_id_dict = {}

        self.model = self.model_init()

    def before_update(self, dets, features):
        return dets, features

    def after_update(self, estimations):
        return estimations

    def run(self, dets, features, warp_matrix, img_info):
        img_info["tracker_speed"] = {"preprocess": 0, "inference": 0, "postprocess": 0}

        preprocess_start_time = time.time()
        dets, features = self.before_update(dets, features)
        preprocess_time = time.time() - preprocess_start_time
        img_info["tracker_speed"]["preprocess"] += preprocess_time

        inference_start_time = time.time()
        estimations = self.update(dets, features, warp_matrix, img_info)
        inference_time = time.time() - inference_start_time
        img_info["tracker_speed"]["inference"] += inference_time

        postprocess_start_time = time.time()
        estimations = self.after_update(estimations)

        for estimation in estimations:
            track_id = estimation[4]
            if track_id not in self.new_track_id_dict:
                self.new_track_id_dict[track_id] = len(self.new_track_id_dict) + 1
            estimation[4] = self.new_track_id_dict[track_id]

        postprocess_time = time.time() - postprocess_start_time
        img_info["tracker_speed"]["postprocess"] += postprocess_time
        self.timestamp += 1
        return estimations

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets, features, warp_matrix)
        return estimations

    def model_init(self):
        model = None
        return model

    def reset_model(self):
        if hasattr(self.exp, "reset"):
            self.exp.reset()
            logger.info("Reset tracker exp")

        if hasattr(self.model, "reset"):
            self.model.reset()
            logger.info("Reset tracker model")
        else:
            self.model = self.model_init()
            logger.info("reinitialize tracker model")

class SORT(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(SORT, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return Sort(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets)
        return estimations

    def after_update(self, estimations):
        online_targets = [track for track in estimations]
        return online_targets


class DeepSORT(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(DeepSORT, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return DeepSort(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features.reshape(-1, 2048)

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets, features)
        return estimations

    def after_update(self, estimations):
        online_targets = [track for track in estimations]
        return online_targets


class BYTE(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(BYTE, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return BYTETracker(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets)
        return estimations

    def after_update(self, estimations):
        online_targets = [
            np.vstack([track.tlbr.reshape(-1, 1),
                       np.array([[track.track_id + 1]]),
                       np.array([[track.category + 1]]) if hasattr(track, 'category') else 1]).squeeze()
            for track in estimations]
        return online_targets


class BoTSORT(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(BoTSORT, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return BoTSORTTracker(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets, features, warp_matrix)
        return estimations

    def after_update(self, estimations):
        online_targets = [
            np.vstack(
                [
                    track.tlbr.reshape(-1, 1),
                    np.array([[track.track_id + 1]]),
                    np.array([[track.cls + 1]]) if hasattr(track, 'cls') else 1
                 ]
            ).squeeze()
            for track in estimations]
        return online_targets


class StrongSORT(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(StrongSORT, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return StrongSORTTracker(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets, features, warp_matrix)
        return estimations

    def after_update(self, estimations):
        online_targets = [
            np.vstack([track.to_tlbr().reshape(-1, 1),
                       np.array([[track.track_id + 1]]),
                       np.array([[track.cls + 1]]) if hasattr(track, 'cls') else 1]).squeeze()
            for track in estimations]
        return online_targets

class HybridSORTReID(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(HybridSORTReID, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return Hybrid_Sort_ReID(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features.reshape(-1, 2048)

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets, id_feature=features, warp_matrix=None)
        return estimations

    def after_update(self, estimations):
        return estimations


class HybridSORT(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(HybridSORT, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return Hybrid_Sort(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets)
        return estimations

    def after_update(self, estimations):
        return estimations


class OCSORT(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(OCSORT, self).__init__(exp)

    def model_init(self):
        tracker_params = {**self.exp.__dict__}
        return OCSort(**tracker_params)

    def before_update(self, dets, features):
        measurements = np.array(
            [
                np.vstack([xyxy.reshape(-1, 1), np.array([[score]]), np.array([[cls]])]).squeeze()
                for xyxy, score, cls in zip(dets['xyxy'], dets['scores'], dets['cls'])
            ]
        ).reshape(-1, 6)
        return measurements, features

    def update(self, dets, features, warp_matrix, img_info):
        estimations = self.model.update(dets)
        return estimations

    def after_update(self, estimations):
        return estimations

class GNNStoneSoup(Tracker):
    def __init__(
            self,
            exp,
    ):
        super(GNNStoneSoup, self).__init__(exp)
        self.start_time = exp.start_time

    def model_init(self):
        self.exp.set_params()
        tracker_params = {**self.exp.__dict__}
        tracker = GNNTracker(**tracker_params)
        return tracker

    def before_update(self, dets, features):
        self.start_time = self.start_time + timedelta(seconds=1 / 30)
        measurement_set = set()
        if dets['scores'].size != 0:
            remain_inds = dets['scores'] > self.exp.det_thresh
            for xyxy, cls, score, feature in zip(dets['xyxy'][remain_inds], dets['cls'][remain_inds],
                                                 dets['scores'][remain_inds], features[remain_inds]):
                xyxyc = np.append(xyxy, score)

                measurement_set.add(VisualDetection(
                    state_vector=self.exp.measurement_model.convert_bbox_to_meas(xyxyc).reshape(-1, 1),
                    timestamp=self.start_time,
                    measurement_model=self.exp.measurement_model,
                    objectness_score=None,
                    class_score=None,
                    class_id=cls,
                    score=score,
                    feature_vector=feature.reshape(-1, 1) if feature is not None else None,
                    metadata=None
                ))
        return measurement_set, None

    def update(self, observations, features, warp_matrix, img_info):
        estimations = self.model.estimate(observations, self.start_time, warp_matrix)
        return estimations

    def after_update(self, estimations):
        online_targets = []
        for track in estimations:
            x = track['x'].squeeze()
            track_id = track['track_id']
            xyxy = self.exp.measurement_model.convert_meas_to_bbox(x)
            object_class = track.get("class_id", 0) + 1
            score = track.get('score', 1)
            bbox = np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], track_id + 1, object_class, score])
            online_targets.append(bbox)
        return online_targets


class JPDAStoneSoup(GNNStoneSoup):
    def __init__(
            self,
            exp,
    ):
        super(JPDAStoneSoup, self).__init__(exp)
        self.start_time = exp.start_time

    def model_init(self):
        self.exp.set_params()
        tracker_params = {**self.exp.__dict__}
        return JPDATracker(**tracker_params)

class MHTStoneSoup(GNNStoneSoup):
    def __init__(
            self,
            exp,
    ):
        super(MHTStoneSoup, self).__init__(exp)
        self.start_time = exp.start_time

    def model_init(self):
        self.exp.set_params()
        tracker_params = {**self.exp.__dict__}
        return MHTTracker(**tracker_params)
