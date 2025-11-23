# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np

from .sort.detection import Detection
from .sort.tracker import Tracker
from .sort.nn_matching import NearestNeighborDistanceMetric


def xyxy2tlwh(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


class StrongSORT(object):
    def __init__(
            self,
            max_iou_distance,
            max_age,
            n_init,
            min_confidence,
            woC=False,
            NSA=False,
            EMA=True,
            EMA_alpha=0.9,
            MC=True,
            MC_lambda=0.98,
            max_dist=0.2,
            category_aware_tracking=False,
            *args,
            **kwargs,
    ):

        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.EMA = EMA
        self.EMA_alpha = EMA_alpha
        self.MC = MC
        self.MC_lambda = MC_lambda
        self.woC = woC
        self.NSA = NSA
        self.max_dist = max_dist
        self.category_aware_tracking = category_aware_tracking
        self.min_confidence = min_confidence
        if self.EMA:
            nn_budget = 1
        else:
            nn_budget = 100

        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init,
            EMA=self.EMA,
            EMA_alpha=self.EMA_alpha,
            MC=self.MC,
            MC_lambda=self.MC_lambda,
            woC=self.woC,
            NSA=self.NSA,
        )

    def update(self, dets, embs=None, warp_matrix=None):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        if len(self.tracker.tracks) >= 1:
            if warp_matrix is not None:
                for track in self.tracker.tracks:
                    track.camera_update(warp_matrix)

        # extract appearance information for each detection
        features = embs

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, feat) for
            box, conf, feat in
            zip(tlwh, confs, features) if conf > self.min_confidence
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        output_stracks = [track for track in self.tracker.tracks if track.is_confirmed()]
        return output_stracks
