# -*- coding: utf-8 -*-

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian


class VisualLinearGaussian(LinearGaussian):
    """Linear Gaussian measurement model with helper utilities for bbox <-> measurement.

    Measurement format: [cx, cy, w, h]
        - cx, cy: centre coordinates
        - w, h:  width and height (in pixels)

    Bounding box format: [x1, y1, x2, y2]
        - (x1, y1): top-left
        - (x2, y2): bottom-right
    """

    @staticmethod
    def convert_bbox_to_meas(bbox, eps: float = 1e-12) -> np.ndarray:
        """Convert a bounding box [x1, y1, x2, y2] to measurement [cx, cy, w, h].

        Parameters
        ----------
        bbox : array_like
            Bounding box in xyxy format (x1, y1, x2, y2).
        eps : float, optional
            Reserved for compatibility; not strictly needed here.

        Returns
        -------
        np.ndarray
            Measurement vector [cx, cy, w, h].
        """
        b = np.asarray(bbox, dtype=float).ravel()
        if b.size < 4:
            raise ValueError(f"bbox must have at least 4 elements, got shape {b.shape}")

        x1, y1, x2, y2 = b[:4]
        w = x2 - x1
        h = y2 - y1

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return np.array([cx, cy, w, h], dtype=float)

    @staticmethod
    def convert_meas_to_bbox(meas) -> np.ndarray:
        """Convert measurement [cx, cy, w, h] to bounding box [x1, y1, x2, y2].

        Parameters
        ----------
        meas : array_like
            Measurement vector [cx, cy, w, h].

        Returns
        -------
        np.ndarray
            Bounding box [x1, y1, x2, y2].
        """
        b = np.asarray(meas, dtype=float).ravel()
        if b.size < 4:
            raise ValueError(f"meas must have at least 4 elements, got shape {b.shape}")

        cx, cy, w, h = b[:4]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return np.array([x1, y1, x2, y2], dtype=float)
