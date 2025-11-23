# -*- coding: utf-8 -*-

from typing import Optional

from stonesoup.base import Property
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.prediction import Prediction
from stonesoup.types.state import GaussianState
from stonesoup.types.update import Update


class VisualDetection(Detection):
    """Detection type enriched with detector scores and optional feature vector.

    This extends :class:`~stonesoup.types.detection.Detection` with common
    fields used in vision-based tracking (objectness, class score, class id,
    final score, and an appearance feature vector).
    """

    objectness_score: Optional[float] = Property(
        default=None,
        doc="Objectness score from the detector (e.g. YOLO box score). "
            "Often multiplied with `class_score` to obtain a final confidence."
    )

    class_score: Optional[float] = Property(
        default=None,
        doc="Class confidence score (e.g. softmax probability for the "
            "predicted class)."
    )

    class_id: Optional[int] = Property(
        default=None,
        doc="Integer class ID of the detection (e.g. COCO category ID)."
    )

    score: Optional[float] = Property(
        default=None,
        doc="Final confidence score of the detection. Typically derived from "
            "`objectness_score` and `class_score`."
    )

    feature_vector: Optional[StateVector] = Property(
        default=None,
        doc="Optional appearance feature vector associated with the detection "
            "(e.g. ReID embedding)."
    )

    def __repr__(self) -> str:
        """Compact string representation for logging and debugging."""
        return (
            f"{self.__class__.__name__}("
            f"state_vector={self.state_vector}, "
            f"objectness_score={self.objectness_score}, "
            f"class_score={self.class_score}, "
            f"class_id={self.class_id}, "
            f"score={self.score}, "
            f"timestamp={self.timestamp}, "
            f"feature_vector_shape="
            f"{None if self.feature_vector is None else self.feature_vector.shape}"
            f")"
        )


class VisualGaussianState(GaussianState):
    """Gaussian state extended with visual detection attributes.

    This state is typically initialised from a :class:`VisualDetection`
    and keeps detector scores and appearance features alongside the
    kinematic state.
    """

    objectness_score: float = Property(
        doc="Objectness score of the detection that initiated this state."
    )

    class_score: float = Property(
        doc="Class confidence score of the detection that initiated this state."
    )

    class_id: int = Property(
        doc="Class ID of the detection that initiated this state."
    )

    score: float = Property(
        doc="Overall confidence score of the detection that initiated this state."
    )

    smoothed_score: Optional[float] = Property(
        default=None,
        doc="Score smoothed over time (e.g. exponential moving average), "
            "updated during tracking."
    )

    feature_vector: Optional[StateVector] = Property(
        default=None,
        doc="Optional appearance feature vector attached to the state "
            "(e.g. aggregated or copied from the latest detection)."
    )


class VisualGaussianStateUpdate(Update, VisualGaussianState):
    """Gaussian state update with visual attributes.

    This represents an updated state after incorporating a new
    :class:`VisualDetection`, while preserving visual meta-data such as
    detector scores and appearance features.
    """


class VisualGaussianStatePrediction(Prediction, VisualGaussianState):
    """Gaussian state prediction with visual attributes.

    This represents a predicted state (prior) before data association
    and update, while still carrying over visual meta-data from the
    previous state.
    """
