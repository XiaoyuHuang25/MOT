# -*- coding: utf-8 -*-

from typing import Iterable, Set

from stonesoup.base import Property
from stonesoup.hypothesiser import Hypothesiser
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import State

from trackers.utils.hypothesiser import MultipleHypothesis


class MHTHypothesiser(Hypothesiser):
    """Wrapper around a base Hypothesiser that operates on a single component.

    It scales each component–measurement hypothesis by the component weight,
    and stores the result both in the prediction weight and in the
    hypothesis.probability field, then packs everything into a MultipleHypothesis.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Underlying hypothesiser used to generate component–detection hypotheses."
    )

    def hypothesise(
        self,
        component: State,
        detections: Set[Detection],
        timestamp,
        **kwargs,
    ) -> MultipleHypothesis:
        """Form hypotheses for associations between detections and a given component.

        Parameters
        ----------
        component : :class:`~.State`
            The state/component to hypothesise on (e.g. a Gaussian component in an MBM).
        detections : set of :class:`~.Detection`
            Retrieved measurements at the given timestamp.
        timestamp : datetime
            Time corresponding to the detections/predicted state.
        **kwargs :
            Passed directly to the underlying ``self.hypothesiser``.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            Container of :class:`~.SingleProbabilityHypothesis` for the given component.
        """

        # Ensure all detections correspond to the same timestamp
        if detections:
            timestamps = {detection.timestamp for detection in detections}
            if len(timestamps) > 1:
                raise ValueError("All detections must share the same timestamp")

        # Delegate to the underlying hypothesiser for component–measurement pairs
        component_hypotheses: Iterable[SingleHypothesis] = self.hypothesiser.hypothesise(
            component, detections, timestamp, **kwargs
        )

        # Re-weight hypotheses by the component weight
        for hypothesis in component_hypotheses:
            # Combine component weight and hypothesis weight
            new_weight = component.weight * hypothesis.weight
            hypothesis.prediction.weight = new_weight
            # Use the same quantity as probability (consistent with existing code)
            hypothesis.probability = new_weight

        # Wrap all hypotheses into a MultipleHypothesis, normalised w.r.t. total weight
        return MultipleHypothesis(
            component_hypotheses,
            normalise=True,
            total_weight=1.0,
        )
