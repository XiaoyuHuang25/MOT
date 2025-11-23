# -*- coding: utf-8 -*-

from math import isfinite

import numpy as np
from scipy.spatial.distance import cdist
from stonesoup.base import Property
from stonesoup.hypothesiser.probability import PDAHypothesiser as BasePDAHypothesiser
from stonesoup.measures import SquaredMahalanobis
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis as BaseMultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.track import Track

from trackers.utils.logpdf import FastLogPDFWithCovCache
from trackers.utils.state import VisualGaussianState


class PDAHypothesiser(BasePDAHypothesiser):
    """Standard PDA hypothesiser with a faster log-pdf implementation.

    This subclass only changes the likelihood computation to use
    `FastLogPDFWithCovCache`, keeping the original PDA logic intact.
    """

    def hypothesise(self, track, detections, timestamp, **kwargs):
        r"""Evaluate and return all track association hypotheses.

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.

        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`i, i \in {1...N}`: detection i is associated
        with the track.

        The probabilities for these detections are calculated as follows:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        Notes
        -----
        Since all probabilities share the same denominator and are normalized
        later, the denominator can be discarded when computing the
        non-normalized probabilities.

        Parameters
        ----------
        track : Track
            Track object to hypothesise on.
        detections : set of Detection
            Available detections at the given timestamp.
        timestamp : datetime.datetime
            Timestamp used for the state prediction when evaluating the track.

        Returns
        -------
        MultipleHypothesis
            Container of :class:`SingleProbabilityHypothesis` objects.
        """
        hypotheses = []
        validated_measurements = 0
        measure = SquaredMahalanobis(state_covar_inv_cache_size=None)

        # Common state prediction for missed detection
        prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)

        # Missed detection hypothesis
        probability = Probability(1 - self.prob_detect * self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability,
            )
        )

        # True detection hypotheses
        for detection in detections:
            # Predict state for the detection timestamp
            prediction = self.predictor.predict(
                track, timestamp=detection.timestamp, **kwargs
            )

            # Measurement prediction
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model, **kwargs
            )

            # Likelihood of detection under the predicted measurement
            log_prob = FastLogPDFWithCovCache.safe_logpdf(
                x=detection.state_vector,
                mean=measurement_prediction.state_vector,
                cov=measurement_prediction.covar,
            )
            probability = Probability(log_prob, log_value=True)

            # Standard Mahalanobis gating
            if measure(measurement_prediction, detection) <= self._gate_threshold(
                self.prob_gate, measurement_prediction.ndim
            ):
                validated_measurements += 1
                valid_measurement = True
            else:
                # Gated out unless include_all is set
                valid_measurement = False

            if self.include_all or valid_measurement:
                probability *= self.prob_detect
                if self.clutter_spatial_density is not None:
                    probability /= self.clutter_spatial_density

                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        detection,
                        probability,
                        measurement_prediction,
                    )
                )

        # Validation volume scaling when clutter density is unknown
        if self.clutter_spatial_density is None and validated_measurements > 0:
            for hypothesis in hypotheses[1:]:  # Skip missed detection
                hypothesis.probability *= (
                    self._validation_region_volume(
                        self.prob_gate, hypothesis.measurement_prediction
                    )
                    / validated_measurements
                )

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class MultipleHypothesis(BaseMultipleHypothesis):
    """MultipleHypothesis that normalises probabilities in log domain.

    This preserves numerical stability when probabilities are small and
    originally represented in log space.
    """

    def normalise_probabilities(self, total_weight=None):
        if total_weight is None:
            total_weight = self.total_weight

        # Verify that all underlying hypotheses carry a Probability
        if any(
            not hasattr(hypothesis, "probability")
            for hypothesis in self.single_hypotheses
        ):
            raise ValueError(
                "MultipleHypothesis is not composed entirely of Probability hypotheses."
            )

        # Sum in log-domain using Probability.sum (StoneSoup utility)
        sum_weights = Probability.sum(
            hypothesis.probability for hypothesis in self.single_hypotheses
        )

        # Rescale each hypothesis so that sum(probabilities) == total_weight
        for hypothesis in self.single_hypotheses:
            hypothesis.probability = Probability(
                hypothesis.probability.log_value
                + np.log(total_weight)
                - sum_weights.log_value,
                log_value=True,
            )
