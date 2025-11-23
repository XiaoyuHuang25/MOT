# -*- coding: utf-8 -*-

from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.prediction import Prediction


class GaussianMixturePrediction(Prediction, GaussianMixture):
    """Gaussian mixture prediction.

    Represents a predicted mixture state prior to measurement update.
    """
    pass
