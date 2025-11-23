
from datetime import datetime

import numpy as np
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState

from trackers.gnn_tracker.configs.base_config import Exp as BaseExp
from trackers.gnn_tracker.configs.base_config import DistanceHypothesiser, Mahalanobis
from trackers.utils.dataassociator import GNNWith2DAssignment


class Exp(BaseExp):
    def __init__(self):

        self.start_time = datetime.now().replace(microsecond=0)
        self.set_common_params()
        self.tracker_name = "GNNStoneSoup"

        self.set_params()

    def set_params(self):
        self.set_predictor_updater()
        self.set_deleter()

        prior_covar = self.set_prior_covar()

        prior_state = GaussianState(
            state_vector=StateVector(np.zeros((self.predictor.transition_model.ndim, 1))),
            covar=prior_covar,
            timestamp=self.start_time,
        )

        self.set_initiator(prior_state, probabilistic_hypothesiser=False)

        self.hypothesiser = DistanceHypothesiser(
            self.predictor, self.updater,
            measure=Mahalanobis(), missed_distance=self.gating_distance
        )
        self.data_associator = GNNWith2DAssignment(
            self.hypothesiser,
        )
