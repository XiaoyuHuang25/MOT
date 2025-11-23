
from datetime import datetime

import numpy as np
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState

from trackers.gnn_tracker.configs.base_config import Exp as BaseExp
from trackers.utils.hypothesiser import PDAHypothesiser
from stonesoup.plugins.pyehm import JPDAWithEHM, JPDAWithEHM2


class Exp(BaseExp):
    def __init__(self):

        self.start_time = datetime.now().replace(microsecond=0)
        self.set_common_params()

        self.tracker_name = "JPDAStoneSoup"

        self.set_params()

    def set_params(self):
        surveillance_area = np.prod(np.diff(self.env_range_c))
        self.clutter_spatial_density = self.clutter_rate / surveillance_area

        self.set_predictor_updater()
        self.set_deleter()

        prior_covar = self.set_prior_covar()

        prior_state = GaussianState(
            state_vector=StateVector(np.zeros((self.predictor.transition_model.ndim, 1))),
            covar=prior_covar,
            timestamp=self.start_time,
        )

        self.set_initiator(prior_state, probabilistic_hypothesiser=True)

        self.hypothesiser = PDAHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            prob_gate=self.prob_gate,
            clutter_spatial_density=self.clutter_spatial_density,
            prob_detect=self.prob_detect,
        )
        self.data_associator = JPDAWithEHM(self.hypothesiser)
