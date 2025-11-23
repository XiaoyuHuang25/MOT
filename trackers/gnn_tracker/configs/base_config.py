
from abc import ABC
from stonesoup.dataassociator import DataAssociator

from stonesoup.deleter import Deleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.hypothesiser import Hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator import Initiator
from stonesoup.initiator.simple import MultiMeasurementInitiator, SimpleMeasurementInitiator

from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor import Predictor
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import CovarianceMatrix
from stonesoup.updater import Updater
from datetime import datetime

from stonesoup.updater.kalman import KalmanUpdater

from trackers.utils.dataassociator import GNNWith2DAssignment, NStateAssignment
from trackers.utils.hypothesiser import PDAHypothesiser
from trackers.utils.measurement import VisualLinearGaussian


import numpy as np

class Exp(ABC):
    det_thresh: float
    q_std_position_xy: float
    q_std_position_w: float
    q_std_position_h: float
    r_std_position_xy: float
    r_std_position_w: float
    r_std_position_h: float
    p_std_position_xy: float
    p_std_position_w: float
    p_std_position_h: float
    p_std_velocity_xy: float
    p_std_velocity_w: float
    p_std_velocity_h: float
    tracker_name: str
    low_thresh: float
    high_thresh: float
    position_q: list
    position_r: list
    position_p: list
    velocity_p: list
    mapping: tuple
    gating_distance: int
    prob_detect: float
    prob_gate: float
    env_range_c: np.ndarray
    clutter_rate: float
    time_steps_since_update: int
    min_initiation_points: int
    using_multi_measurement_initiator: bool
    transition_model: TransitionModel
    measurement_model: MeasurementModel
    predictor: Predictor
    updater: Updater
    start_time: datetime
    initiator: Initiator
    deleter: Deleter
    hypothesiser: Hypothesiser
    data_associator: DataAssociator
    clutter_spatial_density: float
    top_n_hypotheses: int

    def set_common_params(self):
        self.det_thresh = 0.7

        self.q_std_position_xy = 13
        self.q_std_position_w = 10
        self.q_std_position_h = 10

        self.r_std_position_xy = 24
        self.r_std_position_w = 17
        self.r_std_position_h = 17

        self.p_std_position_xy = 37
        self.p_std_position_w = 33
        self.p_std_position_h = 33

        self.p_std_velocity_xy = 19
        self.p_std_velocity_w = 38
        self.p_std_velocity_h = 38

        self.gating_distance = 7
        self.mapping = (0, 2, 4, 6)

        self.prob_detect = 0.99999
        self.prob_gate = 0.999999999

        self.env_range_c = np.array([[0, 1920], [0, 1080]])  # Surveillance area
        self.clutter_rate = 921

        self.time_steps_since_update = 41
        self.min_initiation_points = 8

        self.top_n_hypotheses = 5

        self.using_multi_measurement_initiator = True

    def map_weights_stick(self,beta_meas, beta_feat):
        a = float(beta_meas)
        b = float(beta_feat)
        a = min(max(a, 0.0), 1.0)
        b = min(max(b, 0.0), 1.0)
        weight_meas = a
        weight_feat = (1.0 - a) * b
        weight_iou = (1.0 - a) * (1.0 - b)
        return weight_meas, weight_feat, weight_iou


    def set_predictor_updater(self):
        self.position_q = [
            self.q_std_position_xy ** 2,
            self.q_std_position_xy ** 2,
            self.q_std_position_w ** 2,
            self.q_std_position_h ** 2
        ]
        self.transition_model = CombinedLinearGaussianTransitionModel(
            [
                ConstantVelocity(self.position_q[0]),
                ConstantVelocity(self.position_q[1]),
                ConstantVelocity(self.position_q[2]),
                ConstantVelocity(self.position_q[3])
            ])

        self.position_r = [
            self.r_std_position_xy ** 2,
            self.r_std_position_xy ** 2,
            self.r_std_position_w ** 2,
            self.r_std_position_h ** 2
        ]
        self.measurement_model = VisualLinearGaussian(
            ndim_state=self.transition_model.ndim,
            mapping=self.mapping,
            noise_covar=np.diag(self.position_r)
        )
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = KalmanUpdater(self.measurement_model)

    def set_prior_covar(self):
        self.position_p = [
            self.p_std_position_xy ** 2,
            self.p_std_position_xy ** 2,
            self.p_std_position_w ** 2,
            self.p_std_position_h ** 2
        ]
        self.velocity_p = [
            self.p_std_velocity_xy ** 2,
            self.p_std_velocity_xy ** 2,
            self.p_std_velocity_w ** 2,
            self.p_std_velocity_h ** 2
        ]
        prior_covar = CovarianceMatrix(np.diag(
            [
                self.position_p[0], self.velocity_p[0],
                self.position_p[1], self.velocity_p[1],
                self.position_p[2], self.velocity_p[2],
                self.position_p[3], self.velocity_p[3]
            ]
        ))
        return prior_covar

    def set_deleter(self):
        self.deleter = UpdateTimeStepsDeleter(time_steps_since_update=self.time_steps_since_update)

    def set_initiator(self, prior_state, probabilistic_hypothesiser: bool = False, **kwargs):
        if probabilistic_hypothesiser:
            initiator_hypothesiser = PDAHypothesiser(
                predictor=self.predictor,
                updater=self.updater,
                prob_gate=self.prob_gate,
                clutter_spatial_density=self.clutter_spatial_density,
                prob_detect=self.prob_detect,
            )

        else:
            initiator_hypothesiser = DistanceHypothesiser(
                self.predictor, self.updater,
                measure=Mahalanobis(), missed_distance=self.gating_distance
            )
        initiator_data_associator = GNNWith2DAssignment(
            initiator_hypothesiser,
        )

        initiator_deleter = UpdateTimeStepsDeleter(time_steps_since_update=self.time_steps_since_update)


        self.initiator = SimpleMeasurementInitiator(
            prior_state=prior_state,
            measurement_model=self.updater.measurement_model,
        )
        if self.using_multi_measurement_initiator:
            self.initiator = MultiMeasurementInitiator(
                prior_state=prior_state,
                deleter=initiator_deleter,
                data_associator=initiator_data_associator,
                updater=self.updater,
                measurement_model=self.updater.measurement_model,
                min_points=self.min_initiation_points,
                updates_only=True,
                initiator=self.initiator,
                skip_non_reversible=False,
            )

    @staticmethod
    def update_params(tracker_exp, attribute, value, logger):
        if hasattr(tracker_exp, attribute):
            setattr(tracker_exp, attribute, value)
            tracker_exp.set_params()
            logger.info(f"Updated {attribute} to {value} in {tracker_exp.tracker_name}")
        else:
            logger.error(f"{attribute} is not a valid attribute of {tracker_exp.tracker_name}")
