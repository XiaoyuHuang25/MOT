import numpy as np

from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix
from stonesoup.types.numeric import Probability
from stonesoup.types.update import GaussianStateUpdate


def gm_reduce_single(means, covars, weights, *, eps=1e-12, jitter=1e-9, spd_fix=True):
    """
    Reduce a Gaussian mixture into a single Gaussian (moment matching).

    Parameters
    ----------
    means : StateVectors or array-like of shape (d, K)
        Mean vectors of the Gaussian mixture components.
    covars : array-like of shape (d, d, K)
        Covariance matrices of the mixture components.
    weights : array-like of shape (K,)
        Component weights (not necessarily normalized).
    eps : float, optional
        Minimal eigenvalue and minimal total weight guard.
    jitter : float, optional
        Base jitter magnitude added to the diagonal during SPD fixes.
    spd_fix : bool, optional
        If True, try to enforce the output covariance to be SPD.

    Returns
    -------
    mean : StateVector of shape (d, 1)
        Mean of the reduced Gaussian.
    covar : CovarianceMatrix of shape (d, d)
        Covariance of the reduced Gaussian.

    Notes
    -----
    - This implementation:
        1) Normalizes weights with basic guards.
        2) Computes the moment-matched mean and covariance.
        3) Symmetrizes the covariance and (optionally) projects to SPD if needed.
    """
    # -------- 1) Normalize weights (with guards) --------
    try:
        w_sum = float(Probability.sum(weights))
    except Exception:
        w_sum = float(np.sum(weights))

    if not np.isfinite(w_sum) or w_sum <= eps:
        raise ValueError(f"Invalid weights sum: {w_sum}")

    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError(f"Invalid weights: {weights}")
    weights = weights / w_sum  # ensure sum(weights) == 1

    # -------- 2) Force numeric types --------
    # Keep StoneSoup view for output, but use plain float arrays for computation
    means = means.view(StateVectors)               # (d, K) StoneSoup type
    M = np.asarray(means, dtype=np.float64)        # (d, K) float
    C = np.asarray(covars, dtype=np.float64)       # (d, d, K) float

    # -------- 3) Weighted mean (moment matching) --------
    mean = np.average(M, axis=1, weights=weights).reshape(-1, 1)  # (d, 1)

    # -------- 4) Weighted covariance --------
    #   Cov = Σ_k w_k * (C_k + (m_k - μ)(m_k - μ)ᵀ)
    delta_means = M - mean              # (d, K)
    # Σ_k w_k * C_k
    covar = np.sum(C * weights[None, None, :], axis=2, dtype=np.float64)
    # Σ_k w_k * (m_k - μ)(m_k - μ)ᵀ
    covar += (weights * delta_means) @ delta_means.T

    # -------- 5) Symmetrize + SPD guard --------
    covar = 0.5 * (covar + covar.T)  # remove floating-point asymmetry

    if not np.all(np.isfinite(covar)):
        raise ValueError("Covariance has non-finite entries (NaN/Inf).")

    if spd_fix:
        I = np.eye(covar.shape[0])
        # Try small jitter increments first (cheap fix for near-SDP matrices)
        for k in range(6):
            try:
                np.linalg.cholesky(covar)
                break
            except np.linalg.LinAlgError:
                covar = covar + (10**k) * jitter * I
                covar = 0.5 * (covar + covar.T)
        else:
            # Still failing: project to SPD via eigenvalue clipping
            w, V = np.linalg.eigh(covar)
            w = np.clip(w, eps, None)
            covar = (V * w) @ V.T
            covar = 0.5 * (covar + covar.T)
            covar += eps * I

    return mean.view(StateVector), covar.view(CovarianceMatrix)


class JPDATracker:
    """
    High-level JPDA-based tracker wrapper.

    This class glues together:
      - an updater (e.g., Kalman/Gaussian updater),
      - a track initiator,
      - a JPDA data associator (e.g., JPDAwithNBest),
      - a track deleter,
      - and track state post-processing (Gaussian mixture reduction).
    """

    def __init__(
        self,
        updater,
        initiator,
        data_associator,
        deleter,
        **kwargs,
    ):
        self.updater = updater
        self.initiator = initiator
        self.data_associator = data_associator
        self.deleter = deleter

        self._track_id_counter = 0
        self.tracks = set()

    def _wrap_tracks_with_id(self, tracks):
        """
        Assign incremental integer IDs to newly created tracks.
        """
        for track in tracks:
            track.track_id = self._track_id_counter
            self._track_id_counter += 1
        return tracks

    def estimate(self, measurements, timestamp, warp_matrix=None):

        mapping = self.updater.measurement_model.mapping

        tracks = self.update(measurements, timestamp)

        estimations = []
        for track in tracks:
            estimations.append(
                {
                    "x": track.state.mean[mapping, :],
                    "track_id": track.track_id,
                }
            )

        return estimations

    def update(self, measurements, timestamp):

        hypotheses = self.data_associator.associate(
            self.tracks, measurements, timestamp
        )
        associated_measurements = set()

        # 2) For each track, update and perform GM reduction
        for track in self.tracks:
            track_hypotheses = hypotheses[track]

            posterior_states = []
            posterior_state_weights = []
            any_update = False

            for hypothesis in track_hypotheses:
                if not hypothesis:
                    # Missed detection update
                    state_post = hypothesis.prediction
                    posterior_states.append(state_post)
                else:
                    # Standard measurement update
                    posterior_state = self.updater.update(
                        hypothesis,
                    )
                    any_update = True
                    posterior_states.append(posterior_state)
                    associated_measurements.add(hypothesis.measurement)

                posterior_state_weights.append(hypothesis.probability)

            # 3) Append state to track: either single state or GM reduction
            if not any_update:
                # No detection associated: only one posterior state is expected
                assert len(posterior_states) == 1, (
                    "If no update occurs, exactly one posterior state is expected."
                )
                track.append(posterior_states[0])
            else:
                # Mixture reduction to a single Gaussian
                means = StateVectors(
                    [state.state_vector for state in posterior_states]
                )
                covars = np.stack([state.covar for state in posterior_states], axis=2)
                weights = np.asarray(posterior_state_weights)

                post_mean, post_covar = gm_reduce_single(means, covars, weights)

                track.append(
                    GaussianStateUpdate(
                        state_vector=post_mean,
                        covar=post_covar,
                        hypothesis=track_hypotheses,
                        timestamp=track_hypotheses[0].measurement.timestamp,
                    )
                )


        self.tracks -= self.deleter.delete_tracks(self.tracks)

        unassoc = measurements - associated_measurements
        new_tracks = self.initiator.initiate(unassoc, timestamp)
        new_tracks = self._wrap_tracks_with_id(new_tracks)
        self.tracks |= new_tracks

        return self.tracks
