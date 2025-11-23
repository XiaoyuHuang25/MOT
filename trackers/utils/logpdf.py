# -*- coding: utf-8 -*-

from functools import lru_cache
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal


class FastLogPDFWithCovCache:
    """Numerically robust multivariate Gaussian log-pdf with covariance caching.

    This helper class provides:
    - A robust SPD sanitiser for covariance matrices.
    - Cached Cholesky and log-determinant (via lru_cache) to speed up repeated
      evaluations with the same covariance.
    - A multi-stage fallback strategy:
        1) Cached Cholesky + cho_solve (fast, stable when SPD).
        2) SciPy's multivariate_normal.logpdf.
        3) Pseudo-inverse + slogdet.

    All methods are static; the class is intended as a pure utility.
    """

    # ------------------------------------------------------------------ #
    # SPD sanitisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _spd_sanitize(cov: np.ndarray,
                      max_tries: int = 6,
                      base_eps: float = 1e-8) -> np.ndarray:
        """Ensure the covariance matrix is symmetric positive-definite (SPD).

        This function:
        1) Casts to float64 and copies.
        2) Replaces NaN/Inf with finite values.
        3) Symmetrises cov = 0.5 * (cov + cov.T).
        4) Gradually adds diagonal jitter until Cholesky succeeds,
           or raises a LinAlgError if it cannot be fixed.

        Parameters
        ----------
        cov : np.ndarray
            Input covariance matrix.
        max_tries : int, optional
            Maximum number of jitter attempts (default 6).
        base_eps : float, optional
            Initial jitter magnitude, multiplied by 10 each retry.

        Returns
        -------
        np.ndarray
            Sanitised SPD covariance matrix.

        Raises
        ------
        ValueError
            If cov still contains NaN/Inf after replacement.
        np.linalg.LinAlgError
            If SPD repair fails after max_tries attempts.
        """
        cov = np.array(cov, dtype=np.float64, copy=True)

        # Replace NaN/Inf with finite values
        if not np.isfinite(cov).all():
            warnings.warn(
                "Covariance matrix contains NaN/Inf; replacing with finite values.",
                RuntimeWarning,
            )
            cov = np.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)

        # Symmetrise
        cov = 0.5 * (cov + cov.T)

        if not np.isfinite(cov).all():
            raise ValueError("Covariance matrix contains NaN/Inf after sanitisation.")

        eps = base_eps
        eye = np.eye(cov.shape[0], dtype=np.float64)

        # Try progressively larger diagonal jitter
        for _ in range(max_tries):
            try:
                np.linalg.cholesky(cov)
                return cov
            except np.linalg.LinAlgError:
                cov = cov + eye * eps
                eps *= 10.0

        # Final attempt; if this fails, propagate the error
        np.linalg.cholesky(cov)
        return cov  # If no exception, treat as SPD

    # ------------------------------------------------------------------ #
    # Cached Cholesky + log|Σ|
    # ------------------------------------------------------------------ #
    @staticmethod
    @lru_cache()
    def _cached_cholesky_and_logdet(cov_tuple):
        """Cache Cholesky factorisation and log-determinant for a covariance.

        Parameters
        ----------
        cov_tuple : tuple
            Flattened covariance matrix stored as a tuple of floats.

        Returns
        -------
        cho : np.ndarray
            Cholesky factor.
        lower : bool
            Flag indicating whether `cho` is lower-triangular (SciPy convention).
        log_det : float
            Log-determinant of the covariance.
        """
        # Recover squared dimension from flattened tuple length
        arr = np.asarray(cov_tuple, dtype=np.float64)
        n = int(round(np.sqrt(arr.size)))
        if n * n != arr.size:
            raise ValueError(
                f"Invalid covariance tuple length {arr.size}; "
                f"cannot reshape into a square matrix."
            )

        cov = arr.reshape((n, n))
        cov = FastLogPDFWithCovCache._spd_sanitize(cov)

        cho, lower = cho_factor(cov, check_finite=True)
        log_det = 2.0 * np.sum(np.log(np.diag(cho)))  # log|Σ|
        return cho, lower, log_det

    # ------------------------------------------------------------------ #
    # Main Cholesky-based implementation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logpdf_cholesky_impl(x, mean, cov) -> float:
        """Core log-pdf computation using cached Cholesky + cho_solve.

        Parameters
        ----------
        x : array_like
            Observation vector.
        mean : array_like
            Mean vector.
        cov : np.ndarray
            Covariance matrix.

        Returns
        -------
        float
            Log probability density value.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        mean = np.asarray(mean, dtype=np.float64).ravel()
        diff = x - mean

        # Key for the Cholesky cache
        cov_tuple = tuple(np.asarray(cov, dtype=np.float64).ravel())

        cho, lower, log_det = FastLogPDFWithCovCache._cached_cholesky_and_logdet(cov_tuple)

        # Solve Σ^{-1} * diff via Cholesky factors
        solve = cho_solve((cho, lower), diff, check_finite=True)
        maha = float(diff.dot(solve))

        d = diff.size
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)

    # ------------------------------------------------------------------ #
    # Fallback 1: SciPy multivariate_normal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logpdf_fallback_mvn(x, mean, cov) -> float:
        """Fallback using SciPy's multivariate_normal.logpdf.

        SciPy will perform its own internal checks and stabilisation steps.

        Parameters
        ----------
        x : array_like
        mean : array_like
        cov : np.ndarray

        Returns
        -------
        float
            Log probability density value.
        """
        cov = FastLogPDFWithCovCache._spd_sanitize(cov)
        return float(multivariate_normal.logpdf(
            np.asarray(x, dtype=np.float64).ravel(),
            mean=np.asarray(mean, dtype=np.float64).ravel(),
            cov=cov,
        ))

    # ------------------------------------------------------------------ #
    # Fallback 2: Pseudo-inverse + slogdet
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logpdf_fallback_pinv(x, mean, cov) -> float:
        """Fallback using pseudo-inverse and slogdet.

        This is extremely robust, but statistical interpretation can be
        weaker when `cov` is near-singular.

        Parameters
        ----------
        x : array_like
        mean : array_like
        cov : np.ndarray

        Returns
        -------
        float
            Log probability density value.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        mean = np.asarray(mean, dtype=np.float64).ravel()
        diff = x - mean

        cov = np.asarray(cov, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)  # enforce symmetry

        cov_pinv = np.linalg.pinv(cov)
        maha = float(diff.dot(cov_pinv).dot(diff))

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # Add a small jitter if slogdet indicates non-SPD
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            sign, logdet = np.linalg.slogdet(cov)

        d = diff.size
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)

    # ------------------------------------------------------------------ #
    # Public API: robust log-pdf
    # ------------------------------------------------------------------ #
    @staticmethod
    def safe_logpdf(x, mean, cov) -> float:
        """Compute a numerically robust multivariate Gaussian log-pdf.

        Strategy:
        1) Try cached Cholesky + cho_solve (fast, standard).
        2) If that fails, fall back to SciPy's multivariate_normal.logpdf.
        3) If that also fails, use a pseudo-inverse-based implementation.

        Parameters
        ----------
        x : array_like
            Observation vector.
        mean : array_like
            Mean vector.
        cov : np.ndarray
            Covariance matrix.

        Returns
        -------
        float
            Log probability density value.
        """
        # Primary: cached Cholesky
        try:
            return FastLogPDFWithCovCache._logpdf_cholesky_impl(x, mean, cov)
        except Exception as e:
            warnings.warn(
                f"Cholesky-based logpdf failed: {e}. Falling back to SciPy mvn.",
                RuntimeWarning,
            )

        # Fallback 1: SciPy multivariate_normal
        try:
            return FastLogPDFWithCovCache._logpdf_fallback_mvn(x, mean, cov)
        except Exception as e:
            warnings.warn(
                f"SciPy multivariate_normal.logpdf failed: {e}. "
                f"Falling back to pseudo-inverse method.",
                RuntimeWarning,
            )

        # Fallback 2: pseudo-inverse
        return FastLogPDFWithCovCache._logpdf_fallback_pinv(x, mean, cov)
