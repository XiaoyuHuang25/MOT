# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from typing import Dict, List, Tuple, Set

import numpy as np
from scipy.optimize import linear_sum_assignment
from stonesoup.base import Property
from stonesoup.dataassociator import DataAssociator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment as BaseGNNWith2DAssignment
from stonesoup.types.hypothesis import SingleHypothesis, ProbabilityHypothesis

from trackers.utils.hypothesiser import MultipleHypothesis
from murty import Murty  # noqa


class GNNWith2DAssignment(BaseGNNWith2DAssignment):
    """Standard GNN with 2D assignment, copied from StoneSoup and kept for reference.

    This class performs single-stage 2D assignment (Hungarian algorithm) between
    tracks and detections based on hypothesis distances or probabilities.
    """

    def associate(self, tracks, detections, timestamp, **kwargs):
        """Associate a set of detections with predicted states.

        Parameters
        ----------
        tracks : set of :class:`Track`
            Current tracked objects.
        detections : set of :class:`Detection`
            Retrieved measurements.
        timestamp : datetime.datetime
            Detection time to predict to.

        Returns
        -------
        dict
            Mapping from track to chosen :class:`SingleHypothesis`.
        """
        # Generate hypotheses per track–detection pair
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        associations: Dict = {}

        # Tracks that have at least one non-empty hypothesis
        detected_tracks = [
            track
            for track, track_hyps in hypotheses.items()
            if any(track_hyps)
        ]

        # For tracks with only a missed-detection hypothesis,
        # store that hypothesis directly
        for track in hypotheses.keys() - set(detected_tracks):
            if hypotheses[track]:
                associations[track] = hypotheses[track][0]

        # Nothing to associate if all tracks are missed
        if not detected_tracks:
            return associations

        detections = list(detections)

        # Build hypothesis matrix: rows = tracks, columns = detections + one dummy per track
        hypothesis_matrix = np.empty(
            (len(detected_tracks), len(detections) + len(detected_tracks)),
            dtype=object,
        )
        for i, track in enumerate(detected_tracks):
            row = np.empty(hypothesis_matrix.shape[1], dtype=object)
            for hyp in hypotheses[track]:
                if not hyp:
                    # Missed detection -> track-specific dummy column
                    row[len(detections) + i] = hyp
                else:
                    row[detections.index(hyp.measurement)] = hyp
            hypothesis_matrix[i] = row

        # Decide whether we are using probability or distance hypotheses
        hypothesis_types = {
            isinstance(hyp, ProbabilityHypothesis)
            for row in hypothesis_matrix
            for hyp in row
            if hyp is not None
        }
        if len(hypothesis_types) > 1:
            raise RuntimeError("2D assignment does not support mixed hypothesis types.")
        probability_flag = hypothesis_types.pop()

        # Build cost / distance matrix
        distance_matrix = np.empty(hypothesis_matrix.shape, dtype=float)
        for x in range(hypothesis_matrix.shape[0]):
            for y in range(hypothesis_matrix.shape[1]):
                hyp = hypothesis_matrix[x][y]
                if hyp is None:
                    distance_matrix[x, y] = np.inf if not probability_flag else -np.inf
                else:
                    if probability_flag:
                        # Larger log-probability is better
                        distance_matrix[x, y] = hyp.probability.log_value
                    else:
                        distance_matrix[x, y] = hyp.distance

        # Hungarian algorithm
        try:
            row4col, col4row = linear_sum_assignment(distance_matrix, probability_flag)
        except ValueError as exc:
            raise RuntimeError("Assignment was not feasible") from exc

        # Convert assignment back to per-track association
        for j, track in enumerate(detected_tracks):
            associations[track] = hypothesis_matrix[j][col4row[j]]

        return associations


class NStateAssignment(DataAssociator):
    """Multi-stage association wrapper.

    Runs the inner associator multiple times with gradually relaxed
    detection-score thresholds, so high-confidence pairs are formed first.
    """

    high_score_thresh: float = Property(
        doc="High confidence score threshold for association."
    )
    low_score_thresh: float = Property(
        doc="Low confidence score threshold for association."
    )
    n_stages_assignment: int = Property(
        doc="Number of stages for multi-stage association."
    )
    associator: DataAssociator = Property(
        doc="Base data associator to use at each stage."
    )

    def associate(self, tracks, detections, timestamp, **kwargs):
        assert self.high_score_thresh >= self.low_score_thresh, (
            f"high_score_thresh {self.high_score_thresh} must be "
            f">= low_score_thresh {self.low_score_thresh}"
        )

        # Score thresholds from high -> low
        edges = np.linspace(
            self.high_score_thresh,
            self.low_score_thresh,
            num=self.n_stages_assignment,
        )

        last_detections: Set = set()
        matched_detections: Set = set()
        matched_tracks: Set = set()
        all_hypotheses: Dict = {}

        for stage_idx in range(self.n_stages_assignment):
            # Choose detections for this stage based on score
            if stage_idx == 0:
                current_dets = {
                    det for det in detections if det.score > edges[stage_idx]
                }
            else:
                current_dets = {
                    det
                    for det in detections
                    if edges[stage_idx] < det.score <= edges[stage_idx - 1]
                }

            # Only consider tracks that have not yet been matched
            unmatched_tracks = tracks - matched_tracks
            last_detections.update(current_dets)
            unmatched_detections = last_detections - matched_detections

            # Run underlying associator on this subset
            associations_idx = self.associator.associate(
                unmatched_tracks, unmatched_detections, timestamp, **kwargs
            )

            matched_tracks_idx: Set = set()
            matched_dets_idx: Set = set()

            # Extract which tracks / detections were actually matched
            for t, h in associations_idx.items():
                if isinstance(h, SingleHypothesis):
                    if h.measurement:
                        matched_tracks_idx.add(t)
                        matched_dets_idx.add(h.measurement)
                elif isinstance(h, MultipleHypothesis):
                    if any(sub_h.measurement for sub_h in h):
                        matched_tracks_idx.add(t)
                        matched_dets_idx.add(
                            next(sub_h.measurement for sub_h in h if sub_h.measurement)
                        )
                else:
                    raise RuntimeError(f"Unsupported hypothesis type: {type(h)}")

            matched_detections.update(matched_dets_idx)
            matched_tracks.update(matched_tracks_idx)

            # For intermediate stages keep only matched tracks;
            # in the last stage keep everything (including unmatched)
            if stage_idx == self.n_stages_assignment - 1:
                all_hypotheses.update(associations_idx)
            else:
                all_hypotheses.update({t: associations_idx[t] for t in matched_tracks_idx})

        return all_hypotheses


class VisualGNNWith2DAssignment(GNNWith2DAssignment):
    """GNN with multi-stage 2D assignment and visual scores.

    This extends the standard GNNWith2DAssignment to:
    - use detection scores for multi-stage assignment
    - keep the usual GNN cost-based Hungarian assignment inside each stage
    """

    high_score_thresh: float = Property(
        doc="High confidence score threshold for association."
    )
    low_score_thresh: float = Property(
        doc="Low confidence score threshold for association."
    )
    n_stages_assignment: int = Property(
        doc="Number of stages for multi-stage assignment."
    )

    def associate(self, tracks, detections, timestamp, **kwargs):
        """Associate detections and tracks using multi-stage assignment.

        Steps
        -----
        1. Build hypothesis matrix (track × (detections + dummy)).
        2. Compute distance matrix and score matrix.
        3. Run multi-stage LSAP:
           - first on high-score detections,
           - then gradually include lower-score detections,
           - finally assign remaining tracks to dummy columns.
        """
        # Generate hypotheses per track–detection pair
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        associations: Dict = {}

        # Tracks that have at least one non-empty hypothesis
        detected_tracks = [
            track
            for track, track_hyps in hypotheses.items()
            if any(track_hyps)
        ]

        # For tracks with only a missed-detection hypothesis,
        # store that hypothesis directly
        for track in hypotheses.keys() - set(detected_tracks):
            if hypotheses[track]:
                associations[track] = hypotheses[track][0]

        # Nothing to associate if all tracks are missed
        if not detected_tracks:
            return associations

        detections = list(detections)

        # Hypothesis matrix: rows = tracks, columns = detections + dummy per track
        hypothesis_matrix = np.empty(
            (len(detected_tracks), len(detections) + len(detected_tracks)),
            dtype=object,
        )
        for i, track in enumerate(detected_tracks):
            row = np.empty(hypothesis_matrix.shape[1], dtype=object)
            for hyp in hypotheses[track]:
                if not hyp:
                    row[len(detections) + i] = hyp
                else:
                    row[detections.index(hyp.measurement)] = hyp
            hypothesis_matrix[i] = row

        # Decide type of hypothesis (probability vs distance)
        hypothesis_types = {
            isinstance(hyp, ProbabilityHypothesis)
            for row in hypothesis_matrix
            for hyp in row
            if hyp is not None
        }
        if len(hypothesis_types) > 1:
            raise RuntimeError("2D assignment does not support mixed hypothesis types.")
        probability_flag = hypothesis_types.pop()

        # Distance matrix and measurement-score matrix
        distance_matrix = np.empty(hypothesis_matrix.shape, dtype=float)
        score_matrix = np.zeros((len(detected_tracks), len(detections)), dtype=float)

        for x in range(hypothesis_matrix.shape[0]):
            for y in range(hypothesis_matrix.shape[1]):
                hyp = hypothesis_matrix[x][y]
                if hyp is None:
                    distance_matrix[x, y] = np.inf
                    if y < len(detections):
                        score_matrix[x, y] = 0.0
                else:
                    if probability_flag:
                        # Minimisation: use negative log-probability as cost
                        distance_matrix[x, y] = -hyp.probability.log_value
                    else:
                        distance_matrix[x, y] = hyp.distance
                    if y < len(detections):
                        score_matrix[x, y] = hyp.measurement.score

        # Multi-stage LSAP: returns per-row matched column indices
        row_ids, col_ids = self.multistage_lsap_rowcol(
            distance_matrix,
            score_matrix,
            max_score=self.high_score_thresh,
            min_score=self.low_score_thresh,
            n_stages=self.n_stages_assignment,
        )

        # Convert assignment back to per-track association
        for j, track in enumerate(detected_tracks):
            associations[track] = hypothesis_matrix[j][col_ids[j]]

        return associations

    @staticmethod
    def multistage_lsap_rowcol(
        distance_matrix: np.ndarray,
        score_matrix: np.ndarray,
        min_score: float,
        max_score: float,
        n_stages: int,
        big: float = 1e12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-stage LSAP on a (track × (detections + dummy)) cost matrix.

        Parameters
        ----------
        distance_matrix : ndarray, shape (T, M + T)
            Cost matrix where:
            - first M columns correspond to real detections,
            - last T columns are dummy (missed-detection) columns,
              each dummy column j = M + i belongs to row i only.
        score_matrix : ndarray, shape (T, M)
            Per track–detection score (e.g. detector confidence).
        min_score : float
            Minimum score threshold.
        max_score : float
            Maximum score threshold.
        n_stages : int
            Number of stages for score-based gating.
        big : float, optional
            Large constant used to disallow certain assignments.

        Returns
        -------
        matched_row_ids : ndarray, shape (T,)
            Row indices (0..T-1) in ascending order.
        matched_col_ids : ndarray, shape (T,)
            Matched column index for each row (in 0..M+T-1),
            including dummy assignments for missed detections.
        """
        D = np.array(distance_matrix, copy=True)
        S = np.array(score_matrix, copy=True)
        T, MT = D.shape
        assert S.shape[0] == T
        M = S.shape[1]
        assert MT == M + T, "distance_matrix must be of shape (T, M + T)"
        assert n_stages >= 1 and max_score >= min_score

        row_ids = np.arange(T)
        col_ids = np.arange(M + T)

        # Replace +/-inf with a large finite number to keep LSAP feasible
        D = np.where(np.isfinite(D), D, big)

        # row2col_idx[i] = chosen column index (0..M+T-1) for row i
        row2col_idx = np.full(T, -1, dtype=int)
        used_rows: Set[int] = set()
        used_real_cols: Set[int] = set()  # only real measurements 0..M-1

        def run_stage_real(valid_mask: np.ndarray):
            """Run LSAP for real measurement columns that are valid in this stage.

            Only accepts assignments to real columns; dummy assignments are
            left to the final consolidation step.
            """
            if not np.any(valid_mask):
                return

            cost = D.copy()

            # Block already used rows and columns
            if used_rows:
                cost[list(used_rows), :] = big
            if used_real_cols:
                cost[:, list(used_real_cols)] = big

            # Only allow real columns that are valid for this stage
            cost[:, :M] = np.where(valid_mask, cost[:, :M], big)

            # Safety: if a row is completely blocked, allow its own dummy column
            rows_all_big = np.all(cost >= big, axis=1)
            if rows_all_big.any():
                for i in np.where(rows_all_big)[0]:
                    cost[i, M + i] = D[i, M + i]

            r, c = linear_sum_assignment(cost)

            for i, j in zip(r, c):
                # Accept only real-column matches
                if j < M and (i not in used_rows) and (j not in used_real_cols):
                    if valid_mask[i, j] and cost[i, j] < big / 2:
                        row2col_idx[i] = j
                        used_rows.add(i)
                        used_real_cols.add(j)

        # Stage 0: S >= max_score
        run_stage_real(S >= max_score)

        # Stages 1..N-1: (low, high] intervals from max_score down to min_score
        if n_stages > 1:
            edges = np.linspace(max_score, min_score, num=n_stages)
            for k in range(1, n_stages):
                hi, lo = edges[k - 1], edges[k]
                valid_mask = (S >= lo) & (S < hi)
                run_stage_real(valid_mask)

        # Final dummy stage: assign remaining rows to their own dummy column
        left_rows = np.where(row2col_idx == -1)[0]
        if left_rows.size:
            # Subproblem on left_rows × left_rows,
            # each row i can only take its own dummy M+i
            cost_dummy = np.full((left_rows.size, left_rows.size), big)
            for rr, i in enumerate(left_rows):
                cost_dummy[rr, rr] = D[i, M + i]
            r_loc, c_loc = linear_sum_assignment(cost_dummy)
            for rr, cc in zip(r_loc, c_loc):
                i = left_rows[rr]
                j_global = M + i
                row2col_idx[i] = j_global

        matched_row_ids = row_ids.copy()
        matched_col_ids = col_ids[row2col_idx]

        return matched_row_ids, matched_col_ids


class MurtySolver(DataAssociator):
    """Wrapper around a Murty implementation for N-best data association."""

    def associate(self, tracks, detections, timestamp, **kwargs):
        """Associate a set of tracks and detections using Murty's algorithm."""
        assoc_num = kwargs["assoc_num"]
        all_tracks_hypotheses = kwargs["hypotheses"]
        hypotheses = {track: all_tracks_hypotheses[track] for track in tracks}

        associations_list: List[Dict] = []
        detected_tracks = list(tracks)
        detections = list(detections)

        # Build hypothesis matrix
        hypothesis_matrix = np.empty(
            (len(detected_tracks), len(detections) + len(detected_tracks)),
            dtype=object,
        )
        for i, track in enumerate(detected_tracks):
            row = np.empty(hypothesis_matrix.shape[1], dtype=object)
            for hyp in hypotheses[track]:
                if not hyp:
                    row[len(detections) + i] = hyp
                else:
                    if hyp.measurement in detections:
                        row[detections.index(hyp.measurement)] = hyp
            hypothesis_matrix[i] = row

        hypothesis_types = {
            isinstance(hyp, ProbabilityHypothesis)
            for row in hypothesis_matrix
            for hyp in row
            if hyp is not None
        }
        if len(hypothesis_types) > 1:
            raise RuntimeError("2D assignment does not support mixed hypothesis types.")
        probability_flag = hypothesis_types.pop()
        assert probability_flag, "MurtySolver currently assumes probability-based hypotheses."

        distance_matrix = np.empty(hypothesis_matrix.shape, dtype=float)
        for x in range(hypothesis_matrix.shape[0]):
            for y in range(hypothesis_matrix.shape[1]):
                hyp = hypothesis_matrix[x][y]
                if hyp is None:
                    distance_matrix[x, y] = np.inf
                else:
                    distance_matrix[x, y] = -hyp.probability.log_value

        cost_list, col_list = MurtySolver.data_association(
            distance_matrix, assoc_num=assoc_num
        )

        for col_ind in col_list:
            associations = {}
            for row_idx, col_idx in enumerate(col_ind):
                associations[detected_tracks[row_idx]] = hypothesis_matrix[row_idx][col_idx]
            associations_list.append(associations)

        return cost_list, associations_list

    @staticmethod
    def murty_associate(tracks_list, assoc_num):
        """Murty association for a list of track-like objects with single_hypotheses."""
        detections = set()
        for track in tracks_list:
            for hyp in track.single_hypotheses:
                if hyp:
                    detections.add(hyp.measurement)
        detections = list(detections)

        hypothesis_matrix = np.empty(
            (len(tracks_list), len(detections) + len(tracks_list)),
            dtype=object,
        )
        for i, track in enumerate(tracks_list):
            row = np.empty(hypothesis_matrix.shape[1], dtype=object)
            for hyp in track.single_hypotheses:
                if not hyp:
                    row[len(detections) + i] = hyp
                else:
                    row[detections.index(hyp.measurement)] = hyp
            hypothesis_matrix[i] = row

        distance_matrix = np.empty(hypothesis_matrix.shape, dtype=float)
        for x in range(hypothesis_matrix.shape[0]):
            for y in range(hypothesis_matrix.shape[1]):
                hyp = hypothesis_matrix[x][y]
                if hyp is None:
                    distance_matrix[x, y] = np.inf
                else:
                    distance_matrix[x, y] = -hyp.probability.log_value

        cost_list, col_list = MurtySolver.data_association(
            distance_matrix, assoc_num=assoc_num
        )

        global_hypotheses_indices: List[List[int]] = []
        for col_ind in col_list:
            local_indices: List[int] = []
            for row_idx, col_idx in enumerate(col_ind):
                local_indices.append(
                    tracks_list[row_idx].single_hypotheses.index(
                        hypothesis_matrix[row_idx][col_idx]
                    )
                )
            global_hypotheses_indices.append(local_indices)

        return cost_list, global_hypotheses_indices

    @staticmethod
    def draw_with_timeout(murty_solver: Murty, timeout: float = 1.0):
        """Run ``murty_solver.draw()`` in a thread and enforce a timeout.

        Parameters
        ----------
        murty_solver : Murty
            Murty solver instance.
        timeout : float, optional
            Timeout in seconds.

        Returns
        -------
        status : bool
        cost_iter : float
        col_iter : ndarray

        Raises
        ------
        MurtySolverTimeoutError
            If the draw operation does not finish within the timeout.
        """
        result: List = []

        def run():
            status, cost_iter, col_iter = murty_solver.draw()
            result.extend([status, cost_iter, col_iter])

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Force-stop the thread as a last resort.
            # WARNING: this is generally unsafe in Python and should be used
            # only as a debugging / hard timeout mechanism.
            thread._stop()  # noqa
            raise MurtySolverTimeoutError(
                f"Murty solver timed out after {timeout} seconds. "
                f"Current cost_matrix: {murty_solver.cost_matrix}"
            )

        return result[0], result[1], result[2]

    @staticmethod
    def data_association(cost_matrix: np.ndarray, assoc_num: int, timeout: float = 1.0):
        """Run N-best data association using Murty's algorithm.

        Parameters
        ----------
        cost_matrix : ndarray
            Cost matrix (rows = tracks, columns = detections + dummy).
        assoc_num : int
            Number of N-best hypotheses to extract.
        timeout : float, optional
            Per-draw timeout in seconds.

        Returns
        -------
        cost : list of float
            List of costs for each N-best solution.
        col_ind : list of ndarray
            Column indices for each N-best solution.

        Raises
        ------
        AssertionError
            If no valid association is found.
        """
        INF = 1e6
        cost_matrix = np.where(cost_matrix == np.inf, INF, cost_matrix)

        murty_solver = Murty(cost_matrix)
        status = True
        costs: List[float] = []
        col_inds: List[np.ndarray] = []
        num = 0

        while status and num < assoc_num:
            try:
                status, cost_iter, col_iter = MurtySolver.draw_with_timeout(
                    murty_solver, timeout
                )
            except MurtySolverTimeoutError as exc:
                # Log and stop if we hit the timeout
                print(exc)
                break

            # If cost is too large, treat as invalid and stop
            if cost_iter > 0.5 * INF:
                break

            if status:
                costs.append(cost_iter)
                col_inds.append(col_iter)
                num += 1

        assert len(costs) > 0, (
            f"Invalid cost matrix {cost_matrix} or no valid association found."
        )
        return costs, col_inds


class MurtySolverTimeoutError(Exception):
    """Custom timeout error for Murty solver."""
    pass
