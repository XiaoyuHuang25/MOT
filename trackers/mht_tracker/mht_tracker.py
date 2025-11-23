# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Iterable

import numpy as np
from stonesoup.deleter import Deleter
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianMixtureUpdate

from trackers.mht_tracker.state import GaussianMixturePrediction


# ----------------------------------------------------------------------
# Simple association representations
# ----------------------------------------------------------------------
@dataclass
class AssociationID:
    """Association by (track_id, hypothesis_index) with a log weight.

    This is used when we don't need to carry the full Track object or
    hypothesis instance, only an index into the component list.
    """
    track_id: int
    hypothesis_index: int
    log_weight: float

    def __iter__(self):
        # Allows unpacking: track_id, hypothesis_index = assoc
        return iter((self.track_id, self.hypothesis_index))

    def __repr__(self) -> str:
        return f"{self.track_id}:{self.hypothesis_index}"


@dataclass
class Association:
    """Association storing the full Track and SingleHypothesis object."""
    track: Track
    hypothesis: SingleHypothesis
    log_weight: float

    def __iter__(self):
        # Allows unpacking: track, hypothesis = assoc
        return iter((self.track, self.hypothesis))

    def __repr__(self) -> str:
        return f"{self.track.track_id}:{self.hypothesis}"


@dataclass
class GlobalHypothesis:
    """Global hypothesis over all tracks.

    For MHT, each global hypothesis is a particular combination of
    (track, component_index) or (track, hypothesis) assignments.
    """
    log_weight: float
    associations: List[AssociationID] | List[Association]


class MHTTracker:

    def __init__(
        self,
        updater,
        initiator,
        data_associator,
        deleter: Deleter,
        prob_detect,
        prob_gate,
        **kwargs,
    ):
        self.updater = updater
        self.initiator = initiator
        self.data_associator = data_associator
        self.deleter = deleter

        self._track_id_counter: int = 0
        self.tracks: Set[Track] = set()

        # List of current global hypotheses
        self.global_hypotheses: List[GlobalHypothesis] = []

    def _wrap_tracks_with_id(self, tracks):
        """
        Assign incremental integer IDs to newly created tracks.
        """
        for track in tracks:
            track.track_id = self._track_id_counter
            self._track_id_counter += 1
        return tracks

    def estimate(self, measurements, timestamp, warp_matrix):

        mapping = self.updater.measurement_model.mapping

        tracks = self.update(measurements, timestamp)

        estimations = []
        if self.global_hypotheses:
            most_probable = max(self.global_hypotheses, key=lambda x: x.log_weight)
            for track in tracks:
                for track_id, comp_idx in most_probable.associations:
                    if track_id == track.track_id:
                        component = track.state.components[comp_idx]
                        estimations.append(
                            {
                                "x": component.mean[mapping, :],
                                "track_id": track.track_id,
                            }
                        )

        return estimations

    def update(self, measurements, timestamp):

        # 1) Generate component-level hypotheses for each track
        track_components_hypotheses = {}
        for track in self.tracks:
            components = track.state.components
            track_components_hypotheses[track] = self.data_associator.generate_hypotheses(
                components, measurements, timestamp, mht_track=track
            )

        # 2) Associate via Murty using current global hypotheses
        self.data_associator.associate(
            self.tracks,
            measurements,
            timestamp,
            global_hypotheses=self.global_hypotheses,
            track_components_hypotheses=track_components_hypotheses,
        )

        # Cap global hypotheses to the top N and map them back to per-track hypotheses
        self.global_hypotheses = self.data_associator.cap_global_hypotheses(
            self.global_hypotheses
        )
        association = self.data_associator.prune_hypotheses(self.global_hypotheses)

        associated_measurements = set()

        # 3) For each track, update its VisualGaussianMixture state
        for track, hypotheses in association.items():
            components = []
            has_update = False

            for hypothesis in hypotheses:
                if not hypothesis:
                    # Missed detection
                    state_post = hypothesis.prediction
                    components.append(state_post)
                else:
                    update = self.updater.update(
                        hypothesis,
                    )
                    associated_measurements.add(hypothesis.measurement)
                    components.append(update)
                    has_update = True

            if has_update:
                track.append(
                    GaussianMixtureUpdate(
                        components=components, hypothesis=hypotheses
                    )
                )
            else:
                track.append(
                    GaussianMixturePrediction(components=components)
                )

        # 4) Delete dead tracks
        deleted_tracks = self.deleter.delete_tracks(self.tracks)
        self.tracks -= deleted_tracks

        # Remove deleted tracks from global hypotheses and adjust log weights
        for global_hypothesis in self.global_hypotheses:
            new_associations = []
            log_weight = global_hypothesis.log_weight

            for assoc in global_hypothesis.associations:
                track_id, comp_idx = assoc
                track_exists = any(t.track_id == track_id for t in self.tracks)
                if track_exists:
                    new_associations.append(assoc)
                else:
                    # Component belonging to a deleted track: remove its contribution
                    log_weight -= assoc.log_weight

            global_hypothesis.associations = new_associations
            global_hypothesis.log_weight = log_weight

        # Prune unused components inside each track
        for track in self.tracks:
            self.prune_track_components(track, self.global_hypotheses)

        # Optionally prune tracks that are no longer referenced
        self.prune_tracks()

        # 5) Initiate new tracks from unassociated high-score detections
        unassoc = measurements - associated_measurements
        new_tracks = self.initiator.initiate(unassoc, timestamp)
        new_tracks = self._wrap_tracks_with_id(new_tracks)

        # Extend each global hypothesis with the initial component index for new tracks
        new_track_associations = [
            AssociationID(track.track_id, 0, log_weight=0.0) for track in new_tracks
        ]

        if new_track_associations:
            if self.global_hypotheses:
                for global_hypothesis in self.global_hypotheses:
                    global_hypothesis.associations.extend(new_track_associations)
            else:
                self.global_hypotheses = [
                    GlobalHypothesis(
                        log_weight=0.0,
                        associations=new_track_associations,
                    )
                ]

        self.tracks |= set(new_tracks)

        # Keep a per-track “state_idx” in metadata for later inspection/visualisation
        for global_hypothesis in self.global_hypotheses:
            for track in self.tracks:
                for track_id, comp_idx in global_hypothesis.associations:
                    if track_id == track.track_id:
                        track.metadata["state_idx"] = comp_idx
                    for metadata in track.metadatas:
                        if "state_idx" not in metadata:
                            metadata["state_idx"] = 0

        return self.tracks

    # ------------------------------------------------------------------
    # Pruning utilities
    # ------------------------------------------------------------------
    def prune_track_components(self, track: Track, global_hypotheses: List[GlobalHypothesis]) -> None:
        """Remove components that are never referenced in any global hypothesis.

        Steps:
        1. Collect all component indices that are actually used for this track.
        2. Rebuild track.state.components to keep only those used indices,
           preserving the original component order.
        3. Update all global hypotheses to map old component indices to new ones.
        """

        # 1) Collect used component indices for this track
        used_indices_order: List[int] = []

        def push_used(idx: int | None) -> None:
            if idx is not None and idx not in used_indices_order:
                used_indices_order.append(idx)

        for gh in global_hypotheses:
            for assoc in gh.associations:
                tid, cidx = assoc
                if tid == track.track_id:
                    push_used(cidx)

        # If nothing refers to this track, drop all components
        if not used_indices_order:
            track.state.components = []
            return

        # 2) Compute mapping old_index -> new_index, preserving original order
        used_set = set(used_indices_order)
        ordered_old_indices = [
            i for i, _ in enumerate(track.state.components) if i in used_set
        ]
        old2new = {old: new for new, old in enumerate(ordered_old_indices)}

        # Rebuild component list
        track.state.components = [track.state.components[i] for i in ordered_old_indices]

        # 3) Remap component indices in all global hypotheses
        for gh in global_hypotheses:
            updated_list = []
            for assoc in gh.associations:
                tid, cidx = assoc
                if tid == track.track_id and cidx in old2new:
                    updated_list.append(
                        AssociationID(
                            tid,
                            old2new[cidx],
                            log_weight=assoc.log_weight,
                        )
                    )
                else:
                    updated_list.append(assoc)
            gh.associations = updated_list

    def prune_tracks(self) -> None:
        """Remove tracks that are no longer referenced in any global hypothesis."""
        if not self.global_hypotheses:
            self.tracks = set()
            return

        new_tracks: Set[Track] = set()
        active_ids = {
            track_id
            for gh in self.global_hypotheses
            for track_id, _ in gh.associations
        }

        for track in self.tracks:
            if track.track_id in active_ids:
                new_tracks.add(track)

        self.tracks = new_tracks
