# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Set

import numpy as np
from stonesoup.base import Property
from stonesoup.dataassociator import DataAssociator
from stonesoup.types.hypothesis import SingleHypothesis

from trackers.utils.dataassociator import MurtySolver
from trackers.utils.hypothesiser import MultipleHypothesis
from trackers.mht_tracker.mht_tracker import GlobalHypothesis, Association, AssociationID
from trackers.mht_tracker.hypothesiser import MHTHypothesiser


class MHTDataAssociator(DataAssociator):
    """
    Multi-Hypothesis Tracking (MHT) data associator using Murty's algorithm.

    This class:
      - Holds a hypothesiser that generates single-target hypotheses for
        each component–detection pair.
      - Uses a Murty-based associator to enumerate N-best global hypotheses.
      - Updates the list of global hypotheses in-place.
    """

    hypothesiser: MHTHypothesiser = Property(
        doc="Generate a set of hypotheses for each prediction–detection pair."
    )

    top_n_hypotheses: int = Property(
        doc="Maximum number of global hypotheses to keep from the Murty output."
    )

    associator: MurtySolver = Property(
        doc="Murty-based data associator used for component-level association."
    )

    # --------------------------------------------------------------------- #
    # Core association logic
    # --------------------------------------------------------------------- #
    def associate(self, tracks, detections, timestamp, **kwargs):
        """
        Run Murty-based association on the components of each global hypothesis.

        Parameters
        ----------
        tracks : Iterable[Track]
            Current active tracks.
        detections : Iterable[Detection]
            Detections at the current time step.
        timestamp : datetime-like
            Current timestamp.
        global_hypotheses : list[GlobalHypothesis] (in kwargs)
            Existing global hypotheses to be expanded.
        track_components_hypotheses : dict (optional, in kwargs)
            Optional cache of per-track component hypotheses.

        Returns
        -------
        dict
            Mapping from track -> MultipleHypothesis of component-level hypotheses.
        """
        global_hypotheses: List[GlobalHypothesis] = kwargs["global_hypotheses"]

        if not tracks:
            return {}

        new_global_hypotheses: List[GlobalHypothesis] = []

        # Retrieve or build component-level hypotheses for each track
        if "track_components_hypotheses" not in kwargs:
            track_components_hypotheses: Dict = {}
            # key: track, value: per-component hypotheses
            for track in tracks:
                components = track.state.components
                track_components_hypotheses[track] = self.associator.generate_hypotheses(
                    components, detections, timestamp
                )
        else:
            track_components_hypotheses = kwargs["track_components_hypotheses"]

        # Expand each existing global hypothesis
        for global_hypothesis in global_hypotheses:
            log_weight = global_hypothesis.log_weight

            # Map each component in this hypothesis back to its track
            component_to_track = {}
            components_set: Set = set()
            components_hypotheses: Dict = {}

            for track_id, component_idx in global_hypothesis.associations:
                for track in tracks:
                    if track.track_id == track_id:
                        component = track.state.components[component_idx]
                        component_to_track[component] = track
                        components_set.add(component)
                        components_hypotheses[component] = \
                            track_components_hypotheses[track][component]
                        break

            # Number of Murty solutions for this branch,
            # proportional to the current global hypothesis weight
            assoc_num = int(np.ceil(np.exp(log_weight) * self.top_n_hypotheses))

            cost_list, components_hypothesis_list = self.associator.associate(
                components_set,
                detections,
                timestamp,
                assoc_num=assoc_num,
                hypotheses=components_hypotheses,
            )

            # For each Murty result, build a new global hypothesis
            for cost, components_hypothesis in zip(cost_list, components_hypothesis_list):
                new_global_log_weight = log_weight - cost
                associations_list: List[Association] = []

                for component, hypothesis in components_hypothesis.items():
                    track = component_to_track[component]
                    # Store track–hypothesis pair and use hypothesis probability as local weight
                    associations_list.append(
                        Association(
                            track,
                            hypothesis,
                            log_weight=hypothesis.probability.log_value,
                        )
                    )

                new_global_hypotheses.append(
                    GlobalHypothesis(
                        log_weight=new_global_log_weight,
                        associations=associations_list,
                    )
                )

        # Replace old global hypotheses with the expanded set
        global_hypotheses.clear()
        global_hypotheses.extend(new_global_hypotheses)

        return track_components_hypotheses

    # --------------------------------------------------------------------- #
    # Utilities on global hypotheses
    # --------------------------------------------------------------------- #
    def prune_hypotheses(self, global_hypotheses: List[GlobalHypothesis]):
        """
        Collect unique hypotheses per track and reindex associations.

        Parameters
        ----------
        global_hypotheses : list[GlobalHypothesis]
            Global hypotheses after expansion.

        Returns
        -------
        dict
            Mapping track -> MultipleHypothesis.
        """
        new_track_hypotheses: Dict = {}

        for global_hypothesis in global_hypotheses:
            new_associations: List[AssociationID] = []

            for assoc in global_hypothesis.associations:
                track, hypothesis = assoc

                if track not in new_track_hypotheses:
                    new_track_hypotheses[track] = []

                if hypothesis not in new_track_hypotheses[track]:
                    new_track_hypotheses[track].append(hypothesis)

                new_associations.append(
                    AssociationID(
                        track_id=track.track_id,
                        hypothesis_index=new_track_hypotheses[track].index(hypothesis),
                        log_weight=assoc.log_weight,
                    )
                )

            # Replace associations with track_id/hypothesis_index pairs
            global_hypothesis.associations = new_associations

        # Wrap hypothesis lists into MultipleHypothesis objects
        for track in new_track_hypotheses.keys():
            new_track_hypotheses[track] = MultipleHypothesis(new_track_hypotheses[track])

        return new_track_hypotheses

    def cap_global_hypotheses(self, new_global_hypotheses: List[GlobalHypothesis]):
        """
        Keep at most `top_n_hypotheses` global hypotheses (by log_weight).

        Parameters
        ----------
        new_global_hypotheses : list[GlobalHypothesis]

        Returns
        -------
        list[GlobalHypothesis]
            Normalized and capped list.
        """
        sorted_global_hypotheses = sorted(
            new_global_hypotheses,
            key=lambda x: x.log_weight,
            reverse=True,
        )
        sorted_global_hypotheses = sorted_global_hypotheses[: self.top_n_hypotheses]
        self.normalize_global_hypotheses_weights(sorted_global_hypotheses)
        return sorted_global_hypotheses

    @staticmethod
    def normalize_global_hypotheses_weights(global_hypotheses: List[GlobalHypothesis]):
        """
        Normalize log-weights of global hypotheses in-place (log-sum-exp).

        Parameters
        ----------
        global_hypotheses : list[GlobalHypothesis]
        """
        if not global_hypotheses:
            return

        log_values = np.array([hypo.log_weight for hypo in global_hypotheses])
        max_log_value = float(np.max(log_values))
        value_sum = np.sum(np.exp(log_values - max_log_value))
        log_sum = float(np.log(value_sum) + max_log_value)

        for hypothesis in global_hypotheses:
            hypothesis.log_weight -= log_sum
