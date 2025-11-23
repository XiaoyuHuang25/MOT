import numpy as np

class GNNTracker:
    def __init__(self,
                 updater,
                 initiator,
                 data_associator,
                 deleter,
                 **kwargs):

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
            estimations.append({'x': track.state.mean[mapping, :],
                                'track_id': track.track_id,
                                })
        return estimations

    def update(self, measurements, timestamp):
        hypotheses = self.data_associator.associate(self.tracks, measurements, timestamp)
        associated_measurements = set()

        for track in self.tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = self.updater.update(
                    hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:
                state_post = hypothesis.prediction
                track.append(state_post)

        self.tracks -= self.deleter.delete_tracks(self.tracks)

        unassoc = measurements - associated_measurements
        new_tracks = self.initiator.initiate(unassoc, timestamp)
        new_tracks = self._wrap_tracks_with_id(new_tracks)
        self.tracks |= new_tracks

        return self.tracks
