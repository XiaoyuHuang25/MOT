
class Exp(object):
    def __init__(self,
                 track_high_thresh=0.6,
                 track_low_thresh=0.1,
                 new_track_thresh=0.7,
                 track_buffer=30,
                 match_thresh=0.8,
                 proximity_thresh=0.5,
                 appearance_thresh=0.25,
                 category_aware_tracking=True,
                 fuse_score=True,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = True
        self.with_gmc = True
        self.gmc_method = 'ecc'
        self.gmc_downscale = 10
        self.tracker_name = "BoTSORT"
