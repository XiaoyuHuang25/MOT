
class Exp(object):
    def __init__(self,
                 det_thresh=0.1,
                 det_high_thresh=0.6,
                 track_buffer=30,
                 asso='iou',
                 fuse_score=True,
                 match_thresh=0.9,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = False
        self.tracker_name = "BYTE"
