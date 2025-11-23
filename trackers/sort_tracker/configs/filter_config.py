
class Exp(object):
    def __init__(self,
                 det_thresh=0.6,
                 max_age=30,
                 asso='iou',
                 min_hits=3,
                 iou_threshold=0.3,
                 category_aware_tracking=True,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = False
        self.tracker_name = "SORT"
