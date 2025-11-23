class Exp(object):
    def __init__(self,
                 det_thresh=0.6,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 delta_t=3,
                 inertia=0.2,
                 use_byte=False,
                 asso_func='iou',
                 category_aware_tracking=True,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = True
        self.tracker_name = "OCSORT"
