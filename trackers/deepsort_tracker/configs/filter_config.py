
class Exp(object):
    def __init__(self,
                 det_thresh=0.3,
                 max_cosine_distance=0.1,
                 max_iou_distance=0.7,
                 min_hits=3,
                 max_age=30,
                 nn_budget=100,
                 match_thresh=0.9,
                 asso='iou',
                 category_aware_tracking=True,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = True
        self.tracker_name = "DeepSORT"
