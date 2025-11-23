
class Exp(object):
    def __init__(self,
                 det_thresh=0.6,
                 low_thresh=0.1,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 delta_t=3,
                 asso_func="iou",
                 inertia=0.2,
                 use_byte=True,
                 TCM_first_step=True,
                 TCM_first_step_weight=1.0,
                 TCM_byte_step=True,
                 TCM_byte_step_weight=1.0,
                 EG_weight_high_score=1.3,
                 EG_weight_low_score=1.2,
                 longterm_bank_length=30,
                 adapfs=False,
                 alpha=0.8,
                 ECC=False,
                 with_longterm_reid=False,
                 with_longterm_reid_correction=True,
                 high_score_matching_thresh=0.8,
                 longterm_reid_correction_thresh=0.4,
                 longterm_reid_correction_thresh_low=0.4,
                 longterm_reid_weight=0.0,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = True
        self.tracker_name = "HybridSORTReID"
