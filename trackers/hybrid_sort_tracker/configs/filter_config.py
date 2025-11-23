
class Exp(object):
    def __init__(self,
                 det_thresh=0.6,
                 low_thresh=0.1,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.25,
                 delta_t=3,
                 # track_buffer=30,
                 asso_func="Height_Modulated_IoU",
                 inertia=0.05,
                 use_byte=True,
                 # low_thresh=0.1,
                 # EG_weight_high_score=0.0,
                 TCM_first_step=True,
                 TCM_first_step_weight=1.0,
                 # with_longterm_reid=False,
                 # with_longterm_reid_correction=False,
                 # high_score_matching_thresh=0.8,
                 TCM_byte_step=True,
                 TCM_byte_step_weight=1.0,
                 # ECC=False,
                 # match_thresh=0.9,
                 # min_box_area=100,
                 # EG_weight_low_score=0.0,
                 # adapfs=False,
                 # alpha=0.8,
                 # aspect_ratio_thresh=1.6,
                 # longterm_bank_length=30,
                 # longterm_reid_correction_thresh=1.0,
                 # longterm_reid_correction_thresh_low=1.0,
                 # longterm_reid_weight=0.0,
                 # longterm_reid_weight_low=0.0,
                 # low_score_matching_thresh=0.5,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = False
        self.tracker_name = "HybridSORT"
