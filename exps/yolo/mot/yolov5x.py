# encoding: utf-8
import os


class Exp:
    def __init__(self):

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.1
        # nms threshold
        self.nmsthre = 0.7
        self.half = False
        self.max_det = 300
        self.vid_stride = 1
        self.stream_buffer = True
        self.visualize = False
        self.augment = False
        
        self.classes = None
        self.retina_masks = False
        self.embed = False
        self.single_cls = False
        self.class_agnostic = True

        self.output_dir = "./Outputs/YOLO/yolov5x"
        self.detector_name = "YOLOV5"
