# encoding: utf-8
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 10
        # factor of model depth
        self.depth = 1.33
        # factor of model width
        self.width = 1.25

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 8
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = 'datasets/smd/images/VIS'
        # name of annotation file for training
        self.train_ann = "train.json"
        self.train_dir = "train"
        # name of annotation file for evaluation
        self.val_ann = "val.json"
        self.val_dir = "val"
        # name of annotation file for testing
        self.test_ann = "test.json"
        self.test_dir = "test"

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 100
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15

        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 100
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 5
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.1
        # nms threshold
        self.nmsthre = 0.7
        self.test_fp16 = False
        self.test_fuse = True
        self.test_legacy = False
        self.detector_name = "YOLOX"
        self.output_dir = "./Outputs/YOLO/yolox_x"
        self.single_cls = False
        self.class_agnostic = True

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            name=self.train_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            name=self.val_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
