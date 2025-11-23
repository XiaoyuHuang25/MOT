#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
from ultralytics import YOLO

# https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings


def make_parser():
    parser = argparse.ArgumentParser(description="Train model with configurable parameters")
    parser.add_argument('--weights', default=None, help='initial weights path')
    parser.add_argument("--model", default=None, help="Path to the model file (.pt) or YAML configuration file")
    parser.add_argument("--data", default=None, help="Path to the dataset configuration file (e.g., coco128.yaml)")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--time", type=float, default=None, help="Maximum training time in hours")
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no improvement before stopping")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for training")
    parser.add_argument("--save", action="store_true",
                        help="Whether to save training checkpoints and final model weights")
    parser.add_argument("--save_period", type=int, default=-1, help="Frequency of saving model checkpoints in epochs")
    parser.add_argument("--cache", action="store_true",
                        help="Whether to cache dataset images in memory (ram), disk, or disable it")
    parser.add_argument("--device", default=None, help="Device for training (e.g., 'cpu', 'cuda:0')")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--project", default=None, help="Directory name to save training results")
    parser.add_argument("--name", default=None, help="Name of the training run")
    parser.add_argument("--exist_ok", action="store_true",
                        help="Whether to allow overwriting existing project/name directory")
    parser.add_argument("--pretrained", action="store_true",
                        help="Whether to start training from a pretrained model or "
                             "provide the path to the pretrained model")

    parser.add_argument("--optimizer", default="auto", help="Optimizer selection for training")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose training output")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
    parser.add_argument("--deterministic", action="store_true", help="Whether to enforce deterministic training")
    parser.add_argument("--single_cls", action="store_true",
                        help="Treat all classes in a multi-class dataset as a single class")
    parser.add_argument("--rect", action="store_true", help="Enable rectangular training for batch optimization")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic augmentation for last N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last saved checkpoint")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision training")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use for training")
    parser.add_argument("--profile", action="store_true", help="Profile training speed with ONNX and TensorRT")
    parser.add_argument("--freeze", default=None, help="Freeze the first N layers of the model")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum factor for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="L2 regularization weight")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Number of warmup epochs for learning rate")
    parser.add_argument("--warmup_momentum", type=float, default=0.8, help="Initial momentum for warmup")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1, help="Initial bias learning rate for warmup")
    parser.add_argument("--box", type=float, default=7.5, help="Weight for bounding box loss")
    parser.add_argument("--cls", type=float, default=0.5, help="Weight for classification loss")
    parser.add_argument("--dfl", type=float, default=1.5, help="Weight for distribution focal loss")
    parser.add_argument("--pose", type=float, default=12.0, help="Weight for pose estimation loss")
    parser.add_argument("--kobj", type=float, default=1.0, help="Weight for objectness loss in pose estimation")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Apply label smoothing")
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size for loss normalization")
    parser.add_argument("--overlap_mask", action="store_true", help="Enable overlapping masks for segmentation")
    parser.add_argument("--mask_ratio", type=int, default=4, help="Downsample ratio for segmentation masks")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for classification task")
    parser.add_argument("--val", action="store_true", help="Enable validation during training")
    parser.add_argument("--plots", action="store_true", help="Generate and save training plots")

    # Augmentation Settings and Hyperparameters
    parser.add_argument("--hsv_h", type=float, default=0.015,
                        help="Adjusts the hue of the image by a fraction of the color wheel, introducing color "
                             "variability. Helps the model generalize across different lighting conditions.")
    parser.add_argument("--hsv_s", type=float, default=0.7,
                        help="Alters the saturation of the image by a fraction, affecting the intensity of colors. "
                             "Useful for simulating different environmental conditions.")
    parser.add_argument("--hsv_v", type=float, default=0.4,
                        help="Modifies the value (brightness) of the image by a fraction, helping the model to "
                             "perform well under various lighting conditions.")
    parser.add_argument("--degrees", type=float, default=0.0,
                        help="Rotates the image randomly within the specified degree range, improving the model's "
                             "ability to recognize objects at various orientations.")
    parser.add_argument("--translate", type=float, default=0.1,
                        help="Translates the image horizontally and vertically by a fraction of the image size, "
                             "aiding in learning to detect partially visible objects.")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Scales the image by a gain factor, simulating objects at different distances from the "
                             "camera.")
    parser.add_argument("--shear", type=float, default=0.0,
                        help="Shears the image by a specified degree, mimicking the effect of objects being viewed "
                             "from different angles.")
    parser.add_argument("--perspective", type=float, default=0.0,
                        help="Applies a random perspective transformation to the image, enhancing the model's ability "
                             "to understand objects in 3D space.")
    parser.add_argument("--flipud", type=float, default=0.0,
                        help="Flips the image upside down with the specified probability, increasing the data "
                             "variability without affecting the object's characteristics.")
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="Flips the image left to right with the specified probability, useful for learning "
                             "symmetrical objects and increasing dataset diversity.")
    parser.add_argument("--bgr", type=float, default=0.0,
                        help="Flips the image channels from RGB to BGR with the specified probability, useful for "
                             "increasing robustness to incorrect channel ordering.")
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Combines four training images into one, simulating different scene compositions and "
                             "object interactions. Highly effective for complex scene understanding.")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="Blends two images and their labels, creating a composite image. Enhances the model's "
                             "ability to generalize by introducing label noise and visual variability.")
    parser.add_argument("--copy_paste", type=float, default=0.0,
                        help="Copies objects from one image and pastes them onto another, useful for increasing "
                             "object instances and learning object occlusion.")
    parser.add_argument("--copy_paste_mode", type=str, default="flip",
                        help="Copy-Paste augmentation method selection among the options of ('flip', 'mixup').")
    parser.add_argument("--auto_augment", type=str, default="randaugment",
                        help="Automatically applies a predefined augmentation policy (randaugment, autoaugment, "
                             "augmix), optimizing for classification tasks by diversifying the visual features.")
    parser.add_argument("--erasing", type=float, default=0.4,
                        help="Randomly erases a portion of the image during classification training, encouraging the "
                             "model to focus on less obvious features for recognition.")
    parser.add_argument("--crop_fraction", type=float, default=1.0,
                        help="Crops the classification image to a fraction of its size to emphasize central features "
                             "and adapt to object scales, reducing background distractions.")

    return parser


def main(args):
    if args.pretrained:
        assert args.weights is not None, "Pretrained model weights must be provided"
        print(f"Using pretrained weights from {args.weights}")
        model = YOLO(args.model).load(args.weights)
    elif args.resume:
        assert args.weights is not None, "Checkpoint weights must be provided for resuming training"
        print(f"Resuming training from {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"Training model from scratch")
        model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        time=args.time,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        save=args.save,
        save_period=args.save_period,
        cache=args.cache,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        verbose=args.verbose,
        seed=args.seed,
        deterministic=args.deterministic,
        single_cls=args.single_cls,
        rect=args.rect,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        amp=args.amp,
        fraction=args.fraction,
        profile=args.profile,
        freeze=args.freeze,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        pose=args.pose,
        kobj=args.kobj,
        label_smoothing=args.label_smoothing,
        nbs=args.nbs,
        overlap_mask=args.overlap_mask,
        mask_ratio=args.mask_ratio,
        dropout=args.dropout,
        val=args.val,
        plots=args.plots,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        bgr=args.bgr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        copy_paste_mode=args.copy_paste_mode,
        auto_augment=args.auto_augment,
        erasing=args.erasing,
        crop_fraction=args.crop_fraction
    )

    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
