#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
from ultralytics import YOLO


def make_parser():
    parser = argparse.ArgumentParser(description="Model validation configuration")
    parser.add_argument("--model", type=str, default=None, help="Path to the model file (.pt) or YAML configuration file")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the dataset configuration file (e.g., coco128.yaml)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for processing")
    parser.add_argument("--batch", type=int, default=16, help="Number of images per batch. Use -1 for AutoBatch")
    parser.add_argument("--save_json", action="store_true", help="Save results to a JSON file")
    parser.add_argument("--save_hybrid", action="store_true",
                        help="Save hybrid version of labels combining original annotations with model predictions")
    parser.add_argument("--conf", type=float, default=0.001, help="Minimum confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for Non-Maximum Suppression (NMS)")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum number of detections per image")
    parser.add_argument("--half", action="store_true", help="Enable half-precision (FP16) computation")
    parser.add_argument("--device", type=str, default=None, help="Device for validation (cpu, cuda:0, etc.)")
    parser.add_argument("--dnn", action="store_true", help="Use OpenCV DNN module for ONNX model inference")
    parser.add_argument("--plots", action="store_true",
                        help="Generate and save plots of predictions versus ground truth")
    parser.add_argument("--rect", action="store_true", help="Use rectangular inference for batching")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"],
                        help="Dataset split for validation")
    parser.add_argument("--project", default=None, help="Directory name to save training results")
    parser.add_argument("--name", default=None, help="Name of the training run")

    return parser


def main(args):
    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        half=args.half,
        device=args.device,
        dnn=args.dnn,
        plots=args.plots,
        rect=args.rect,
        split=args.split,
        project=args.project,
        name=args.name
    )
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # a list contains map50-95 of each category


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
