import argparse
import copy
import json
import os
import configparser
import random
import shutil
from os import listdir

import cv2
import numpy as np


classes_dict = {
    1: "swimmer",
    2: "floater",
    3: "boat",
    4: "swimmer on boat",
    5: "floater on boat",
    6: "life jacket",
    7: "ignored",
}

RELATIVE_PATH = "datasets/sds/"
ANNOTATION_PATH = RELATIVE_PATH + "annotations/"

PATHS_TO_DATA = {
    "Object": {
        "train": {
            "annotations": ANNOTATION_PATH + "instances_train_objects_in_water.json",
            "img": RELATIVE_PATH + "train",
        },
        "val": {
            "annotations": ANNOTATION_PATH + "instances_val_objects_in_water.json",
            "img": RELATIVE_PATH + "val",
        },
        "test": {
            "annotations": ANNOTATION_PATH + "instances_test_objects_in_water.json",
            "img": RELATIVE_PATH + "test",
        },
    },
    "Swimmer": {
        "train": {
            "annotations": ANNOTATION_PATH + "instances_train_swimmer.json",
            "img": RELATIVE_PATH + "train",
        },
        "val": {
            "annotations": ANNOTATION_PATH + "instances_val_swimmer.json",
            "img": RELATIVE_PATH + "val",
        },
        "test": {
            "annotations": ANNOTATION_PATH + "instances_test_swimmer.json",
            "img": RELATIVE_PATH + "test",
        },
    },
}

PATH_TO_OUTPUT = {
    "Object": {
        "train": RELATIVE_PATH + "images/Object/train",
        "val": RELATIVE_PATH + "images/Object/val",
        "train_val": RELATIVE_PATH + "images/Object/train_val",
        "test": RELATIVE_PATH + "images/Object/test",
    },
    "Swimmer": {
        "train": RELATIVE_PATH + "images/Swimmer/train",
        "val": RELATIVE_PATH + "images/Swimmer/val",
        "train_val": RELATIVE_PATH + "images/Swimmer/train_val",
        "test": RELATIVE_PATH + "images/Swimmer/test",
    },
}


def bbox_visualization_video(video_name, video_path, show_interval=10):
    """
    Visualize MOT-style sequences with bounding boxes.

    Parameters
    ----------
    video_name : str
        Window name and sequence name (e.g., MVI_XXXX).
    video_path : str
        Path to the MOT-formatted sequence (containing img1/ and gt/gt.txt).
    show_interval : int
        Show every `show_interval`-th frame to speed up visualization.
    """

    def generate_random_colors(num_ids):
        random.seed(0)  # deterministic colors for reproducibility
        colors = set()
        while len(colors) < num_ids:
            colors.add(
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
            )
        return sorted(colors)

    img_dir = os.path.join(video_path, "img1")
    if not os.path.isdir(img_dir):
        print(f"Image dir not found: {img_dir}")
        return

    image_file_list = sorted(os.listdir(img_dir))
    if not image_file_list:
        print(f"No images found in {img_dir}")
        return

    gt_file_path = os.path.join(video_path, "gt", "gt.txt")
    if os.path.exists(gt_file_path):
        with open(gt_file_path, "r") as file:
            content_list = file.read().split("\n")
    else:
        print(f"GT file not found: {gt_file_path}")
        content_list = []

    bbox_dict = {}
    max_id = 0
    for content in content_list:
        if not content:
            continue
        (
            frame_number,
            identity_number,
            bbox_left,
            bbox_top,
            bbox_width,
            bbox_height,
            confidence,
            category,
            visibility,
        ) = content.split(",")
        frame_number = int(frame_number)
        identity_number = int(identity_number)
        max_id = max(max_id, identity_number)

        bbox_dict.setdefault(frame_number, []).append(
            [
                identity_number,
                int(bbox_left),
                int(bbox_top),
                int(bbox_width),
                int(bbox_height),
                int(category),
            ]
        )

    if max_id == 0:
        print(f"No bbox in {gt_file_path}")
        return

    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(video_name, 0, 0)
    cv2.resizeWindow(video_name, 800, 600)

    colors = generate_random_colors(max_id)

    for image_path in image_file_list:
        frame_number = int(os.path.splitext(image_path)[0])
        if frame_number % show_interval != 0:
            continue

        image = cv2.imread(os.path.join(img_dir, image_path))
        if image is None:
            continue

        if frame_number in bbox_dict:
            bboxes = bbox_dict[frame_number]
            for bbox in bboxes:
                (
                    identity_number,
                    bbox_left,
                    bbox_top,
                    bbox_width,
                    bbox_height,
                    category,
                ) = bbox
                x1, y1 = bbox_left, bbox_top
                x2, y2 = bbox_left + bbox_width, bbox_top + bbox_height
                color = colors[identity_number - 1]
                class_label = classes_dict.get(category, str(category))

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                text = f"Class: {class_label}, ID: {identity_number}"
                text_size, _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                text_x = max(x1, 0)
                text_y = max(y1 - 10, text_size[1])
                cv2.putText(
                    image,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        cv2.putText(
            image,
            f"Frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imshow(video_name, image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


def convert_dataset_to_mot_format(sub_dataset, overwrite_images, merge_train_val):
    """
    Convert SeaDronesSee COCO-style annotations to MOT format.

    This will:
      - Reorganize images into MOT-style sequences: seq_name/img1/000001.jpg, ...
      - Generate MOT gt.txt and seqinfo.ini for each sequence.
      - Optionally create a merged train_val split.
    """
    dataset_dict = PATHS_TO_DATA[sub_dataset]
    splits = list(dataset_dict.keys())  # ['train', 'val', 'test']
    if merge_train_val:
        splits.append("train_val")

    for split in splits:
        # Remove existing split folder if required
        if overwrite_images and os.path.exists(PATH_TO_OUTPUT[sub_dataset][split]):
            shutil.rmtree(PATH_TO_OUTPUT[sub_dataset][split])

        num_data = {}
        print(f"\nProcessing {split} split")
        if split == "train_val":
            # Merge train + val for images / annotations / videos
            ann_path_train = dataset_dict["train"]["annotations"]
            with open(ann_path_train) as f:
                data_train = json.load(f)

            ann_path_val = dataset_dict["val"]["annotations"]
            with open(ann_path_val) as f:
                data_val = json.load(f)

            data = {
                "images": data_train["images"] + data_val["images"],
                "annotations": data_train["annotations"] + data_val["annotations"],
                "split": ["train"] * len(data_train["images"])
                + ["val"] * len(data_val["images"]),
                "videos": [],
            }

            videos_list = list(data_train["videos"])
            train_video_ids = {video["id"] for video in data_train["videos"]}
            videos_list.extend(
                [video for video in data_val["videos"] if video["id"] not in train_video_ids]
            )
            data["videos"] = videos_list

            # Ensure both image roots exist
            for s in ("train", "val"):
                data_path = dataset_dict[s]["img"]
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Path {data_path} does not exist")
        else:
            ann_path = dataset_dict[split]["annotations"]
            with open(ann_path) as f:
                data = json.load(f)
            data["split"] = [split] * len(data["images"])

            data_path = dataset_dict[split]["img"]
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Path {data_path} does not exist")

        # Build per-video containers
        videos_dict = {}
        for video_dict in data["videos"]:
            # Some annotations use "name:", some use "name"; support both.
            video_id = video_dict["id"]
            raw_name = video_dict.get("name:", video_dict.get("name", str(video_id)))
            video_name = os.path.basename(raw_name).replace(".", "_")
            height = video_dict["height"]
            width = video_dict["width"]
            videos_dict[video_id] = {
                "video_name": video_name,
                "images": {},
                "tracks": {},
                "height": height,
                "width": width,
            }
            num_data[video_id] = {"video_name": video_name}

        # Re-map images into MOT-style per-sequence folders
        for image_dict, data_split in zip(data["images"], data["split"]):
            video_id = image_dict["video_id"]
            if video_id not in videos_dict:
                # This video_id has no entry in data['videos'], mark as skipped
                num_data[video_id] = None
                continue

            image_id = image_dict["id"]
            # Ensure .jpg extension
            file_name = os.path.splitext(image_dict["file_name"])[0] + ".jpg"
            new_file_name = os.path.join(
                str(videos_dict[video_id]["video_name"]), "img1", f"{image_id:06d}.jpg"
            )
            height = image_dict["height"]
            width = image_dict["width"]
            frame_index = image_dict["frame_index"]

            assert image_id not in videos_dict[video_id]["images"]
            assert height == videos_dict[video_id]["height"]
            assert width == videos_dict[video_id]["width"]

            videos_dict[video_id]["images"][image_id] = {
                "file_name": file_name,
                "new_file_name": new_file_name,
                "frame_index": frame_index,
            }

            source_image_path = os.path.join(
                PATHS_TO_DATA[sub_dataset][data_split]["img"], file_name
            )
            destination_image_path = os.path.join(
                PATH_TO_OUTPUT[sub_dataset][split], new_file_name
            )
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
            if not os.path.exists(destination_image_path):
                shutil.copy(str(source_image_path), str(destination_image_path))

        # Build MOT gt.txt tracks
        if data.get("annotations") is not None:
            for anna_dict in data["annotations"]:
                anna_id = anna_dict["id"]
                image_id = anna_dict["image_id"]
                bbox = anna_dict["bbox"]
                area = anna_dict["area"]
                category_id = anna_dict["category_id"]
                video_id = anna_dict["video_id"]
                track_id = anna_dict["track_id"]

                if video_id not in videos_dict:
                    # Annotation refers to a video not in videos_dict (very rare), skip
                    continue

                if track_id not in videos_dict[video_id]["tracks"]:
                    videos_dict[video_id]["tracks"][track_id] = {}
                if image_id not in videos_dict[video_id]["tracks"][track_id]:
                    videos_dict[video_id]["tracks"][track_id][image_id] = []
                videos_dict[video_id]["tracks"][track_id][image_id].append(
                    {"bbox": bbox, "area": area, "category_id": category_id}
                )

            for video_id in videos_dict:
                num_data[video_id].update(
                    {"annotations": {i: 0 for i in range(1, len(classes_dict) + 1)}}
                )
                num_data[video_id]["invalid_annotations"] = 0
                videos_dict[video_id]["gt"] = ""

                track_id_list = sorted(videos_dict[video_id]["tracks"].keys())
                for track_id in track_id_list:
                    image_id_list = sorted(
                        videos_dict[video_id]["tracks"][track_id].keys()
                    )
                    for image_id in image_id_list:
                        bboxes = [
                            annotations["bbox"]
                            for annotations in videos_dict[video_id]["tracks"][track_id][
                                image_id
                            ]
                        ]
                        track_ids = [track_id] * len(bboxes)
                        categories = [
                            annotations["category_id"]
                            for annotations in videos_dict[video_id]["tracks"][track_id][
                                image_id
                            ]
                        ]
                        areas = [
                            annotations["area"]
                            for annotations in videos_dict[video_id]["tracks"][track_id][
                                image_id
                            ]
                        ]

                        # If multiple annotations for same track/frame, keep the largest one
                        if len(track_ids) > 1:
                            num_data[video_id]["invalid_annotations"] += len(track_ids) - 1
                            assert all(
                                x == track_ids[0] for x in track_ids
                            ), f"{track_ids} != {track_ids[0]}"
                            assert all(
                                x == categories[0] for x in categories
                            ), f"{categories} != {categories[0]}"
                            bbox_idx = int(np.argmax(areas))
                            bboxes = [bboxes[bbox_idx]]
                            categories = [categories[bbox_idx]]

                        videos_dict[video_id]["gt"] += (
                            f"{image_id},{track_id},"
                            f"{bboxes[0][0]},{bboxes[0][1]},"
                            f"{bboxes[0][2]},{bboxes[0][3]},"
                            f"1,{categories[0]},1\n"
                        )
                        num_data[video_id]["annotations"][categories[0]] += 1
        else:
            print(f"No annotations for {split} split")

        # Print per-video statistics
        record_txt = ""
        total_frames_num = 0
        total_tracks_num = 0
        total_labels_num = 0

        for video_id in num_data:
            if num_data[video_id] is not None:
                video_name = num_data[video_id]["video_name"]
                num_images = len(videos_dict[video_id]["images"])
                if "annotations" in num_data[video_id]:
                    num_tracks = len(videos_dict[video_id]["tracks"])
                    num_invalid_annotations = num_data[video_id]["invalid_annotations"]
                    record_txt += f"{video_name} {video_id} {num_images} {num_tracks}"
                    for i in range(1, len(classes_dict) + 1):
                        record_txt += f' {num_data[video_id]["annotations"][i]}'
                        total_labels_num += num_data[video_id]["annotations"][i]
                    record_txt += f" {num_invalid_annotations}\n"
                    total_tracks_num += num_tracks
                else:
                    # No annotations for this video (but still images)
                    record_txt += (
                        f"{video_name} {video_id} {num_images} 0 0 0 0 0 0 0 0 0\n"
                    )
            else:
                # Video id appears only in images, not in videos_dict; just log as skipped
                image_list_tmp = [
                    image for image in data["images"] if image["video_id"] == video_id
                ]
                num_images = len(image_list_tmp)
                if not image_list_tmp:
                    # Should not happen, but guard anyway
                    video_name = f"{video_id}_skip"
                else:
                    video_name = image_list_tmp[0]["source"]["video"].replace(".", "_")
                record_txt += (
                    f"{video_name}(skip) {video_id} {num_images} 0 0 0 0 0 0 0 0 0\n"
                )
            total_frames_num += num_images

        print(record_txt)
        print(
            f"Total number of frames: {total_frames_num}, "
            f"total number of labels: {total_labels_num}, "
            f"total number of objects: {total_tracks_num}"
        )

        # Write MOT gt.txt and seqinfo.ini
        for video_id, video_dict in videos_dict.items():
            video_name = video_dict["video_name"]
            images_dict = video_dict["images"]

            # Write gt.txt only if gt field exists
            if "gt" in videos_dict[video_id]:
                gt_txt = videos_dict[video_id]["gt"]
                if gt_txt != "" or (
                    gt_txt == "" and split in ("train", "val", "train_val")
                ):
                    gt_file_path = os.path.join(
                        PATH_TO_OUTPUT[sub_dataset][split], video_name, "gt", "gt.txt"
                    )
                    os.makedirs(os.path.dirname(gt_file_path), exist_ok=True)
                    with open(gt_file_path, "w") as file:
                        file.write(gt_txt)

            config_file_path = os.path.join(
                PATH_TO_OUTPUT[sub_dataset][split], video_name, "seqinfo.ini"
            )
            os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
            config = configparser.ConfigParser()

            config["Sequence"] = {
                "name": video_name,
                "imDir": "img1",
                "seqLength": len(images_dict),
                "imWidth": videos_dict[video_id]["width"],
                "imHeight": videos_dict[video_id]["height"],
                "imExt": ".jpg",
            }

            with open(config_file_path, "w") as configfile:
                config.write(configfile)


def convert_gt_to_yolox_and_ultralytics_format(sub_dataset="Object"):
    """
    Convert MOT-style SDD splits into:
      - COCO-like JSON (with videos / images / annotations),
      - YOLOX / Ultralytics-style label .txt files,
      - category-agnostic JSON (*_cag.json).
    """
    dataset_dict = PATH_TO_OUTPUT[sub_dataset]
    splits = dataset_dict.keys()
    info_show = {split: "" for split in splits}

    for split in splits:
        print(
            f"Generating COCO-format annotations for {split} split of {sub_dataset} dataset"
        )
        data_path = PATH_TO_OUTPUT[sub_dataset][split]
        if not os.path.exists(data_path):
            print(f"Path {data_path} does not exist, skipping...")
            continue

        base_path = os.path.dirname(data_path)
        ann_root = os.path.join(base_path, "annotations")
        out_path = os.path.join(ann_root, f"{split}.json")
        category_agnostic_out_path = os.path.join(ann_root, f"{split}_cag.json")
        os.makedirs(ann_root, exist_ok=True)

        out = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [
                {"id": i, "name": classes_dict[i]}
                for i in range(1, len(classes_dict) + 1)
            ],
        }

        seqs = sorted(os.listdir(data_path))
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1

        # image_id -> list[annotation dict], used later to write YOLO labels
        labels_per_image = {}

        for seq in seqs:
            video_cnt += 1  # video sequence id
            out["videos"].append({"id": video_cnt, "file_name": seq})
            seq_path = os.path.join(data_path, seq)
            img1_path = os.path.join(seq_path, "img1")
            ann_path = os.path.join(seq_path, "gt", "gt.txt")

            if not os.path.isdir(img1_path):
                print(f"Sequence {seq} has no img1 directory, skipping...")
                continue

            images_file_name = sorted(os.listdir(img1_path))
            images_file_name = [
                f for f in images_file_name if f.lower().endswith(".jpg")
            ]
            num_images = len(images_file_name)
            if num_images == 0:
                print(f"Sequence {seq} has no jpg images, skipping...")
                continue

            config = configparser.ConfigParser()
            config.read(os.path.join(seq_path, "seqinfo.ini"))
            width = int(config["Sequence"]["imWidth"])
            height = int(config["Sequence"]["imHeight"])

            img_num_in_dataset = {}
            for i, image_file_name in enumerate(images_file_name):
                image_id_local = int(os.path.splitext(image_file_name)[0])
                image_id_global = image_cnt + i + 1
                img_num_in_dataset[image_id_local] = image_id_global
                image_info = {
                    "file_name": f"{seq}/img1/{image_file_name}",  # relative path
                    "id": image_id_global,
                    "frame_id": image_id_local,
                    "video_id": video_cnt,
                    "height": height,
                    "width": width,
                }
                out["images"].append(image_info)
            print(f"{seq}: {num_images} images")

            # Only train/val/train_val have GT labels
            if split != "test":
                if not os.path.exists(ann_path):
                    print(f"GT file not found for {seq} at {ann_path}, skipping...")
                else:
                    # Handle empty-file / one-line / multi-line cases robustly
                    try:
                        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
                    except ValueError:
                        # Empty file or invalid format
                        print(f"No annotations (empty GT) for {seq}")
                        anns = np.empty((0, 9), dtype=np.float32)

                    if anns.ndim == 1 and anns.size > 0:
                        anns = np.atleast_2d(anns)

                    if anns.size == 0:
                        print(f"No annotations for {seq}")
                    else:
                        print(f"{seq}: {int(anns[:, 0].max())} annotated frames")
                        for row in anns:
                            image_id_local = int(row[0])
                            track_id_raw = int(row[1])
                            bbox_tlwh = row[2:6]
                            conf = float(row[6])
                            category_id = int(row[7])

                            if track_id_raw != tid_last:
                                tid_curr += 1
                                tid_last = track_id_raw

                            image_id_global = img_num_in_dataset.get(image_id_local)
                            # If GT refers to a frame not in img1, skip it
                            if image_id_global is None:
                                continue

                            ann_cnt += 1
                            ann = {
                                "id": ann_cnt,
                                "category_id": category_id,
                                "image_id": image_id_global,
                                "track_id": tid_curr,
                                "bbox": bbox_tlwh.tolist(),
                                "conf": conf,
                                "iscrowd": 0,
                                "area": float(bbox_tlwh[2] * bbox_tlwh[3]),
                            }
                            out["annotations"].append(ann)
                            labels_per_image.setdefault(image_id_global, []).append(ann)

            image_cnt += num_images
            print(tid_curr, tid_last)

        # Write YOLO label txt files (one per image) for splits with GT
        if split != "test":
            for image in out["images"]:
                image_id = image["id"]
                seq_img_rel_path = image["file_name"]  # e.g., seq/img1/000001.jpg
                file_name = os.path.join(data_path, seq_img_rel_path)
                height, width = image["height"], image["width"]

                anns_for_image = labels_per_image.get(image_id, [])
                lines = []
                if anns_for_image:
                    dw = 1.0 / width
                    dh = 1.0 / height
                    for ann in anns_for_image:
                        x, y, w, h = ann["bbox"]
                        cx = (x + w / 2.0) * dw
                        cy = (y + h / 2.0) * dh
                        ww = w * dw
                        hh = h * dh
                        cls = int(ann["category_id"]) - 1  # YOLO uses 0-based class index
                        lines.append(
                            f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}"
                        )

                label_path = (
                    file_name.replace("/images/", "/labels/")
                    .replace("\\images\\", "\\labels\\")
                    .replace(".jpg", ".txt")
                )
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                with open(label_path, "w") as f:
                    f.write("\n".join(lines))

        info_show[split] = (
            f"loaded {split} for {len(out['images'])} images "
            f"and {len(out['annotations'])} samples"
        )
        print(info_show[split])

        # Save COCO-style json
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"Annotations saved to {out_path}")

        # Save category-agnostic json
        out_cag = copy.deepcopy(out)
        out_cag["categories"] = [{"id": 1, "name": "object"}]
        for ann in out_cag["annotations"]:
            ann["category_id"] = 1
        with open(category_agnostic_out_path, "w") as f:
            json.dump(out_cag, f)
        print(
            f"Category-agnostic annotations saved to {category_agnostic_out_path}"
        )

    print(f"\nDataset {sub_dataset} loaded.")
    for split in splits:
        if info_show[split]:
            print(info_show[split])


def generate_seqmaps_for_mot(sub_dataset):
    """
    Generate MOTChallenge-style seqmaps files listing all sequences
    in each split (train / val / train_val / test).
    """
    dataset_dict = PATH_TO_OUTPUT[sub_dataset]
    splits = dataset_dict.keys()

    for split in splits:
        data_path = PATH_TO_OUTPUT[sub_dataset][split]
        if not os.path.exists(data_path):
            print(f"Path {data_path} does not exist, skipping...")
            continue
        base_path = os.path.dirname(data_path)
        seqmaps_path = os.path.join(base_path, "seqmaps", f"{split}.txt")
        os.makedirs(os.path.join(base_path, "seqmaps"), exist_ok=True)
        seqs = sorted(os.listdir(data_path))
        txt_tmp = "name\n" + "\n".join(seqs) + "\n"
        with open(seqmaps_path, "w") as f:
            f.write(txt_tmp)


def main(args):
    overwrite_images = args.overwrite_images
    convert_to_mot = args.convert_to_mot
    visualization = args.visualization
    merge_train_val = args.merge_train_val
    convert_gt_to_coco_ultralytics = args.convert_gt_to_coco_ultralytics
    generate_seqmaps_flag = args.generate_seqmaps

    assert args.process_datasets in ["Object", "Swimmer", "ALL"], "Invalid dataset"
    if args.process_datasets == "ALL":
        datasets = ["Object", "Swimmer"]
    else:
        datasets = [args.process_datasets]

    if convert_to_mot:
        for sub_dataset in datasets:
            print(f"\nProcessing {sub_dataset} dataset, converting to MOT format...")
            convert_dataset_to_mot_format(sub_dataset, overwrite_images, merge_train_val)

    if convert_gt_to_coco_ultralytics:
        for sub_dataset in datasets:
            print(
                f"Processing {sub_dataset} dataset, converting to COCO and ultralytics format..."
            )
            convert_gt_to_yolox_and_ultralytics_format(sub_dataset)

    if visualization:
        for sub_dataset in datasets:
            for split in ["train", "test", "val"]:
                data_split_path = PATH_TO_OUTPUT[sub_dataset][split]
                if not os.path.exists(data_split_path):
                    continue
                video_name_list = os.listdir(data_split_path)
                for video_name in video_name_list:
                    print(
                        f"Visualizing {video_name} video in {split} split "
                        f"of {sub_dataset} dataset"
                    )
                    video_path = os.path.join(data_split_path, video_name)
                    bbox_visualization_video(video_name, video_path, show_interval=50)

    if generate_seqmaps_flag:
        for dataset in datasets:
            print(f"Processing {dataset} dataset, generating seqmaps...")
            generate_seqmaps_for_mot(dataset)


def make_parser():
    parser = argparse.ArgumentParser("Sea Drone See Dataset Converter")
    parser.add_argument(
        "--overwrite_images",
        default=False,
        action="store_true",
        help="Overwrite existing MOT image folders",
    )
    parser.add_argument(
        "--convert_to_mot",
        default=False,
        action="store_true",
        help="Convert the dataset to MOT format",
    )
    parser.add_argument(
        "--visualization",
        default=False,
        action="store_true",
        help="Visualize the MOT-formatted dataset",
    )
    parser.add_argument(
        "--merge_train_val",
        default=False,
        action="store_true",
        help="Merge train and val splits into an additional train_val split",
    )
    parser.add_argument(
        "--convert_gt_to_coco_ultralytics",
        default=False,
        action="store_true",
        help="Convert MOT GT to COCO-like JSON and YOLO label format",
    )
    parser.add_argument(
        "--generate_seqmaps",
        default=False,
        action="store_true",
        help="Generate MOTChallenge-style seqmaps files",
    )
    parser.add_argument(
        "--process_datasets",
        default="ALL",
        type=str,
        help="Datasets to process: Object, Swimmer, or ALL",
    )

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
