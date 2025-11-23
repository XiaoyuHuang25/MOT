import argparse
import copy
import json
import os
import configparser
import random
import shutil
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from scipy.io import loadmat


classes_dict = {
    1: "Ferry",
    2: "Buoy",
    3: "Vessel/ship",
    4: "Speed boat",
    5: "Boat",
    6: "Kayak",
    7: "Sail boat",
    8: "Swimming person",
    9: "Flying bird/plane",
    10: "Other",
}

RELATIVE_PATH = "datasets/smd/"

PATHS_TO_DATA = {
    "NIR": {
        "ObjectGT": RELATIVE_PATH + "NIR/ObjectGT",
        "TrackGT": RELATIVE_PATH + "NIR/TrackGT",
        "Videos": RELATIVE_PATH + "NIR/Videos",
        "Data": RELATIVE_PATH + "Track/NIR",
    },
    "VIS_Onshore": {
        "ObjectGT": RELATIVE_PATH + "VIS_Onshore/ObjectGT",
        "TrackGT": RELATIVE_PATH + "VIS_Onshore/TrackGT",
        "Videos": RELATIVE_PATH + "VIS_Onshore/Videos",
        "Data": RELATIVE_PATH + "Track/VIS_Onshore",
    },
    "VIS_Onboard": {
        "ObjectGT": RELATIVE_PATH + "VIS_Onboard/ObjectGT",
        "TrackGT": RELATIVE_PATH + "VIS_Onboard/TrackGT",
        "Videos": RELATIVE_PATH + "VIS_Onboard/Videos",
        "Data": RELATIVE_PATH + "Track/VIS_Onboard",
    },
}

PATH_TO_OUTPUT = {
    "NIR": {
        "train": RELATIVE_PATH + "images/NIR/train",
        "val": RELATIVE_PATH + "images/NIR/val",
        "train_val": RELATIVE_PATH + "images/NIR/train_val",
        "test": RELATIVE_PATH + "images/NIR/test",
    },
    "VIS": {
        "train": RELATIVE_PATH + "images/VIS/train",
        "val": RELATIVE_PATH + "images/VIS/val",
        "train_val": RELATIVE_PATH + "images/VIS/train_val",
        "test": RELATIVE_PATH + "images/VIS/test",
    },
    "VISNIR": {
        "train": RELATIVE_PATH + "images/VISNIR/train",
        "val": RELATIVE_PATH + "images/VISNIR/val",
        "train_val": RELATIVE_PATH + "images/VISNIR/train_val",
        "test": RELATIVE_PATH + "images/VISNIR/test",
    },
}

SPLIT_DATASETS = {
    "VIS": {
        "datasets": ["VIS_Onshore", "VIS_Onboard"],
        "test": [
            "MVI_1448_VIS_Haze",
            "MVI_1469_VIS",
            "MVI_1481_VIS",
            "MVI_1587_VIS",
            "MVI_1592_VIS",
            "MVI_1610_VIS",
            "MVI_1612_VIS",
            "MVI_1613_VIS",
        ],
        "val": [
            "MVI_1478_VIS",
            "MVI_1484_VIS",
            "MVI_1486_VIS",
            "MVI_1578_VIS",
            "MVI_1582_VIS",
            "MVI_1622_VIS",
            "MVI_1623_VIS",
            "MVI_0790_VIS_OB",
        ],
    },
    "NIR": {
        "datasets": ["NIR"],
        "test": [
            "MVI_0895_NIR_Haze",
            "MVI_1468_NIR",
            "MVI_1520_NIR",
            "MVI_1527_NIR",
            "MVI_1551_NIR",
        ],
        "val": [
            "MVI_1522_NIR",
            "MVI_1524_NIR",
            "MVI_1525_NIR",
            "MVI_1528_NIR",
            "MVI_1539_NIR",
        ],
    },
    "VISNIR": {
        "datasets": ["VIS_Onshore", "VIS_Onboard", "NIR"],
        "test": [],
        "val": [],
    },
}
SPLIT_DATASETS["VISNIR"]["test"] = (
    SPLIT_DATASETS["VIS"]["test"] + SPLIT_DATASETS["NIR"]["test"]
)
SPLIT_DATASETS["VISNIR"]["val"] = (
    SPLIT_DATASETS["VIS"]["val"] + SPLIT_DATASETS["NIR"]["val"]
)

# Basic sanity checks for split definitions
assert (
    len(set(SPLIT_DATASETS["VIS"]["test"]) & set(SPLIT_DATASETS["VIS"]["val"])) == 0
), "Overlapping videos"
assert (
    len(set(SPLIT_DATASETS["NIR"]["test"]) & set(SPLIT_DATASETS["NIR"]["val"])) == 0
), "Overlapping videos"
assert (
    len(set(SPLIT_DATASETS["VISNIR"]["test"])
        & set(SPLIT_DATASETS["VISNIR"]["val"]))
    == 0
), "Overlapping videos"
assert len(set(SPLIT_DATASETS["VIS"]["test"])) == 8, "Invalid number of test videos for VIS"
assert len(set(SPLIT_DATASETS["VIS"]["val"])) == 8, "Invalid number of val videos for VIS"
assert len(set(SPLIT_DATASETS["NIR"]["test"])) == 5, "Invalid number of test videos for NIR"
assert len(set(SPLIT_DATASETS["NIR"]["val"])) == 5, "Invalid number of val videos for NIR"
assert len(set(SPLIT_DATASETS["VISNIR"]["test"])) == len(
    set(SPLIT_DATASETS["VIS"]["test"])
) + len(set(SPLIT_DATASETS["NIR"]["test"])), "Invalid number of test videos for VISNIR"
assert len(set(SPLIT_DATASETS["VISNIR"]["val"])) == len(
    set(SPLIT_DATASETS["VIS"]["val"])
) + len(set(SPLIT_DATASETS["NIR"]["val"])), "Invalid number of val videos for VISNIR"


def generate_list_as_txt(
    frame_number, identity_number, bbox_tlwh, confidence, category, visibility
):
    """Format one MOT line as CSV: frame,id,x,y,w,h,conf,cls_id,visibility."""
    return (
        f"{frame_number},{identity_number},"
        f"{bbox_tlwh[0]},{bbox_tlwh[1]},"
        f"{bbox_tlwh[2]},{bbox_tlwh[3]},"
        f"{confidence},{category},{visibility}\n"
    )


def generate_gt_files_dict(path_to_gt_files, gt_file_type="ObjectGT"):
    """Scan GT directory and return a dict {video_name: mat_path}."""
    return {
        f.split(".")[0].replace(f"_{gt_file_type}", ""): join(path_to_gt_files, f)
        for f in listdir(path_to_gt_files)
        if isfile(join(path_to_gt_files, f))
    }


def convert_mat_to_gt(gt_files_dict, path_to_output=None):
    """
    Convert SMD .mat annotations into MOT-style gt.txt files.
    Also collects some statistics and returns a per-video dict.
    """
    gt_dict = {}
    category_id_dict = {}

    record = "\nvideo_name frames objects labels"
    class_num_keys = list(range(1, len(classes_dict) + 1))
    for key in class_num_keys:
        record += f" {classes_dict[key]}"
    record += "\n"

    total_frames_num = 0
    total_labels_num = 0
    total_objects_num = 0

    video_name_list = sorted(gt_files_dict.keys())
    for video_name in video_name_list:
        gt_dict[video_name] = {}
        file_name = gt_files_dict[video_name]

        gt = loadmat(file_name)
        track_struct = gt["Track"][0]

        objects_number = len(track_struct)
        frames_number = len(track_struct["BB"][0])
        labels_num = 0
        class_num_dict = {i: 0 for i in class_num_keys}

        # Collect all lines in a list for performance, then join at the end.
        lines = []

        for track_id in range(objects_number):
            invalid_frames = []
            for frame_index in range(frames_number):
                bb = track_struct["BB"][track_id][frame_index]
                if np.all(bb == -1) or np.all(bb == 0):
                    invalid_frames.append(frame_index)
                    continue

                bbox_tlwh = bb.astype(int)
                category_id = track_struct["Object"][track_id][0][frame_index]

                # Count category usage globally
                category_id_dict[category_id] = category_id_dict.get(category_id, 0) + 1

                # Uncomment these if you need motion / distance
                # motion = track_struct["Motion"][track_id][0][frame_index]
                # distance = track_struct["Distance"][track_id][0][frame_index]

                lines.append(
                    generate_list_as_txt(
                        frame_index + 1,
                        track_id + 1,
                        bbox_tlwh,
                        1,
                        category_id,
                        1,
                    )
                )
                labels_num += 1
                class_num_dict[int(category_id)] += 1

            if len(invalid_frames) == frames_number:
                print(f"Invalid track {track_id} in {video_name}.")

        txt_content = "".join(lines)
        gt_dict[video_name]["txt"] = txt_content

        if path_to_output is not None:
            path_to_gt_txt = os.path.join(path_to_output, video_name, "gt", "gt.txt")
            os.makedirs(os.path.dirname(path_to_gt_txt), exist_ok=True)
            with open(path_to_gt_txt, "w") as f:
                f.write(txt_content)

        gt_dict[video_name]["frames"] = frames_number
        gt_dict[video_name]["objects"] = objects_number
        gt_dict[video_name]["path"] = gt_files_dict[video_name]
        print(
            f"Loaded {video_name} with {frames_number} frames, "
            f"{labels_num} labels and {objects_number} objects."
        )

        record += f"{video_name} {frames_number} {objects_number} {labels_num} "
        for key in class_num_keys:
            record += f"{class_num_dict[key]} "
        record += "\n"

        total_labels_num += labels_num
        total_frames_num += frames_number
        total_objects_num += objects_number

    print(f"{record}\n")
    print(
        f"Total number of classes: {len(category_id_dict)}, "
        f"max class id: {max(category_id_dict.keys())}, "
        f"min class id: {min(category_id_dict.keys())}"
    )
    print(
        f"Total number of frames: {total_frames_num}, "
        f"total number of labels: {total_labels_num}, "
        f"total number of objects: {total_objects_num}"
    )
    return gt_dict


def convert_video_to_images(
    video_name, gt_dict, video_path, path_to_output, overwrite_images
):
    """
    Convert a single video into frames (img1 folder) and write seqinfo.ini.
    """
    config = configparser.ConfigParser()
    output_path = os.path.join(path_to_output, video_name, "img1")
    config_file_path = os.path.join(path_to_output, video_name, "seqinfo.ini")

    # If not overwriting and existing frames + seqinfo look consistent, skip.
    if not overwrite_images and os.path.exists(config_file_path) and os.path.isdir(
        output_path
    ):
        config.read(config_file_path)
        try:
            count = config.getint("Sequence", "seqLength")
        except (configparser.Error, ValueError):
            count = -1
        if count == len(os.listdir(output_path)):
            return config

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0
    os.makedirs(output_path, exist_ok=True)
    image_shape = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        cv2.imwrite(f"{output_path}/{count:06d}.jpg", frame)

        if image_shape is None:
            image_shape = frame.shape
        else:
            if image_shape != frame.shape:
                print(
                    f"Image shape changed from {image_shape} "
                    f"to {frame.shape} at frame {count} in {video_name}."
                )
                image_shape = frame.shape

        if count == gt_dict[video_name]["frames"]:
            break

    cap.release()

    if image_shape is None:
        raise RuntimeError(f"No frames extracted from {video_path}")

    config["Sequence"] = {
        "name": video_name,
        "imDir": "img1",
        "frameRate": int(fps),
        "seqLength": count,
        "imWidth": image_shape[1],
        "imHeight": image_shape[0],
        "imExt": ".jpg",
    }

    with open(config_file_path, "w") as configfile:
        config.write(configfile)

    return config


def convert_all_videos_to_images(
    video_names, gt_dict, path_to_videos, path_to_output, overwrite_images=False
):
    """
    Convert all given videos in a dataset to frames and write seqinfo.ini
    for each of them.
    """
    videos_dict = {}
    for f in listdir(path_to_videos):
        full_path = join(path_to_videos, f)
        if isfile(full_path):
            videos_dict[os.path.splitext(f)[0]] = {"path": full_path}

    for video_name in video_names:
        if video_name not in videos_dict:
            print(
                f"Warning: video file for {video_name} not found in {path_to_videos}, skip."
            )
            continue
        video_path = videos_dict[video_name]["path"]
        config = convert_video_to_images(
            video_name, gt_dict, video_path, path_to_output, overwrite_images
        )
        frames = config.getint("Sequence", "seqLength")
        videos_dict[video_name]["frames"] = frames
        print(f"Converted {video_name} to {frames} images.")
    return videos_dict


def bbox_visualization_video(video_name, video_path, show_interval=10):
    """
    Visualize bounding boxes in a MOT-style sequence with OpenCV.
    Only show every `show_interval`-th frame.
    """

    def generate_random_colors(num_ids):
        # Fix the RNG seed for reproducible colors
        random.seed(0)
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
    gt_file_path = os.path.join(video_path, "gt", "gt.txt")

    if os.path.exists(gt_file_path):
        with open(gt_file_path, "r") as file:
            content_list = file.read().split("\n")
    else:
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


def convert_gt_to_yolox_and_ultralytics_format(split_dataset):
    """
    Generate:
      - COCO-like json (with videos / images / annotations)
      - YOLOX / ultralytics-style label .txt files
      - category-agnostic json (*_cag.json)
    """
    base_path = os.path.dirname(PATH_TO_OUTPUT[split_dataset]["train"])
    splits = PATH_TO_OUTPUT[split_dataset].keys()
    info_show = {split: "" for split in splits}
    ann_root = os.path.join(base_path, "annotations")
    os.makedirs(ann_root, exist_ok=True)

    for split in splits:
        data_path = os.path.join(base_path, split)
        if not os.path.exists(data_path):
            print(f"Path {data_path} does not exist, skipping...")
            continue

        out_path = os.path.join(ann_root, f"{split}.json")
        category_agnostic_out_path = os.path.join(ann_root, f"{split}_cag.json")

        out = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [
                {"id": i, "name": classes_dict[i]}
                for i in range(1, len(classes_dict) + 1)
            ],
        }

        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1

        # image_id -> list[annotation dict], used later to write YOLO label txt
        labels_per_image = {}

        seqs = sorted(os.listdir(data_path))
        for seq in seqs:
            seq_path = os.path.join(data_path, seq)
            img1_path = os.path.join(seq_path, "img1")
            ann_path = os.path.join(seq_path, "gt", "gt.txt")

            if not os.path.isdir(img1_path):
                print(f"Seq {seq} has no img1 directory, skip.")
                continue

            images_file_name = sorted(
                f for f in os.listdir(img1_path) if f.lower().endswith(".jpg")
            )
            num_images = len(images_file_name)
            if num_images == 0:
                print(f"Seq {seq} has no jpg images, skip.")
                continue

            config = configparser.ConfigParser()
            config.read(os.path.join(seq_path, "seqinfo.ini"))
            width = int(config["Sequence"]["imWidth"])
            height = int(config["Sequence"]["imHeight"])

            video_cnt += 1
            out["videos"].append({"id": video_cnt, "file_name": seq})

            # Map per-sequence frame_id -> global image_id
            img_num_in_dataset = {}
            for i, image_file_name in enumerate(images_file_name):
                local_frame_id = int(os.path.splitext(image_file_name)[0])
                global_image_id = image_cnt + i + 1
                img_num_in_dataset[local_frame_id] = global_image_id

                image_info = {
                    "file_name": f"{seq}/img1/{image_file_name}",
                    "id": global_image_id,
                    "frame_id": local_frame_id,
                    "video_id": video_cnt,
                    "height": height,
                    "width": width,
                }
                out["images"].append(image_info)

            print(f"{seq}: {num_images} images")
            image_cnt += num_images

            # Read per-sequence GT annotations
            if not os.path.exists(ann_path):
                print(f"Warning: gt file not found for {seq} at {ann_path}")
                continue

            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
            # Handle 1-row case (np.loadtxt returns 1D array)
            if anns.ndim == 1:
                anns = np.atleast_2d(anns)

            if anns.size == 0:
                print(f"Warning: empty ann file for {seq}")
                continue

            print(f"{seq}: {int(anns[:, 0].max())} ann frames")

            for row in anns:
                frame_local = int(row[0])
                track_id_raw = int(row[1])
                bbox_tlwh = row[2:6]
                conf = float(row[6])
                category_id = int(row[7])

                # Global track id (across all sequences)
                if track_id_raw != tid_last:
                    tid_curr += 1
                    tid_last = track_id_raw

                global_image_id = img_num_in_dataset.get(frame_local)
                # Some GT entries may refer to frames that do not exist in img1
                if global_image_id is None:
                    continue

                ann_cnt += 1
                ann = {
                    "id": ann_cnt,
                    "category_id": category_id,
                    "image_id": global_image_id,
                    "track_id": tid_curr,
                    "bbox": bbox_tlwh.tolist(),
                    "conf": conf,
                    "iscrowd": 0,
                    "area": float(bbox_tlwh[2] * bbox_tlwh[3]),
                }
                out["annotations"].append(ann)
                labels_per_image.setdefault(global_image_id, []).append(ann)

        # Write YOLO label .txt files (one per image)
        for image in out["images"]:
            image_id = image["id"]
            seq_img_rel_path = image["file_name"]  # e.g., MVI_xxx/img1/000001.jpg
            file_name = os.path.join(data_path, seq_img_rel_path)
            height, width = image["height"], image["width"]

            anns_for_image = labels_per_image.get(image_id, [])
            if not anns_for_image:
                gt_lines = []
            else:
                dw = 1.0 / width
                dh = 1.0 / height
                gt_lines = []
                for ann in anns_for_image:
                    x, y, w, h = ann["bbox"]
                    cx = (x + w / 2.0) * dw
                    cy = (y + h / 2.0) * dh
                    ww = w * dw
                    hh = h * dh
                    cls = int(ann["category_id"]) - 1  # YOLO class index starts from 0
                    gt_lines.append(
                        f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}"
                    )

            label_path = (
                file_name.replace("/images/", "/labels/")
                .replace("\\images\\", "\\labels\\")
                .replace(".jpg", ".txt")
            )
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w") as f:
                f.write("\n".join(gt_lines))

        info_show[split] = (
            f"loaded {split} for {len(out['images'])} images "
            f"and {len(out['annotations'])} samples"
        )
        print(info_show[split])

        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"Annotations saved to {out_path}")

        out_cag = copy.deepcopy(out)
        out_cag["categories"] = [{"id": 1, "name": "object"}]
        for ann in out_cag["annotations"]:
            ann["category_id"] = 1

        with open(category_agnostic_out_path, "w") as f:
            json.dump(out_cag, f)
        print(f"Category agnostic annotations saved to {category_agnostic_out_path}")

    print(f"\nDataset {split_dataset} loaded.")
    for split in splits:
        if info_show[split]:
            print(info_show[split])


def generate_train_val_test_splits(split_dataset, merge_train_val):
    """
    Copy Track folders into train / val / test / (train_val) splits
    according to SPLIT_DATASETS.
    """
    videos_path_dict = {
        split: {"video_name": [], "src": [], "dst": []}
        for split in PATH_TO_OUTPUT[split_dataset].keys()
    }

    # Remove old split directories if they exist
    for path in PATH_TO_OUTPUT[split_dataset].values():
        if os.path.exists(path):
            shutil.rmtree(path)

    for sub_dataset in SPLIT_DATASETS[split_dataset]["datasets"]:
        path_to_data = PATHS_TO_DATA[sub_dataset]["Data"]
        video_name_list = listdir(path_to_data)

        test_video_name_list = [
            video_name
            for video_name in video_name_list
            if video_name in SPLIT_DATASETS[split_dataset]["test"]
        ]
        val_video_name_list = [
            video_name
            for video_name in video_name_list
            if video_name in SPLIT_DATASETS[split_dataset]["val"]
        ]
        train_video_name_list = [
            video_name
            for video_name in video_name_list
            if video_name not in SPLIT_DATASETS[split_dataset]["test"]
            and video_name not in SPLIT_DATASETS[split_dataset]["val"]
        ]

        # Make sure we do not accidentally duplicate any video across splits
        assert all(
            video_name not in videos_path_dict["test"]["video_name"]
            for video_name in test_video_name_list
        ), "Overlapping videos in test split"
        assert all(
            video_name not in videos_path_dict["val"]["video_name"]
            for video_name in val_video_name_list
        ), "Overlapping videos in val split"
        assert all(
            video_name not in videos_path_dict["train"]["video_name"]
            for video_name in train_video_name_list
        ), "Overlapping videos in train split"

        videos_path_dict["test"]["video_name"] += test_video_name_list
        videos_path_dict["test"]["src"] += [
            os.path.join(path_to_data, video_name)
            for video_name in test_video_name_list
        ]
        videos_path_dict["test"]["dst"] += [
            os.path.join(PATH_TO_OUTPUT[split_dataset]["test"], video_name)
            for video_name in test_video_name_list
        ]

        videos_path_dict["val"]["video_name"] += val_video_name_list
        videos_path_dict["val"]["src"] += [
            os.path.join(path_to_data, video_name)
            for video_name in val_video_name_list
        ]
        videos_path_dict["val"]["dst"] += [
            os.path.join(PATH_TO_OUTPUT[split_dataset]["val"], video_name)
            for video_name in val_video_name_list
        ]

        videos_path_dict["train"]["video_name"] += train_video_name_list
        videos_path_dict["train"]["src"] += [
            os.path.join(path_to_data, video_name)
            for video_name in train_video_name_list
        ]
        videos_path_dict["train"]["dst"] += [
            os.path.join(PATH_TO_OUTPUT[split_dataset]["train"], video_name)
            for video_name in train_video_name_list
        ]

        if merge_train_val:
            videos_path_dict["train_val"]["video_name"] += (
                train_video_name_list + val_video_name_list
            )
            videos_path_dict["train_val"]["src"] += [
                os.path.join(path_to_data, video_name)
                for video_name in train_video_name_list + val_video_name_list
            ]
            videos_path_dict["train_val"]["dst"] += [
                os.path.join(PATH_TO_OUTPUT[split_dataset]["train_val"], video_name)
                for video_name in train_video_name_list + val_video_name_list
            ]

    # Copy directories to split destinations
    for split, info in videos_path_dict.items():
        print(f"Copying {split} data...")
        for src, dst in zip(info["src"], info["dst"]):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            assert os.path.basename(src) == os.path.basename(dst), "Video names do not match."
            print(f"  Adding {os.path.basename(src)} to {split}...")
            shutil.copytree(src, dst)

    for split, info in videos_path_dict.items():
        print(f"Total number of videos in {split}: {len(info['src'])}")
    return videos_path_dict


def generate_seqmaps_for_mot(split_dataset):
    """
    Generate MOTChallenge-style seqmaps files that list
    all sequences in each split.
    """
    base_path = os.path.dirname(PATH_TO_OUTPUT[split_dataset]["train"])
    splits = PATH_TO_OUTPUT[split_dataset].keys()
    for split in splits:
        data_path = os.path.join(str(base_path), split)
        if not os.path.exists(data_path):
            print(f"Path {data_path} does not exist, skipping...")
            continue
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
    split_data = args.split_data
    merge_train_val = args.merge_train_val
    convert_gt_to_coco_ultralytics = args.convert_gt_to_coco_ultralytics
    generate_seqmaps_flag = args.generate_seqmaps
    process_datasets = args.process_datasets

    assert process_datasets in ["VIS", "NIR", "VISNIR", "ALL"], "Invalid dataset"
    if process_datasets == "ALL":
        datasets = ["VIS", "NIR", "VISNIR"]
    else:
        datasets = [process_datasets]

    if convert_to_mot:
        for sub_dataset, paths in PATHS_TO_DATA.items():
            path_to_object_gt = paths["TrackGT"]
            path_to_videos = paths["Videos"]
            path_to_output = paths["Data"]
            gt_files_dict = generate_gt_files_dict(
                path_to_object_gt, gt_file_type="TrackGT"
            )

            print(f"Processing {sub_dataset}, converting to MOT format...")
            print(f"Loading GT files from {path_to_object_gt}...")
            gt_dict = convert_mat_to_gt(gt_files_dict, path_to_output=path_to_output)
            print("Converting videos to images...")
            videos_dict = convert_all_videos_to_images(
                gt_files_dict.keys(),
                gt_dict,
                path_to_videos,
                path_to_output,
                overwrite_images,
            )
            for video_name in gt_files_dict.keys():
                if video_name not in videos_dict:
                    print(f"Warning: {video_name} not in videos_dict.")
                    continue
                if "frames" not in videos_dict[video_name]:
                    print(
                        f"Warning: {video_name} has no 'frames' in videos_dict, "
                        "maybe video failed to convert."
                    )
                    continue
                try:
                    assert gt_dict[video_name]["frames"] == int(
                        videos_dict[video_name]["frames"]
                    )
                except AssertionError:
                    print(f"Frames do not match for {video_name}.")

    if split_data:
        for split_dataset in datasets:
            print(f"\nProcessing {split_dataset}, splitting data...")
            generate_train_val_test_splits(split_dataset, merge_train_val)

    if visualization:
        for sub_dataset, paths in PATHS_TO_DATA.items():
            path_to_object_gt = paths["TrackGT"]
            path_to_output = paths["Data"]
            gt_files_dict = generate_gt_files_dict(
                path_to_object_gt, gt_file_type="TrackGT"
            )
            video_name_dic = {sub_dataset: gt_files_dict}
            if sub_dataset in video_name_dic:
                for video_name in video_name_dic[sub_dataset]:
                    video_path = os.path.join(path_to_output, video_name)
                    bbox_visualization_video(video_name, video_path)

    if convert_gt_to_coco_ultralytics:
        for dataset in datasets:
            print(f"Processing {dataset} dataset, converting to COCO and ultralytics format...")
            convert_gt_to_yolox_and_ultralytics_format(dataset)

    if generate_seqmaps_flag:
        for dataset in datasets:
            print(f"Processing {dataset} dataset, generating seqmaps...")
            generate_seqmaps_for_mot(dataset)


def make_parser():
    parser = argparse.ArgumentParser("Singapore Maritime Dataset")

    parser.add_argument(
        "--overwrite_images",
        default=False,
        action="store_true",
        help="Overwrite existing extracted images",
    )
    parser.add_argument(
        "--convert_to_mot",
        default=False,
        action="store_true",
        help="Convert .mat GT files and videos to MOT format",
    )
    parser.add_argument(
        "--visualization",
        default=False,
        action="store_true",
        help="Visualize the dataset with bounding boxes",
    )
    parser.add_argument(
        "--merge_train_val",
        default=False,
        action="store_true",
        help="Additionally create a train_val split (train + val)",
    )
    parser.add_argument(
        "--split_data",
        default=False,
        action="store_true",
        help="Split the dataset into train / val / test (and train_val)",
    )
    parser.add_argument(
        "--generate_seqmaps",
        default=False,
        action="store_true",
        help="Generate MOTChallenge-style seqmaps files",
    )
    parser.add_argument(
        "--convert_gt_to_coco_ultralytics",
        default=False,
        action="store_true",
        help="Convert ground truth to COCO-like json + YOLO label files",
    )
    parser.add_argument(
        "--process_datasets",
        default="ALL",
        type=str,
        help="Which logical dataset to process: VIS, NIR, VISNIR, or ALL",
    )

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
