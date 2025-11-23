import os
import argparse
import re

import cv2
import numpy as np
from tqdm import tqdm


def generate_trajectories(file_path: str, filter_visible: bool = True) -> np.ndarray:
    """
    Read MOT-style ground-truth or detection file and convert TLWH to TLBR.

    File format (per line):
        frame, id, x, y, w, h, score, class_id, visibility

    Parameters
    ----------
    file_path : str
        Path to the gt/det text file.
    filter_visible : bool
        If True, filter out objects with low visibility (visibility <= 0.4).

    Returns
    -------
    np.ndarray
        An array of shape [N, 9] with columns:
        [frame, id, x1, y1, x2, y2, score, class_id, visibility].
        Returns an empty array if no valid lines exist.
    """
    values = []

    with open(file_path, "r") as f:
        for line in f.read().split("\n"):
            line = line.strip()
            if not line:
                # skip empty lines
                continue
            split = line.split(",")
            if len(split) < 2:
                # skip malformed lines
                continue
            numbers = [float(i) for i in split]
            values.append(numbers)

    if not values:
        return np.array([], dtype=np.float32)

    values = np.array(values, dtype=np.float32)

    if filter_visible:
        # visibility is at index 8
        values = values[values[:, 8] > 0.4, :]

    if values.size != 0:
        # convert TLWH -> TLBR
        values[:, 4] += values[:, 2]  # x2 = x + w
        values[:, 5] += values[:, 3]  # y2 = y + h

    return values


def split_data(data, train_ratio: float):
    """
    Split a list of ids into train/test subsets in groups.

    The ids are divided into groups of size 5, then we assign the first
    round(5 * train_ratio) ids to train and the rest to test for each group.

    Parameters
    ----------
    data : list
        A list of identity ids.
    train_ratio : float
        Train ratio in [0, 1].

    Returns
    -------
    (list, list)
        train_ids, test_ids
    """
    train_list = []
    test_list = []
    group_size = 5

    if len(data) < group_size:
        group_size = len(data)

    train_size = round(group_size * train_ratio)
    test_size = group_size - train_size

    for i in range(0, len(data), group_size):
        group_data = data[i : i + group_size]
        # in case of last group with < group_size elements
        actual_group_size = len(group_data)
        actual_train_size = min(train_size, actual_group_size)
        train_data = group_data[:actual_train_size]
        test_data = group_data[actual_train_size:]

        assert len(train_data) + len(test_data) == actual_group_size

        train_list.extend(train_data)
        test_list.extend(test_data)

    return train_list, test_list


def get_gt_data(data_path: str, reid_img_dict: dict, id_offset: int):
    """
    Parse MOT sequences under `data_path` and collect ReID samples.

    For each sequence:
      - Read gt/gt.txt.
      - For each frame, collect bounding boxes and metadata.
      - Assign new global ids by offsetting per-sequence ids.

    Parameters
    ----------
    data_path : str
        Root directory of the MOT-formatted data.
    reid_img_dict : dict
        Nested dict {category_id: {new_id: [records...]}} to be filled.
    id_offset : int
        Current offset for identity ids (to make ids globally unique across seqs).

    Returns
    -------
    (dict, int)
        Updated reid_img_dict and updated id_offset.
    """
    seqs = sorted(os.listdir(data_path))
    for seq in seqs:
        print(f"Current sequence: {seq}")

        gt_path = os.path.join(data_path, seq, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            continue

        gt = generate_trajectories(gt_path, filter_visible=True)
        if gt.size == 0:
            continue

        images_path = os.path.join(data_path, seq, "img1")
        img_files = sorted(os.listdir(images_path))

        num_frames = len(img_files)
        max_id_per_seq = 0
        num_labels_per_seq = 0

        for img_file in img_files:
            frame_id = int(os.path.splitext(img_file)[0])
            # columns: [id, x1, y1, x2, y2, score, class_id, visibility]
            det = gt[frame_id == gt[:, 0], 1:].astype(np.int32)
            for d in range(det.shape[0]):
                id_ = det[d, 0]
                x1 = det[d, 1]
                y1 = det[d, 2]
                x2 = det[d, 3]
                y2 = det[d, 4]
                category_id = det[d, 6]

                num_labels_per_seq += 1
                max_id_per_seq = max(max_id_per_seq, id_)

                if category_id not in reid_img_dict:
                    reid_img_dict[category_id] = {}

                image_path = os.path.join(images_path, img_file)
                new_id = id_ + id_offset

                if new_id not in reid_img_dict[category_id]:
                    reid_img_dict[category_id][new_id] = []

                reid_img_dict[category_id][new_id].append(
                    {
                        "image_path": image_path,
                        "seq_name": seq,
                        "frame": frame_id,
                        "bbox": [x1, y1, x2, y2],
                        "new_id": new_id,
                        "type": "train",  # temporary, will be overwritten later
                    }
                )

        id_offset += max_id_per_seq
        print(
            f"Current id_offset: {id_offset}, "
            f"num_ids_per_seq: {max_id_per_seq}, "
            f"num_labels_per_seq: {num_labels_per_seq}, "
            f"num_frames: {num_frames}"
        )

    return reid_img_dict, id_offset


def generate_patches(
    reid_img_dict: dict,
    pattern: re.Pattern,
    train_save_path: str,
    gallery_save_path: str,
    query_save_path: str,
    train_ratio: float,
    query_ratio: float,
    save_patches: bool,
    num_data: dict,
):
    """
    Split trajectories into train / query / gallery and optionally save cropped patches.

    For each category:
      1) Sort ids by trajectory length.
      2) Use `split_data` to derive train ids and test ids.
      3) For train ids: all frames -> train.
         For test ids: the first (query_ratio * len(trajectory)) frames -> query,
                       the rest -> gallery.
      4) When `save_patches` is True, crop patches and save them to disk.
         Otherwise, only check that the expected files already exist.

    Parameters
    ----------
    reid_img_dict : dict
        Nested dict {category_id: {new_id: [frame_records...]}}.
    pattern : re.Pattern
        Compiled regex to rewrite sequence names for query camera ids.
    train_save_path : str
        Output directory for bounding_box_train.
    gallery_save_path : str
        Output directory for bounding_box_test.
    query_save_path : str
        Output directory for query.
    train_ratio : float
        Train ratio used in `split_data` for trajectory-level split.
    query_ratio : float
        Fraction of each test trajectory to assign to query (the rest to gallery).
    save_patches : bool
        If True, crop and write patches; if False, only verify files.
    num_data : dict
        Dict to store statistics per category (will be modified in-place).
    """
    for category_id in reid_img_dict:
        # sort ids by trajectory length (descending)
        sorted_new_ids = sorted(
            reid_img_dict[category_id],
            key=lambda k: len(reid_img_dict[category_id][k]),
            reverse=True,
        )

        train_new_ids_list, test_new_ids_list = split_data(
            sorted_new_ids, train_ratio
        )

        if category_id not in num_data:
            num_data[category_id] = {
                "train_trajectories": 0,
                "test_trajectories": 0,
                "train": 0,
                "query": 0,
                "gallery": 0,
            }

        num_data[category_id]["train_trajectories"] += len(train_new_ids_list)
        num_data[category_id]["test_trajectories"] += len(test_new_ids_list)

        # Collect samples grouped by image_path
        reid_res = {}
        for id_ in sorted_new_ids:
            trajectory = reid_img_dict[category_id][id_]
            trajectory_len = len(trajectory)

            for j in range(trajectory_len):
                img_dict_tmp = trajectory[j]
                image_path = img_dict_tmp["image_path"]
                reid_res.setdefault(image_path, [])

                if id_ in train_new_ids_list:
                    img_dict_tmp["type"] = "train"
                    num_data[category_id]["train"] += 1
                else:
                    if j < query_ratio * trajectory_len:
                        img_dict_tmp["type"] = "query"
                        num_data[category_id]["query"] += 1
                    else:
                        img_dict_tmp["type"] = "gallery"
                        num_data[category_id]["gallery"] += 1

                reid_res[image_path].append(img_dict_tmp)

        if save_patches:
            # Actually crop and save patches
            for image_path in tqdm(reid_res, desc=f"Saving patches (class {category_id})"):
                img = cv2.imread(image_path)
                if img is None:
                    print(f"ERROR: Empty frame: {image_path}")
                    continue

                H, W, _ = img.shape

                for img_dict_tmp in reid_res[image_path]:
                    assert img_dict_tmp["image_path"] == image_path
                    id_ = img_dict_tmp["new_id"]

                    # Clamp bbox to image boundaries
                    x1, y1, x2, y2 = img_dict_tmp["bbox"]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(x2, W)
                    y2 = min(y2, H)

                    patch = img[y1:y2, x1:x2, :]  # cropped patch

                    if img_dict_tmp["type"] == "train":
                        file_name = (
                            str(id_).zfill(7)
                            + "_"
                            + img_dict_tmp["seq_name"]
                            + "_"
                            + str(img_dict_tmp["frame"] + 1).zfill(7)
                            + "_acc_data.bmp"
                        )
                        cv2.imwrite(os.path.join(train_save_path, file_name), patch)

                    elif img_dict_tmp["type"] == "query":
                        # For query images, shift cam id by an offset
                        query_camid_offset = 2000
                        seq_name = img_dict_tmp["seq_name"]
                        match = pattern.search(seq_name)
                        if match is None:
                            raise ValueError(
                                f"Cannot parse camera id from seq_name: {seq_name}"
                            )
                        camid = int(match.group(2)) + query_camid_offset
                        separator = match.group(1)
                        seq_name = pattern.sub(
                            separator + str(camid), seq_name, count=1
                        )

                        file_name = (
                            str(id_).zfill(7)
                            + "_"
                            + seq_name
                            + "_"
                            + str(img_dict_tmp["frame"] + 1).zfill(7)
                            + "_acc_data.bmp"
                        )
                        cv2.imwrite(os.path.join(query_save_path, file_name), patch)

                    else:  # gallery
                        file_name = (
                            str(id_).zfill(7)
                            + "_"
                            + img_dict_tmp["seq_name"]
                            + "_"
                            + str(img_dict_tmp["frame"] + 1).zfill(7)
                            + "_acc_data.bmp"
                        )
                        cv2.imwrite(os.path.join(gallery_save_path, file_name), patch)
        else:
            # Only verify that expected patches already exist (sanity check mode)
            print("Skip saving images, checking file existence only...")
            for image_path in tqdm(reid_res, desc=f"Checking patches (class {category_id})"):
                for img_dict_tmp in reid_res[image_path]:
                    assert img_dict_tmp["image_path"] == image_path
                    id_ = img_dict_tmp["new_id"]

                    if img_dict_tmp["type"] == "train":
                        file_name = (
                            str(id_).zfill(7)
                            + "_"
                            + img_dict_tmp["seq_name"]
                            + "_"
                            + str(img_dict_tmp["frame"] + 1).zfill(7)
                            + "_acc_data.bmp"
                        )
                        path = os.path.join(train_save_path, file_name)
                        if not os.path.exists(path):
                            raise ValueError(f"File {path} does not exist")

                    elif img_dict_tmp["type"] == "query":
                        query_camid_offset = 2000
                        seq_name = img_dict_tmp["seq_name"]
                        match = pattern.search(seq_name)
                        if match is None:
                            raise ValueError(
                                f"Cannot parse camera id from seq_name: {seq_name}"
                            )
                        camid = int(match.group(2)) + query_camid_offset
                        separator = match.group(1)
                        seq_name = pattern.sub(
                            separator + str(camid), seq_name, count=1
                        )
                        file_name = (
                            str(id_).zfill(7)
                            + "_"
                            + seq_name
                            + "_"
                            + str(img_dict_tmp["frame"] + 1).zfill(7)
                            + "_acc_data.bmp"
                        )
                        path = os.path.join(query_save_path, file_name)
                        if not os.path.exists(path):
                            raise ValueError(f"File {path} does not exist")

                    else:  # gallery
                        file_name = (
                            str(id_).zfill(7)
                            + "_"
                            + img_dict_tmp["seq_name"]
                            + "_"
                            + str(img_dict_tmp["frame"] + 1).zfill(7)
                            + "_acc_data.bmp"
                        )
                        path = os.path.join(gallery_save_path, file_name)
                        if not os.path.exists(path):
                            raise ValueError(f"File {path} does not exist")

    # Print per-class statistics
    record = "Class Train_trajectories Test_trajectories Train Query Gallery\n"
    for category_id in num_data:
        record += (
            f'{category_id} '
            f'{num_data[category_id]["train_trajectories"]} '
            f'{num_data[category_id]["test_trajectories"]} '
            f'{num_data[category_id]["train"]} '
            f'{num_data[category_id]["query"]} '
            f'{num_data[category_id]["gallery"]}\n'
        )
    print(record)


def make_parser():
    parser = argparse.ArgumentParser("Sea Drone See ReID Dataset Builder")

    parser.add_argument(
        "--data_path",
        default="datasets/sds/images",
        help="Path to SDS MOT-style data root",
    )
    parser.add_argument(
        "--save_path",
        default="datasets/ReID",
        help="Path to save the SDS-ReID dataset",
    )
    parser.add_argument(
        "--dataset_type",
        default="Object",
        help="Dataset type, e.g. Object or Swimmer",
    )
    parser.add_argument(
        "--query_ratio",
        type=float,
        default=0.5,
        help="Query ratio for each test trajectory (0~1)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train ratio for each class (0~1, used when merge_train_val=True)",
    )
    parser.add_argument(
        "--save_patches",
        default=False,
        action="store_true",
        help="If set, crop and save patches; otherwise only check existing files",
    )
    parser.add_argument(
        "--merge_train_val",
        default=False,
        action="store_true",
        help="If set, use train_val split instead of separate train/val",
    )
    return parser


def main(args):
    if not args.merge_train_val:
        save_path = os.path.join(
            args.save_path, f"SDS{args.dataset_type}-ReID"
        )
    else:
        save_path = os.path.join(
            args.save_path, f"SDS{args.dataset_type}-TrainVal-ReID"
        )
        # If train_ratio == 1, there will be no query/gallery when merge_train_val
        assert args.train_ratio != 1.0, "train_ratio must be < 1 when merge_train_val=True"

    os.makedirs(save_path, exist_ok=True)

    train_save_path = os.path.join(save_path, "bounding_box_train")
    os.makedirs(train_save_path, exist_ok=True)

    gallery_save_path = os.path.join(save_path, "bounding_box_test")
    os.makedirs(gallery_save_path, exist_ok=True)

    query_save_path = os.path.join(save_path, "query")
    os.makedirs(query_save_path, exist_ok=True)

    num_data = {}
    pattern = re.compile(r"([\-_])(\d+)")

    if not args.merge_train_val:
        # 1) Use all train trajectories as training data.
        id_offset = 0
        reid_img_dict = {}
        print("Processing train split...")
        data_path = os.path.join(args.data_path, str(args.dataset_type), "train")
        reid_img_dict, id_offset = get_gt_data(data_path, reid_img_dict, id_offset)

        generate_patches(
            reid_img_dict,
            pattern,
            train_save_path,
            gallery_save_path,
            query_save_path,
            train_ratio=1.0,
            query_ratio=0.0,
            save_patches=args.save_patches,
            num_data=num_data,
        )

        # 2) Split val trajectories into query/gallery only.
        id_offset = 0
        reid_img_dict = {}
        print("Processing val split...")
        data_path = os.path.join(args.data_path, str(args.dataset_type), "val")
        reid_img_dict, id_offset = get_gt_data(data_path, reid_img_dict, id_offset)

        generate_patches(
            reid_img_dict,
            pattern,
            train_save_path,
            gallery_save_path,
            query_save_path,
            train_ratio=0.0,
            query_ratio=args.query_ratio,
            save_patches=args.save_patches,
            num_data=num_data,
        )
    else:
        # Use a merged train_val split.
        id_offset = 0
        reid_img_dict = {}
        print("Processing train_val split...")
        data_path = os.path.join(args.data_path, str(args.dataset_type), "train_val")
        reid_img_dict, id_offset = get_gt_data(data_path, reid_img_dict, id_offset)

        generate_patches(
            reid_img_dict,
            pattern,
            train_save_path,
            gallery_save_path,
            query_save_path,
            train_ratio=args.train_ratio,
            query_ratio=args.query_ratio,
            save_patches=args.save_patches,
            num_data=num_data,
        )

    # Global statistics
    num_train_data = sum(num_data[c]["train"] for c in num_data)
    num_query_data = sum(num_data[c]["query"] for c in num_data)
    num_gallery_data = sum(num_data[c]["gallery"] for c in num_data)

    print(
        f"Total number of train data: {num_train_data}",
        f"Total number of query data: {num_query_data}",
        f"Total number of gallery data: {num_gallery_data}, "
        f"Total number of data: {num_train_data + num_query_data + num_gallery_data}",
    )

    # Sanity checks: number of files == counted samples
    assert len(os.listdir(train_save_path)) == num_train_data, (
        f"len(os.listdir(train_save_path))={len(os.listdir(train_save_path))}, "
        f"num_train_data={num_train_data}"
    )
    assert len(os.listdir(query_save_path)) == num_query_data, (
        f"len(os.listdir(query_save_path))={len(os.listdir(query_save_path))}, "
        f"num_query_data={num_query_data}"
    )
    assert len(os.listdir(gallery_save_path)) == num_gallery_data, (
        f"len(os.listdir(gallery_save_path))={len(os.listdir(gallery_save_path))}, "
        f"num_gallery_data={num_gallery_data}"
    )


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
