import os
import copy
import random
import contextlib
import io
import itertools
import json
import tempfile
import shutil
from abc import ABCMeta, abstractmethod
from functools import partial, wraps
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import psutil
import torch
from loguru import logger
from pycocotools.coco import COCO
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader, default_collate, SequentialSampler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utility: pretty tables for per-class metrics
# ---------------------------------------------------------------------------

def list_to_table(data, title: str = "", separator: str = " ") -> str:
    """
    Format a 2D list into a human-readable, aligned text table.

    Parameters
    ----------
    data : list[list]
        Table content in row-major form.
    title : str
        Optional title shown above the table.
    separator : str
        Separator between columns.

    Returns
    -------
    str
        Formatted table string.
    """
    separator = " " + separator + " "
    max_lengths = [max(len(str(item)) for item in column) for column in zip(*data)]
    table = "\n"
    if title:
        table += title.center(sum(max_lengths) + len(separator) * (len(max_lengths) - 1)) + "\n"
    for row in data:
        formatted_row = [str(item).ljust(length) for item, length in zip(row, max_lengths)]
        table += separator.join(formatted_row) + "\n"
    return table


def per_class_AR_table(coco_eval, class_names, headers=None, colums=2):
    """
    Build per-class AR (Average Recall) table from COCOeval object.

    Parameters
    ----------
    coco_eval : COCOeval
        COCO evaluation result object (bbox mode).
    class_names : list[str]
        Category names in the same order as COCO categories.
    headers : list[str] or None
        Column headers, defaults to ["class", "AR50:95"].
    colums : int
        Number of columns in the tabulated view (for markdown).

    Returns
    -------
    str
        Tabulated AR table in markdown format (pipe style).
    """
    if headers is None:
        headers = ["class", "AR50:95"]

    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # recalls shape: [T x K x A x M] (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names + ["all"]):
        if name == "all":
            # all classes, all iou, area=0, max_dets=-1
            recall = recalls[:, :, 0, -1]
        else:
            recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100.0)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(
        *[result_pair[i::num_cols] for i in range(num_cols)]
    )
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=table_headers,
        numalign="left",
    )

    # Also log a simple text table with loguru
    info = [headers]
    for name in class_names + ["all"]:
        ap = per_class_AR[name]
        info.append([name, "-" if np.isnan(ap) else f"{ap:.3f}"])
    logger.info(list_to_table(info, separator=" "))

    return table


def per_class_AP_table(coco_eval, class_names, headers=None, colums=4):
    """
    Build per-class AP table from COCOeval object.

    AP metrics:
      - AP50     : IoU = 0.50
      - AP75     : IoU = 0.75
      - AP50:95  : mean over IoU in [0.50:0.95]

    Parameters
    ----------
    coco_eval : COCOeval
        COCO evaluation result object (bbox mode).
    class_names : list[str]
        Category names in the same order as COCO categories.
    headers : list[str] or None
        Column headers, defaults to ["class", "AP50", "AP75", "AP50:95"].
    colums : int
        Number of columns in the tabulated view (for markdown).

    Returns
    -------
    (str, list[float])
        - Markdown AP table string.
        - A list [AP50_all, AP75_all, AP50:95_all] for the "all" category.
    """
    if headers is None:
        headers = ["class", "AP50", "AP75", "AP50:95"]

    # iou index ranges for AP50, AP75, AP[.5:.95]
    iou_idxs = [(0, 1), (5, 6), (0, 10)]
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # precisions shape: [T x R x K x A x M]
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names + ["all"]):
        per_class_AP[name] = []
        for iou_idx in iou_idxs:
            if name == "all":
                precision = precisions[iou_idx[0]: iou_idx[1], :, :, 0, -1]
            else:
                precision = precisions[iou_idx[0]: iou_idx[1], :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_AP[name].append(float(ap * 100.0))

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for name, ap_list in per_class_AP.items() for x in [name] + ap_list]
    row_pair = itertools.zip_longest(
        *[result_pair[i::num_cols] for i in range(num_cols)]
    )
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=table_headers,
        numalign="left",
    )

    # Also log a simple text table with loguru
    info = [headers]
    for name in class_names + ["all"]:
        row = [name]
        for ap in per_class_AP[name]:
            row.append("-" if np.isnan(ap) else f"{ap:.3f}")
        info.append(row)
    logger.info(list_to_table(info, separator=" "))

    # Last row is 'all', skip the name
    return table, info[-1][1:]


# ---------------------------------------------------------------------------
# Dataset + caching
# ---------------------------------------------------------------------------

class Dataset(TorchDataset):
    """
    Thin wrapper over torch.utils.data.Dataset that stores the current input_dim.

    This is mainly used so that transforms can read a single source of truth
    for network input dimensions.
    """

    def __init__(self, input_dimension):
        """
        Parameters
        ----------
        input_dimension : tuple(int, int)
            (width, height) default network input dimensions.
        """
        super().__init__()
        self.__input_dim = input_dimension[:2]

    @property
    def input_dim(self):
        """
        Returns
        -------
        list[int, int]
            Current (width, height) input dimensions used by transforms.
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim


class CacheDataset(Dataset, metaclass=ABCMeta):
    """
    Dataset subclass that supports caching images to RAM or disk.

    Parameters
    ----------
    input_dimension : tuple(int, int)
        Default network input dimension (width, height).
    num_imgs : int
        Number of images in the dataset.
    data_dir : str
        Root directory of the dataset, e.g. '/path/to/COCO'.
    cache_dir_name : str
        Directory name used to store cached images on disk,
        e.g. 'custom_cache' => '/path/to/COCO/custom_cache'.
    path_filename : list[str]
        List of relative image paths (relative to data_dir).
    cache : bool
        If True, enable caching.
    cache_type : {'ram', 'disk'}
        Cache type:
          - 'ram' : cache images in memory.
          - 'disk': cache images as .npy files on disk.
    """

    def __init__(
        self,
        input_dimension,
        num_imgs=None,
        data_dir=None,
        cache_dir_name=None,
        path_filename=None,
        cache=False,
        cache_type="ram",
    ):
        super().__init__(input_dimension)
        self.cache = cache
        self.cache_type = cache_type
        self.num_threads_disk_write = 2

        if self.cache and self.cache_type == "disk":
            self.cache_dir = os.path.join(data_dir, cache_dir_name)
            self.path_filename = path_filename

        if self.cache and self.cache_type == "ram":
            self.imgs = None

        if self.cache:
            self.cache_images(
                num_imgs=num_imgs,
                data_dir=data_dir,
                cache_dir_name=cache_dir_name,
                path_filename=path_filename,
            )

    def __del__(self):
        # Explicitly free cached images in RAM
        if self.cache and self.cache_type == "ram":
            del self.imgs

    @abstractmethod
    def read_img(self, index):
        """
        Read image given index.

        Must be implemented in subclasses.

        Parameters
        ----------
        index : int
            Image index.

        Returns
        -------
        np.ndarray
            Loaded image (H, W, C).
        """
        raise NotImplementedError

    def cache_images(
        self,
        num_imgs=None,
        data_dir=None,
        cache_dir_name=None,
        path_filename=None,
    ):
        """
        Cache images to RAM or disk depending on `self.cache_type`.

        When caching to disk:
         - images are saved as .npy under self.cache_dir
        """
        assert num_imgs is not None, "num_imgs must be specified as the size of the dataset"
        if self.cache_type == "disk":
            assert (data_dir and cache_dir_name and path_filename) is not None, (
                "data_dir, cache_dir_name and path_filename must be specified "
                "if cache_type is 'disk'"
            )
            self.path_filename = path_filename

        mem = psutil.virtual_memory()
        mem_required = self.cal_cache_occupy(num_imgs)
        gb = 1 << 30

        # Decide whether RAM caching is feasible
        if self.cache_type == "ram":
            if mem_required > mem.available:
                logger.warning(
                    f"RAM cache disabled: required {mem_required / gb:.1f}GB, "
                    f"available {mem.available / gb:.1f}GB."
                )
                self.cache = False
            else:
                logger.info(
                    f"{mem_required / gb:.1f}GB RAM required, "
                    f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB RAM available.\n"
                    f"Note: since caching happens first, there is no guarantee that "
                    f"the remaining memory is sufficient for the rest of the training."
                )

        if not self.cache:
            return

        # Create cache container
        if self.cache_type == "ram":
            self.imgs = [None] * num_imgs
            logger.info("Using cached images in RAM to accelerate training!")
        else:  # 'disk'
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.warning(
                    "\n*******************************************************************\n"
                    "You are using cached images on DISK to accelerate training.\n"
                    "This requires large DISK space.\n"
                    f"Make sure you have {mem_required / gb:.1f} GB of free space.\n"
                    "*******************************************************************\n"
                )
            else:
                logger.info(f"Found disk cache at {self.cache_dir}")
                cache_exist_flags = []
                for i in range(num_imgs):
                    cache_filename = f'{self.path_filename[i].split(".")[0]}.npy'
                    cache_path_filename = os.path.join(self.cache_dir, cache_filename)
                    cache_exist_flags.append(os.path.exists(cache_path_filename))
                if all(cache_exist_flags) and len([i for i in cache_exist_flags if i]) == num_imgs:
                    logger.info("All images are already cached on disk, skip re-caching.")
                    return
                else:
                    logger.info("Not all images are cached on disk, re-caching...")
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                    os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("Caching images... This might take some time for your dataset.")

        num_threads = min(self.num_threads_disk_write, max(1, (os.cpu_count() or 1) - 1))
        bytes_cached = 0

        # NOTE: read_img is decorated with cache_read_img, we must force use_cache=False
        load_imgs = ThreadPool(num_threads).imap(
            partial(self.read_img, use_cache=False),
            range(num_imgs),
        )

        pbar = tqdm(enumerate(load_imgs), total=num_imgs)
        for i, img in pbar:
            if self.cache_type == "ram":
                self.imgs[i] = img
            else:  # 'disk'
                cache_filename = f'{self.path_filename[i].split(".")[0]}.npy'
                cache_path_filename = os.path.join(self.cache_dir, cache_filename)
                os.makedirs(os.path.dirname(cache_path_filename), exist_ok=True)
                np.save(cache_path_filename, img)

            bytes_cached += img.nbytes
            pbar.desc = (
                f"Caching images ({bytes_cached / gb:.1f}/{mem_required / gb:.1f}GB "
                f"{self.cache_type})"
            )
        pbar.close()

    def cal_cache_occupy(self, num_imgs: int) -> float:
        """
        Estimate memory cost for caching all images.

        It randomly samples up to 32 images, sums their sizes, and scales to
        the full dataset size.

        Parameters
        ----------
        num_imgs : int
            Total number of images.

        Returns
        -------
        float
            Estimated required bytes for caching all images.
        """
        cache_bytes = 0
        num_samples = min(num_imgs, 32)
        for _ in range(num_samples):
            img = self.read_img(index=random.randint(0, num_imgs - 1), use_cache=False)
            cache_bytes += img.nbytes
        mem_required = cache_bytes * num_imgs / max(num_samples, 1)
        return mem_required


def cache_read_img(use_cache: bool = True):
    """
    Decorator factory for caching-aware `read_img` functions.

    When `use_cache=True` and dataset.cache is enabled, images are read from
    RAM or disk cache instead of the original source.

    Usage
    -----
    @cache_read_img(use_cache=True)
    def read_img(self, index):
        ...
    """

    def decorator(read_img_fn):
        @wraps(read_img_fn)
        def wrapper(self, index, use_cache=use_cache):
            cache_enabled = self.cache and use_cache
            if cache_enabled:
                if self.cache_type == "ram":
                    img = self.imgs[index]
                    # avoid in-place modifications affecting cache
                    img = copy.deepcopy(img)
                elif self.cache_type == "disk":
                    img = np.load(
                        os.path.join(
                            self.cache_dir,
                            f"{self.path_filename[index].split('.')[0]}.npy",
                        )
                    )
                else:
                    raise ValueError(f"Unknown cache type: {self.cache_type}")
            else:
                img = read_img_fn(self, index)
            return img

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# COCO-style dataset with optional caching
# ---------------------------------------------------------------------------

class CustomDataset(CacheDataset):
    """
    COCO-style dataset with optional RAM/disk caching and COCO evaluation.

    - If `json_file` is provided, annotations are loaded from COCO json.
    - If `json_file` is None, dataset is treated as unlabeled image folder.
    """

    def __init__(
        self,
        data_dir=None,
        json_file=None,
        name="test",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
        load_data=True,
    ):
        if not load_data:
            cache = False
            logger.warning("load_data is set to False, cache will be disabled.")

        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.load_data = load_data

        if self.json_file is not None:
            # Standard COCO-style dataset
            self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
            self.ids = self.coco.getImgIds()
            self.class_ids = sorted(self.coco.getCatIds())
            self.cats = self.coco.loadCats(self.coco.getCatIds())
            self._classes = tuple([c["name"] for c in self.cats])
            self.annotations = self._load_coco_annotations()
        else:
            # Unlabeled dataset: just image paths
            self.coco = None
            self.cats = None
            self.annotations = self._load_coco_annotations()
            self.ids = [anno[1]["image_id"] for anno in self.annotations]

        assert len(self.ids) == len(self.annotations), (
            f"Error: {len(self.ids)} != {len(self.annotations)}"
        )

        self.num_imgs = len(self.ids)
        path_filename = [os.path.join(name, anno[1]["file_name"]) for anno in self.annotations]

        # Log basic dataset info
        num_anns = sum(len(annotation[0]) for annotation in self.annotations)
        num_videos = len(
            set(
                [
                    anno[1]["video_name"] if "video_name" in anno[1] else 0
                    for anno in self.annotations
                ]
            )
        )
        logger.info(
            f"Loaded {num_anns} annotations from {self.num_imgs} images in "
            f"{self.data_dir}/{self.name} with {num_videos} videos"
        )
        logger.info(f"image_id: {min(self.ids)} - {max(self.ids)}")

        video_ids = [
            anno[1]["video_id"]
            for anno in self.annotations
            if "video_id" in anno[1]
        ]
        logger.info(f"video_ids: {set(video_ids)}")

        categories = [
            category for anno in self.annotations for category in anno[0][:, 4]
        ]
        if self.cats is not None:
            logger.info(f"Cats with the specified ids: {self.cats}")
        logger.info(f"categories: {set(categories)}")

        track_ids = [
            track_id for anno in self.annotations for track_id in anno[0][:, 5]
        ]
        if len(track_ids) != 0:
            logger.info(f"track_ids: min {min(track_ids)}, max {max(track_ids)}")

        # Initialize caching
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type,
        )

    # --------------------- utilities ---------------------

    def get_image_paths(self):
        """
        Collect all image file paths under `data_dir / name`.

        Returns
        -------
        list[(video_name, relative_path, full_path)]
        """
        path = os.path.join(self.data_dir, self.name)
        image_paths = []
        image_extensions = (".jpg", ".jpeg", ".png", ".gif")

        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)
                    relative = os.path.relpath(file_path, path)
                    parts = relative.split(os.sep)
                    video_name = parts[0] if len(parts) > 1 else "None"
                    image_paths.append((video_name, relative, file_path))

        return image_paths

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        """
        Load annotations for all images.

        Returns
        -------
        list[(np.ndarray, dict)]
            List of (bbox_array, img_info) pairs.
        """
        if self.json_file is None:
            # Unlabeled: build minimal img_info only from image file paths
            annotations = []
            image_paths = sorted(self.get_image_paths(), key=lambda x: x[1])
            for image_id, (video_name, relative, _) in enumerate(image_paths):
                res = np.zeros((0, 6), dtype=np.float32)
                img_info = {
                    "image_size": None,
                    "file_name": relative,
                    "video_id": None,
                    "frame_id": None,
                    "image_id": image_id,
                    "video_name": video_name,
                    "split_name": "",
                }
                annotations.append((res, img_info))
            return annotations

        # Standard COCO json
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        """
        Load annotations for a single image by COCO image id.

        Parameters
        ----------
        id_ : int
            COCO image id.

        Returns
        -------
        (np.ndarray, dict)
            (labels, img_info)
        """
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))

            # Valid bounding boxes only
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj.get("track_id", -1)

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        img_info = {
            "image_size": (height, width),
            "file_name": file_name,
            "video_id": im_ann.get("video_id", -1),
            "frame_id": im_ann.get("frame_id", -1),
            "image_id": id_,
            "video_name": file_name.split("/")[0] if "/" in file_name else "None",
            "split_name": self.name,
        }

        assert id_ == im_ann["id"], f"Error: {id_} != {im_ann['id']}"
        return res, img_info

    # --------------------- image IO ---------------------

    def load_resized_img(self, index):
        """
        Load image and resize it so that the longer side matches `self.img_size`
        while keeping aspect ratio.
        """
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index, load=True):
        """
        Low-level image loader.

        Parameters
        ----------
        index : int
            Dataset index.
        load : bool
            If False, return a zero-sized image instead of reading from disk.

        Returns
        -------
        np.ndarray
            Loaded BGR image, or empty image if load=False.
        """
        file_name = self.annotations[index][1]["file_name"]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        if load:
            img = cv2.imread(img_file)
            assert img is not None, f"file named {img_file} not found"
        else:
            img = np.zeros((0, 0, 3), dtype=np.uint8)
        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        """
        Caching-aware image reader for a dataset index.
        """
        return self.load_image(index, load=self.load_data)

    @cache_read_img(use_cache=True)
    def read_img_from_id(self, image_id):
        """
        Caching-aware image reader for a COCO image id.
        """
        index = self.ids.index(image_id)
        return self.load_image(index, load=True)

    # --------------------- Dataset protocol ---------------------

    def __getitem__(self, index):
        if index >= len(self.ids):
            logger.warning(
                f"Index {index} is out of range. The length of self.ids is "
                f"{len(self.ids)}, using the last index instead."
            )
            index = len(self.ids) - 1

        image_id = self.ids[index]
        label, img_info = self.annotations[index]

        assert image_id == img_info["image_id"], (
            f"Error: {image_id} != {img_info['image_id']}"
        )

        img = self.read_img(index)
        target = copy.deepcopy(label)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        img_info["target"] = target
        return img, img_info

    # --------------------- collate functions ---------------------

    @staticmethod
    def custom_collate_merge(batch):
        """
        Collate function that merges nested containers but keeps dicts/lists.

        This can be useful when you want to keep a more flexible structure
        than the default PyTorch collate.
        """
        items = list(zip(*batch))

        for i in range(len(items)):
            if isinstance(items[i][0], (list, tuple, dict)):
                items[i] = list(items[i])
            else:
                items[i] = default_collate(items[i])

        return items

    @staticmethod
    def custom_collate(batch):
        """
        Collate function that returns raw batch (no merging).
        """
        return batch

    # --------------------- COCO evaluation ---------------------

    def evaluate_prediction(self, data_dict, statistics):
        """
        Evaluate detection results with COCO metrics.

        Parameters
        ----------
        data_dict : list[dict]
            List of detection results (COCO json style, but in-memory).
        statistics : tuple(float, float, int)
            (inference_time_sum, track_time_sum, num_samples)

        Returns
        -------
        (float, float, str, list[float])
            - mAP50:95
            - mAP50
            - info string (including per-class tables)
            - AP results for 'all' category: [AP50, AP75, AP50:95]
        """
        logger.info(f"Evaluate in total {len(data_dict)} annotations")

        annType = ["segm", "bbox", "keypoints"]

        inference_time, track_time, n_samples = statistics
        a_infer_time = 1000.0 * inference_time / max(n_samples, 1)
        a_track_time = 1000.0 * track_time / max(n_samples, 1)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        if len(data_dict) == 0:
            # No detections: return zeros and timing info only
            return 0.0, 0.0, info, [0.0, 0.0, 0.0]

        if self.coco is None:
            logger.warning("self.coco is None, cannot run COCO evaluation.")
            return 0.0, 0.0, info, [0.0, 0.0, 0.0]

        cocoGt = self.coco

        # pycocotools can't handle python dict directly in some versions,
        # so we dump to a temp json file.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(data_dict, tmp_file)
            tmp_file_path = tmp_file.name

        cocoDt = cocoGt.loadRes(tmp_file_path)
        from yolox.layers import COCOeval_opt as COCOeval

        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info += redirect_string.getvalue()

        cat_ids = list(cocoGt.cats.keys())
        cat_names = [cocoGt.cats[cat_id]["name"] for cat_id in sorted(cat_ids)]

        AP_table, AP_res = per_class_AP_table(cocoEval, class_names=cat_names)
        info += "per class AP:\n" + AP_table + "\n"

        AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
        info += "per class AR:\n" + AR_table + "\n"

        # cocoEval.stats:
        # [0] mAP50:95, [1] mAP50, [2] mAP75, ...
        return cocoEval.stats[0], cocoEval.stats[1], info, AP_res


# ---------------------------------------------------------------------------
# Data pipeline wrapper
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Simple wrapper that builds a CustomDataset + DataLoader pair.

    This is mainly for convenient construction of evaluation / inference
    dataloaders without touching training code.
    """

    def __init__(
        self,
        data_dir,
        json_file,
        name,
        img_size=(416, 416),
        preproc=None,
        cache=True,
        cache_type="disk",
        load_data=True,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    ):
        self.dataset = CustomDataset(
            data_dir=data_dir,
            json_file=json_file,
            name=name,
            img_size=img_size,
            preproc=preproc,
            cache=cache,
            cache_type=cache_type,
            load_data=load_data,
        )

        self.sampler = SequentialSampler(self.dataset)
        self.dataloader = TorchDataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.sampler,
            pin_memory=pin_memory,
            collate_fn=self.dataset.custom_collate,
        )

    def __call__(self):
        """
        Returns
        -------
        (CustomDataset, DataLoader)
        """
        return self.dataset, self.dataloader
