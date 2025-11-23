# -*- coding: utf-8 -*-
import argparse
import csv
import json

import os
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List

from loguru import logger

from tools.batch_run.tracking import tracking

# ============================== Utility functions ==============================
def _to_jsonable(obj):
    """Recursively convert numpy types to native Python types for json/csv/log output."""
    try:
        import numpy as _np
    except Exception:
        _np = None

    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _load_params(path: str) -> "OrderedDict[str, Dict[str, Any]]":
    """
    Load JSON/YAML params: { tracker_name: { param: value, ... } }.

    Keeps the tracker declaration order in the YAML/JSON (OrderedDict).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found: {path}")
    ext = os.path.splitext(path.lower())[1]
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("Reading YAML requires PyYAML: pip install pyyaml") from e

        class OrderedLoader(yaml.SafeLoader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return OrderedDict(loader.construct_pairs(node))

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping,
        )
        data = yaml.load(txt, Loader=OrderedLoader)
    elif ext == ".json":
        data = json.loads(txt, object_pairs_hook=OrderedDict)
    else:
        raise ValueError("Params file must be .json/.yaml/.yml")

    if not isinstance(data, dict):
        raise ValueError("Params root must be a mapping (tracker -> params)")

    clean = OrderedDict()
    for k, v in data.items():
        clean[k] = _to_jsonable(v)
    return clean


def _extract_metrics(tracking_out: Dict[str, Any]):
    """
    Extract HOTA/IDF1/MOTA from the third object returned by tracking(...).

    Compatible with your previous structure:
    - find a key containing 'Simple'
    - then read indices [1][1], [2][1], [3][1] as HOTA/IDF1/MOTA.
    """
    key_simple = [k for k in tracking_out.keys() if 'Simple' in k][0]
    hota = float(tracking_out[key_simple][1][1])
    idf1 = float(tracking_out[key_simple][2][1])
    mota = float(tracking_out[key_simple][3][1])
    return hota, idf1, mota


def _unique_preserve(seq: List[str]) -> List[str]:
    """Return unique elements while preserving their first-seen order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ============================== Single evaluation run ==============================
def _run_single_eval(exp: Dict[str, Any], static_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    exp: {
      'tracker_name', 'dataset_name', 'detector_category', 'tracker_params', 'data_split_name'
    }

    Returns: exp keys + evaluation metrics + error info (if any).
    """
    tracker_name = exp["tracker_name"]
    dataset_name = exp["dataset_name"]
    detector_category = exp["detector_category"]
    tracker_params = exp.get("tracker_params", {}) or {}
    split_name = exp["data_split_name"]

    # Assemble parameters for tracking(...)
    run_params = static_params.copy()
    run_params.update({
        'dataset_name': dataset_name,
        'tracker_name': tracker_name,
        'detector_single_cls': (detector_category == 'single_cls'),
        'tracker_update_params': tracker_params,
        'logger': logger,  # pass logger into tracking
        'data_split_name': split_name,  # explicitly set split per experiment
    })

    logger.info(f"[Eval] {tracker_name} | {dataset_name} | {detector_category} | split={split_name} | params={tracker_params}")
    try:
        _, _, result_obj = tracking(**run_params)
        if result_obj is None:
            return {
                "dataset_name": dataset_name,
                "data_split_name": split_name,
                "detector_category": detector_category,
                "tracker_name": tracker_name,
                "tracker_params": _to_jsonable(tracker_params),
                "HOTA": 0.0,
                "IDF1": 0.0,
                "MOTA": 0.0,
                "ok": False,
                "error": "None result returned from tracking()",
            }
        hota, idf1, mota = _extract_metrics(result_obj)
        return {
            "dataset_name": dataset_name,
            "data_split_name": split_name,
            "detector_category": detector_category,
            "tracker_name": tracker_name,
            "tracker_params": _to_jsonable(tracker_params),
            "HOTA": hota,
            "IDF1": idf1,
            "MOTA": mota,
            "ok": True,
            "error": "",
        }
    except Exception as e:
        logger.exception(f"[Eval][FAILED] {tracker_name} | {dataset_name} | {detector_category} | split={split_name}: {e}")
        return {
            "dataset_name": dataset_name,
            "data_split_name": split_name,
            "detector_category": detector_category,
            "tracker_name": tracker_name,
            "tracker_params": _to_jsonable(tracker_params),
            "HOTA": 0.0,
            "IDF1": 0.0,
            "MOTA": 0.0,
            "ok": False,
            "error": str(e),
        }


# ============================== Parallel executor ==============================
def _worker_entry(job: Dict[str, Any], static_params: Dict[str, Any],
                  blas_threads_per_proc: int, q):
    # Limit BLAS threads per process to avoid oversubscription
    if blas_threads_per_proc is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(blas_threads_per_proc))
        os.environ.setdefault("MKL_NUM_THREADS", str(blas_threads_per_proc))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads_per_proc))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(blas_threads_per_proc))

    out = _run_single_eval(job, static_params)
    q.put(out)


def launch_jobs_in_order(experiments: List[Dict[str, Any]],
                         static_params: Dict[str, Any],
                         num_processes: int,
                         blas_threads_per_proc: int) -> List[Dict[str, Any]]:
    """
    Launch all experiments, possibly in parallel.

    Even though processes finish in arbitrary order, we *preserve the input
    experiments order* by using a placeholder array and filling in results
    by matching (tracker, dataset, category, split).
    """

    if num_processes == 1:
        logger.info("[Main] num_processes=1, running sequentially without multiprocessing.")
        results_in_order: List[Dict[str, Any]] = []
        for exp in experiments:
            result = _run_single_eval(exp, static_params)
            results_in_order.append(result)
        return results_in_order

    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        mp.set_start_method('spawn', force=True)

    results_in_order: List[Dict[str, Any]] = [None] * len(experiments)  # placeholders
    q: mp.Queue = mp.Queue()
    running: List[mp.Process] = []
    idx = 0
    total = len(experiments)

    jobs = [{"__idx": i, **exp} for i, exp in enumerate(experiments)]

    while idx < total or running:
        # Launch new jobs until we hit num_processes or run out of experiments
        while idx < total and len(running) < num_processes:
            job = jobs[idx]
            p = mp.Process(target=_worker_entry,
                           args=(job, static_params, blas_threads_per_proc, q))
            p.start()
            logger.info(
                f"[Main] Launched PID={p.pid} for "
                f"{job['tracker_name']} | {job['dataset_name']} | {job['detector_category']} | split={job['data_split_name']}"
            )
            running.append(p)
            idx += 1

        # Collect finished results from the queue
        while not q.empty():
            r = q.get()
            # match (tracker, dataset, category, split) and fill first empty slot
            for j, exp in enumerate(experiments):
                if (results_in_order[j] is None and
                    r["tracker_name"] == exp["tracker_name"] and
                    r["dataset_name"] == exp["dataset_name"] and
                    r["detector_category"] == exp["detector_category"] and
                    r["data_split_name"] == exp["data_split_name"]):
                    results_in_order[j] = r
                    break

        # Clean up finished processes
        alive = []
        for p in running:
            if p.is_alive():
                alive.append(p)
            else:
                p.join(timeout=0.1)
                logger.info(f"[Main] PID={p.pid} finished.")
        running = alive

        time.sleep(0.1)

    # Drain remaining results from the queue if any
    while not q.empty():
        r = q.get()
        for j, exp in enumerate(experiments):
            if (results_in_order[j] is None and
                r["tracker_name"] == exp["tracker_name"] and
                r["dataset_name"] == exp["dataset_name"] and
                r["detector_category"] == exp["detector_category"] and
                r["data_split_name"] == exp["data_split_name"]):
                results_in_order[j] = r
                break

    # Fill in any missing entries with a default "no result returned"
    for j, exp in enumerate(experiments):
        if results_in_order[j] is None:
            results_in_order[j] = {
                "dataset_name": exp["dataset_name"],
                "data_split_name": exp["data_split_name"],
                "detector_category": exp["detector_category"],
                "tracker_name": exp["tracker_name"],
                "tracker_params": _to_jsonable(exp.get("tracker_params", {})),
                "HOTA": 0.0, "IDF1": 0.0, "MOTA": 0.0,
                "ok": False, "error": "no result returned"
            }
    return results_in_order


# ============================== Save CSV in ordered form ==============================
def save_csv_ordered(results: List[Dict[str, Any]],
                     out_csv: str,
                     datasets_order: List[str],
                     splits_order: List[str],
                     categories_order: List[str],
                     trackers_order: List[str]):
    """
    Save results to CSV with a specific ordering:

    - By dataset (datasets_order)
    - Then by split (splits_order)
    - Then by detector category (categories_order)
    - Then by tracker name (trackers_order, typically YAML order)
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Build index maps for sorting
    idx_d = {d: i for i, d in enumerate(datasets_order)}
    idx_s = {s: i for i, s in enumerate(splits_order)}
    idx_c = {c: i for i, c in enumerate(categories_order)}
    idx_t = {t: i for i, t in enumerate(trackers_order)}

    results_sorted = sorted(
        results,
        key=lambda r: (
            idx_d.get(r["dataset_name"], 10**9),
            idx_s.get(r.get("data_split_name", ""), 10**9),
            idx_c.get(r["detector_category"], 10**9),
            idx_t.get(r["tracker_name"], 10**9),
        )
    )

    # Column order
    fields = [
        "dataset_name", "data_split_name", "detector_category",
        "tracker_name", "HOTA", "IDF1", "MOTA", "ok", "error", "tracker_params"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results_sorted:
            row = dict(r)
            row["tracker_params"] = json.dumps(
                _to_jsonable(r.get("tracker_params", {})),
                ensure_ascii=False
            )
            w.writerow(row)

    logger.success(f"[Save] CSV  -> {out_csv}")


# ============================== JSON view (optional) ==============================
def save_json(results: List[Dict[str, Any]], out_path: str, meta: Dict[str, Any]):
    """
    Save results as a JSON file, including a 'meta' block describing the run.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "meta": _to_jsonable(meta),
        "results": _to_jsonable(results),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.success(f"[Save] JSON -> {out_path}")


# ============================== CLI ==============================
def make_parser():
    ps = argparse.ArgumentParser(
        "Batch evaluation for trackers with given params "
        "(CSV ordered by dataset->split->category, then tracker(YAML))"
    )
    ps.add_argument("--params", required=True, type=str,
                    help="Path to tracker params JSON/YAML (tracker -> params)")
    ps.add_argument("--detector_name", default='yolox_x', type=str)
    ps.add_argument("--dataset_names", default='SMDNIR', type=str)

    # Backward compatibility: support both old --data_split_name and
    # new --data_split_names (both can be comma-separated).
    ps.add_argument(
        "--data_split_name",
        default='val',
        type=str,
        help="Single or comma-separated split names (e.g., 'val,test'). "
             "Prefer --data_split_names for clarity."
    )
    ps.add_argument(
        "--data_split_names",
        default=None,
        type=str,
        help="Comma-separated split names (e.g., 'train,val,test'). "
             "If provided, overrides --data_split_name."
    )

    ps.add_argument("--tracker_names", default='sort,bytetrack', type=str)
    ps.add_argument("--detector_categories", default='single_cls', type=str)
    ps.add_argument("--num_processes", default=2, type=int)
    ps.add_argument("--blas_threads_per_proc", default=None, type=int)
    ps.add_argument("--storage_tag", default=None, type=str,
                    help="Optional tag for output filenames")
    ps.add_argument("--save_json", action="store_true",
                    help="Also save a JSON alongside CSV")
    ps.add_argument("--save_cache", action="store_true",
                    help="Whether to save intermediate cache files (may be large)")
    ps.add_argument("--device", default='0', type=str,
                    help="GPU device ID (default=0), only for detector/reid/gmc")
    return ps


# ============================== Main entry ==============================
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True)

    args = make_parser().parse_args()

    # 1) Load tracker params from YAML/JSON (preserve YAML order)
    tracker_params_map_od = _load_params(args.params)  # OrderedDict
    yaml_trackers_order = list(tracker_params_map_od.keys())

    # 2) Parse CLI lists
    cli_trackers = [s.strip() for s in args.tracker_names.split(',') if s.strip()]
    datasets_order = [s.strip() for s in args.dataset_names.split(',') if s.strip()]
    categories_order = [s.strip() for s in args.detector_categories.split(',') if s.strip()]

    # Splits: prefer --data_split_names if provided; otherwise parse --data_split_name (comma-separated)
    splits_raw = args.data_split_names if args.data_split_names is not None else args.data_split_name
    splits_order = _unique_preserve([s.strip() for s in splits_raw.split(',') if s.strip()])
    if not splits_order:
        splits_order = ['val']

    # 3) Final tracker order = YAML order intersected with CLI trackers (if any)
    if cli_trackers:
        trackers_order = [t for t in yaml_trackers_order if t in cli_trackers]
        # Add trackers that are only in CLI but not in YAML, with empty params
        for t in cli_trackers:
            if t not in tracker_params_map_od:
                tracker_params_map_od[t] = {}
                trackers_order.append(t)
    else:
        trackers_order = yaml_trackers_order

    # 4) Static parameters (passed to tracking)
    static_params = {
        # Data args
        'data_cache': False,
        'data_cache_type': 'disk',
        'data_pin_memory': False,
        'data_workers': 16,
        'data_batch_size': 1,
        'data_split_name': None,  # will be overridden per experiment

        # Detector args
        'detector_name': args.detector_name,
        'conf': None,
        'nms': None,
        'tsize': None,
        'detector_device': args.device,
        'detected_cache_path': None,
        'detector_joint_nir_vis_train': True,
        'detector_cache': True,
        'detector_update_params': {'class_agnostic': False},
        'eval_detector': False,

        # ReID / GMC / Tracker cache
        'reid_device': args.device,
        'reid_batch_size': 16,
        'reid_cache': True,
        'gmc_cache': True,
        'tracker_cache': False,

        # Eval args
        'eval_tracker': True,
        'eval_tracker_metrics': "HOTA CLEAR Identity",
        'vis_folder': None,
        'save_cache': False,
        'test_eval': False,
    }

    if args.save_cache:
        static_params['save_cache'] = True

    # 5) Build experiment list: tracker(YAML) × dataset(CLI) × split(CLI) × category(CLI)
    experiments: List[Dict[str, Any]] = []
    for t in trackers_order:
        t_params = tracker_params_map_od.get(t, {})
        for d in datasets_order:
            for s in splits_order:
                for c in categories_order:
                    experiments.append({
                        "tracker_name": t,
                        "dataset_name": d,
                        "detector_category": c,
                        "tracker_params": t_params,
                        "data_split_name": s,
                    })

    logger.info(
        f"[Main] Found {len(experiments)} experiments to run "
        f"(T={len(trackers_order)} × D={len(datasets_order)} × "
        f"S={len(splits_order)} × C={len(categories_order)})."
    )
    logger.info(f"[Main] Starting with {args.num_processes} concurrent processes.")

    # 6) Run experiments in parallel (but keep output aligned with input order)
    results_flat_in_order = launch_jobs_in_order(
        experiments=experiments,
        static_params=static_params,
        num_processes=args.num_processes,
        blas_threads_per_proc=args.blas_threads_per_proc
    )

    # 7) Save CSV (sorted by dataset->split->category, then tracker(YAML order))
    tag = args.storage_tag or (
        f"T_{'-'.join(trackers_order)}__D_{'-'.join(datasets_order)}__"
        f"S_{'-'.join(splits_order)}__Det_{args.detector_name}__C_{'-'.join(categories_order)}"
    )
    out_csv = os.path.join("tools", "batch_run", "res", f"batch_eval_{tag}.csv")
    save_csv_ordered(
        results=results_flat_in_order,
        out_csv=out_csv,
        datasets_order=datasets_order,
        splits_order=splits_order,
        categories_order=categories_order,
        trackers_order=trackers_order
    )

    # 8) Optional JSON (flat list, same order as CSV sorting)
    if args.save_json:
        idx_d = {d: i for i, d in enumerate(datasets_order)}
        idx_s = {s: i for i, s in enumerate(splits_order)}
        idx_c = {c: i for i, c in enumerate(categories_order)}
        idx_t = {t: i for i, t in enumerate(trackers_order)}

        results_sorted_for_json = sorted(
            results_flat_in_order,
            key=lambda r: (
                idx_d.get(r["dataset_name"], 10**9),
                idx_s.get(r.get("data_split_name", ""), 10**9),
                idx_c.get(r["detector_category"], 10**9),
                idx_t.get(r["tracker_name"], 10**9),
            )
        )
        out_json = os.path.join("tools", "batch_run", "res", f"batch_eval_{tag}.json")
        save_json(results_sorted_for_json, out_json, meta={
            "detector_name": args.detector_name,
            "datasets": datasets_order,
            "data_splits": splits_order,
            "trackers": trackers_order,
            "detector_categories": categories_order,
            "params_file": os.path.abspath(args.params),
            "num_processes": args.num_processes,
            "blas_threads_per_proc": args.blas_threads_per_proc,
            "order_policy": "dataset -> split -> category, then tracker (YAML order)",
        })

    logger.success("Batch evaluation completed.")
