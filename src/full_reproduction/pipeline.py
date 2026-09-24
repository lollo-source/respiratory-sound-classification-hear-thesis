"""Orchestration for auditable harmonised HeAR Full Reproduction stages."""

from __future__ import print_function

import json
import hashlib
import os
import platform
import sys
from pathlib import Path

import numpy as np

from .datasets import load_biocas, load_hf_lung, public_record_frame
from .downstream import run_downstream
from .features import (
    extract_dataset,
    infer_batches,
    load_feature_paths,
    write_csv_gz,
    write_json,
)
from .hear import load_model, release_model, sha256_file
from src.progress import run_stage


DATASET_KEYS = ("hf_lung", "biocas_harmonised")


def _cache_contract(release_root, dataset, records):
    source_name = "hf_lung_records.csv.gz" if dataset == "hf_lung" else "biocas_records.csv.gz"
    public_csv = public_record_frame(records).to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "source_manifest_sha256": sha256_file(Path(release_root) / "manifests" / source_name),
        "record_manifest_sha256": hashlib.sha256(public_csv).hexdigest(),
    }


def _load_records(release_root, dataset, hf_root, sprsound_root):
    manifests = Path(release_root) / "manifests"
    if dataset == "hf_lung":
        if hf_root is None:
            raise RuntimeError("--hf-lung-root is required")
        return load_hf_lung(hf_root, manifests / "hf_lung_records.csv.gz")
    if sprsound_root is None:
        raise RuntimeError("--sprsound-root is required")
    return load_biocas(sprsound_root, manifests / "biocas_records.csv.gz")


def _dataset_dir(output_root, dataset):
    return Path(output_root) / "datasets" / dataset


def _public_model_provenance(provenance):
    return {key: value for key, value in provenance.items() if not key.endswith("_path")}


def environment_payload():
    import pandas
    import scipy
    import sklearn
    import torch
    import transformers

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "numerical_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }


def preflight(release_root, output_root, datasets, hf_root, sprsound_root):
    summary = {}
    for dataset in datasets:
        records = _load_records(release_root, dataset, hf_root, sprsound_root)
        expected = 9765 if dataset == "hf_lung" else 3323
        if len(records) != expected:
            raise RuntimeError("record count mismatch for %s" % dataset)
        public = public_record_frame(records)
        dataset_dir = _dataset_dir(output_root, dataset)
        write_csv_gz(dataset_dir / "record_manifest.csv.gz", public)
        source_name = "hf_lung_records.csv.gz" if dataset == "hf_lung" else "biocas_records.csv.gz"
        summary[dataset] = {
            "records": len(records),
            "train_records": int((records["fold_id"].astype(int) >= 0).sum()),
            "evaluation_records": int((records["fold_id"].astype(int) < 0).sum()),
            "source_manifest": "manifests/" + source_name,
            "source_manifest_sha256": sha256_file(Path(release_root) / "manifests" / source_name),
            "record_ids_are_release_local_pseudonyms": True,
        }
    write_json(Path(output_root) / "preflight.json", {"status": "complete", "datasets": summary})
    return summary


def first_batch(release_root, output_root, datasets, hf_root, sprsound_root, model_path, hear_repo, device):
    validation_dir = Path(output_root) / "validation" / "first_batch"
    validation_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for dataset in datasets:
        records = _load_records(release_root, dataset, hf_root, sprsound_root)
        model, preprocess, torch_device, provenance = load_model(model_path, hear_repo, device)
        try:
            cls, tg, metadata = infer_batches(records, "hf_lung" if dataset == "hf_lung" else "biocas", model, preprocess, torch_device, 1)
        finally:
            release_model(model)
        np.savez(validation_dir / (dataset + ".npz"), cls=cls, temporal_grid=tg)
        write_csv_gz(validation_dir / (dataset + "_metadata.csv.gz"), metadata)
        result[dataset] = {
            "windows": len(cls),
            "cls_shape": list(cls.shape),
            "temporal_grid_shape": list(tg.shape),
            "model": _public_model_provenance(provenance),
        }
    write_json(validation_dir / "summary.json", {"status": "complete", "datasets": result})
    return result


def subset_repeatability(release_root, output_root, datasets, hf_root, sprsound_root, model_path, hear_repo, device):
    validation_dir = Path(output_root) / "validation" / "subset_repeatability"
    validation_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for dataset in datasets:
        records = _load_records(release_root, dataset, hf_root, sprsound_root)
        outputs = []
        for _pass in range(2):
            model, preprocess, torch_device, _provenance = load_model(model_path, hear_repo, device)
            try:
                cls, tg, metadata = infer_batches(
                    records,
                    "hf_lung" if dataset == "hf_lung" else "biocas",
                    model,
                    preprocess,
                    torch_device,
                    32,
                )
            finally:
                release_model(model)
            outputs.append((cls, tg, metadata))
        first, second = outputs
        metadata_equal = first[2].equals(second[2])
        result[dataset] = {
            "batches": 32,
            "windows": len(first[0]),
            "fresh_model_reload": True,
            "metadata_equal": bool(metadata_equal),
            "cls_exact_equal": bool(np.array_equal(first[0], second[0])),
            "temporal_grid_exact_equal": bool(np.array_equal(first[1], second[1])),
            "cls_max_abs_diff": float(np.max(np.abs(first[0].astype(np.float64) - second[0].astype(np.float64)))),
            "temporal_grid_max_abs_diff": float(np.max(np.abs(first[1].astype(np.float64) - second[1].astype(np.float64)))),
        }
        if not metadata_equal or not result[dataset]["cls_exact_equal"] or not result[dataset]["temporal_grid_exact_equal"]:
            raise RuntimeError("subset repeatability failed for %s" % dataset)
        write_json(validation_dir / (dataset + ".json"), result[dataset])
    write_json(validation_dir / "summary.json", {"status": "complete", "datasets": result})
    return result


def extract_full(release_root, output_root, datasets, hf_root, sprsound_root, model_path, hear_repo, device, resume):
    validation_summary = Path(output_root) / "validation" / "subset_repeatability" / "summary.json"
    if not validation_summary.is_file():
        raise RuntimeError("full extraction requires a successful subset-repeatability stage")
    results = {}
    for dataset in datasets:
        records = _load_records(release_root, dataset, hf_root, sprsound_root)
        model, preprocess, torch_device, provenance = load_model(model_path, hear_repo, device)
        try:
            paths = extract_dataset(
                records,
                "hf_lung" if dataset == "hf_lung" else "biocas",
                _dataset_dir(output_root, dataset) / "features",
                model,
                preprocess,
                torch_device,
                _public_model_provenance(provenance),
                _cache_contract(release_root, dataset, records),
                resume=resume,
            )
        finally:
            release_model(model)
        results[dataset] = paths["manifest"]
    return results


def downstream(output_root, datasets, release_root, hf_root, sprsound_root):
    results = {}
    for dataset in datasets:
        records = _load_records(release_root, dataset, hf_root, sprsound_root)
        paths = load_feature_paths(_dataset_dir(output_root, dataset) / "features")
        if paths["manifest"].get("status") != "complete":
            raise RuntimeError("feature cache is incomplete for %s" % dataset)
        result = run_downstream(
            "hf_lung" if dataset == "hf_lung" else "biocas",
            records,
            np.load(paths["single_cls"], mmap_mode="r"),
            np.load(paths["cls_record"], mmap_mode="r"),
            np.load(paths["tg_record"], mmap_mode="r"),
            _dataset_dir(output_root, dataset) / "downstream",
        )
        results[dataset] = result
    return results


def run(args, release_root):
    datasets = list(args.datasets)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    progress = getattr(args, "progress", None)
    if args.stage in ("preflight", "all"):
        run_stage(progress, "preflight", lambda: preflight(
            release_root, output_root, datasets, args.hf_lung_root, args.sprsound_root
        ))
    if args.stage in ("first-batch", "all"):
        run_stage(progress, "first-batch", lambda: first_batch(
            release_root, output_root, datasets, args.hf_lung_root, args.sprsound_root,
            args.hear_model_path, args.hear_repo, args.device,
        ))
    if args.stage in ("subset", "all"):
        run_stage(progress, "subset", lambda: subset_repeatability(
            release_root, output_root, datasets, args.hf_lung_root, args.sprsound_root,
            args.hear_model_path, args.hear_repo, args.device,
        ))
    extraction = None
    replay = None
    if args.stage in ("extract", "all"):
        extraction = run_stage(progress, "extract", lambda: extract_full(
            release_root, output_root, datasets, args.hf_lung_root, args.sprsound_root,
            args.hear_model_path, args.hear_repo, args.device, args.resume,
        ))
    if args.stage in ("downstream", "all"):
        replay = run_stage(progress, "downstream", lambda: downstream(
            output_root, datasets, release_root, args.hf_lung_root, args.sprsound_root
        ))
    final = {
        "status": "complete",
        "scope": "HeAR harmonised progression only",
        "datasets": datasets,
        "stage": args.stage,
        "implemented": ["HF Lung progression", "SPRSound/BioCAS harmonised progression"],
        "out_of_scope_for_this_workflow": ["official BioCAS Challenge", "OPERA"],
        "environment": environment_payload(),
        "extraction": extraction,
        "downstream": replay,
    }
    write_json(output_root / "run_manifest.json", final)
    return final
