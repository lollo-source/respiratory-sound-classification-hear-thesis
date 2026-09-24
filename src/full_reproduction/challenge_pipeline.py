"""Staged public orchestration for official BioCAS Challenge Full Reproduction."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .challenge_compare import compare_published
from .challenge_data import TASKS, dataset_records, load_challenge, public_record_frame
from .challenge_downstream import run_challenge_downstream
from .challenge_features import extract_challenge, feature_paths
from .features import infer_batches, write_csv_gz, write_json
from .hear import load_model, release_model, sha256_file
from .pipeline import environment_payload
from src.progress import run_stage


DATASETS = ("biocas2022", "biocas2023")


def _records(release_root, sprsound_root):
    if not sprsound_root:
        raise RuntimeError("--sprsound-root is required")
    return load_challenge(sprsound_root, Path(release_root) / "manifests" / "biocas_records.csv.gz")


def _contract(release_root, records):
    payload = public_record_frame(records).to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {"source_manifest_sha256": sha256_file(Path(release_root) / "manifests" / "biocas_records.csv.gz"),
            "record_manifest_sha256": hashlib.sha256(payload).hexdigest()}


def preflight(release_root, root, sprsound_root):
    records = _records(release_root, sprsound_root); summaries = {}
    for dataset in DATASETS:
        subset = dataset_records(records, dataset)
        write_csv_gz(root / "datasets" / dataset / "record_manifest.csv.gz", public_record_frame(subset))
        summaries[dataset] = {"records": len(subset), "windows_expected": 15691 if dataset == "biocas2022" else 5452,
                              "splits": subset["official_split"].value_counts().sort_index().to_dict()}
    fold_sizes = {}
    train = records[records["official_split"] == "train2022"]
    for task, spec in TASKS.items():
        fold_sizes[task] = train[spec["fold_column"]].astype(int).value_counts().sort_index().to_dict()
    result = {"status": "complete", "records": len(records), "datasets": summaries,
              "fold_validation_sizes": fold_sizes, "seed": 42, "groups_are_release_local_pseudonyms": True}
    write_json(root / "preflight.json", result); return result


def first_batch(release_root, root, sprsound_root, model_path, hear_repo, device):
    records = _records(release_root, sprsound_root); out = root / "validation" / "first_batch"; result = {}
    for dataset in DATASETS:
        model, preprocess, torch_device, provenance = load_model(model_path, hear_repo, device)
        try:
            cls, tg, meta = infer_batches(dataset_records(records, dataset), "biocas", model, preprocess, torch_device, 1)
        finally: release_model(model)
        out.mkdir(parents=True, exist_ok=True); np.savez(out / (dataset + ".npz"), cls=cls, temporal_grid=tg)
        write_csv_gz(out / (dataset + "_metadata.csv.gz"), meta)
        result[dataset] = {"windows": len(cls), "cls_shape": list(cls.shape), "temporal_grid_shape": list(tg.shape),
                           "model": {k: v for k, v in provenance.items() if not k.endswith("_path")}}
    write_json(out / "summary.json", {"status": "complete", "datasets": result}); return result


def subset(release_root, root, sprsound_root, model_path, hear_repo, device):
    records = _records(release_root, sprsound_root); out = root / "validation" / "subset_repeatability"; result = {}
    for dataset in DATASETS:
        values = []
        for _ in range(2):
            model, preprocess, torch_device, _ = load_model(model_path, hear_repo, device)
            try: values.append(infer_batches(dataset_records(records, dataset), "biocas", model, preprocess, torch_device, 32))
            finally: release_model(model)
        a, b = values
        item = {"batches": 32, "windows": len(a[0]), "fresh_model_reload": True,
                "metadata_equal": bool(a[2].equals(b[2])), "cls_exact_equal": bool(np.array_equal(a[0], b[0])),
                "temporal_grid_exact_equal": bool(np.array_equal(a[1], b[1])),
                "cls_max_abs_diff": float(np.max(np.abs(a[0].astype(np.float64)-b[0].astype(np.float64)))),
                "temporal_grid_max_abs_diff": float(np.max(np.abs(a[1].astype(np.float64)-b[1].astype(np.float64))))}
        if not all(item[key] for key in ("metadata_equal", "cls_exact_equal", "temporal_grid_exact_equal")):
            raise RuntimeError("Challenge subset repeatability failed for %s" % dataset)
        result[dataset] = item; write_json(out / (dataset + ".json"), item)
    write_json(out / "summary.json", {"status": "complete", "datasets": result}); return result


def extract(release_root, root, sprsound_root, model_path, hear_repo, device, resume):
    if not (root / "validation" / "subset_repeatability" / "summary.json").is_file():
        raise RuntimeError("Challenge extraction requires successful subset stage")
    records = _records(release_root, sprsound_root); contract = _contract(release_root, records); result = {}
    for dataset in DATASETS:
        model, preprocess, torch_device, provenance = load_model(model_path, hear_repo, device)
        try:
            paths = extract_challenge(dataset_records(records, dataset), dataset,
                                      root / "datasets" / dataset / "features", model, preprocess, torch_device,
                                      {k: v for k, v in provenance.items() if not k.endswith("_path")}, contract, resume)
        finally: release_model(model)
        result[dataset] = paths["manifest"]
    return result


def downstream(release_root, root, sprsound_root):
    records = _records(release_root, sprsound_root)
    by_dataset = {dataset: dataset_records(records, dataset) for dataset in DATASETS}
    paths = {dataset: feature_paths(root / "datasets" / dataset / "features") for dataset in DATASETS}
    return run_challenge_downstream(by_dataset, paths, root / "downstream")


def run(args, release_root):
    root = Path(args.output_dir).expanduser().resolve() / "challenge"; root.mkdir(parents=True, exist_ok=True)
    extraction = replay = comparison = None
    progress = getattr(args, "progress", None)
    if args.stage in ("preflight", "all"):
        run_stage(progress, "preflight", lambda: preflight(release_root, root, args.sprsound_root))
    if args.stage in ("first-batch", "all"):
        run_stage(progress, "first-batch", lambda: first_batch(
            release_root, root, args.sprsound_root, args.hear_model_path, args.hear_repo, args.device
        ))
    if args.stage in ("subset", "all"):
        run_stage(progress, "subset", lambda: subset(
            release_root, root, args.sprsound_root, args.hear_model_path, args.hear_repo, args.device
        ))
    if args.stage in ("extract", "all"):
        extraction = run_stage(progress, "extract", lambda: extract(
            release_root, root, args.sprsound_root, args.hear_model_path, args.hear_repo, args.device, args.resume
        ))
    if args.stage in ("downstream", "all"):
        replay = run_stage(progress, "downstream", lambda: downstream(
            release_root, root, args.sprsound_root
        ))
    if args.stage in ("published", "all"):
        comparison = run_stage(progress, "published", lambda: compare_published(
            root / "downstream" / "metrics.csv",
            Path(release_root) / "published_challenge_reference" / "published_biocas_task2_scores.csv",
            root / "published_comparison",
        ))
    final = {"status": "complete", "scope": "HeAR official BioCAS Challenge", "stage": args.stage,
             "implemented": ["Task 2-1", "Task 2-2"], "out_of_scope_for_this_workflow": ["OPERA"],
             "environment": environment_payload(), "extraction": extraction, "downstream": replay,
             "published_comparison": comparison}
    write_json(root / "run_manifest.json", final); return final
