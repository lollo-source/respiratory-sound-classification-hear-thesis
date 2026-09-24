"""Staged OPERA CE/CT/GT Full Reproduction orchestration."""

import json
from pathlib import Path

import numpy as np

from .datasets import load_biocas, public_record_frame
from .features import write_csv_gz, write_json
from .hear import sha256_file
from .opera import MODEL_SPECS, infer_records, load_model, release_model, validate_installation
from .opera_downstream import run_downstream
from .pipeline import environment_payload
from src.progress import cache_reused, run_stage


def _records(release_root, sprsound_root):
    if not sprsound_root: raise RuntimeError("--sprsound-root is required")
    return load_biocas(sprsound_root, Path(release_root) / "manifests" / "biocas_records.csv.gz")


def _paths(root, model_name):
    directory = Path(root) / "features" / model_name
    return directory, directory / "embeddings.npy", directory / "manifest.json"


def _manifest_hashes(release_root, root):
    source = Path(release_root) / "manifests" / "biocas_records.csv.gz"
    generated = Path(root) / "record_manifest.csv.gz"
    if not source.is_file() or not generated.is_file():
        raise RuntimeError("OPERA record manifests are incomplete; run preflight first")
    return {"source_manifest_sha256": sha256_file(source), "record_manifest_sha256": sha256_file(generated)}


def preflight(release_root, root, args):
    records = _records(release_root, args.sprsound_root); provenance = validate_installation(args.opera_root, args.opera_checkpoint_root)
    write_csv_gz(root / "record_manifest.csv.gz", public_record_frame(records))
    result = {"status": "complete", "records": len(records), "splits": records.split.value_counts().to_dict(),
              "classes": {"binary": records.binary_label.value_counts().to_dict(),
                          "coarse": records.coarse_label.value_counts().to_dict()}, "opera": provenance}
    write_json(root / "preflight.json", result); return result


def validation(release_root, root, args, repeat=False):
    records = _records(release_root, args.sprsound_root).head(8).copy(); destination = root / "validation" / ("repeatability" if repeat else "first_batch")
    result = {}
    for model_name in MODEL_SPECS:
        passes = []
        for _ in range(2 if repeat else 1):
            model, device = load_model(model_name, args.opera_root, args.opera_checkpoint_root, args.device)
            try: passes.append(infer_records(model, model_name, records, args.opera_root, device))
            finally: release_model(model)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / (model_name + ".npy"), passes[0])
        item = {"shape": list(passes[0].shape), "dtype": str(passes[0].dtype), "fresh_model_reload": repeat}
        if repeat:
            item.update({"exact_equal": bool(np.array_equal(passes[0], passes[1])),
                         "max_abs_diff": float(np.max(np.abs(passes[0].astype(float)-passes[1].astype(float))))})
            if not item["exact_equal"]: raise RuntimeError("OPERA repeatability failed for %s" % model_name)
        result[model_name] = item
    write_json(destination / "summary.json", {"status": "complete", "models": result}); return result


def extract(release_root, root, args):
    repeat_path = root / "validation" / "repeatability" / "summary.json"
    if not repeat_path.is_file():
        raise RuntimeError("OPERA full extraction requires successful repeatability stage")
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
    repeat_models = repeat.get("models", {})
    repeat_ok = repeat.get("status") == "complete" and all(
        repeat_models.get(name, {}).get("exact_equal") is True
        and repeat_models.get(name, {}).get("shape") == [8, spec["dimension"]]
        for name, spec in MODEL_SPECS.items()
    )
    if not repeat_ok:
        raise RuntimeError("OPERA repeatability evidence is incomplete or invalid")
    records = _records(release_root, args.sprsound_root)
    manifest_hashes = _manifest_hashes(release_root, root)
    result = {}
    for model_name, spec in MODEL_SPECS.items():
        directory, array_path, manifest_path = _paths(root, model_name)
        if args.resume and array_path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text());
            identity = (
                manifest.get("status") == "complete"
                and manifest.get("model") == model_name
                and manifest.get("records") == len(records)
                and manifest.get("shape") == [len(records), spec["dimension"]]
                and manifest.get("dtype") == "float32"
                and manifest.get("checkpoint_sha256") == spec["sha256"]
                and manifest.get("opera_revision") == validate_installation(
                    args.opera_root, args.opera_checkpoint_root
                )["revision"]
                and all(manifest.get(key) == value for key, value in manifest_hashes.items())
            )
            if identity and sha256_file(array_path) == manifest.get("embedding_sha256"):
                cache_reused("%s extraction" % spec["method"])
                result[model_name] = manifest; continue
        if directory.exists() and any(directory.iterdir()): raise RuntimeError("refusing nonempty OPERA feature directory without valid --resume")
        directory.mkdir(parents=True, exist_ok=True); write_json(manifest_path, {"status": "incomplete", "model": model_name})
        model, device = load_model(model_name, args.opera_root, args.opera_checkpoint_root, args.device)
        try: embeddings = infer_records(model, model_name, records, args.opera_root, device)
        finally: release_model(model)
        np.save(array_path, embeddings)
        manifest = {"status": "complete", "model": model_name, "method": spec["method"],
                    "records": len(records), "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
                    "input_seconds": 8.0, "sample_rate": 16000, "checkpoint_sha256": spec["sha256"],
                    "opera_revision": validate_installation(args.opera_root, args.opera_checkpoint_root)["revision"],
                    "preprocessing": "official OPERA librosa mono/16k, silence trim, mel-dB min-max, repeat padding",
                    "readout": "extract_feature for CE/CT; mean forward_feature over GT 50%-overlap splits",
                    "embedding_sha256": sha256_file(array_path), **manifest_hashes}
        write_json(manifest_path, manifest); result[model_name] = manifest
    return result


def downstream(release_root, root, args):
    records = _records(release_root, args.sprsound_root)
    expected_hashes = _manifest_hashes(release_root, root)
    paths = {}
    for name, spec in MODEL_SPECS.items():
        _directory, path, manifest_path = _paths(root, name)
        if not path.is_file() or not manifest_path.is_file():
            raise RuntimeError("fresh OPERA feature cache is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = (
            manifest.get("status") == "complete"
            and manifest.get("model") == name
            and manifest.get("records") == len(records)
            and manifest.get("shape") == [len(records), spec["dimension"]]
            and manifest.get("checkpoint_sha256") == spec["sha256"]
            and manifest.get("opera_revision") == validate_installation(
                args.opera_root, args.opera_checkpoint_root
            )["revision"]
            and all(manifest.get(key) == value for key, value in expected_hashes.items())
            and manifest.get("embedding_sha256") == sha256_file(path)
        )
        if not valid:
            raise RuntimeError("fresh OPERA feature provenance mismatch for %s" % name)
        paths[name] = path
    return run_downstream(records, paths, root / "downstream", args.hear_metrics)


def run(args, release_root):
    root = Path(args.output_dir).expanduser().resolve() / "opera"; root.mkdir(parents=True, exist_ok=True)
    result = None
    progress = getattr(args, "progress", None)
    if args.stage in ("preflight", "all"):
        result = run_stage(progress, "preflight", lambda: preflight(release_root, root, args))
    if args.stage in ("first-batch", "all"):
        result = run_stage(progress, "first-batch", lambda: validation(release_root, root, args, False))
    if args.stage in ("subset", "all"):
        result = run_stage(progress, "subset", lambda: validation(release_root, root, args, True))
    if args.stage in ("extract", "all"):
        result = run_stage(progress, "extract", lambda: extract(release_root, root, args))
    if args.stage in ("downstream", "all"):
        result = run_stage(progress, "downstream", lambda: downstream(release_root, root, args))
    final = {"status": "complete", "scope": "OPERA CE/CT/GT BioCAS harmonised thesis comparison",
             "stage": args.stage, "models": list(MODEL_SPECS), "result": result,
             "environment": environment_payload(),
             "out_of_scope_for_this_workflow": ["repository-wide clean-room certification"]}
    write_json(root / "run_manifest.json", final); return final
