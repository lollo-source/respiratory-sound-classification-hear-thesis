"""Fresh same-forward feature production for the official BioCAS Challenge."""

from __future__ import print_function

from pathlib import Path

import numpy as np
import pandas as pd

from . import BATCH_SIZE, CLS_DIM, TG_DIM
from .features import iter_full_batches, write_csv_gz, write_json
from .hear import infer_dual, sha256_file
from src.progress import LoopProgress, cache_reused


EXPECTED_WINDOWS = {"biocas2022": 15691, "biocas2023": 5452}


def _valid_cache(directory, dataset, provenance, contract):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("status") == "complete"
        and manifest.get("dataset") == dataset
        and manifest.get("same_forward_cls_and_temporal_grid") is True
        and manifest.get("global_record_major_stream") is True
        and int(manifest.get("windows", -1)) == EXPECTED_WINDOWS[dataset]
        and manifest.get("source_manifest_sha256") == contract["source_manifest_sha256"]
        and manifest.get("record_manifest_sha256") == contract["record_manifest_sha256"]
    ):
        return False
    for key in ("model_id", "model_revision", "weights_sha256", "config_sha256", "preprocessing_sha256"):
        if manifest.get("model", {}).get(key) != provenance.get(key):
            return False
    for name in ("cls_windows.npy", "temporal_grid_windows.npy", "window_metadata.csv.gz"):
        path = directory / name
        item = manifest.get("files", {}).get(name, {})
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            return False
        if sha256_file(path) != item.get("sha256"):
            return False
    return True


def feature_paths(directory):
    import json
    directory = Path(directory)
    return {
        "directory": directory,
        "manifest": json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
        "cls_windows": directory / "cls_windows.npy",
        "tg_windows": directory / "temporal_grid_windows.npy",
        "window_metadata": directory / "window_metadata.csv.gz",
    }


def extract_challenge(records, dataset, output_dir, model, preprocess, device, provenance, contract, resume=False):
    output_dir = Path(output_dir)
    expected = EXPECTED_WINDOWS[dataset]
    if resume and _valid_cache(output_dir, dataset, provenance, contract):
        cache_reused("%s Challenge extraction" % dataset)
        return feature_paths(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing nonempty Challenge feature directory without valid --resume cache: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", {"status": "incomplete", "dataset": dataset})
    cls_path = output_dir / "cls_windows.npy"
    tg_path = output_dir / "temporal_grid_windows.npy"
    cls = np.lib.format.open_memmap(cls_path, mode="w+", dtype=np.float32, shape=(expected, CLS_DIM))
    tg = np.lib.format.open_memmap(tg_path, mode="w+", dtype=np.float32, shape=(expected, TG_DIM))
    rows, cursor, batches = [], 0, 0
    extraction_progress = LoopProgress(
        "%s Challenge extraction" % dataset,
        (expected + BATCH_SIZE - 1) // BATCH_SIZE,
    )
    for waveforms, metadata in iter_full_batches(records, "biocas"):
        batch_cls, batch_tg = infer_dual(model, preprocess, waveforms, device)
        end = cursor + len(metadata)
        if [int(row["global_window_index"]) for row in metadata] != list(range(cursor, end)):
            raise RuntimeError("Challenge global record-major window order changed")
        cls[cursor:end], tg[cursor:end] = batch_cls, batch_tg
        rows.extend(metadata)
        cursor, batches = end, batches + 1
        extraction_progress.update(batches)
    if cursor != expected:
        raise RuntimeError("%s Challenge window count mismatch: %d/%d" % (dataset, cursor, expected))
    extraction_progress.complete()
    cls.flush(); tg.flush()
    metadata_path = output_dir / "window_metadata.csv.gz"
    metadata = pd.DataFrame(rows)
    if metadata["record_id"].astype(str).drop_duplicates().tolist() != records["record_id"].astype(str).tolist():
        raise RuntimeError("Challenge record/window ordering mismatch")
    write_csv_gz(metadata_path, metadata)
    files = {}
    for path in (cls_path, tg_path, metadata_path):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(output_dir / "manifest.json", {
        "status": "complete", "dataset": dataset, "records": int(len(records)), "windows": int(expected),
        "batch_size": BATCH_SIZE, "batch_count": batches,
        "final_batch_size": int(expected - (batches - 1) * BATCH_SIZE),
        "global_record_major_stream": True, "same_forward_cls_and_temporal_grid": True,
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "record_manifest_sha256": contract["record_manifest_sha256"], "model": provenance, "files": files,
    })
    return feature_paths(output_dir)
