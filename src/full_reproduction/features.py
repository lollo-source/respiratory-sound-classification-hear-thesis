"""Canonical global-stream feature production and regenerable cache handling."""

from __future__ import print_function

import gzip
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import BATCH_SIZE, CLS_DIM, TG_DIM, WINDOW_SAMPLES
from .audio import complete_record_windows, hf_fixed_windows, load_resampled, maximum_energy_window
from .hear import infer_dual, sha256_file
from src.progress import LoopProgress, cache_reused


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_csv_gz(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")
    os.replace(str(temporary), str(path))


def _record_windows(record, dataset):
    source_rate, audio = load_resampled(record.audio_path)
    windows = hf_fixed_windows(audio) if dataset == "hf_lung" else complete_record_windows(audio)
    rows = []
    for index, window in enumerate(windows):
        start = index * WINDOW_SAMPLES
        valid = max(0, min(WINDOW_SAMPLES, len(audio) - start))
        rows.append(
            {
                "record_id": str(record.record_id),
                "group_id": str(record.group_id),
                "split": str(record.split),
                "window_index": int(index),
                "valid_samples": int(valid),
                "right_padding_samples": int(WINDOW_SAMPLES - valid),
                "source_sample_rate": int(source_rate),
            }
        )
    return windows, rows


def iter_full_batches(records, dataset, limit_batches=None):
    pending = []
    metadata = []
    global_index = 0
    yielded = 0
    for record in records.itertuples(index=False):
        windows, rows = _record_windows(record, dataset)
        for waveform, row in zip(windows, rows):
            row["global_window_index"] = int(global_index)
            global_index += 1
            pending.append(waveform)
            metadata.append(row)
            if len(pending) == BATCH_SIZE:
                yield np.stack(pending).astype(np.float32), metadata
                yielded += 1
                if limit_batches is not None and yielded >= limit_batches:
                    return
                pending, metadata = [], []
    if pending and (limit_batches is None or yielded < limit_batches):
        yield np.stack(pending).astype(np.float32), metadata


def iter_single_batches(records, limit_batches=None):
    pending = []
    metadata = []
    yielded = 0
    for record_index, record in enumerate(records.itertuples(index=False)):
        source_rate, audio = load_resampled(record.audio_path)
        waveform, start = maximum_energy_window(audio)
        pending.append(waveform)
        metadata.append(
            {
                "record_index": int(record_index),
                "record_id": str(record.record_id),
                "group_id": str(record.group_id),
                "split": str(record.split),
                "crop_start_sample": int(start),
                "crop_end_sample": int(min(start + WINDOW_SAMPLES, len(audio))),
                "padding_samples": int(max(0, WINDOW_SAMPLES - len(audio))),
                "source_sample_rate": int(source_rate),
            }
        )
        if len(pending) == BATCH_SIZE:
            yield np.stack(pending).astype(np.float32), metadata
            yielded += 1
            if limit_batches is not None and yielded >= limit_batches:
                return
            pending, metadata = [], []
    if pending and (limit_batches is None or yielded < limit_batches):
        yield np.stack(pending).astype(np.float32), metadata


def infer_batches(records, dataset, model, preprocess, device, limit_batches):
    cls_parts = []
    tg_parts = []
    rows = []
    for waveforms, metadata in iter_full_batches(records, dataset, limit_batches=limit_batches):
        cls, tg = infer_dual(model, preprocess, waveforms, device)
        cls_parts.append(cls)
        tg_parts.append(tg)
        rows.extend(metadata)
    return np.concatenate(cls_parts), np.concatenate(tg_parts), pd.DataFrame(rows)


def _aggregate_records(values, counts):
    output = np.empty((len(counts), values.shape[1]), dtype=np.float32)
    cursor = 0
    for index, count in enumerate(counts):
        end = cursor + int(count)
        output[index] = np.asarray(values[cursor:end]).mean(axis=0).astype(np.float32)
        cursor = end
    if cursor != len(values):
        raise RuntimeError("record aggregation did not consume every window")
    return output


def _cache_complete(directory, expected_windows, expected_records, dataset, model_provenance, cache_contract):
    manifest_path = Path(directory) / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_ok = (
        manifest.get("status") == "complete"
        and manifest.get("dataset") == dataset
        and int(manifest.get("windows", -1)) == int(expected_windows)
        and int(manifest.get("records", -1)) == int(expected_records)
        and int(manifest.get("batch_size", -1)) == int(BATCH_SIZE)
        and manifest.get("global_record_major_stream") is True
        and manifest.get("same_forward_cls_and_temporal_grid") is True
        and manifest.get("record_aggregation") == "arithmetic_mean"
        and manifest.get("source_manifest_sha256") == cache_contract["source_manifest_sha256"]
        and manifest.get("record_manifest_sha256") == cache_contract["record_manifest_sha256"]
    )
    if not identity_ok:
        return False
    model = manifest.get("model", {})
    for key in ("model_id", "model_revision", "weights_sha256", "config_sha256", "preprocessing_sha256"):
        if model.get(key) != model_provenance.get(key):
            return False
    files = manifest.get("files", {})
    required = (
        "cls_windows.npy",
        "temporal_grid_windows.npy",
        "cls_record_mean.npy",
        "temporal_grid_record_mean.npy",
        "single_window_cls.npy",
        "window_metadata.csv.gz",
        "single_window_metadata.csv.gz",
    )
    for name in required:
        path = Path(directory) / name
        expected = files.get(name, {})
        if not path.is_file() or path.stat().st_size != int(expected.get("bytes", -1)):
            return False
        if sha256_file(path) != expected.get("sha256"):
            return False
    return True


def extract_dataset(
    records,
    dataset,
    output_dir,
    model,
    preprocess,
    device,
    model_provenance,
    cache_contract,
    resume=False,
):
    output_dir = Path(output_dir)
    expected_records = len(records)
    expected_windows = expected_records * 8 if dataset == "hf_lung" else 19895
    counts = [8] * expected_records if dataset == "hf_lung" else None
    if resume and _cache_complete(
        output_dir,
        expected_windows,
        expected_records,
        dataset,
        model_provenance,
        cache_contract,
    ):
        cache_reused("%s extraction" % ("HF Lung" if dataset == "hf_lung" else "BioCAS harmonised"))
        return load_feature_paths(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing nonempty feature directory without a valid --resume cache: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", {"status": "incomplete", "dataset": dataset})

    cls_path = output_dir / "cls_windows.npy"
    tg_path = output_dir / "temporal_grid_windows.npy"
    cls = np.lib.format.open_memmap(cls_path, mode="w+", dtype=np.float32, shape=(expected_windows, CLS_DIM))
    tg = np.lib.format.open_memmap(tg_path, mode="w+", dtype=np.float32, shape=(expected_windows, TG_DIM))
    rows = []
    cursor = 0
    batch_count = 0
    display_name = "HF Lung" if dataset == "hf_lung" else "BioCAS harmonised"
    full_progress = LoopProgress(
        "%s full-window extraction" % display_name,
        (expected_windows + BATCH_SIZE - 1) // BATCH_SIZE,
    )
    for waveforms, metadata in iter_full_batches(records, dataset):
        batch_cls, batch_tg = infer_dual(model, preprocess, waveforms, device)
        end = cursor + len(metadata)
        expected_indices = list(range(cursor, end))
        actual_indices = [int(row["global_window_index"]) for row in metadata]
        if actual_indices != expected_indices:
            raise RuntimeError("global record-major window order changed")
        cls[cursor:end] = batch_cls
        tg[cursor:end] = batch_tg
        rows.extend(metadata)
        cursor = end
        batch_count += 1
        full_progress.update(batch_count)
    if cursor != expected_windows:
        raise RuntimeError("window count mismatch: %d/%d" % (cursor, expected_windows))
    full_progress.complete()
    cls.flush()
    tg.flush()
    metadata_path = output_dir / "window_metadata.csv.gz"
    metadata_frame = pd.DataFrame(rows)
    write_csv_gz(metadata_path, metadata_frame)

    if counts is None:
        observed = metadata_frame.groupby("record_id", sort=False).size()
        counts = [int(observed.get(record_id, 0)) for record_id in records["record_id"].astype(str)]
        if not counts or min(counts) < 1 or sum(counts) != expected_windows:
            raise RuntimeError("BioCAS per-record window inventory mismatch")

    cls_record = _aggregate_records(cls, counts)
    tg_record = _aggregate_records(tg, counts)
    cls_record_path = output_dir / "cls_record_mean.npy"
    tg_record_path = output_dir / "temporal_grid_record_mean.npy"
    np.save(str(cls_record_path), cls_record)
    np.save(str(tg_record_path), tg_record)

    single_path = output_dir / "single_window_cls.npy"
    single = np.lib.format.open_memmap(single_path, mode="w+", dtype=np.float32, shape=(expected_records, CLS_DIM))
    single_rows = []
    single_cursor = 0
    single_batch_count = 0
    single_progress = LoopProgress(
        "%s single-window extraction" % display_name,
        (expected_records + BATCH_SIZE - 1) // BATCH_SIZE,
    )
    for waveforms, metadata in iter_single_batches(records):
        batch_cls, _batch_tg = infer_dual(model, preprocess, waveforms, device)
        end = single_cursor + len(metadata)
        single[single_cursor:end] = batch_cls
        single_rows.extend(metadata)
        single_cursor = end
        single_batch_count += 1
        single_progress.update(single_batch_count)
    if single_cursor != expected_records:
        raise RuntimeError("single-window record count mismatch")
    single_progress.complete()
    single.flush()
    single_metadata_path = output_dir / "single_window_metadata.csv.gz"
    write_csv_gz(single_metadata_path, pd.DataFrame(single_rows))

    files = {}
    for path in [cls_path, tg_path, cls_record_path, tg_record_path, single_path, metadata_path, single_metadata_path]:
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "status": "complete",
        "dataset": dataset,
        "records": int(expected_records),
        "windows": int(expected_windows),
        "batch_size": BATCH_SIZE,
        "batch_count": int(batch_count),
        "final_batch_size": int(expected_windows - (batch_count - 1) * BATCH_SIZE),
        "global_record_major_stream": True,
        "same_forward_cls_and_temporal_grid": True,
        "record_aggregation": "arithmetic_mean",
        "source_manifest_sha256": cache_contract["source_manifest_sha256"],
        "record_manifest_sha256": cache_contract["record_manifest_sha256"],
        "model": model_provenance,
        "files": files,
    }
    write_json(output_dir / "manifest.json", manifest)
    return load_feature_paths(output_dir)


def load_feature_paths(directory):
    directory = Path(directory)
    return {
        "directory": directory,
        "manifest": json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
        "cls_windows": directory / "cls_windows.npy",
        "tg_windows": directory / "temporal_grid_windows.npy",
        "cls_record": directory / "cls_record_mean.npy",
        "tg_record": directory / "temporal_grid_record_mean.npy",
        "single_cls": directory / "single_window_cls.npy",
        "window_metadata": directory / "window_metadata.csv.gz",
        "single_metadata": directory / "single_window_metadata.csv.gz",
    }
