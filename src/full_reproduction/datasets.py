"""Explicit, pseudonym-preserving adapters for HF Lung and SPRSound/BioCAS."""

from __future__ import print_function

from pathlib import Path

import numpy as np
import pandas as pd


HF_RECORDS = 9765
HF_TRAIN = 7809
HF_TEST = 1956
BIO_ALL_RECORDS = 3554
BIO_INCLUDED_RECORDS = 3323
BIO_WINDOWS = 19895


def _unique_named_files(directory, suffix):
    paths = sorted(directory.rglob("*" + suffix), key=lambda path: (path.name, str(path)))
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate filenames below explicit dataset directory: %s" % directory)
    return paths


def _folds(records, split_name, seed):
    from sklearn.model_selection import StratifiedGroupKFold

    frame = records.copy()
    frame["fold_id"] = -1
    train = frame[frame["split"].astype(str) == split_name]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_id, (_fit, validation) in enumerate(
        splitter.split(np.zeros((len(train), 1)), train["coarse_label"].astype(str), train["group_id"].astype(str))
    ):
        frame.loc[train.index[validation], "fold_id"] = int(fold_id)
    frame["fold_id"] = frame["fold_id"].astype(int)
    leakage = frame.loc[frame["fold_id"] >= 0].groupby("group_id")["fold_id"].nunique()
    if (leakage > 1).any():
        raise RuntimeError("group leakage across generated folds")
    return frame


def load_hf_lung(root, manifest_path):
    root = Path(root).expanduser().resolve()
    train_root = root / "train"
    test_root = root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise RuntimeError("HF root must contain train/ and test/ directories: %s" % root)
    train_paths = _unique_named_files(train_root, ".wav")
    test_paths = _unique_named_files(test_root, ".wav")
    if len(train_paths) != HF_TRAIN or len(test_paths) != HF_TEST:
        raise RuntimeError(
            "HF WAV count mismatch: train=%d/%d test=%d/%d"
            % (len(train_paths), HF_TRAIN, len(test_paths), HF_TEST)
        )
    manifest = pd.read_csv(manifest_path, dtype=str)
    if len(manifest) != HF_RECORDS or manifest["record_id"].duplicated().any():
        raise RuntimeError("invalid public HF record manifest")
    expected_split = ["test"] * HF_TEST + ["train"] * HF_TRAIN
    if manifest["split"].astype(str).tolist() != expected_split:
        raise RuntimeError("public HF manifest order/split contract mismatch")
    paths = test_paths + train_paths
    records = manifest.rename(columns={"binary_label": "binary_label", "coarse_label": "coarse_label"}).copy()
    records["audio_path"] = [str(path) for path in paths]
    records = _folds(records, "train", 20260712)
    return records


def load_biocas(root, manifest_path):
    root = Path(root).expanduser().resolve()
    locations = {
        "train2022": root / "BioCAS2022" / "train2022_wav",
        "test2022": root / "BioCAS2022" / "test2022_wav",
        "test2023": root / "BioCAS2023" / "test2023_wav",
    }
    missing = [name for name, path in locations.items() if not path.is_dir()]
    if missing:
        raise RuntimeError("SPRSound root is missing required directories: %s" % ", ".join(missing))
    by_stem = {}
    stem_split = {}
    for split_name, directory in locations.items():
        for path in _unique_named_files(directory, ".wav"):
            if path.stem in by_stem:
                raise RuntimeError("duplicate BioCAS recording stem: %s" % path.stem)
            by_stem[path.stem] = path
            stem_split[path.stem] = split_name
    if len(by_stem) != BIO_ALL_RECORDS:
        raise RuntimeError("BioCAS WAV count mismatch: %d/%d" % (len(by_stem), BIO_ALL_RECORDS))
    stem_to_pseudonym = {
        stem: "bio_record_%05d" % index
        for index, stem in enumerate(sorted(by_stem), start=1)
    }
    pseudonym_to_stem = {value: key for key, value in stem_to_pseudonym.items()}
    manifest = pd.read_csv(manifest_path, dtype=str)
    if len(manifest) != BIO_ALL_RECORDS or manifest["record_id"].duplicated().any():
        raise RuntimeError("invalid public BioCAS record manifest")
    if set(manifest["record_id"].astype(str)) != set(pseudonym_to_stem):
        raise RuntimeError("BioCAS raw inventory does not match the public pseudonymous universe")
    audio_paths = []
    for row in manifest.itertuples(index=False):
        stem = pseudonym_to_stem[str(row.record_id)]
        if stem_split[stem] != str(row.split):
            raise RuntimeError("BioCAS split mismatch for pseudonymous record %s" % row.record_id)
        audio_paths.append(str(by_stem[stem]))
    records = manifest.rename(columns={"patient_group": "group_id"}).copy()
    records["audio_path"] = audio_paths
    records = records[records["inclusion_status"].astype(str) == "included"].copy().reset_index(drop=True)
    if len(records) != BIO_INCLUDED_RECORDS:
        raise RuntimeError("BioCAS included-record count mismatch: %d" % len(records))
    records = _folds(records, "train2022", 20260717)
    return records


def public_record_frame(records):
    columns = ["record_id", "split", "group_id", "binary_label", "coarse_label", "fold_id"]
    # HF device is canonical public provenance needed to reconstruct the
    # acquisition-device distribution table. BioCAS has no corresponding
    # field, so preserve it only when supplied by the audited source manifest.
    if "device" in records.columns:
        columns.insert(3, "device")
    return records[columns].copy()
