"""Public official-BioCAS Challenge adapter with release-local identifiers."""

from __future__ import print_function

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


EXPECTED_RAW_RECORDS = 3554
EXPECTED_COUNTS = {
    "train2022": 1949,
    "test_inter": 355,
    "test_intra": 379,
    "test2023": 871,
}
TASKS = {
    "T2-1": {
        "column": "challenge_t2_1_label",
        "fold_column": "fold_t2_1",
        "classes": ["Normal", "Adventitious", "Poor Quality"],
        "sensitivity": ["Adventitious"],
    },
    "T2-2": {
        "column": "challenge_t2_2_label",
        "fold_column": "fold_t2_2",
        "classes": ["Normal", "CAS", "DAS", "CAS & DAS", "Poor Quality"],
        "sensitivity": ["CAS", "DAS", "CAS & DAS"],
    },
}


def _wav_inventory(root):
    root = Path(root).expanduser().resolve()
    locations = {
        "train2022": root / "BioCAS2022" / "train2022_wav",
        "test2022": root / "BioCAS2022" / "test2022_wav",
        "test2023": root / "BioCAS2023" / "test2023_wav",
    }
    missing = [name for name, path in locations.items() if not path.is_dir()]
    if missing:
        raise RuntimeError("SPRSound root is missing required directories: %s" % ", ".join(missing))
    paths = {}
    source_split = {}
    for split, directory in locations.items():
        candidates = sorted(directory.rglob("*.wav"), key=lambda path: (path.name, str(path)))
        names = [path.name for path in candidates]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate filenames below explicit Challenge directory: %s" % directory)
        for path in candidates:
            if path.stem in paths:
                raise RuntimeError("duplicate BioCAS recording stem: %s" % path.stem)
            paths[path.stem] = path
            source_split[path.stem] = split
    if len(paths) != EXPECTED_RAW_RECORDS:
        raise RuntimeError("BioCAS Challenge WAV count mismatch: %d/%d" % (len(paths), EXPECTED_RAW_RECORDS))
    pseudonyms = {stem: "bio_record_%05d" % index for index, stem in enumerate(sorted(paths), start=1)}
    return paths, source_split, pseudonyms


def prepare_protocol(manifest):
    """Derive official splits/order/folds using only the public manifest."""
    frame = manifest.copy()
    if len(frame) != EXPECTED_RAW_RECORDS or frame["record_id"].duplicated().any():
        raise RuntimeError("invalid public BioCAS Challenge record manifest")
    frame = frame.rename(columns={"patient_group": "group_id"})
    train_groups = set(frame.loc[frame["split"].astype(str) == "train2022", "group_id"].astype(str))
    frame["official_split"] = frame["split"].astype(str)
    test2022 = frame["split"].astype(str) == "test2022"
    frame.loc[test2022, "official_split"] = np.where(
        frame.loc[test2022, "group_id"].astype(str).isin(train_groups),
        "test_intra",
        "test_inter",
    )
    observed = frame["official_split"].value_counts().to_dict()
    if observed != EXPECTED_COUNTS:
        raise RuntimeError("official Challenge split mismatch: %s" % observed)
    order = {"train2022": 0, "test_inter": 1, "test_intra": 2, "test2023": 3}
    frame["_split_order"] = frame["official_split"].map(order)
    frame = frame.sort_values(["_split_order", "record_id"], kind="mergesort").reset_index(drop=True)
    frame["dataset"] = np.where(frame["year"].astype(str) == "2022", "biocas2022", "biocas2023")
    for task, spec in TASKS.items():
        invalid = sorted(set(frame[spec["column"]].astype(str)) - set(spec["classes"]))
        if invalid:
            raise RuntimeError("%s invalid Challenge labels: %s" % (task, invalid))
        frame[spec["fold_column"]] = -1
        train = frame.index[frame["official_split"].astype(str) == "train2022"].to_numpy(dtype=int)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (_fit, validation) in enumerate(
            splitter.split(
                np.zeros((len(train), 1)),
                frame.loc[train, spec["column"]].astype(str),
                frame.loc[train, "group_id"].astype(str),
            )
        ):
            frame.loc[train[validation], spec["fold_column"]] = int(fold)
        for fold in range(5):
            fit_groups = set(frame.loc[train[frame.loc[train, spec["fold_column"]].to_numpy() != fold], "group_id"])
            val_groups = set(frame.loc[train[frame.loc[train, spec["fold_column"]].to_numpy() == fold], "group_id"])
            if fit_groups & val_groups:
                raise RuntimeError("%s group leakage in fold %d" % (task, fold))
    return frame.drop(columns=["_split_order"])


def load_challenge(root, manifest_path):
    paths, source_split, pseudonyms = _wav_inventory(root)
    reverse = {pseudonym: stem for stem, pseudonym in pseudonyms.items()}
    manifest = pd.read_csv(manifest_path, dtype=str)
    if set(manifest["record_id"].astype(str)) != set(reverse):
        raise RuntimeError("BioCAS raw inventory does not match the public pseudonymous universe")
    audio_paths = []
    for row in manifest.itertuples(index=False):
        stem = reverse[str(row.record_id)]
        if source_split[stem] != str(row.split):
            raise RuntimeError("raw/public split mismatch for pseudonymous record %s" % row.record_id)
        audio_paths.append(str(paths[stem]))
    manifest["audio_path"] = audio_paths
    return prepare_protocol(manifest)


def public_record_frame(records):
    columns = [
        "record_id",
        "dataset",
        "official_split",
        "group_id",
        "challenge_t2_1_label",
        "challenge_t2_2_label",
        "fold_t2_1",
        "fold_t2_2",
    ]
    return records[columns].copy()


def dataset_records(records, dataset):
    frame = records.loc[records["dataset"].astype(str) == dataset].copy().reset_index(drop=True)
    frame["split"] = frame["official_split"].astype(str)
    return frame
