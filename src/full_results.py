"""Reference-blind thesis-table assembly from completed Full outputs."""

from __future__ import print_function

import json
from collections import Counter
from pathlib import Path

from .challenge_score import official_challenge_metrics
from .io_utils import read_csv
from .protocol import BINARY_CLASSES, CHALLENGE_CLASSES, CHALLENGE_SENSITIVITY, COARSE_CLASSES
from .reproduction import LABELS, _group, _indexed, _metric_rows, _prediction_from_probabilities, _recall_rows, _table


QUICK_ONLY_TABLES = [LABELS[6], LABELS[15], LABELS[16]]
FULL_TABLE_LABELS = [label for label in LABELS if label not in QUICK_ONLY_TABLES]


def harmonised_tables(hf_records, biocas_records, predictions):
    """Apply the existing Quick metric/table logic to explicit fresh Full rows."""
    tables = {}
    rows = []
    for task, column, classes in (
        ("binary", "binary_label", BINARY_CLASSES),
        ("coarse", "coarse_label", COARSE_CLASSES),
    ):
        for split in ("train", "test"):
            counts = Counter(row[column] for row in hf_records if row["split"] == split)
            for label in classes:
                rows.append({"task": task, "split": split, "class": label, "count": counts[label]})
    tables[LABELS[0]] = _table(LABELS[0], rows)

    rows = []
    for split in ("train", "test"):
        part = [row for row in hf_records if row["split"] == split]
        if part and "device" not in part[0]:
            raise RuntimeError("fresh Full HF record manifest does not preserve device metadata")
        counts = Counter(row["device"] for row in part)
        for device in sorted(counts):
            rows.append({
                "split": split,
                "device": device,
                "count": counts[device],
                "percentage": 100.0 * counts[device] / len(part),
            })
    tables[LABELS[1]] = _table(LABELS[1], rows)

    rows = []
    for task, column, classes in (
        ("binary", "binary_label", BINARY_CLASSES),
        ("coarse", "coarse_label", COARSE_CLASSES),
    ):
        for split in ("train2022", "test2022", "test2023"):
            counts = Counter(row[column] for row in biocas_records if row["split"] == split)
            for label in classes:
                rows.append({"task": task, "split": split, "class": label, "count": counts[label]})
    tables[LABELS[2]] = _table(LABELS[2], rows)

    phase1 = _metric_rows(predictions, "phase1")
    phase2 = _metric_rows(predictions, "phase2")
    phase3 = _metric_rows(predictions, "phase3")
    recall1 = _recall_rows(predictions, "phase1")
    recall2 = _recall_rows(predictions, "phase2")
    recall3 = _recall_rows(predictions, "phase3")
    tables[LABELS[4]] = _table(LABELS[4], phase1)
    tables[LABELS[5]] = _table(LABELS[5], recall1)

    metric_keys = ["dataset", "evaluation", "task"]
    m1, m2, m3 = _indexed(phase1, metric_keys), _indexed(phase2, metric_keys), _indexed(phase3, metric_keys)
    for current, baseline in ((phase2, m1), (phase3, m2)):
        for row in current:
            old = baseline[tuple(row[key] for key in metric_keys)]
            for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
                row["delta_" + metric] = row[metric] - old[metric]
    recall_keys = ["dataset", "evaluation", "task", "class"]
    r1, r2 = _indexed(recall1, recall_keys), _indexed(recall2, recall_keys)
    for current, baseline in ((recall2, r1), (recall3, r2)):
        for row in current:
            old = baseline[tuple(row[key] for key in recall_keys)]
            row["delta_recall"] = row["recall"] - old["recall"]
    tables[LABELS[7]] = _table(LABELS[7], phase2)
    tables[LABELS[8]] = _table(LABELS[8], recall2)
    tables[LABELS[9]] = _table(LABELS[9], phase3)
    tables[LABELS[10]] = _table(LABELS[10], recall3)

    rows = []
    for key in sorted(m3):
        row = {name: value for name, value in zip(metric_keys, key)}
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            row["single_" + metric] = m1[key][metric]
            row["dual_" + metric] = m3[key][metric]
            row["delta_" + metric] = m3[key][metric] - m1[key][metric]
        rows.append(row)
    tables[LABELS[11]] = _table(LABELS[11], rows)
    return tables


def challenge_tables(record_rows, predictions, published_rows):
    tables = {}
    prepared_records = []
    for row in record_rows:
        item = dict(row)
        official = item["official_split"]
        item["split"] = "test2022" if official in ("test_inter", "test_intra") else official
        prepared_records.append(item)
    rows = []
    for task, column in (("T2-1", "challenge_t2_1_label"), ("T2-2", "challenge_t2_2_label")):
        for split in ("train2022", "test2022", "test2023"):
            counts = Counter(row[column] for row in prepared_records if row["split"] == split)
            for label in CHALLENGE_CLASSES[task]:
                rows.append({"task": task, "split": split, "class": label, "count": counts[label]})
    tables[LABELS[3]] = _table(LABELS[3], rows)

    normalized = []
    for row in predictions:
        item = dict(row)
        item["evaluation"] = item["evaluation"].lower()
        normalized.append(item)
    metric_rows = []
    scores = {}
    for key, part in sorted(_group(normalized, ["evaluation", "task", "method"]).items()):
        evaluation, task, method = key
        values = official_challenge_metrics(
            [row["y_true"] for row in part],
            [_prediction_from_probabilities(row) for row in part],
            CHALLENGE_SENSITIVITY[task],
        )
        result = {"evaluation": evaluation, "task": task, "method": method, "record_count": len(part)}
        result.update(values)
        metric_rows.append(result)
        scores[key] = values["Score"]
    tables[LABELS[14]] = _table(LABELS[14], metric_rows)

    placements = []
    keys = (
        ("biocas2022", "T2-1", "Table II"),
        ("biocas2022", "T2-2", "Table II"),
        ("biocas2023", "T2-1", "Table II"),
        ("biocas2023", "T2-2", "Table II"),
        ("biocas2023", "T2-1", "Table III"),
        ("biocas2023", "T2-2", "Table III"),
    )
    for evaluation, task, published_table in keys:
        score = scores[(evaluation, task, "Fusion")]
        comparison = [
            float(row["score"])
            for row in published_rows
            if row["evaluation"] == evaluation and row["task"] == task and row["table"] == published_table
        ]
        placements.append({
            "evaluation": evaluation,
            "task": task,
            "published_table": published_table,
            "score": score,
            "estimated_position": 1 + sum(value > score for value in comparison),
            "comparison_set_size": len(comparison) + 1,
            "status": "retrospective estimate, not an official submission",
        })
    tables[LABELS[13]] = _table(LABELS[13], placements)
    return tables


def opera_comparison_table(payload):
    if payload.get("label") != LABELS[12] or not isinstance(payload.get("rows"), list):
        raise RuntimeError("fresh Full HeAR/OPERA comparison table is missing or invalid")
    return _table(LABELS[12], payload["rows"])


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_complete_manifest(path, identity):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("completed %s manifest is missing: %s" % (identity, path))
    payload = _read_json(path)
    if payload.get("status") != "complete" or payload.get("stage") != "all":
        raise RuntimeError("%s must be a completed --stage all workflow" % identity)
    return payload


def assemble_full_tables(full_output_root, release_root):
    full_output_root = Path(full_output_root)
    release_root = Path(release_root)
    workflow_manifests = {
        "hear_harmonised": _require_complete_manifest(full_output_root / "run_manifest.json", "HeAR harmonised"),
        "hear_challenge": _require_complete_manifest(full_output_root / "challenge" / "run_manifest.json", "HeAR Challenge"),
        "opera": _require_complete_manifest(full_output_root / "opera" / "run_manifest.json", "OPERA"),
    }

    hf_manifest = full_output_root / "datasets" / "hf_lung" / "record_manifest.csv.gz"
    bio_manifest = full_output_root / "datasets" / "biocas_harmonised" / "record_manifest.csv.gz"
    progression_paths = [
        full_output_root / "datasets" / dataset / "downstream" / "predictions.csv.gz"
        for dataset in ("hf_lung", "biocas_harmonised")
    ]
    progression = []
    for path in progression_paths:
        progression.extend(read_csv(str(path)))
    tables = harmonised_tables(read_csv(str(hf_manifest)), read_csv(str(bio_manifest)), progression)

    challenge_records = []
    challenge_record_paths = [
        full_output_root / "challenge" / "datasets" / dataset / "record_manifest.csv.gz"
        for dataset in ("biocas2022", "biocas2023")
    ]
    for path in challenge_record_paths:
        challenge_records.extend(read_csv(str(path)))
    challenge_predictions = full_output_root / "challenge" / "downstream" / "predictions.csv.gz"
    published = release_root / "published_challenge_reference" / "published_biocas_task2_scores.csv"
    tables.update(challenge_tables(
        challenge_records,
        read_csv(str(challenge_predictions)),
        read_csv(str(published)),
    ))

    opera_comparison = full_output_root / "opera" / "downstream" / "hear_opera_comparison.json"
    if not opera_comparison.is_file():
        raise RuntimeError("complete Full assembly requires fresh OPERA --hear-metrics comparison output")
    opera = opera_comparison_table(_read_json(opera_comparison))
    tables[opera["label"]] = opera
    ordered = {label: tables[label] for label in FULL_TABLE_LABELS}
    if set(ordered) != set(FULL_TABLE_LABELS):
        raise RuntimeError("Full table assembly did not produce the declared applicable subset")

    alpha_manifests = {
        dataset: _read_json(full_output_root / "datasets" / dataset / "downstream" / "manifest.json")
        for dataset in ("hf_lung", "biocas_harmonised")
    }
    challenge_manifest = _read_json(full_output_root / "challenge" / "downstream" / "manifest.json")
    hear_feature_manifest = _read_json(full_output_root / "datasets" / "hf_lung" / "features" / "manifest.json")
    opera_manifests = {
        name: _read_json(full_output_root / "opera" / "features" / name / "manifest.json")
        for name in ("operaCE", "operaCT", "operaGT")
    }
    provenance = {
        "completed_workflow_manifests": workflow_manifests,
        "selected_harmonised_alphas": {
            dataset: payload["selected_alpha"] for dataset, payload in alpha_manifests.items()
        },
        "selected_challenge_temperatures": challenge_manifest["selected_temperature"],
        "hear_identity": hear_feature_manifest["model"],
        "opera_identity": {
            name: {
                "model": payload["model"],
                "method": payload["method"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "opera_revision": payload["opera_revision"],
            }
            for name, payload in opera_manifests.items()
        },
    }
    source_paths = [hf_manifest, bio_manifest] + progression_paths + challenge_record_paths + [
        challenge_predictions,
        full_output_root / "challenge" / "downstream" / "metrics.csv",
        published,
        opera_comparison,
        full_output_root / "opera" / "downstream" / "predictions.csv.gz",
        full_output_root / "opera" / "downstream" / "metrics.csv",
    ]
    source_artifacts = {}
    for path in source_paths:
        if not path.is_file():
            raise RuntimeError("required Full result artifact is missing: %s" % path)
        try:
            key = str(path.relative_to(full_output_root))
        except ValueError:
            key = "published-literature/" + path.name
        source_artifacts[key] = path
    return ordered, provenance, source_artifacts
