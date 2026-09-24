from __future__ import print_function

import json
import os
from collections import Counter, defaultdict

import numpy as np

from .bootstrap import patient_confusions, resampled_confusions, classification_replicates, challenge_replicates, percentile_interval
from .challenge_score import official_challenge_metrics
from .io_utils import read_csv, root_path
from .metrics import classification_metrics
from .protocol import BINARY_CLASSES, COARSE_CLASSES, CHALLENGE_CLASSES, CHALLENGE_SENSITIVITY, assert_exclusion_contract


LABELS = [
    "tab:hflung_class_distribution", "tab:hflung_device_distribution",
    "tab:biocas_harmonised_class_distribution", "tab:biocas_challenge_class_distribution",
    "tab:single_window_results", "tab:single_window_per_class_recall",
    "tab:single_window_event_capture", "tab:complete_record_results",
    "tab:complete_record_per_class_recall", "tab:dual_readout_results",
    "tab:dual_readout_per_class_recall", "tab:overall_harmonised_comparison",
    "tab:hear_opera_comparison", "tab:biocas_challenge_placements",
    "tab:challenge_readout_comparison", "tab:bootstrap_harmonised",
    "tab:bootstrap_challenge_readouts",
]


def _table(label, rows):
    return {"label": label, "rows": rows}


def _group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def _classes(task):
    return BINARY_CLASSES if task == "binary" else COARSE_CLASSES


def _prediction_from_probabilities(row):
    classes = json.loads(row["class_order"])
    probabilities = json.loads(row["probabilities"])
    if len(classes) != len(probabilities):
        raise ValueError("probability/class-order length mismatch")
    prediction = classes[max(range(len(classes)), key=lambda index: probabilities[index])]
    if row.get("y_pred") and prediction != row["y_pred"]:
        raise ValueError("stored label differs from probability argmax")
    return prediction


def _metric_rows(predictions, phase):
    output = []
    groups = _group([row for row in predictions if row["phase"] == phase], ["dataset", "evaluation", "task"])
    for key in sorted(groups):
        dataset, evaluation, task = key
        part = groups[key]
        values = classification_metrics([row["y_true"] for row in part], [_prediction_from_probabilities(row) for row in part], _classes(task))
        output.append({
            "dataset": dataset, "evaluation": evaluation, "task": task,
            "balanced_accuracy": values["balanced_accuracy"], "macro_f1": values["macro_f1"],
            "accuracy": values["accuracy"], "record_count": len(part),
        })
    return output


def _recall_rows(predictions, phase):
    output = []
    groups = _group([row for row in predictions if row["phase"] == phase], ["dataset", "evaluation", "task"])
    for key in sorted(groups):
        dataset, evaluation, task = key
        part = groups[key]
        values = classification_metrics([row["y_true"] for row in part], [_prediction_from_probabilities(row) for row in part], _classes(task))
        for label in _classes(task):
            output.append({"dataset": dataset, "evaluation": evaluation, "task": task, "class": label, "recall": values["per_class_recall"][label]})
    return output


def _indexed(rows, keys):
    return {tuple(row[key] for key in keys): row for row in rows}


def progression_tables():
    hf = read_csv(root_path("manifests", "hf_lung_records.csv.gz"))
    bio = read_csv(root_path("manifests", "biocas_records.csv.gz"))
    predictions = read_csv(root_path("artifacts", "predictions", "progression.csv.gz"))
    assert_exclusion_contract(bio)
    tables = {}

    rows = []
    for task, column, classes in (("binary", "binary_label", BINARY_CLASSES), ("coarse", "coarse_label", COARSE_CLASSES)):
        for split in ("train", "test"):
            counts = Counter(row[column] for row in hf if row["split"] == split)
            for label in classes:
                rows.append({"task": task, "split": split, "class": label, "count": counts[label]})
    tables[LABELS[0]] = _table(LABELS[0], rows)

    rows = []
    for split in ("train", "test"):
        part = [row for row in hf if row["split"] == split]
        counts = Counter(row["device"] for row in part)
        for device in sorted(counts):
            rows.append({"split": split, "device": device, "count": counts[device], "percentage": 100.0 * counts[device] / len(part)})
    tables[LABELS[1]] = _table(LABELS[1], rows)

    rows = []
    included = [row for row in bio if row["inclusion_status"] == "included"]
    for task, column, classes in (("binary", "binary_label", BINARY_CLASSES), ("coarse", "coarse_label", COARSE_CLASSES)):
        for split in ("train2022", "test2022", "test2023"):
            counts = Counter(row[column] for row in included if row["split"] == split)
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

    capture = read_csv(root_path("artifacts", "predictions", "event_capture.csv.gz"))
    rows = []
    for dataset in sorted(set(row["dataset"] for row in capture)):
        part = [row for row in capture if row["dataset"] == dataset and row["binary_label"] == "Adventitious"]
        captured = sum(row["overlap_ge_50ms"] == "True" for row in part)
        rows.append({"dataset": dataset, "pathological_records": len(part), "captured_records": captured, "capture_rate_percent": 100.0 * captured / len(part)})
    tables[LABELS[6]] = _table(LABELS[6], rows)

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


def challenge_distribution_table():
    bio = read_csv(root_path("manifests", "biocas_records.csv.gz"))
    rows = []
    for task, column in (("T2-1", "challenge_t2_1_label"), ("T2-2", "challenge_t2_2_label")):
        for split in ("train2022", "test2022", "test2023"):
            counts = Counter(row[column] for row in bio if row["split"] == split)
            for label in CHALLENGE_CLASSES[task]:
                rows.append({"task": task, "split": split, "class": label, "count": counts[label]})
    return _table(LABELS[3], rows)


def opera_table():
    opera = read_csv(root_path("artifacts", "predictions", "opera.csv.gz"))
    progression = read_csv(root_path("artifacts", "predictions", "progression.csv.gz"))
    combined = list(opera)
    for row in progression:
        if row["phase"] == "phase3" and row["dataset"].startswith("BioCAS"):
            combined.append({"dataset": row["dataset"], "evaluation": row["evaluation"], "task": row["task"], "method": "HeAR", "record_id": row["record_id"], "y_true": row["y_true"], "y_pred": row["y_pred"], "class_order": row["class_order"], "probabilities": row["probabilities"]})
    rows = []
    for key, part in sorted(_group(combined, ["dataset", "evaluation", "task", "method"]).items()):
        values = classification_metrics([row["y_true"] for row in part], [_prediction_from_probabilities(row) for row in part], _classes(key[2]))
        rows.append({"dataset": key[0], "evaluation": key[1], "task": key[2], "method": key[3], "balanced_accuracy": values["balanced_accuracy"], "macro_f1": values["macro_f1"], "accuracy": values["accuracy"], "record_count": len(part)})
    return _table(LABELS[12], rows)


def challenge_tables():
    predictions = read_csv(root_path("artifacts", "predictions", "challenge.csv.gz"))
    rows = []
    scores = {}
    for key, part in sorted(_group(predictions, ["evaluation", "task", "method"]).items()):
        evaluation, task, method = key
        metrics = official_challenge_metrics([row["y_true"] for row in part], [_prediction_from_probabilities(row) for row in part], CHALLENGE_SENSITIVITY[task])
        result = {"evaluation": evaluation, "task": task, "method": method, "record_count": len(part)}
        result.update(metrics)
        rows.append(result)
        scores[key] = metrics["Score"]

    published = read_csv(root_path("published_challenge_reference", "published_biocas_task2_scores.csv"))
    placement_rows = []
    for evaluation, task, table in (("biocas2022", "T2-1", "Table II"), ("biocas2022", "T2-2", "Table II"), ("biocas2023", "T2-1", "Table II"), ("biocas2023", "T2-2", "Table II"), ("biocas2023", "T2-1", "Table III"), ("biocas2023", "T2-2", "Table III")):
        score = scores[(evaluation, task, "Fusion")]
        comparison = [float(row["score"]) for row in published if row["evaluation"] == evaluation and row["task"] == task and row["table"] == table]
        position = 1 + sum(value > score for value in comparison)
        placement_rows.append({"evaluation": evaluation, "task": task, "published_table": table, "score": score, "estimated_position": position, "comparison_set_size": len(comparison) + 1, "status": "retrospective estimate, not an official submission"})
    return {_table(LABELS[13], placement_rows)["label"]: _table(LABELS[13], placement_rows), LABELS[14]: _table(LABELS[14], rows)}


def bootstrap_tables():
    draws = np.load(root_path("artifacts", "predictions", "bootstrap_draw_indices.npz"))
    harm = read_csv(root_path("artifacts", "predictions", "harmonised_bootstrap.csv.gz"))
    harm_rows = []
    harm_replicates = {}
    for task in ("binary", "coarse"):
        classes = _classes(task)
        draw = draws["harmonised_" + task]
        for method in ("Single Window", "Dual Readout"):
            part = [row for row in harm if row["task"] == task and row["method"] == method]
            point = classification_metrics([row["y_true"] for row in part], [row["y_pred"] for row in part], classes)
            patient = patient_confusions(part, classes, draw.shape[1])
            replicates = classification_replicates(resampled_confusions(patient, draw))
            for metric in ("balanced_accuracy", "macro_f1"):
                lower, upper = percentile_interval(replicates[metric])
                harm_rows.append({"task": task, "method": method, "metric": metric, "point_estimate": point[metric], "ci_lower": lower, "ci_upper": upper, "replicates": len(draw)})
                harm_replicates[(task, method, metric)] = replicates[metric]
        for metric in ("balanced_accuracy", "macro_f1"):
            values = harm_replicates[(task, "Dual Readout", metric)] - harm_replicates[(task, "Single Window", metric)]
            lower, upper = percentile_interval(values)
            harm_rows.append({"task": task, "method": "Dual Readout - Single Window", "metric": metric, "point_estimate": next(row["point_estimate"] for row in harm_rows if row["task"] == task and row["method"] == "Dual Readout" and row["metric"] == metric) - next(row["point_estimate"] for row in harm_rows if row["task"] == task and row["method"] == "Single Window" and row["metric"] == metric), "ci_lower": lower, "ci_upper": upper, "replicates": len(draw)})

    challenge = read_csv(root_path("artifacts", "predictions", "challenge.csv.gz"))
    challenge_rows = []
    challenge_rep = {}
    for task, draw_key in (("T2-1", "challenge_t2_1"), ("T2-2", "challenge_t2_2")):
        draw = draws[draw_key]
        for method in ("CLS", "Temporal Grid", "Fusion"):
            part = [dict(row, y_pred=_prediction_from_probabilities(row)) for row in challenge if row["evaluation"] == "biocas2022" and row["task"] == task and row["method"] == method]
            point = official_challenge_metrics([row["y_true"] for row in part], [row["y_pred"] for row in part], CHALLENGE_SENSITIVITY[task])
            patient = patient_confusions(part, CHALLENGE_CLASSES[task], draw.shape[1])
            replicates = challenge_replicates(resampled_confusions(patient, draw), CHALLENGE_CLASSES[task], CHALLENGE_SENSITIVITY[task])["Score"]
            lower, upper = percentile_interval(replicates)
            challenge_rows.append({"task": task, "method": method, "metric": "Score", "point_estimate": point["Score"], "ci_lower": lower, "ci_upper": upper, "replicates": len(draw)})
            challenge_rep[(task, method)] = replicates
        for left, right in (("Temporal Grid", "CLS"), ("Fusion", "CLS"), ("Fusion", "Temporal Grid")):
            values = challenge_rep[(task, left)] - challenge_rep[(task, right)]
            lower, upper = percentile_interval(values)
            left_point = next(row["point_estimate"] for row in challenge_rows if row["task"] == task and row["method"] == left)
            right_point = next(row["point_estimate"] for row in challenge_rows if row["task"] == task and row["method"] == right)
            challenge_rows.append({"task": task, "method": left + " - " + right, "metric": "Score", "point_estimate": left_point - right_point, "ci_lower": lower, "ci_upper": upper, "replicates": len(draw)})
    return {LABELS[15]: _table(LABELS[15], harm_rows), LABELS[16]: _table(LABELS[16], challenge_rows)}
