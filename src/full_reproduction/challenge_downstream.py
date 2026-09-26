"""Training-only selection and fresh official BioCAS Challenge evaluation."""

from __future__ import print_function

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info
from src.progress import downstream_progress

from .challenge_data import TASKS
from .features import write_csv_gz, write_json


TEMPERATURES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
REPLACEMENT_TOLERANCE = 1e-4
FUSION_ALPHA_CLS = 0.5


def make_classifier():
    return Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, class_weight=None))])


def official_metrics(y_true, y_pred, sensitivity_labels):
    truth, prediction = np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
    normal = truth == "Normal"
    sensitive = np.isin(truth, list(sensitivity_labels))
    if not normal.any() or not sensitive.any():
        raise ValueError("undefined official Challenge score population")
    sp = float(((prediction == truth) & normal).sum() / normal.sum())
    se = float(((prediction == truth) & sensitive).sum() / sensitive.sum())
    average = (se + sp) / 2.0
    harmonic = 0.0 if se + sp == 0.0 else 2.0 * se * sp / (se + sp)
    return {"SE": se, "SP": sp, "AS": average, "HS": harmonic, "Score": (average + harmonic) / 2.0}


def aggregate_sdp(windows, groups, temperature):
    """Historical float32 SDP arithmetic used only for temperature selection."""
    rows = []
    for indices in groups:
        values = np.asarray(windows[indices])
        mean = np.mean(values, axis=0)
        distances = np.linalg.norm(values - mean, axis=1)
        logits = distances.astype(np.float64) / float(temperature)
        logits -= np.max(logits)
        weights = np.exp(logits); weights /= weights.sum()
        rows.append(np.sum(values * weights[:, None], axis=0))
    return np.stack(rows).astype(np.float32)


def aggregate_sdp_final(windows, groups, temperature):
    """Canonical final-fit SDP arithmetic (window values promoted to float64)."""
    rows = []
    for indices in groups:
        values = np.asarray(windows[indices], dtype=np.float64)
        mean = values.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(values - mean, axis=1)
        logits = distances / float(temperature); logits -= logits.max()
        weights = np.exp(logits); weights /= weights.sum()
        rows.append((values * weights[:, None]).sum(axis=0).astype(np.float32))
    return np.stack(rows)


def aggregate_q75(windows, groups):
    return np.stack([np.quantile(np.asarray(windows[idx]), 0.75, axis=0, method="linear").astype(np.float32)
                     for idx in groups]).astype(np.float32)


def grouped_window_indices(records, metadata):
    positions = {}
    for index, record_id in enumerate(metadata["record_id"].astype(str)):
        positions.setdefault(record_id, []).append(index)
    if list(positions) != records["record_id"].astype(str).tolist():
        raise RuntimeError("record/window alignment mismatch")
    return [np.asarray(positions[record_id], dtype=int) for record_id in records["record_id"].astype(str)]


def _probabilities(train_x, train_y, evaluation_x, classes):
    model = make_classifier().fit(train_x, np.asarray(train_y, dtype=str))
    probability = model.predict_proba(evaluation_x)
    fitted = list(model.named_steps["logreg"].classes_)
    if set(fitted) != set(classes):
        raise RuntimeError("classifier class order/inventory mismatch")
    return probability[:, [fitted.index(label) for label in classes]]


def select_temperature(records, features_by_temperature, task):
    spec, rows, means = TASKS[task], [], {}
    labels = records[spec["column"]].astype(str).to_numpy()
    folds = records[spec["fold_column"]].astype(int).to_numpy()
    training = np.flatnonzero(folds >= 0)
    for temperature in TEMPERATURES:
        scores = []
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            fit = training[folds[training] != fold]
            if set(records.iloc[fit]["group_id"].astype(str)) & set(records.iloc[validation]["group_id"].astype(str)):
                raise RuntimeError("group leakage in %s fold %d" % (task, fold))
            probability = _probabilities(features_by_temperature[temperature][fit], labels[fit],
                                         features_by_temperature[temperature][validation], spec["classes"])
            prediction = np.asarray(spec["classes"], dtype=str)[probability.argmax(axis=1)]
            metrics = official_metrics(labels[validation], prediction, spec["sensitivity"])
            scores.append(metrics["Score"])
            rows.append({"task": task, "temperature": temperature, "fold": fold, **metrics})
        means[temperature] = float(np.mean(scores))
    incumbent = select_incumbent(means)
    summary = [{"task": task, "temperature": value, "mean_Score": means[value],
                "selected": value == incumbent} for value in TEMPERATURES]
    return float(incumbent), rows, summary


def select_incumbent(mean_scores):
    """Ascending-grid incumbent replacement with the audited strict tolerance."""
    incumbent = TEMPERATURES[0]
    for candidate in TEMPERATURES[1:]:
        if float(mean_scores[candidate]) > float(mean_scores[incumbent]) + REPLACEMENT_TOLERANCE:
            incumbent = candidate
    return float(incumbent)


def run_challenge_downstream(records_by_dataset, feature_paths_by_dataset, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    aggregated, selections, fold_rows, summary_rows = {}, {}, [], []
    for dataset, records in records_by_dataset.items():
        dataset_label = "BioCAS 2022" if dataset == "biocas2022" else "BioCAS 2023"
        downstream_progress("%s: aggregating CLS and Temporal Grid readouts" % dataset_label)
        paths = feature_paths_by_dataset[dataset]
        metadata = pd.read_csv(paths["window_metadata"], dtype={"record_id": str})
        groups = grouped_window_indices(records, metadata)
        cls = np.load(paths["cls_windows"], mmap_mode="r")
        tg = np.load(paths["tg_windows"], mmap_mode="r")
        aggregated[dataset] = {
            "cls_selection": {temperature: aggregate_sdp(cls, groups, temperature) for temperature in TEMPERATURES},
            "cls_final": {}, "cls_windows": cls, "groups": groups,
            "tg": aggregate_q75(tg, groups),
        }
    train_records = records_by_dataset["biocas2022"]
    for task in TASKS:
        downstream_progress("BioCAS 2022 — %s task: selecting SDP temperature by grouped CV" % task)
        selected, rows, summary = select_temperature(train_records, aggregated["biocas2022"]["cls_selection"], task)
        selections[task] = selected; fold_rows.extend(rows); summary_rows.extend(summary)
    for dataset in records_by_dataset:
        dataset_label = "BioCAS 2022" if dataset == "biocas2022" else "BioCAS 2023"
        downstream_progress("%s: preparing selected CLS aggregation" % dataset_label)
        for temperature in set(selections.values()):
            aggregated[dataset]["cls_final"][temperature] = aggregate_sdp_final(
                aggregated[dataset]["cls_windows"], aggregated[dataset]["groups"], temperature)
    prediction_rows, metric_rows = [], []
    train = np.flatnonzero(train_records["official_split"].astype(str).to_numpy() == "train2022")
    for task, spec in TASKS.items():
        classes, column = list(spec["classes"]), spec["column"]
        train_y = train_records[column].astype(str).to_numpy()
        temperature = selections[task]
        for dataset, records in records_by_dataset.items():
            if dataset == "biocas2022":
                indices = np.flatnonzero(np.isin(records["official_split"].astype(str), ["test_inter", "test_intra"]))
                evaluation = "BioCAS2022"
            else:
                indices = np.arange(len(records), dtype=int); evaluation = "BioCAS2023"
            downstream_progress("%s — %s task: final CLS/Temporal Grid/Fusion evaluation" % (evaluation, task))
            truth = records[column].astype(str).to_numpy()[indices]
            p_cls = _probabilities(aggregated["biocas2022"]["cls_final"][temperature][train], train_y[train],
                                   aggregated[dataset]["cls_final"][temperature][indices], classes)
            p_tg = _probabilities(aggregated["biocas2022"]["tg"][train], train_y[train],
                                  aggregated[dataset]["tg"][indices], classes)
            branches = {"CLS": p_cls, "Temporal Grid": p_tg,
                        "Fusion": FUSION_ALPHA_CLS * p_cls + (1.0 - FUSION_ALPHA_CLS) * p_tg}
            for method, probability in branches.items():
                pred = np.asarray(classes, dtype=str)[probability.argmax(axis=1)]
                metric_rows.append({"evaluation": evaluation, "task": task, "method": method,
                                    "temperature": temperature if method != "Temporal Grid" else "",
                                    "alpha_cls": FUSION_ALPHA_CLS if method == "Fusion" else "",
                                    **official_metrics(truth, pred, spec["sensitivity"])})
                for record_id, y_true, y_pred, row in zip(records.iloc[indices]["record_id"], truth, pred, probability):
                    prediction_rows.append({"evaluation": evaluation, "task": task, "method": method,
                                            "record_id": str(record_id), "y_true": y_true, "y_pred": y_pred,
                                            "class_order": json.dumps(classes),
                                            "probabilities": json.dumps([float(v) for v in row])})
    folds = pd.DataFrame(fold_rows); summary = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(prediction_rows); metrics = pd.DataFrame(metric_rows)
    folds.to_csv(output_dir / "sdp_temperature_cv_folds.csv", index=False)
    summary.to_csv(output_dir / "sdp_temperature_cv_summary.csv", index=False)
    write_csv_gz(output_dir / "predictions.csv.gz", predictions)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    write_json(output_dir / "metrics.json", {"metrics": metrics.to_dict(orient="records")})
    write_json(output_dir / "manifest.json", {
        "status": "complete", "selected_temperature": selections,
        "temperature_grid": list(TEMPERATURES), "replacement_tolerance": REPLACEMENT_TOLERANCE,
        "temperature_selection_uses_training_only": True, "fusion_alpha_cls": FUSION_ALPHA_CLS,
        "tg_aggregation": "dimension-wise q75, NumPy linear quantile",
        "classifier": {"scaler": "StandardScaler", "estimator": "LogisticRegression", "C": 1.0,
                       "penalty": "l2", "solver": "lbfgs", "max_iter": 5000, "class_weight": None},
        "selection_sdp_arithmetic": "historical float32 mean/distance; float64 softmax weights",
        "final_sdp_arithmetic": "float64 window values/mean/distance/weights; float32 output",
        "threadpools": [{k: v for k, v in item.items() if k != "filepath"} for item in threadpool_info()],
        "prediction_rows": int(len(predictions)), "metric_rows": int(len(metrics)),
    })
    return {"selected_temperature": selections, "prediction_rows": len(predictions),
            "metrics": metrics.to_dict(orient="records")}
