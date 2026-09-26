"""Fresh one-thread downstream fitting and training-only fusion selection."""

from __future__ import print_function

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits
from src.progress import downstream_progress

from .features import write_csv_gz, write_json


BINARY_CLASSES = ["Normal", "Adventitious"]
COARSE_CLASSES = ["Normal", "CAS only", "DAS only", "CAS and DAS"]
TASKS = {
    "binary": {"column": "binary_label", "classes": BINARY_CLASSES},
    "coarse": {"column": "coarse_label", "classes": COARSE_CLASSES},
}


def alpha_grid():
    return [round(float(value), 10) for value in np.arange(0.0, 1.0 + 0.025, 0.05)]


def make_classifier():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight=None,
                ),
            ),
        ]
    )


def _fit_probabilities(train_x, train_y, eval_x, classes):
    with threadpool_limits(limits=1):
        classifier = make_classifier()
        classifier.fit(train_x, train_y.astype(str))
        probabilities = classifier.predict_proba(eval_x)
    source_classes = list(classifier.named_steps["logreg"].classes_)
    order = [source_classes.index(label) for label in classes]
    return probabilities[:, order]


def _metrics(y_true, probabilities, classes):
    prediction = np.asarray(classes, dtype=str)[np.argmax(probabilities, axis=1)]
    recalls = recall_score(y_true, prediction, labels=classes, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, labels=classes, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "per_class_recall": {label: float(value) for label, value in zip(classes, recalls)},
    }


def select_alpha(candidates):
    frame = pd.DataFrame(candidates)
    frame["distance_from_0_50"] = (frame["alpha"].astype(float) - 0.5).abs()
    ranked = frame.sort_values(
        ["balanced_accuracy", "macro_f1", "distance_from_0_50", "alpha"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return float(ranked.iloc[0]["alpha"]), ranked


def _oof_probabilities(features, records, task):
    spec = TASKS[task]
    classes = list(spec["classes"])
    labels = records[spec["column"]].astype(str).to_numpy()
    train_indices = np.flatnonzero(records["fold_id"].astype(int).to_numpy() >= 0)
    train_position = {int(index): position for position, index in enumerate(train_indices)}
    output = np.full((len(train_indices), len(classes)), np.nan, dtype=np.float64)
    for fold_id in range(5):
        validation = np.flatnonzero(records["fold_id"].astype(int).to_numpy() == fold_id)
        validation_set = set(validation.tolist())
        fit = np.asarray([index for index in train_indices if int(index) not in validation_set], dtype=int)
        fit_groups = set(records.iloc[fit]["group_id"].astype(str))
        validation_groups = set(records.iloc[validation]["group_id"].astype(str))
        if fit_groups & validation_groups:
            raise RuntimeError("group leakage in fold %d" % fold_id)
        probability = _fit_probabilities(features[fit], labels[fit], features[validation], classes)
        output[[train_position[int(index)] for index in validation]] = probability
    if not np.isfinite(output).all():
        raise RuntimeError("incomplete OOF probabilities")
    return train_indices, labels[train_indices], output


def _prediction_rows(dataset_name, evaluation, phase, task, record_ids, truth, probabilities, classes, alpha=None):
    prediction = np.asarray(classes, dtype=str)[np.argmax(probabilities, axis=1)]
    rows = []
    for record_id, y_true, y_pred, probability in zip(record_ids, truth, prediction, probabilities):
        rows.append(
            {
                "dataset": dataset_name,
                "evaluation": evaluation,
                "phase": phase,
                "task": task,
                "record_id": str(record_id),
                "y_true": str(y_true),
                "y_pred": str(y_pred),
                "class_order": json.dumps(classes),
                "probabilities": json.dumps([float(value) for value in probability]),
                "selected_alpha": "" if alpha is None else float(alpha),
            }
        )
    return rows


def _evaluation_splits(dataset, records):
    split = records["split"].astype(str).to_numpy()
    if dataset == "hf_lung":
        return [("HF_Lung", "official_test", np.flatnonzero(split == "test"))]
    return [
        ("BioCAS2022", "test2022", np.flatnonzero(split == "test2022")),
        ("BioCAS2023", "test2023", np.flatnonzero(split == "test2023")),
    ]


def run_downstream(dataset, records, single_cls, cls_record, tg_record, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    single_cls = np.asarray(single_cls)
    cls_record = np.asarray(cls_record)
    tg_record = np.asarray(tg_record)
    if len(single_cls) != len(records) or len(cls_record) != len(records) or len(tg_record) != len(records):
        raise RuntimeError("feature/record row mismatch")
    train_indices = np.flatnonzero(records["fold_id"].astype(int).to_numpy() >= 0)
    predictions = []
    metrics = []
    alpha_rows = []
    selected = {}
    dataset_label = "HF Lung" if dataset == "hf_lung" else "BioCAS harmonised"

    for task, spec in TASKS.items():
        classes = list(spec["classes"])
        labels = records[spec["column"]].astype(str).to_numpy()
        downstream_progress("%s — %s task: CLS grouped cross-validation" % (dataset_label, task))
        oof_indices_cls, oof_truth, oof_cls = _oof_probabilities(cls_record, records, task)
        downstream_progress("%s — %s task: Temporal Grid grouped cross-validation" % (dataset_label, task))
        oof_indices_tg, oof_truth_tg, oof_tg = _oof_probabilities(tg_record, records, task)
        if not np.array_equal(oof_indices_cls, oof_indices_tg) or not np.array_equal(oof_truth, oof_truth_tg):
            raise RuntimeError("branch OOF alignment mismatch")
        candidates = []
        for alpha in alpha_grid():
            fused = alpha * oof_cls + (1.0 - alpha) * oof_tg
            candidate = {"task": task, "alpha": float(alpha)}
            candidate.update(_metrics(oof_truth, fused, classes))
            candidate.pop("per_class_recall")
            candidates.append(candidate)
        alpha, ranking = select_alpha(candidates)
        selected[task] = alpha
        ranking["task"] = task
        ranking["selected"] = ranking["rank"].astype(int) == 1
        alpha_rows.append(ranking)

        for dataset_name, evaluation, eval_indices in _evaluation_splits(dataset, records):
            downstream_progress("%s — %s task: %s final evaluation" % (dataset_label, task, dataset_name))
            phase1_probability = _fit_probabilities(
                single_cls[train_indices], labels[train_indices], single_cls[eval_indices], classes
            )
            cls_probability = _fit_probabilities(
                cls_record[train_indices], labels[train_indices], cls_record[eval_indices], classes
            )
            tg_probability = _fit_probabilities(
                tg_record[train_indices], labels[train_indices], tg_record[eval_indices], classes
            )
            fused_probability = alpha * cls_probability + (1.0 - alpha) * tg_probability
            evaluations = [
                ("phase1", phase1_probability, None),
                ("phase2", cls_probability, None),
                ("phase3", fused_probability, alpha),
            ]
            for phase, probability, phase_alpha in evaluations:
                predictions.extend(
                    _prediction_rows(
                        dataset_name,
                        evaluation,
                        phase,
                        task,
                        records.iloc[eval_indices]["record_id"].astype(str).to_numpy(),
                        labels[eval_indices],
                        probability,
                        classes,
                        phase_alpha,
                    )
                )
                item = {
                    "dataset": dataset_name,
                    "evaluation": evaluation,
                    "phase": phase,
                    "task": task,
                }
                item.update(_metrics(labels[eval_indices], probability, classes))
                metrics.append(item)

    prediction_path = output_dir / "predictions.csv.gz"
    metric_path = output_dir / "metrics.json"
    alpha_path = output_dir / "alpha_selection.csv"
    write_csv_gz(prediction_path, pd.DataFrame(predictions))
    write_json(metric_path, {"metrics": metrics})
    pd.concat(alpha_rows, ignore_index=True).to_csv(alpha_path, index=False)
    # threadpoolctl reports absolute shared-library paths. They are useful
    # locally but disclose workstation layout and are not needed to reproduce
    # the numerical environment, so retain only portable fields.
    threadpools = [
        {key: value for key, value in pool.items() if key != "filepath"}
        for pool in threadpool_info()
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "status": "complete",
            "dataset": dataset,
            "selected_alpha": selected,
            "selection": "five-fold grouped training-only OOF; balanced accuracy, macro-F1, distance from 0.50, smaller alpha",
            "test_data_used_for_selection": False,
            "classifier": {
                "scaler": "StandardScaler",
                "estimator": "LogisticRegression",
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 5000,
                "class_weight": None,
            },
            "blas_threads": 1,
            "threadpools": threadpools,
            "prediction_rows": len(predictions),
        },
    )
    return {"selected_alpha": selected, "prediction_rows": len(predictions), "metrics": metrics}
