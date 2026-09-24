"""Fresh downstream evaluation for public OPERA representations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import write_csv_gz, write_json
from .opera import MODEL_SPECS


TASKS = {"binary": {"column": "binary_label", "classes": ["Normal", "Adventitious"]},
         "coarse": {"column": "coarse_label", "classes": ["Normal", "CAS only", "DAS only", "CAS and DAS"]}}


def make_classifier():
    return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, class_weight=None))])


def metrics(truth, prediction, classes):
    return {"balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, prediction))}


def run_downstream(records, feature_paths, output_dir, hear_metrics_path=None):
    split = records["split"].astype(str).to_numpy(); train = np.flatnonzero(split == "train2022")
    evaluations = [("BioCAS2022", "test2022", np.flatnonzero(split == "test2022")),
                   ("BioCAS2023", "test2023", np.flatnonzero(split == "test2023"))]
    prediction_rows, metric_rows = [], []
    for model_name, path in feature_paths.items():
        values = np.load(path, mmap_mode="r"); method = MODEL_SPECS[model_name]["method"]
        if values.shape != (len(records), MODEL_SPECS[model_name]["dimension"]): raise RuntimeError("feature row/shape mismatch")
        for task, task_spec in TASKS.items():
            labels = records[task_spec["column"]].astype(str).to_numpy(); classes = task_spec["classes"]
            classifier = make_classifier().fit(values[train], labels[train])
            fitted = list(classifier.named_steps["classifier"].classes_)
            for dataset, evaluation, indices in evaluations:
                probability = classifier.predict_proba(values[indices])[:, [fitted.index(label) for label in classes]]
                prediction = np.asarray(classes)[probability.argmax(axis=1)]
                metric_rows.append({"dataset": dataset, "evaluation": evaluation, "task": task, "method": method,
                                    "record_count": int(len(indices)), **metrics(labels[indices], prediction, classes)})
                for rid, truth, pred, prob in zip(records.iloc[indices].record_id, labels[indices], prediction, probability):
                    prediction_rows.append({"dataset": dataset, "evaluation": evaluation, "task": task, "method": method,
                                            "record_id": rid, "y_true": truth, "y_pred": pred,
                                            "class_order": json.dumps(classes), "probabilities": json.dumps([float(x) for x in prob])})
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows); predictions = pd.DataFrame(prediction_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False); write_csv_gz(output_dir / "predictions.csv.gz", predictions)
    comparison = []
    if hear_metrics_path:
        payload = json.loads(Path(hear_metrics_path).read_text(encoding="utf-8"))
        phase3 = [row for row in payload["metrics"] if row.get("phase") == "phase3"]
        expected_hear = {
            ("BioCAS2022", "test2022", "binary"), ("BioCAS2022", "test2022", "coarse"),
            ("BioCAS2023", "test2023", "binary"), ("BioCAS2023", "test2023", "coarse"),
        }
        observed_hear = {(row.get("dataset"), row.get("evaluation"), row.get("task")) for row in phase3}
        if len(phase3) != 4 or observed_hear != expected_hear:
            raise RuntimeError("--hear-metrics does not contain the four certified fresh phase-3 BioCAS rows")
        hear = [
            {"dataset": row["dataset"], "evaluation": row["evaluation"], "task": row["task"],
             "method": "HeAR", "record_count": int((split == row["evaluation"]).sum()),
             "balanced_accuracy": row["balanced_accuracy"], "macro_f1": row["macro_f1"],
             "accuracy": row["accuracy"]}
            for row in phase3
        ]
        comparison = sorted(hear + metric_rows, key=lambda row: (
            row["dataset"], row["task"], ("HeAR", "OPERA-CE", "OPERA-CT", "OPERA-GT").index(row["method"])
        ))
        write_json(output_dir / "hear_opera_comparison.json", {
            "label": "tab:hear_opera_comparison", "rows": comparison,
            "hear_source": "certified fresh Full Reproduction"
        })
    write_json(output_dir / "manifest.json", {"status": "complete", "prediction_rows": len(predictions),
               "metrics": metric_rows, "classifier": {"scaler": "StandardScaler", "C": 1.0, "penalty": "l2",
               "solver": "lbfgs", "max_iter": 5000, "class_weight": None}, "test_data_used_for_selection": False})
    return {"metrics": metric_rows, "prediction_rows": len(predictions), "comparison_rows": len(comparison)}
