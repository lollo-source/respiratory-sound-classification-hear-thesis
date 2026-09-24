import numpy as np


def patient_confusions(rows, classes, patient_count):
    position = {label: index for index, label in enumerate(classes)}
    result = np.zeros((patient_count, len(classes), len(classes)), dtype=np.int64)
    for row in rows:
        result[int(row["patient_index"]), position[row["y_true"]], position[row["y_pred"]]] += 1
    return result


def resampled_confusions(patient_level, draw_indices):
    return patient_level[np.asarray(draw_indices, dtype=np.int64)].sum(axis=1)


def classification_replicates(confusion):
    true_total = confusion.sum(axis=2)
    predicted_total = confusion.sum(axis=1)
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    recall = true_positive / true_total.astype(np.float64)
    denominator = true_total + predicted_total
    per_class_f1 = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=np.float64), where=denominator != 0)
    return {
        "balanced_accuracy": recall.mean(axis=1),
        "macro_f1": per_class_f1.mean(axis=1),
        "accuracy": true_positive.sum(axis=1) / confusion.sum(axis=(1, 2)).astype(np.float64),
    }


def challenge_replicates(confusion, classes, sensitivity_labels):
    position = {label: index for index, label in enumerate(classes)}
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    true_total = confusion.sum(axis=2)
    normal = position["Normal"]
    sensitive = [position[label] for label in sensitivity_labels]
    sp = diagonal[:, normal] / true_total[:, normal].astype(np.float64)
    se = diagonal[:, sensitive].sum(axis=1) / true_total[:, sensitive].sum(axis=1).astype(np.float64)
    average = (se + sp) / 2.0
    harmonic = np.divide(2.0 * se * sp, se + sp, out=np.zeros_like(se), where=(se + sp) != 0)
    return {"SE": se, "SP": sp, "AS": average, "HS": harmonic, "Score": (average + harmonic) / 2.0}


def percentile_interval(values):
    lower, upper = np.percentile(np.asarray(values), [2.5, 97.5], interpolation="linear")
    return float(lower), float(upper)

