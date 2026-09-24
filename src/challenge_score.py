def official_challenge_metrics(y_true, y_pred, sensitivity_labels):
    """Official score with true Poor Quality excluded from SE/SP denominators."""
    sensitivity_labels = set(sensitivity_labels)
    normal_total = sum(value == "Normal" for value in y_true)
    sensitivity_total = sum(value in sensitivity_labels for value in y_true)
    if normal_total == 0 or sensitivity_total == 0:
        raise ValueError("undefined official score population")
    sp = sum(t == "Normal" and p == t for t, p in zip(y_true, y_pred)) / float(normal_total)
    se = sum(t in sensitivity_labels and p == t for t, p in zip(y_true, y_pred)) / float(sensitivity_total)
    average = (se + sp) / 2.0
    harmonic = 0.0 if se + sp == 0.0 else 2.0 * se * sp / (se + sp)
    return {"SE": se, "SP": sp, "AS": average, "HS": harmonic, "Score": (average + harmonic) / 2.0}

