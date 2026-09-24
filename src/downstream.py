"""Level-B downstream contract and probability operations."""

import numpy as np


def reconcile_class_order(probabilities, source_classes, target_classes):
    values = np.asarray(probabilities)
    positions = {label: index for index, label in enumerate(source_classes)}
    missing = [label for label in target_classes if label not in positions]
    if missing:
        raise ValueError("missing classes: %s" % missing)
    return values[:, [positions[label] for label in target_classes]]


def fuse_probabilities(cls_probabilities, temporal_grid_probabilities, alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    return alpha * np.asarray(cls_probabilities) + (1.0 - alpha) * np.asarray(temporal_grid_probabilities)


def make_classifier():
    """Create the audited Level-B estimator when scikit-learn is installed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, class_weight=None)),
    ])

