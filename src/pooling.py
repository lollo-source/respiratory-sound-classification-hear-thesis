import numpy as np


def arithmetic_mean(window_features):
    return np.asarray(window_features).mean(axis=0)


def sdp(window_features, temperature):
    values = np.asarray(window_features, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("window_features must be a non-empty matrix")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    centre = values.mean(axis=0)
    distance = np.linalg.norm(values - centre, axis=1)
    logits = distance / float(temperature)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return np.sum(values * weights[:, None], axis=0)


def dimensionwise_q75(window_features):
    return np.percentile(np.asarray(window_features), 75.0, axis=0, interpolation="linear")

