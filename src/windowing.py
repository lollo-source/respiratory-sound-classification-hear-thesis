import numpy as np


def right_pad(values, size):
    x = np.asarray(values)
    if len(x) >= size:
        return x[:size].copy()
    return np.pad(x, (0, size - len(x)), mode="constant")


def complete_record_windows(values, window_samples=32000):
    x = np.asarray(values)
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    count = max(1, int(np.ceil(len(x) / float(window_samples))))
    return np.stack([
        right_pad(x[index * window_samples:(index + 1) * window_samples], window_samples)
        for index in range(count)
    ])


def max_energy_crop(values, window_samples=32000):
    """Squared-energy crop; np.argmax gives the earliest maximum on ties."""
    x = np.asarray(values)
    if len(x) <= window_samples:
        return right_pad(x, window_samples), 0
    energy = np.square(x.astype(np.float64))
    prefix = np.concatenate(([0.0], np.cumsum(energy)))
    scores = prefix[window_samples:] - prefix[:-window_samples]
    start = int(np.argmax(scores))
    return x[start:start + window_samples].copy(), start

