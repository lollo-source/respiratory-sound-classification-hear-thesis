"""Level-A audio contract utilities (not exercised by Level C)."""

import numpy as np


def to_mono(waveform):
    values = np.asarray(waveform)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError("waveform must be [samples] or [channels, samples]")
    return values.mean(axis=0)


def linear_resample(waveform, source_rate, target_rate=16000):
    """Deterministic dependency-light reference resampler for interface tests.

    Certified Level A will use the pinned canonical resampler; this helper makes
    the data contract explicit and is not claimed as Level-A certified.
    """
    x = np.asarray(waveform, dtype=np.float32)
    if source_rate == target_rate:
        return x.copy()
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    target_length = int(round(len(x) * float(target_rate) / float(source_rate)))
    if not len(x) or not target_length:
        return np.zeros(target_length, dtype=np.float32)
    old = np.linspace(0.0, 1.0, len(x), endpoint=False)
    new = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(new, old, x).astype(np.float32)

