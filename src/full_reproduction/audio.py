"""Canonical PCM decoding, polyphase resampling, and deterministic windowing."""

from __future__ import print_function

import math
import wave

import numpy as np
from scipy.signal import resample_poly

from . import SAMPLE_RATE, WINDOW_SAMPLES


def read_wav_float32(path):
    """Decode integer PCM with Python ``wave`` and mix channels by arithmetic mean."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        audio = ((values ^ sign_bit) - sign_bit).astype(np.float32) / float(1 << 23)
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError("unsupported WAV sample width: %d bytes" % sample_width)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return int(sample_rate), audio.astype(np.float32)


def canonical_resample(audio, source_rate, target_rate=SAMPLE_RATE):
    values = np.asarray(audio, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return values.astype(np.float32, copy=True)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(values, target_rate // divisor, source_rate // divisor).astype(np.float32)


def right_pad(values, size=WINDOW_SAMPLES):
    array = np.asarray(values, dtype=np.float32)
    if len(array) >= size:
        return array[:size].astype(np.float32, copy=True)
    return np.pad(array, (0, size - len(array)), mode="constant").astype(np.float32)


def maximum_energy_window(values, size=WINDOW_SAMPLES):
    """Return the earliest exhaustive maximum-squared-energy crop."""
    array = np.asarray(values, dtype=np.float32)
    if len(array) <= size:
        return right_pad(array, size), 0
    squared = array.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    scores = cumulative[size:] - cumulative[:-size]
    start = int(np.argmax(scores))
    return array[start:start + size].astype(np.float32, copy=True), start


def hf_fixed_windows(values, size=WINDOW_SAMPLES):
    """Eight windows at [0,2),...,[14,16), padding every incomplete region."""
    array = np.asarray(values, dtype=np.float32)
    return [right_pad(array[index * size:(index + 1) * size], size) for index in range(8)]


def complete_record_windows(values, size=WINDOW_SAMPLES):
    """Consecutive windows; only the final incomplete window is padded."""
    array = np.asarray(values, dtype=np.float32)
    count = max(1, int(math.ceil(len(array) / float(size))))
    return [right_pad(array[index * size:(index + 1) * size], size) for index in range(count)]


def load_resampled(path):
    source_rate, audio = read_wav_float32(path)
    values = canonical_resample(audio, source_rate, SAMPLE_RATE)
    if not len(values):
        raise RuntimeError("empty audio after resampling")
    return source_rate, values

