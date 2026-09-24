"""Audited HeAR Temporal Grid readout."""

import numpy as np

PATCH_TIME = 12
PATCH_MEL = 8
TOKEN_DIM = 1024
PATCH_TOKENS = PATCH_TIME * PATCH_MEL
OUTPUT_DIM = 2 * PATCH_TIME * TOKEN_DIM


def temporal_grid_readout(patch_tokens):
    values = np.asarray(patch_tokens)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (PATCH_TOKENS, TOKEN_DIM):
        raise ValueError("expected [96,1024] or [B,96,1024]")
    grid = values.reshape(values.shape[0], PATCH_TIME, PATCH_MEL, TOKEN_DIM)
    means = grid.mean(axis=2).reshape(values.shape[0], -1)
    stds = grid.std(axis=2, ddof=0).reshape(values.shape[0], -1)
    result = np.concatenate((means, stds), axis=1)
    return result[0] if squeeze else result

