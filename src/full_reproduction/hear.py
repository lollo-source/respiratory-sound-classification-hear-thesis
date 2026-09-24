"""Pinned HeAR loading and same-forward CLS/Temporal-Grid inference."""

from __future__ import print_function

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from . import (
    CLS_DIM,
    MODEL_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PREPROCESSING_SHA256,
    TG_DIM,
    WINDOW_SAMPLES,
)


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_grid_readout(patch_tokens):
    values = np.asarray(patch_tokens, dtype=np.float32)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (96, 1024):
        raise ValueError("expected patch tokens [96,1024] or [B,96,1024]")
    grid = values.reshape(values.shape[0], 12, 8, 1024)
    means = grid.mean(axis=2).reshape(values.shape[0], -1)
    stds = grid.std(axis=2, ddof=0).reshape(values.shape[0], -1)
    output = np.concatenate((means, stds), axis=1).astype(np.float32)
    if output.shape[1] != TG_DIM:
        raise RuntimeError("unexpected Temporal Grid dimension")
    return output[0] if squeeze else output


def validate_model_and_preprocessing(model_path, hear_repo):
    model_path = Path(model_path).expanduser().resolve()
    hear_repo = Path(hear_repo).expanduser().resolve()
    weights = model_path / "pytorch_model.bin"
    config = model_path / "config.json"
    preprocessing = hear_repo / "python" / "data_processing" / "audio_utils.py"
    required = [weights, config, preprocessing]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing pinned HeAR files: %s" % missing)
    actual = {
        "weights": sha256_file(weights),
        "config": sha256_file(config),
        "preprocessing": sha256_file(preprocessing),
    }
    expected = {
        "weights": MODEL_WEIGHTS_SHA256,
        "config": MODEL_CONFIG_SHA256,
        "preprocessing": PREPROCESSING_SHA256,
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise RuntimeError("pinned HeAR hash mismatch: %s" % ", ".join(mismatches))
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_sha256": actual["weights"],
        "config_sha256": actual["config"],
        "preprocessing_sha256": actual["preprocessing"],
    }


def _load_preprocess(path):
    spec = importlib.util.spec_from_file_location("full_reproduction_hear_audio_utils", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned HeAR preprocessing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_audio


def load_model(model_path, hear_repo, device_name):
    provenance = validate_model_and_preprocessing(model_path, hear_repo)
    import torch
    from transformers import AutoModel

    if not str(device_name).startswith("cuda:"):
        raise RuntimeError("canonical full extraction requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device_index = int(str(device_name).split(":", 1)[1])
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable: %s" % device_name)
    torch.cuda.set_device(device_index)
    device = torch.device(device_name)
    resolved_model_path = Path(model_path).expanduser().resolve()
    preprocessing_path = Path(hear_repo).expanduser().resolve() / "python" / "data_processing" / "audio_utils.py"
    preprocess = _load_preprocess(preprocessing_path)
    model = AutoModel.from_pretrained(
        str(resolved_model_path),
        use_safetensors=False,
        trust_remote_code=False,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance["device"] = str(device)
    provenance["device_name"] = torch.cuda.get_device_name(device)
    provenance["device_capability"] = list(torch.cuda.get_device_capability(device))
    return model, preprocess, device, provenance


def infer_dual(model, preprocess, waveforms, device):
    import torch

    batch = np.asarray(waveforms, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != WINDOW_SAMPLES:
        raise RuntimeError("expected waveform batch [B,%d], got %s" % (WINDOW_SAMPLES, batch.shape))
    raw = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.inference_mode():
        processed = preprocess(raw).to(device)
        output = model.forward(processed, return_dict=True)
        cls = output.pooler_output.detach().cpu().numpy().astype(np.float32)
        patch = output.last_hidden_state[:, 1:, :].detach().cpu().numpy().astype(np.float32)
    temporal_grid = temporal_grid_readout(patch)
    if cls.shape != (len(batch), CLS_DIM) or temporal_grid.shape != (len(batch), TG_DIM):
        raise RuntimeError("unexpected HeAR outputs: CLS=%s TG=%s" % (cls.shape, temporal_grid.shape))
    return cls, temporal_grid


def release_model(model):
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
