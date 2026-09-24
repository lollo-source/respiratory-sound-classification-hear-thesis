"""Pinned official OPERA loading, preprocessing, and canonical readouts."""

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np

from .hear import sha256_file
from src.progress import LoopProgress


OPERA_REVISION = "3622310e667afb8aa40169050b4dd45de75946a2"
MODEL_UTIL_SHA256 = "bc637deacfde3154d2290eb063fb6b3a6c5dd33fc0c26e79a39a08dd0e5f185c"
UTIL_SHA256 = "96bd9692d85fe82a033dcc86e8d97fa4971ba9d04ea2437888f88d01e0e81af5"
MODEL_SPECS = {
    "operaCE": {"method": "OPERA-CE", "dimension": 1280, "batch_size": 64,
                "checkpoint": "encoder-operaCE.ckpt", "sha256": "2e18765ad90e584f7122dd739cd9e9b56d9013ce25660042ab8f6690cce74d39"},
    "operaCT": {"method": "OPERA-CT", "dimension": 768, "batch_size": 32,
                "checkpoint": "encoder-operaCT.ckpt", "sha256": "83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae"},
    "operaGT": {"method": "OPERA-GT", "dimension": 384, "batch_size": 8,
                "checkpoint": "encoder-operaGT.ckpt", "sha256": "64dff121368e30ee7b1c9b2e526eaf2af93ac63ce4a8712efd60ecda84618cb8"},
}


def validate_installation(opera_root, checkpoint_root):
    opera_root = Path(opera_root).expanduser().resolve()
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    if not opera_root.is_dir() or not checkpoint_root.is_dir():
        raise RuntimeError("explicit OPERA source and checkpoint directories are required")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=opera_root, text=True,
                              capture_output=True, check=True).stdout.strip()
    if revision != OPERA_REVISION:
        raise RuntimeError("pinned OPERA revision mismatch: %s" % revision)
    sources = {"model_util": opera_root / "src/benchmark/model_util.py", "util": opera_root / "src/util.py"}
    expected_sources = {"model_util": MODEL_UTIL_SHA256, "util": UTIL_SHA256}
    for name, path in sources.items():
        if not path.is_file() or sha256_file(path) != expected_sources[name]:
            raise RuntimeError("pinned OPERA source mismatch: %s" % name)
    models = {}
    for name, spec in MODEL_SPECS.items():
        path = checkpoint_root / spec["checkpoint"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise RuntimeError("pinned OPERA checkpoint mismatch: %s" % name)
        models[name] = {"method": spec["method"], "dimension": spec["dimension"],
                        "checkpoint_sha256": spec["sha256"]}
    return {"repository": "https://github.com/evelyn0414/OPERA.git", "revision": revision,
            "model_util_sha256": MODEL_UTIL_SHA256, "util_sha256": UTIL_SHA256, "models": models}


def _add_source(opera_root):
    root = Path(opera_root).resolve()
    opera_src = str(root / "src")
    package = sys.modules.get("src")
    if package is not None and hasattr(package, "__path__") and opera_src not in package.__path__:
        package.__path__.insert(0, opera_src)
    if str(root) not in sys.path: sys.path.insert(0, str(root))


def load_model(model_name, opera_root, checkpoint_root, device_name):
    validate_installation(opera_root, checkpoint_root)
    import torch
    if not str(device_name).startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("canonical OPERA extraction requires an explicit available CUDA device")
    _add_source(opera_root)
    if model_name == "operaGT":
        import timm
        from timm.models.swin_transformer import SwinTransformerBlock
        if "feat_size" not in inspect.signature(SwinTransformerBlock.__init__).parameters:
            raise RuntimeError("OPERA-GT requires the official OPERA timm patch; run upstream prepare_code.sh")
    from src.benchmark.model_util import initialize_pretrained_model
    device = torch.device(device_name); torch.cuda.set_device(device)
    model = initialize_pretrained_model(model_name)
    checkpoint = torch.load(Path(checkpoint_root) / MODEL_SPECS[model_name]["checkpoint"], map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("OPERA checkpoint/model state mismatch")
    model.to(device); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, device


def spectrograms(audio_path, model_name, opera_root, input_sec=8.0):
    _add_source(opera_root)
    from src.util import get_entire_signal_librosa, get_split_signal_librosa
    path = str(Path(audio_path)); stem = path[:-4] if path.lower().endswith(".wav") else path
    if model_name == "operaGT":
        return get_split_signal_librosa("", stem, spectrogram=True, input_sec=input_sec)
    return get_entire_signal_librosa("", stem, spectrogram=True, input_sec=input_sec, pad=True)


def infer_records(model, model_name, records, opera_root, device, input_sec=8.0):
    import torch

    spec = MODEL_SPECS[model_name]
    rows = []
    extraction_progress = LoopProgress("%s extraction" % spec["method"], len(records))
    for record_index, record in enumerate(records.itertuples(index=False), start=1):
        value = spectrograms(record.audio_path, model_name, opera_root, input_sec)
        if model_name == "operaGT":
            valid = [item for item in value if item.shape[0] >= 16]
            if not valid:
                raise RuntimeError("no valid OPERA-GT spectrogram for %s" % record.record_id)
            chunks = []
            for start in range(0, len(valid), spec["batch_size"]):
                tensor = torch.tensor(np.stack(valid[start:start + spec["batch_size"]]), dtype=torch.float32, device=device)
                with torch.inference_mode(): chunks.append(model.forward_feature(tensor).detach().cpu().numpy())
            rows.append(np.concatenate(chunks).mean(axis=0))
        else:
            tensor = torch.tensor(np.expand_dims(value, 0), dtype=torch.float32, device=device)
            with torch.inference_mode():
                rows.append(model.extract_feature(tensor, dim=spec["dimension"]).detach().cpu().numpy()[0])
        extraction_progress.update(record_index)
    result = np.stack(rows).astype(np.float32)
    if result.shape != (len(records), spec["dimension"]) or not np.isfinite(result).all():
        raise RuntimeError("unexpected OPERA representation")
    extraction_progress.complete()
    return result


def release_model(model):
    import torch
    del model; torch.cuda.empty_cache()
