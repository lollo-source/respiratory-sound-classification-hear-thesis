# Environment setup

Run commands in this guide from the repository root. Python **3.10** is required; use a separate virtual environment so the pinned scientific packages do not interact with system packages or another project.

## System prerequisites

The setup commands assume a POSIX-like Linux environment with Git, Python 3.10 and its `venv` support, `curl`, and a 7-Zip-compatible `7z` command. Git is used for repository acquisition, `7z` for HF Lung extraction, and `curl` for OPERA checkpoint downloads.

Full Reproduction additionally requires an NVIDIA GPU, a compatible NVIDIA driver, and a CUDA-capable PyTorch installation from the pinned requirements. The required driver depends on the CUDA runtime supplied by the installed PyTorch build; no single driver version is claimed for every supported machine.

## Quick Verification environment

Quick Verification is CPU-only and has the minimum dependency set:

```bash
python3.10 -m venv .venv-quick
source .venv-quick/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-quick-verification.txt
python -c "import sys, numpy; assert sys.version_info[:2] == (3, 10); print(sys.version.split()[0], numpy.__version__)"
```

## Full Reproduction environment

Full Reproduction uses the certified versions in `requirements-full-reproduction.txt`:

```bash
python3.10 -m venv .venv-full
source .venv-full/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-full-reproduction.txt
python -c "import sys, torch, numpy, sklearn; assert sys.version_info[:2] == (3, 10); assert torch.cuda.is_available(), 'CUDA is not visible to PyTorch'; print('python', sys.version.split()[0], 'torch', torch.__version__, 'torch-cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
python scripts/full_reproduce.py --help
```

The minimum documented platform requirements are Python 3.10, a POSIX-like environment for the supplied setup commands, and an NVIDIA CUDA-capable GPU visible to PyTorch for Full encoder inference. The dependency file pins PyTorch `2.6.0`, torchvision `0.21.0` and torchaudio `2.6.0`. The host NVIDIA driver must support the CUDA runtime reported by the installed PyTorch build. Driver version, CUDA wheel availability and GPU memory are hardware/platform dependent; this repository does not require or claim one universal NVIDIA driver version.

Approximately 32 GB system RAM is recommended for high-dimensional Temporal Grid fitting. A complete harmonised HeAR feature cache can require about 11 GB of storage, in addition to Challenge, OPERA and result artifacts.

## OPERA patch ordering

Install `requirements-full-reproduction.txt` first so that the target `timm` package exists. Then clone and check out OPERA, activate `.venv-full`, and run OPERA's `prepare_code.sh`. That script copies OPERA's patched Swin Transformer implementation into the active environment's `timm` installation; running it in a different environment patches the wrong installation.

After the model steps in [MODEL_SETUP.md](MODEL_SETUP.md), validate the required interface:

```bash
python -c "import inspect; from timm.models.swin_transformer import SwinTransformerBlock; assert 'feat_size' in inspect.signature(SwinTransformerBlock.__init__).parameters; print('OPERA timm interface: OK')"
```

Full Reproduction repeats this interface validation before loading OPERA-GT. Source, model and checkpoint hashes are also checked at runtime; passing this environment check alone does not replace those integrity checks.

## Certified versus portable details

The exact package versions in `requirements-full-reproduction.txt`, pinned model/checkpoint hashes and workflow contracts define the certified software setup. Python 3.10 and CUDA-capable PyTorch execution are minimum requirements. GPU model, driver packaging, available memory, filesystem location and wall-clock duration are machine-dependent; keep those local choices out of configuration committed for review.
