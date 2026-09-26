# Model setup for Full Reproduction

Complete [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) first. Model source and weights are not redistributed and are never downloaded implicitly by Full Reproduction.

## HeAR implementation

`--hear-repo` is a local checkout of the official [Google-Health/hear](https://github.com/Google-Health/hear) implementation. The validated checkout was `36bbb90879545a2f1106668048b1a12a61ff2a0d`:

```bash
git clone https://github.com/Google-Health/hear.git /path/to/hear
git -C /path/to/hear checkout 36bbb90879545a2f1106668048b1a12a61ff2a0d
```

The runtime does not trust a repository name or Git branch. It requires this file and verifies its content hash:

```text
/path/to/hear/
└── python/data_processing/audio_utils.py
```

```text
1f2dcc0406e86d1162c97c91782e8488f290b9f41006d7f5fd5035bbf4c88957  python/data_processing/audio_utils.py
```

The code validates the file hash rather than comparing the HeAR checkout's Git `HEAD`, so another checkout is accepted only if the required preprocessing file is byte-identical.

## HeAR model snapshot

HeAR weights are intentionally not stored in this repository.
`--hear-model-path` must instead point to a local snapshot of
[google/hear-pytorch](https://huggingface.co/google/hear-pytorch) at exact
revision `f791cd42437c3e268c8ac84707e3508900f65f1a`. Access is gated by the Health
AI Developer Foundations terms. Using your own Hugging Face account, sign in
and accept the upstream terms on the model page before continuing.

With the Full environment active, authenticate through the Python API already
provided by the pinned `huggingface-hub` package, then download the exact
revision. These commands do not require the `hf` CLI or an additional `click`
installation:

```bash
python -c "from huggingface_hub import interpreter_login; interpreter_login()"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/hear-pytorch', revision='f791cd42437c3e268c8ac84707e3508900f65f1a', local_dir='/path/to/hear-model')"
```

The two required files must be directly below the path passed to `--hear-model-path`:

```text
/path/to/hear-model/
├── config.json
└── pytorch_model.bin
```

The repository's runtime SHA-256 checks then verify that the downloaded
snapshot contains the certified files:

```text
9774a77892e39ca8798aacfe68287a7cbd280993175ea93ca8424ffc75130e60  config.json
d44d355816ee4315f67d7810da274409e9b1a6570325fc5ba9ae27555fd81723  pytorch_model.bin
```

Verify all three HeAR files before a long run:

```bash
sha256sum \
  /path/to/hear-model/config.json \
  /path/to/hear-model/pytorch_model.bin \
  /path/to/hear/python/data_processing/audio_utils.py
```

## OPERA source

`--opera-root` is a local checkout of the official [evelyn0414/OPERA](https://github.com/evelyn0414/OPERA) repository at pinned commit `3622310e667afb8aa40169050b4dd45de75946a2`:

```bash
git clone https://github.com/evelyn0414/OPERA.git /path/to/OPERA
git -C /path/to/OPERA checkout 3622310e667afb8aa40169050b4dd45de75946a2
```

The loader verifies Git `HEAD` and these source hashes:

```text
bc637deacfde3154d2290eb063fb6b3a6c5dd33fc0c26e79a39a08dd0e5f185c  src/benchmark/model_util.py
96bd9692d85fe82a033dcc86e8d97fa4971ba9d04ea2437888f88d01e0e81af5  src/util.py
```

With `.venv-full` active and the Full requirements already installed, apply the upstream patch and validate `timm`:

```bash
cd /path/to/OPERA
bash prepare_code.sh
cd -
python -c "import inspect; from timm.models.swin_transformer import SwinTransformerBlock; assert 'feat_size' in inspect.signature(SwinTransformerBlock.__init__).parameters; print('OPERA timm interface: OK')"
```

The patch must be applied after `timm` is installed and repeated if the environment's `timm` installation is replaced.

## OPERA checkpoints

`--opera-checkpoint-root` is a directory containing the three official CE, CT and GT files from [evelyn0414/OPERA on Hugging Face](https://huggingface.co/evelyn0414/OPERA/tree/main). The official repository provides direct downloads:

```bash
mkdir -p /path/to/opera-checkpoints
curl -L https://huggingface.co/evelyn0414/OPERA/resolve/d8de4322870b596f0a6ff6ea907b9a6996cd243a/encoder-operaCE.ckpt -o /path/to/opera-checkpoints/encoder-operaCE.ckpt
curl -L https://huggingface.co/evelyn0414/OPERA/resolve/d8de4322870b596f0a6ff6ea907b9a6996cd243a/encoder-operaCT.ckpt -o /path/to/opera-checkpoints/encoder-operaCT.ckpt
curl -L https://huggingface.co/evelyn0414/OPERA/resolve/d8de4322870b596f0a6ff6ea907b9a6996cd243a/encoder-operaGT.ckpt -o /path/to/opera-checkpoints/encoder-operaGT.ckpt
```

Expected layout and SHA-256 values:

```text
/path/to/opera-checkpoints/
├── encoder-operaCE.ckpt  2e18765ad90e584f7122dd739cd9e9b56d9013ce25660042ab8f6690cce74d39
├── encoder-operaCT.ckpt  83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae
└── encoder-operaGT.ckpt  64dff121368e30ee7b1c9b2e526eaf2af93ac63ce4a8712efd60ecda84618cb8
```

```bash
sha256sum /path/to/opera-checkpoints/encoder-operaCE.ckpt \
  /path/to/opera-checkpoints/encoder-operaCT.ckpt \
  /path/to/opera-checkpoints/encoder-operaGT.ckpt
```

Full Reproduction rejects a wrong OPERA commit, altered source file, missing patch interface, renamed/missing checkpoint, or checkpoint hash mismatch before scientific extraction proceeds.
