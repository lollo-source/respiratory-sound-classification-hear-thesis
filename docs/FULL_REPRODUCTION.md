# Full Reproduction

Full Reproduction starts from authorised raw audio and pinned external model resources. It runs HeAR Harmonised, HeAR Challenge and OPERA, then assembles and verifies the 14 scientifically applicable thesis tables. It does not read packaged Quick predictions, frozen thesis values, historical embeddings or earlier result tables as scientific inputs.

## Prerequisites

Complete these guides before running Full:

1. [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) — Python 3.10, isolated environment, Full requirements and CUDA checks
2. [DATASET_SETUP.md](DATASET_SETUP.md) — HF Lung and SPRSound/BioCAS acquisition, layout and counts
3. [MODEL_SETUP.md](MODEL_SETUP.md) — HeAR source/model snapshot and OPERA source/checkpoints

All six input roots in the complete command must already exist with the documented contents. Full Reproduction validates them; it does not download, discover or repair external inputs.

## Complete clean-room run

From the repository root:

```bash
python scripts/full_reproduce.py \
  --workflow all \
  --stage all \
  --hf-lung-root /path/to/HF_Lung \
  --sprsound-root /path/to/SPRSound \
  --hear-repo /path/to/hear \
  --hear-model-path /path/to/hear-model \
  --opera-root /path/to/OPERA \
  --opera-checkpoint-root /path/to/opera-checkpoints \
  --device cuda:0 \
  --output-dir /path/to/full-output
```

`--output-dir` may be nonexistent; the program creates it. For the strongest clean-room reproduction, choose a brand-new output path, preferably outside the Git checkout, and omit `--resume`.

### Arguments

| Argument | Meaning and expected contents | Existing input or generated output |
|---|---|---|
| `--workflow all` | Run HeAR Harmonised, HeAR Challenge and OPERA, then assemble/verify/publish the final Full bundle. Other accepted scopes are `hear-harmonised`, `hear-challenge` (also accepted as `biocas-challenge`) and `opera`. | Selection |
| `--stage all` | Run every stage supported by the selected workflow. With `--workflow all`, this value is required. | Selection |
| `--hf-lung-root` | HF_Lung_V1 root containing `train/` and `test/` with 7,809 and 1,956 WAV files. | Existing input directory |
| `--sprsound-root` | SPRSound root containing the BioCAS 2022 train/test and BioCAS 2023 test WAV directories, 3,554 raw records total. | Existing input directory |
| `--hear-repo` | Google-Health/hear checkout containing the hash-validated `python/data_processing/audio_utils.py`. | Existing input directory |
| `--hear-model-path` | Pinned `google/hear-pytorch` snapshot containing hash-validated `config.json` and `pytorch_model.bin` at its root. | Existing input directory |
| `--opera-root` | OPERA checkout at commit `3622310e667afb8aa40169050b4dd45de75946a2`, prepared with `prepare_code.sh`. | Existing input directory |
| `--opera-checkpoint-root` | Directory containing the three hash-validated `encoder-operaCE.ckpt`, `encoder-operaCT.ckpt` and `encoder-operaGT.ckpt` files. | Existing input directory |
| `--device cuda:0` | Explicit CUDA device used for encoder inference. It must exist and be visible to PyTorch. | Runtime selection |
| `--output-dir` | Root for generated runtime/intermediate artifacts: manifests, validation arrays, feature caches, predictions and metrics. The directory is created if absent. | Generated output directory |

When `CUDA_VISIBLE_DEVICES` is set, `--device cuda:0` selects the first logically visible GPU, which is not necessarily physical GPU 0.

Additional CLI options used for scoped or resumed runs are:

| Argument | Meaning |
|---|---|
| `--datasets hf_lung biocas_harmonised` | Select harmonised datasets; applies only to `--workflow hear-harmonised` and defaults to both. |
| `--hear-metrics /path/to/metrics.json` | For standalone `--workflow opera`, points to freshly generated harmonised HeAR downstream metrics. `--workflow all` connects this automatically. |
| `--resume` | Reuse only complete feature caches that pass all identity, protocol, size and hash checks. It does not mean “continue any partial file.” |

## Workflows and stages

| Workflow | Ordered stages |
|---|---|
| HeAR Harmonised (`hear-harmonised`) | `preflight` → `first-batch` → `subset` → `extract` → `downstream` |
| HeAR Challenge (`hear-challenge`) | `preflight` → `first-batch` → `subset` → `extract` → `downstream` → `published` |
| OPERA (`opera`) | `preflight` → `first-batch` → `subset` → `extract` → `downstream` |
| Final result handling in `workflow all` | assembly → verification → publication |

OPERA's `first-batch` stage validates CE, CT and GT on the first eight harmonised records; `subset` reloads every model twice and requires exact equality before full extraction.

When `--workflow all --stage all` is used, each top-level scientific workflow runs in a fresh Python subprocess. This isolates numerical runtime state between HeAR Harmonised, HeAR Challenge and OPERA. Final assembly and verification run only after all three subprocesses succeed.

Scoped examples:

```bash
python scripts/full_reproduce.py \
  --workflow hear-harmonised --stage all \
  --hf-lung-root /path/to/HF_Lung \
  --sprsound-root /path/to/SPRSound \
  --hear-repo /path/to/hear \
  --hear-model-path /path/to/hear-model \
  --device cuda:0 \
  --output-dir /path/to/full-output

python scripts/full_reproduce.py \
  --workflow hear-challenge --stage all \
  --sprsound-root /path/to/SPRSound \
  --hear-repo /path/to/hear \
  --hear-model-path /path/to/hear-model \
  --device cuda:0 \
  --output-dir /path/to/full-output

python scripts/full_reproduce.py \
  --workflow opera --stage all \
  --sprsound-root /path/to/SPRSound \
  --opera-root /path/to/OPERA \
  --opera-checkpoint-root /path/to/opera-checkpoints \
  --hear-metrics /path/to/full-output/datasets/biocas_harmonised/downstream/metrics.json \
  --device cuda:0 \
  --output-dir /path/to/full-output
```

An individual stage can replace `--stage all`, but later stages require the validated outputs of their prerequisites.

## Resume, caches and interruptions

- A new output directory without `--resume` performs completely fresh extraction.
- With `--resume`, an existing complete feature cache is reused only after validation of its source/model/checkpoint identity, protocol manifest, record/window counts, shapes, file sizes and file hashes.
- Invalid, incomplete or non-empty feature caches are rejected; they are never silently accepted as complete.
- Without `--resume`, reaching a non-empty feature directory causes refusal rather than automatic deletion.
- Downstream fitting/evaluation is recomputed whenever that stage is reached, even when valid feature caches are reused.
- Stale unrelated files under `--output-dir` are not automatically cleaned.

An interrupted extraction can leave an incomplete non-empty feature directory. `--resume` does not promise recovery of arbitrary partial arrays, so the incomplete cache will be rejected. The conservative response is to diagnose the interruption and begin again with a new output directory. A fully fresh reviewer test should always use a previously nonexistent `--output-dir` and omit `--resume`.

Extraction loops report progress. Downstream fitting and cross-validation can take several minutes without intermediate updates; the command prints:

```text
Fitting, cross-validation and evaluation are running; this stage may take
several minutes without intermediate output.
```

Do not treat that expected quiet interval alone as a failure.

## Outputs

Runtime and intermediate artifacts live under `--output-dir`:

```text
/path/to/full-output/
├── datasets/               # harmonised manifests, validation, features and downstream outputs
├── challenge/              # Challenge manifests, validation, features and downstream outputs
├── opera/                  # OPERA validation, features and downstream outputs
└── run_manifest.json
```

The final reviewer-facing result bundle is separate and is always published below the repository root:

```text
results/full/
├── RESULTS.md
├── tables/*.csv
├── machine_readable/*.json
├── manifest.json
└── verification.json
```

`results/full/` is **not** inside `--output-dir`. To verify an already published bundle:

```bash
python scripts/verify_results.py --mode full --results-dir results/full
```

Expected successful complete-run summary:

```text
Full Reproduction complete
14/17 applicable thesis tables reproduced
14/14 verification PASS
results/full/RESULTS.md
```

Full reports 14/17 because `tab:single_window_event_capture` and the two patient-cluster bootstrap tables require packaged provenance intentionally handled by Quick Verification only. See [TABLE_MAP.md](TABLE_MAP.md), [EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md), [RESULT_LINEAGE.md](RESULT_LINEAGE.md) and [VALIDATION.md](VALIDATION.md).
