# Validation status

The public implementations for Quick Verification and all Full Reproduction scientific scopes have been exercised successfully. Current release evidence is:

- Quick Verification: 17/17 thesis tables reproduced and 17/17 verification PASS.
- Unit/regression discovery: 79 tests. In the latest targeted run, 77/77 non-Quick tests passed; two tests that directly invoke Quick Verification were not rerun because that workflow was explicitly excluded from the task.
- Protected release baseline: 77/77 files PASS size and SHA-256 verification.
- Recovered/resumed Full run used during intermediate validation: 14/17 applicable thesis tables reproduced and 14/14 verification PASS.
- Final second fully fresh end-to-end Full Reproduction after the top-level workflow process-isolation fix: 14/17 applicable thesis tables reproduced and 14/14 verification PASS.

The top-level workflow process-isolation fix uses fresh Python subprocesses for HeAR Harmonised, HeAR Challenge and OPERA, preventing numerical runtime state from leaking between workflows. `--resume` was exercised successfully with validated complete feature caches during intermediate validation. It was followed by a second fully fresh end-to-end Full Reproduction from a previously nonexistent output directory, with `--resume` omitted. That final run completed HeAR Harmonised, the official BioCAS Challenge workflow, OPERA CE/CT/GT, final result assembly and post-computation verification, and reported:

```text
Full Reproduction complete
14/17 applicable thesis tables reproduced
14/14 verification PASS
```

The generated Full report recorded 14 comparable tables, consistency at thesis display precision, and a maximum absolute numerical difference of `1.1102230246251565e-16`.

## Meaning of verification PASS

Verification runs only after the producer has emitted the declared table set. Overall `PASS` requires an error-free integrity preflight and every applicable machine-readable table to match its frozen reference structurally, numerically and at thesis display precision. Non-numeric values, Boolean values, dictionary key sets, list lengths and list ordering must match exactly. Numeric values must differ by no more than the implemented absolute tolerance of `1e-12` and must format identically at the table's declared display precision: four decimal places for most tables and one decimal place for the HF Lung device-distribution and Quick-only event-capture tables.

The calculation producers do not read frozen thesis values. Only the separate post-computation verifier reads `reference_results/`, after fresh tables exist, and records the reference hashes and per-table comparison status in `verification.json`.

The “Quick versus Full” line in a Full `RESULTS.md` is a separate comparison between the 14 freshly assembled Full table objects and the corresponding current `results/quick/machine_readable/*.json` files. It is not the frozen-reference verification result.

## Protected release baseline and Quick Verification

The protected release manifest covers 77 files; all 77 passed their recorded size and SHA-256 checks. Protected scientific artifacts, references, protocol constants and thesis identity metadata are unchanged by this documentation update.

Quick Verification reproduced all 17 thesis tables and passed all 17 post-computation table checks. It calculates in an isolated temporary workspace and publishes only after all newly generated tables are present and verification passes.

## Harmonised HeAR evidence

- Public preflight reconstructed 9,765 HF Lung records and 3,323 included BioCAS records solely from the public manifests and explicit dataset roots.
- Two fresh model loads over the first 32 batches/1,024 windows of each dataset produced bit-identical CLS and Temporal Grid arrays.
- Full extraction produced 78,120 HF windows and 19,895 BioCAS windows. Complete fresh arrays and record aggregates were bit-identical to the independent audit evidence.
- Fresh downstream fitting produced 21,048 prediction rows. Ground truth, class order, selected alpha, predicted class, and all metric cells matched the certified evidence.
- The maximum probability difference was zero except for an already audited BioCAS fusion last-bit difference of `2.220446049250313e-16`; no class, decision, displayed value, or metric changed.
- A diagnostic multithreaded HF replay initially exposed 14 solver-level differences. Reapplying the canonical one-thread downstream contract removed all differences without altering data or expected values.

The internal validation feature arrays were approximately 9.84 GB and are deliberately not distributed or consumed by the public workflow.

| Dataset | CLS window SHA-256 | Temporal Grid window SHA-256 |
|---|---|---|
| HF Lung | `56c9134326d35bbbb82c5cf1a624fce8c6a0f3d88fe9e4083a87075e5825a7be` | `455e7a24d98f614f704ab0374eed49f244edbde92ed9c91f9780415b71650360` |
| BioCAS harmonised | `61a12f5f5ca1b1a875c645fde0fde21c661af2fb7a55f1131c2da34dffdd46b6` | `afa2447dee80fc784341e0755bff9197e6286687fedbd13e72817fdd1aaf7b04` |

## Official BioCAS Challenge evidence

- Public preflight reconstructed all 3,554 official records with deterministic task-specific grouped folds and no group leakage.
- First-batch and two-reload subset checks were bit-identical for both representations.
- Complete fresh feature SHA-256 values were:

| Dataset | CLS | Temporal Grid |
|---|---|---|
| BioCAS2022 | `1bc6638786062950d3fb246a19886b6df0a746c64a75985994c1ca27c3d441d7` | `b6652fa8e13fe526c30b48e3bc0e3416e72d878093d47cdcf1ba8cbce061001a` |
| BioCAS2023 | `f6f93c9d172ec3b9bd57e3e62579187d7ba6cc84aea1fc3f302a9106109ccf01` | `f6648be71bedca3dc794a44845adc1eeb5b5fcfd5fea3a0a60bc7d2ec195c287` |

- Training-only selection independently recovered SDP temperatures 100 for T2-1 and 50 for T2-2. The T2-2 result depends on the documented strict incumbent threshold and canonical Challenge thread environment.
- Fresh downstream output contained 9,630 prediction rows. Keys, truths, class orders, predictions, probabilities, and all 12 branch metric rows matched exactly.
- Published comparison values were accessed only after fresh scoring and only to derive retrospective, non-official placement estimates.

## OPERA evidence

- The public adapter reconstructed the 3,323-record harmonised BioCAS order using only public pseudonyms.
- First-eight-record extraction and two complete checkpoint reloads were bit-identical for CE, CT, and GT.
- Full fresh feature files had these SHA-256 values:

| Model | Shape | SHA-256 |
|---|---:|---|
| OPERA-CE | `[3323,1280]` | `b9c4bcfd3d6c22425900b2a94314c31024e7387dfca11a738d820c4dc09588f1` |
| OPERA-CT | `[3323,768]` | `d94b9579e83013b9dafcda8f1cf08e75f21f41a5e108acbee6629508f5967060` |
| OPERA-GT | `[3323,384]` | `0ddc0cb7e6c5b4e37fffd9f8a0e7ac4a20f4806d75f6fa55255c2416084e7db8` |

- Fresh downstream fitting produced 9,312 prediction rows. All keys, truths, class orders, predictions, and 27,936 probability cells matched exactly.
- Confusion matrices and displayed thesis values were identical. Six raw metric floats differed only by reduction order, with maximum absolute difference `1.1102230246251565e-16`; no scientific conclusion changed.

## Independence, integrity, and privacy

The Full calculation modules do not read frozen thesis references, packaged Quick predictions, historical result directories, or internal validation features. Expected thesis metrics and selected values are not embedded as correction constants. References are accessed only by the separate post-computation verifier.

Generated metadata contains only release-local pseudonyms. Validation found no exported private path, credential, source identifier, reversible mapping, or unexpected symlink. Full runtime caches and result bundles are Git-ignored.

The pinned public dependency files and scientific source/model/checkpoint hashes define the certified software inputs. GPU model, NVIDIA driver packaging and other hardware-dependent details can vary as long as the documented Python/CUDA checks and all runtime identity validations pass.

Protected provenance files retain some legacy internal labels and relative development paths for hash/audit continuity. As explained in [RESULT_LINEAGE.md](RESULT_LINEAGE.md), these entries are historical metadata, not runtime paths or reviewer setup requirements.

## Final clean-room Full validation

No scientific workflow or clean-room Full rerun remains pending. The second fully fresh Full Reproduction completed successfully from a previously nonexistent `--output-dir`, with `--resume` omitted, after the top-level workflow process-isolation fix. Final result assembly and post-computation verification completed with 14/17 applicable thesis tables reproduced and 14/14 verification PASS.
