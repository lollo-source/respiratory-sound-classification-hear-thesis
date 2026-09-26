# Thesis table map

The 17 tables listed below constitute the complete set of quantitative thesis
tables covered by this reproducibility release. Quick Verification reproduces
and verifies all 17; Full Reproduction independently reconstructs the 14 tables
for which the complete raw-data-to-result workflow is scientifically
applicable, while the remaining three are explicitly Quick-only for the
documented event-annotation and frozen-bootstrap-provenance reasons.

Quick Verification calculates all thesis-formatted tables in an isolated temporary workspace, requires all 17 newly generated tables, then publishes them to `results/quick/{tables,machine_readable}/` and renders `results/quick/RESULTS.md`. Frozen references under `reference_results/` are read only by the post-computation verifier.

Full runtime paths below are relative to `<FULL_OUTPUT_DIR>`. Harmonised HeAR writes below `datasets/<dataset>/`, Challenge below `challenge/`, and OPERA below `opera/`. The complete `--workflow all --stage all` run publishes the 14 scientifically applicable tables to repository-root `results/full/`, not below `<FULL_OUTPUT_DIR>`. The remaining three tables depend on packaged event annotations or frozen bootstrap draws and are explicitly Quick-only.

| Thesis label | Quick Verification origin | Full Reproduction origin | Status |
|---|---|---|---|
| `tab:hflung_class_distribution` | Packaged HF labels | `datasets/hf_lung/record_manifest.csv.gz` and `preflight.json` | Quick + Full |
| `tab:hflung_device_distribution` | Packaged HF device metadata | `datasets/hf_lung/record_manifest.csv.gz` | Quick + Full |
| `tab:biocas_harmonised_class_distribution` | Packaged harmonised BioCAS labels | `datasets/biocas_harmonised/record_manifest.csv.gz` and `preflight.json` | Quick + Full |
| `tab:biocas_challenge_class_distribution` | Packaged official Challenge labels | `challenge/datasets/biocas2022/record_manifest.csv.gz` and `challenge/datasets/biocas2023/record_manifest.csv.gz` | Quick + Full |
| `tab:single_window_results` | Packaged predictions and labels | Harmonised downstream phase-1 predictions and metrics | Quick + Full |
| `tab:single_window_per_class_recall` | Packaged predictions and labels | Harmonised downstream phase-1 predictions | Quick + Full |
| `tab:single_window_event_capture` | Packaged predictions and event-overlap annotations | No Full stage emits the annotation-overlap input | Quick-only |
| `tab:complete_record_results` | Packaged predictions and labels | Harmonised downstream phase-1/phase-2 predictions and metrics | Quick + Full |
| `tab:complete_record_per_class_recall` | Packaged predictions and labels | Harmonised downstream phase-1/phase-2 predictions | Quick + Full |
| `tab:dual_readout_results` | Packaged predictions and labels | Harmonised downstream phase-2/phase-3 predictions and metrics | Quick + Full |
| `tab:dual_readout_per_class_recall` | Packaged predictions and labels | Harmonised downstream phase-2/phase-3 predictions | Quick + Full |
| `tab:overall_harmonised_comparison` | Packaged predictions and labels | Harmonised downstream phase-1/phase-3 predictions and metrics | Quick + Full |
| `tab:hear_opera_comparison` | Packaged HeAR/OPERA predictions | Fresh OPERA metrics plus explicit fresh harmonised HeAR metrics | Quick + Full |
| `tab:biocas_challenge_placements` | Packaged Challenge predictions plus static literature subsets | Fresh Challenge Fusion metrics plus static literature subsets | Quick + Full; placements are retrospective |
| `tab:challenge_readout_comparison` | Packaged Challenge predictions | `challenge/downstream/{predictions.csv.gz,metrics.json}` | Quick + Full |
| `tab:bootstrap_harmonised` | Packaged predictions, patient groups, and frozen draws | No Full bootstrap/draw-index stage | Quick-only |
| `tab:bootstrap_challenge_readouts` | Packaged predictions, patient groups, and frozen draws | No Full bootstrap/draw-index stage | Quick-only |

Coverage is **17/17** for Quick Verification and **14/17** for complete Full Reproduction. Both modes verify only after fresh table generation; Full records the three intentional exclusions in `RESULTS.md` and `manifest.json`.
