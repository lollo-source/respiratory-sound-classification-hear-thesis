# Result lineage

Full HeAR lineage: authorised audio → mono/canonical resampling → maximum-energy crop or complete-record windows → frozen HeAR → CLS and Temporal Grid readouts → record aggregation → independent scaled logistic-regression branches → training-only alpha selection → probabilities/predictions → metrics/deltas → tables.

The implemented harmonised, official-Challenge, and OPERA Full Reproduction calculations follow those lineages through fresh metrics. Runtime inputs are only authorised raw-audio roots, included public manifests, explicit model source/checkpoint paths, and public configuration. Generated features and predictions are stored below the selected `--output-dir`, physically separate from Quick Verification artifacts and frozen thesis references. The final reviewer-facing bundle is published at repository-root `results/full/`, never below `--output-dir`.

Full OPERA lineage: authorised BioCAS audio → native OPERA CE/CT/GT extraction → model-specific embeddings → harmonised split/downstream contract → predictions → balanced accuracy/macro-F1/accuracy → comparison table.

The OPERA producer writes and verifies hashes for the public source manifest, generated pseudonymous record manifest, official checkpoint, and fresh embedding array before downstream fitting. The HeAR rows used in the final comparison come from the already-certified fresh harmonised Full Reproduction metrics, never from the frozen thesis table.

Quick Verification begins at the included pseudonymised predictions. It computes every metric and derived table independently, then invokes the separate verifier. It never uses frozen thesis results as computational inputs.

Challenge lineage: authorised audio → deterministic 2 s windows → one shared HeAR forward → CLS SDP or Temporal Grid q75 → independent classifiers → 0.5 probability fusion → official SE/SP/AS/HS/Score → readout table. Fusion scores are inserted into separately stored organiser-published comparison subsets to derive explicitly non-official placements.

Bootstrap lineage: saved BioCAS2022 record predictions + pseudonymous patient-group index + frozen patient draw indices → per-patient confusion matrices → multiplicity-preserving paired replicate matrices → metrics/differences → linear percentile intervals → tables. No model is retrained.

Every packaged Quick Verification input has source and packaged hashes in `manifests/ARTIFACTS.json`. Quick calculations run in an isolated temporary workspace and publish validated output to `results/quick/`; frozen comparison values remain under `reference_results/`.

The Full Reproduction calculation modules do not import or read `reference_results/`, `artifacts/predictions/`, historical result directories, or internal feature evidence. Comparisons against frozen evidence are a separate post-computation verification activity and are not part of the public producer.

## Protected historical provenance

Some protected provenance metadata retains development-era names, including the `reproduction_levels` field in `THESIS_IDENTITY.json`, relative source paths in `manifests/ARTIFACTS.json` and the `artifacts/generated/level_c/` directory name. They record how frozen Quick artifacts and references were assembled and are retained for baseline integrity. The `github_release_status` field in `THESIS_IDENTITY.json` is likewise a pre-push historical snapshot retained for protected-baseline integrity, not the current repository publication state; the live GitHub repository state supersedes that descriptive field. These entries are not external paths, runtime inputs, workflow names or setup requirements. Reviewers should use only the public Quick Verification and Full Reproduction instructions.
