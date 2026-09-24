# Final experimental protocol

HeAR is frozen at revision `f791cd42437c3e268c8ac84707e3508900f65f1a`. Audio is mono, 16 kHz, and represented in 32,000-sample (2 s) windows. Single Window selects the earliest maximum squared-energy crop and right-pads short records. Complete Record starts at sample zero, uses consecutive non-overlapping windows, right-pads only a final partial window, and adds no window for an exact multiple; CLS record pooling is the arithmetic mean.

Temporal Grid maps 96 patch tokens to `12 × 8 × 1024`, computes mean and population standard deviation (`ddof=0`) over eight mel positions, preserves 12 time positions, and concatenates to 24,576 dimensions. Harmonised record pooling is arithmetic mean. The classifier is `StandardScaler` followed by L2 `LogisticRegression(C=1, lbfgs, max_iter=5000, class_weight=None)`.

Harmonised fusion uses `alpha*p_CLS + (1-alpha)*p_TG`. Training-only grouped OOF selection produced HF Binary/Multiclass alpha 0.85/0.65 and SPRSound Binary/Multiclass 0.75/0.65.

The Challenge retains Poor Quality as a fitted/predicted class. CLS uses SDP with T=100 for T2-1 and T=50 for T2-2; Temporal Grid uses dimension-wise q75 with NumPy linear interpolation; fusion alpha is 0.5. True Poor Quality is excluded from SE/SP denominators. Placements are retrospective estimates.

Bootstrap uses BioCAS2022 patient clusters, 10,000 ordinary resamples with replacement, every recording and multiplicity of sampled patients, paired draws, 95% linear-percentile intervals, and no retraining.

## Harmonised exclusions

The corrected harmonised exclusion accounting is 230 Poor-Quality-only records plus one exact train/evaluation duplicate, total 231. It is asserted in code and tests.

| Exclusion | Count | Reason |
|---|---:|---|
| Poor-Quality-only records | 230 | The harmonised respiratory-sound formulation excludes records that do not provide an eligible Normal/adventitious target. |
| Exact train/evaluation duplicate | 1 | Removed before fitting/evaluation to prevent duplicate-content leakage across the split boundary. |
| Total | 231 | Fixed protocol total; not selected from evaluation performance. |

Poor Quality is retained in the separate official BioCAS Challenge workflow. These exclusions apply only to the harmonised analysis.
