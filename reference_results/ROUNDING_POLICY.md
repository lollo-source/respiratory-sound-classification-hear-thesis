# Rounding policy

Reference JSON retains Python/JSON full precision from the audited canonical artifacts. Verification first requires absolute agreement within `1e-12`, then independently compares thesis-display strings.

Metrics, recalls, deltas, Challenge components/Scores, and bootstrap bounds use four digits after the decimal point. Device percentages and event-capture percentages use one digit. Counts and estimated positions are integers. Python's format operation is the executable display rule; no pre-rounding or post-hoc correction is applied.

