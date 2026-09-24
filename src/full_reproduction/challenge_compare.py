"""Post-computation comparison with static published Challenge subsets."""

from pathlib import Path

import pandas as pd

from .features import write_json


def compare_published(metrics_path, published_path, output_dir):
    metrics = pd.read_csv(metrics_path)
    fresh = metrics.loc[metrics["method"].astype(str) == "Fusion"].copy()
    if len(fresh) != 4:
        raise RuntimeError("fresh Fusion metrics must exist before published comparison")
    published = pd.read_csv(published_path)
    rows = []
    for row in fresh.itertuples(index=False):
        key = str(row.evaluation).lower()
        subset = published.loc[(published["evaluation"].astype(str) == key) &
                               (published["task"].astype(str) == str(row.task))]
        score = float(row.Score)
        rows.append({"evaluation": row.evaluation, "task": row.task, "fresh_score": score,
                     "retrospective_insertion_position": int((subset["score"].astype(float) > score).sum() + 1),
                     "comparison_set_size_including_fresh_system": int(len(subset) + 1),
                     "interpretation": "retrospective insertion within published subset; not an official rank"})
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "published_comparison.csv", index=False)
    write_json(output_dir / "manifest.json", {"status": "complete", "fresh_metrics_read_first": True,
                                               "published_values_used_for_scientific_computation": False,
                                               "rows": frame.to_dict(orient="records")})
    return frame.to_dict(orient="records")
