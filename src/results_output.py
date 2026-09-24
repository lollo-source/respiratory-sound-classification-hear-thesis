"""Reference-blind creation and publication of public results bundles."""

from __future__ import print_function

import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TABLE_TITLES = {
    "tab:hflung_class_distribution": "HF Lung class distribution",
    "tab:hflung_device_distribution": "HF Lung acquisition-device distribution",
    "tab:biocas_harmonised_class_distribution": "BioCAS harmonised class distribution",
    "tab:biocas_challenge_class_distribution": "BioCAS Challenge class distribution",
    "tab:single_window_results": "Single Window results",
    "tab:single_window_per_class_recall": "Single Window per-class recall",
    "tab:single_window_event_capture": "Single Window event capture",
    "tab:complete_record_results": "Complete Record results",
    "tab:complete_record_per_class_recall": "Complete Record per-class recall",
    "tab:dual_readout_results": "Dual Readout results",
    "tab:dual_readout_per_class_recall": "Dual Readout per-class recall",
    "tab:overall_harmonised_comparison": "Overall harmonised comparison",
    "tab:hear_opera_comparison": "HeAR and OPERA comparison",
    "tab:biocas_challenge_placements": "BioCAS Challenge retrospective placements",
    "tab:challenge_readout_comparison": "BioCAS Challenge readout comparison",
    "tab:bootstrap_harmonised": "Harmonised patient-cluster bootstrap",
    "tab:bootstrap_challenge_readouts": "Challenge readout patient-cluster bootstrap",
}

TABLE_NOTES = {
    "tab:biocas_challenge_placements": (
        "Positions are retrospective estimates within the published comparison subsets "
        "and do not represent official Challenge submissions."
    ),
}

DISPLAY_DECIMALS = {label: 4 for label in TABLE_TITLES}
DISPLAY_DECIMALS["tab:hflung_device_distribution"] = 1
DISPLAY_DECIMALS["tab:single_window_event_capture"] = 1


FIELD_LABELS = {
    "dataset": "Dataset",
    "evaluation": "Evaluation",
    "task": "Task",
    "class": "Class",
    "split": "Split",
    "device": "Device",
    "method": "Method",
    "metric": "Metric",
    "published_table": "Published table",
    "status": "Note",
    "count": "Count",
    "percentage": "Percentage (%)",
    "record_count": "N",
    "pathological_records": "Pathological",
    "captured_records": "Captured",
    "capture_rate_percent": "Capture rate (%)",
    "balanced_accuracy": "BA",
    "macro_f1": "Macro-F1",
    "accuracy": "Acc.",
    "delta_balanced_accuracy": "Δ BA",
    "delta_macro_f1": "Δ Macro-F1",
    "delta_accuracy": "Δ Acc.",
    "recall": "Recall",
    "delta_recall": "Δ Recall",
    "single_balanced_accuracy": "Single",
    "dual_balanced_accuracy": "Dual",
    "single_macro_f1": "Single",
    "dual_macro_f1": "Dual",
    "single_accuracy": "Single",
    "dual_accuracy": "Dual",
    "SE": "SE",
    "SP": "SP",
    "AS": "AS",
    "HS": "HS",
    "Score": "Score",
    "score": "Score",
    "estimated_position": "Position",
    "comparison_set_size": "Comparison N",
    "point_estimate": "Estimate",
    "ci_lower": "CI Lower",
    "ci_upper": "CI Upper",
    "replicates": "Replicates",
}

INTEGER_FIELDS = {
    "count", "record_count", "pathological_records", "captured_records",
    "estimated_position", "comparison_set_size", "replicates",
}
NUMERIC_FIELDS = INTEGER_FIELDS | {
    "percentage", "capture_rate_percent", "balanced_accuracy", "macro_f1", "accuracy",
    "delta_balanced_accuracy", "delta_macro_f1", "delta_accuracy", "recall", "delta_recall",
    "single_balanced_accuracy", "dual_balanced_accuracy", "single_macro_f1", "dual_macro_f1",
    "single_accuracy", "dual_accuracy", "SE", "SP", "AS", "HS", "Score", "score",
    "point_estimate", "ci_lower", "ci_upper",
}


def _columns(*fields):
    return list(fields)


DISPLAY_SCHEMAS = {
    "tab:hflung_class_distribution": [(None, _columns("split", "task", "class", "count"))],
    "tab:hflung_device_distribution": [(None, _columns("split", "device", "count", "percentage"))],
    "tab:biocas_harmonised_class_distribution": [(None, _columns("split", "task", "class", "count"))],
    "tab:biocas_challenge_class_distribution": [(None, _columns("split", "task", "class", "count"))],
    "tab:single_window_results": [(
        None, _columns("dataset", "evaluation", "task", "balanced_accuracy", "macro_f1", "accuracy", "record_count")
    )],
    "tab:single_window_per_class_recall": [(
        None, _columns("dataset", "evaluation", "task", "class", "recall")
    )],
    "tab:single_window_event_capture": [(
        None, _columns("dataset", "pathological_records", "captured_records", "capture_rate_percent")
    )],
    "tab:complete_record_results": [
        ("Balanced Accuracy", _columns("dataset", "evaluation", "task", "balanced_accuracy", "delta_balanced_accuracy", "record_count")),
        ("Macro-F1", _columns("dataset", "evaluation", "task", "macro_f1", "delta_macro_f1", "record_count")),
        ("Accuracy", _columns("dataset", "evaluation", "task", "accuracy", "delta_accuracy", "record_count")),
    ],
    "tab:complete_record_per_class_recall": [(
        None, _columns("dataset", "evaluation", "task", "class", "recall", "delta_recall")
    )],
    "tab:dual_readout_results": [
        ("Balanced Accuracy", _columns("dataset", "evaluation", "task", "balanced_accuracy", "delta_balanced_accuracy", "record_count")),
        ("Macro-F1", _columns("dataset", "evaluation", "task", "macro_f1", "delta_macro_f1", "record_count")),
        ("Accuracy", _columns("dataset", "evaluation", "task", "accuracy", "delta_accuracy", "record_count")),
    ],
    "tab:dual_readout_per_class_recall": [(
        None, _columns("dataset", "evaluation", "task", "class", "recall", "delta_recall")
    )],
    "tab:overall_harmonised_comparison": [
        ("Balanced Accuracy", _columns("dataset", "evaluation", "task", "single_balanced_accuracy", "dual_balanced_accuracy", "delta_balanced_accuracy")),
        ("Macro-F1", _columns("dataset", "evaluation", "task", "single_macro_f1", "dual_macro_f1", "delta_macro_f1")),
        ("Accuracy", _columns("dataset", "evaluation", "task", "single_accuracy", "dual_accuracy", "delta_accuracy")),
    ],
    "tab:hear_opera_comparison": [(
        None, _columns("dataset", "task", "method", "balanced_accuracy", "macro_f1", "accuracy")
    )],
    "tab:biocas_challenge_placements": [(
        None, _columns("evaluation", "task", "published_table", "score", "estimated_position", "comparison_set_size")
    )],
    "tab:challenge_readout_comparison": [(
        None, _columns("evaluation", "task", "method", "SE", "SP", "AS", "HS", "Score")
    )],
    "tab:bootstrap_harmonised": [(
        None, _columns("task", "method", "metric", "point_estimate", "ci_lower", "ci_upper", "replicates")
    )],
    "tab:bootstrap_challenge_readouts": [(
        None, _columns("task", "method", "metric", "point_estimate", "ci_lower", "ci_upper", "replicates")
    )],
}

PRESENTATION_LABELS = {
    "binary": "Binary",
    "coarse": "Coarse",
    "train": "Train",
    "test": "Test",
    "train2022": "Train 2022",
    "test2022": "Test 2022",
    "test2023": "Test 2023",
    "official_test": "Official test",
    "biocas2022": "BioCAS 2022",
    "biocas2023": "BioCAS 2023",
    "BioCAS2022": "BioCAS 2022",
    "BioCAS2023": "BioCAS 2023",
    "HF_Lung": "HF Lung",
    "balanced_accuracy": "BA",
    "macro_f1": "Macro-F1",
    "accuracy": "Acc.",
}


def table_filename(label):
    return label.replace("tab:", "").replace(":", "_")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _machine_columns(rows):
    result = []
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _machine_columns(rows)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})
    os.replace(str(temporary), str(path))


def _markdown_value(value, field=None, decimals=4):
    if field in INTEGER_FIELDS and isinstance(value, (int, float)):
        text = str(int(value))
    elif field in NUMERIC_FIELDS and isinstance(value, (int, float)):
        text = format(float(value), ".%df" % decimals)
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = PRESENTATION_LABELS.get(str(value), str(value))
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(rows, columns, decimals):
    if not columns:
        return "_No rows._\n"
    headers = [FIELD_LABELS[field] for field in columns]
    displayed_rows = [
        [_markdown_value(row.get(field, ""), field, decimals) for field in columns]
        for row in rows
    ]
    widths = []
    for index, field in enumerate(columns):
        minimum = 4 if field in NUMERIC_FIELDS else 3
        widths.append(max([minimum, len(headers[index])] + [len(row[index]) for row in displayed_rows]))

    def aligned(values):
        cells = []
        for index, field in enumerate(columns):
            value = values[index]
            cells.append(value.rjust(widths[index]) if field in NUMERIC_FIELDS else value.ljust(widths[index]))
        return "| " + " | ".join(cells) + " |"

    separator = []
    for index, field in enumerate(columns):
        if field in NUMERIC_FIELDS:
            separator.append("-" * (widths[index] - 1) + ":")
        else:
            separator.append("-" * widths[index])
    lines = [aligned(headers), "| " + " | ".join(separator) + " |"]
    lines.extend(aligned(row) for row in displayed_rows)
    return "\n".join(lines) + "\n"


def _source_hashes(source_artifacts):
    output = {}
    for identity, value in sorted(source_artifacts.items()):
        if isinstance(value, (str, Path)):
            path = Path(value)
            output[identity] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        else:
            output[identity] = value
    return output


def _release_identity(release_root):
    release_root = Path(release_root)
    identity_path = release_root / "THESIS_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    return {
        "title": identity.get("title"),
        "release_version": identity.get("release_version"),
        "manuscript_sha256": identity.get("manuscript_sha256"),
        "hear_revision": identity.get("hear_revision"),
        "opera_revision": identity.get("opera_revision"),
        "thesis_identity_sha256": sha256_file(identity_path),
        "python": sys.version.split()[0],
    }


def create_bundle(
    destination,
    tables,
    mode,
    applicable_count,
    source_workflows,
    source_artifacts,
    release_root,
    quick_only_tables=None,
    extra_manifest=None,
    quick_comparison=None,
):
    """Write fresh tables and provenance without consulting thesis references."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    machine = destination / "machine_readable"
    csv_dir = destination / "tables"
    machine.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    labels = list(tables)
    for label in labels:
        payload = tables[label]
        if payload.get("label") != label or not isinstance(payload.get("rows"), list):
            raise ValueError("invalid table payload: %s" % label)
        basename = table_filename(label)
        json_path = machine / (basename + ".json")
        csv_path = csv_dir / (basename + ".csv")
        _write_json(json_path, payload)
        _write_csv(csv_path, payload["rows"])
        for path in (json_path, csv_path):
            relative = str(path.relative_to(destination))
            file_hashes[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "public-results-bundle-v1",
        "mode": mode,
        "generated_table_labels": labels,
        "generated_table_count": len(labels),
        "applicable_table_count": int(applicable_count),
        "total_thesis_table_count": 17,
        "source_workflow_identities": list(source_workflows),
        "source_artifact_hashes": _source_hashes(source_artifacts),
        "software_release_identity": _release_identity(release_root),
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "table_file_hashes": file_hashes,
        "quick_only_tables": list(quick_only_tables or []),
    }
    if quick_comparison is not None:
        manifest["quick_vs_full_comparison"] = quick_comparison
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_json(destination / "manifest.json", manifest)
    render_results_markdown(destination, verification_status="PENDING")
    return manifest


def render_results_markdown(bundle_dir, verification_status=None):
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    verification_path = bundle_dir / "verification.json"
    verification = None
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification_status = verification.get("overall_status", verification_status)
    verification_status = verification_status or "NOT RUN"
    mode_name = "Quick Verification" if manifest["mode"] == "quick" else "Full Reproduction"
    lines = ["# Reproduced thesis results", "", "Mode: **%s**  " % mode_name]
    if manifest["mode"] == "quick":
        lines.extend([
            "Tables reproduced: **%d/17**  " % manifest["generated_table_count"],
            "Verification: **%s**  " % verification_status,
            "Generated from: packaged public prediction/statistical inputs",
        ])
    else:
        lines.extend([
            "Applicable thesis tables reproduced: **%d/17**  " % manifest["applicable_table_count"],
            "Verification: **%s**  " % verification_status,
            "Scientific source: raw external datasets + pinned pretrained models",
        ])
        excluded = manifest.get("quick_only_tables", [])
        lines.extend(["", "## Quick-only tables under the current Full scope", ""])
        lines.extend("- `%s`" % label for label in excluded)
        comparison = manifest.get("quick_vs_full_comparison", {})
        lines.extend(["", "## Quick versus Full comparison", ""])
        if comparison.get("performed"):
            lines.extend([
                "- Comparable tables: %d" % comparison["comparable_table_count"],
                "- Consistent at thesis display precision: %s" % ("yes" if comparison["display_consistent"] else "no"),
                "- Maximum absolute numerical difference: %.4g" % comparison["maximum_absolute_difference"],
            ])
        else:
            lines.append("Not performed because no valid `results/quick/` bundle was available.")
    for label in manifest["generated_table_labels"]:
        payload = json.loads(
            (bundle_dir / "machine_readable" / (table_filename(label) + ".json")).read_text(encoding="utf-8")
        )
        lines.extend(["", "## %s" % TABLE_TITLES.get(label, label), "", "Thesis label: `%s`" % label, ""])
        if label in TABLE_NOTES:
            lines.extend([TABLE_NOTES[label], ""])
        schema = DISPLAY_SCHEMAS.get(label)
        if not schema:
            raise ValueError("missing Markdown display schema: %s" % label)
        for section_index, (section_title, columns) in enumerate(schema):
            if section_title:
                if section_index:
                    lines.append("")
                lines.extend(["### %s" % section_title, ""])
            lines.append(_markdown_table(payload["rows"], columns, DISPLAY_DECIMALS[label]).rstrip())
    lines.extend(["", "## Verification", ""])
    if verification and verification.get("overall_status") == "PASS":
        lines.extend([
            "All reproduced thesis tables passed post-computation verification.",
            "",
            "**%d/%d PASS**" % (
                verification.get("passed_table_count", 0),
                verification.get("checked_table_count", manifest["generated_table_count"]),
            ),
        ])
    elif verification:
        lines.extend([
            "Post-computation verification did not pass for every reproduced table.",
            "",
            "**%d/%d PASS**" % (
                verification.get("passed_table_count", 0),
                verification.get("checked_table_count", manifest["generated_table_count"]),
            ),
        ])
    else:
        lines.append("Post-computation verification status: **%s**" % verification_status)
    lines.extend([
        "",
        "Full-precision CSV/JSON outputs and provenance metadata are stored alongside this report for automated verification and further analysis.",
        "",
    ])
    temporary = bundle_dir / "RESULTS.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(str(temporary), str(bundle_dir / "RESULTS.md"))


def _compare_values(left, right, decimals, state):
    if isinstance(left, bool) or isinstance(right, bool):
        state["display"] = state["display"] and left == right
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        state["maximum"] = max(state["maximum"], difference)
        state["display"] = state["display"] and format(float(left), ".%df" % decimals) == format(float(right), ".%df" % decimals)
        return
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        for key in sorted(left):
            _compare_values(left[key], right[key], decimals, state)
        return
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        for a, b in zip(left, right):
            _compare_values(a, b, decimals, state)
        return
    state["display"] = state["display"] and left == right


def compare_with_quick(full_tables, quick_bundle):
    quick_bundle = Path(quick_bundle)
    quick_manifest = quick_bundle / "manifest.json"
    if not quick_manifest.is_file():
        return {"performed": False, "reason": "quick bundle unavailable"}
    metadata = json.loads(quick_manifest.read_text(encoding="utf-8"))
    if metadata.get("mode") != "quick" or metadata.get("generated_table_count") != 17:
        return {"performed": False, "reason": "quick bundle is incomplete"}
    state = {"display": True, "maximum": 0.0}
    checked = []
    for label, payload in full_tables.items():
        path = quick_bundle / "machine_readable" / (table_filename(label) + ".json")
        if not path.is_file():
            state["display"] = False
            continue
        quick = json.loads(path.read_text(encoding="utf-8"))
        _compare_values(payload, quick, DISPLAY_DECIMALS[label], state)
        checked.append(label)
    return {
        "performed": True,
        "comparable_table_count": len(checked),
        "checked_labels": checked,
        "display_consistent": bool(state["display"] and len(checked) == len(full_tables)),
        "maximum_absolute_difference": float(state["maximum"]),
        "quick_values_used_as_computational_inputs": False,
    }


def prepare_temporary_slot(results_root, mode):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    temporary = results_root / (".%s_tmp" % mode)
    if temporary.exists():
        shutil.rmtree(str(temporary))
    temporary.mkdir(parents=True)
    return temporary


def publish_slot(results_root, mode):
    """Transactionally replace one fixed result slot, restoring it on failure."""
    results_root = Path(results_root)
    temporary = results_root / (".%s_tmp" % mode)
    destination = results_root / mode
    previous = results_root / (".%s_previous" % mode)
    if not temporary.is_dir():
        raise RuntimeError("completed temporary results slot is missing: %s" % temporary)
    if previous.exists():
        shutil.rmtree(str(previous))
    moved_previous = False
    try:
        if destination.exists():
            os.replace(str(destination), str(previous))
            moved_previous = True
        os.replace(str(temporary), str(destination))
    except Exception:
        if moved_previous and not destination.exists() and previous.exists():
            os.replace(str(previous), str(destination))
        raise
    if previous.exists():
        shutil.rmtree(str(previous))
    return destination
