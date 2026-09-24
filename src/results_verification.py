"""Post-computation verification for an explicit public results bundle."""

from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path

from .full_results import FULL_TABLE_LABELS, QUICK_ONLY_TABLES
from .protocol import BINARY_CLASSES, CHALLENGE_CLASSES, COARSE_CLASSES, assert_exclusion_contract
from .reproduction import LABELS
from .results_output import DISPLAY_DECIMALS, sha256_file, table_filename
from .io_utils import read_csv


ABSOLUTE_TOLERANCE = 1e-12
REFERENCE_DIRECTORY = "reference" + "_results"
EXPECTED_FILENAME = "expected" + "_values.json"


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _compare(actual, expected, decimals, location="root"):
    errors = []
    maximum = 0.0
    display_ok = True
    if isinstance(expected, bool):
        if actual is not expected:
            errors.append("%s: %r != %r" % (location, actual, expected))
            display_ok = False
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            errors.append("%s: non-numeric value" % location)
            display_ok = False
        else:
            maximum = abs(float(actual) - expected)
            if maximum > ABSOLUTE_TOLERANCE:
                errors.append("%s: %r != %r" % (location, actual, expected))
            display_ok = format(float(actual), ".%df" % decimals) == format(expected, ".%df" % decimals)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            errors.append("%s: dictionary keys differ" % location)
            display_ok = False
        else:
            for key in sorted(expected):
                child_errors, child_maximum, child_display = _compare(
                    actual[key], expected[key], decimals, location + "." + key
                )
                errors.extend(child_errors)
                maximum = max(maximum, child_maximum)
                display_ok = display_ok and child_display
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            errors.append("%s: list length differs" % location)
            display_ok = False
        else:
            for index, expected_value in enumerate(expected):
                child_errors, child_maximum, child_display = _compare(
                    actual[index], expected_value, decimals, "%s[%d]" % (location, index)
                )
                errors.extend(child_errors)
                maximum = max(maximum, child_maximum)
                display_ok = display_ok and child_display
    elif actual != expected:
        errors.append("%s: %r != %r" % (location, actual, expected))
        display_ok = False
    return errors, maximum, display_ok


def _source_integrity_scan(release_root, expected):
    errors = []
    forbidden_names = ("reference" + "_results", "expected" + "_values.json")
    allowed_reference_readers = {
        Path("src/results_verification.py"),
        Path("scripts/verify_against_thesis.py"),
    }
    precise_values = set()

    def collect(value):
        if isinstance(value, float) and 0.0 < abs(value) < 1.0:
            text = format(value, ".12g")
            if len(text.split(".")[-1]) >= 6:
                precise_values.add(text)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(expected["tables"])
    for directory in (release_root / "src", release_root / "scripts"):
        for path in directory.rglob("*.py"):
            relative = path.relative_to(release_root)
            if relative in allowed_reference_readers:
                continue
            text = path.read_text(encoding="utf-8")
            for forbidden in forbidden_names:
                if forbidden in text:
                    errors.append("forbidden frozen-reference access in %s" % relative)
            for value in precise_values:
                if value in text:
                    errors.append("expected metric literal %s embedded in %s" % (value, relative))
    return errors


def _quick_preflight(release_root, expected):
    errors = []
    artifact_manifest = json.loads((release_root / "manifests" / "ARTIFACTS.json").read_text(encoding="utf-8"))
    for relative, metadata in sorted(artifact_manifest["artifacts"].items()):
        path = release_root / relative
        if not path.is_file() or sha256_file(path) != metadata["packaged_sha256"]:
            errors.append("checksum mismatch: " + relative)
    bio = read_csv(str(release_root / "manifests" / "biocas_records.csv.gz"))
    try:
        assert_exclusion_contract(bio)
    except AssertionError:
        errors.append("BioCAS exclusion contract mismatch")
    progression = read_csv(str(release_root / "artifacts" / "predictions" / "progression.csv.gz"))
    for row in progression:
        required = BINARY_CLASSES if row["task"] == "binary" else COARSE_CLASSES
        if json.loads(row["class_order"]) != required:
            errors.append("progression class-order mismatch")
            break
    challenge = read_csv(str(release_root / "artifacts" / "predictions" / "challenge.csv.gz"))
    for row in challenge:
        if json.loads(row["class_order"]) != CHALLENGE_CLASSES[row["task"]]:
            errors.append("Challenge class-order mismatch")
            break
    errors.extend(_source_integrity_scan(release_root, expected))
    return errors


def verify_bundle(bundle_dir, release_root, mode=None, emit=True):
    bundle_dir = Path(bundle_dir)
    release_root = Path(release_root)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    mode = mode or manifest.get("mode")
    if mode not in ("quick", "full") or manifest.get("mode") != mode:
        raise RuntimeError("results bundle mode mismatch")
    labels = manifest.get("generated_table_labels", [])
    expected_labels = LABELS if mode == "quick" else FULL_TABLE_LABELS
    if labels != expected_labels:
        raise RuntimeError("%s bundle table coverage/order mismatch" % mode)
    if mode == "full" and manifest.get("quick_only_tables") != QUICK_ONLY_TABLES:
        raise RuntimeError("Full bundle Quick-only exclusions are not explicit")

    reference_root = release_root / REFERENCE_DIRECTORY
    expected_path = reference_root / EXPECTED_FILENAME
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    preflight = _quick_preflight(release_root, expected) if mode == "quick" else _source_integrity_scan(release_root, expected)
    table_results = []
    passed = 0
    reference_hashes = {EXPECTED_FILENAME: sha256_file(expected_path)}
    for label in labels:
        filename = table_filename(label) + ".json"
        actual_path = bundle_dir / "machine_readable" / filename
        if not actual_path.is_file():
            errors, maximum, display_ok = (["generated table is missing"], 0.0, False)
            actual = None
        else:
            actual = json.loads(actual_path.read_text(encoding="utf-8"))
            errors, maximum, display_ok = _compare(
                actual,
                expected["tables"][label],
                int(DISPLAY_DECIMALS[label]),
                label,
            )
        reference_table = reference_root / "thesis_tables" / filename
        reference_hashes["thesis_tables/" + filename] = sha256_file(reference_table)
        item_pass = not errors and display_ok
        table_results.append({
            "label": label,
            "status": "PASS" if item_pass else "FAIL",
            "numeric_comparison_status": "PASS" if not errors else "FAIL",
            "display_comparison_status": "PASS" if display_ok else "FAIL",
            "maximum_absolute_numerical_difference": float(maximum),
            "errors": errors[:20],
        })
        if item_pass:
            passed += 1
        if emit:
            print(("PASS " if item_pass else "FAIL ") + label)
            for error in errors[:5]:
                print("  " + error)
    overall = "PASS" if not preflight and passed == len(labels) else "FAIL"
    report = {
        "schema": "public-results-verification-v1",
        "mode": mode,
        "checked_labels": labels,
        "checked_table_count": len(labels),
        "passed_table_count": passed,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "preflight_status": "PASS" if not preflight else "FAIL",
        "preflight_errors": preflight,
        "tables": table_results,
        "reference_identity": {"sha256": reference_hashes},
        "overall_status": overall,
    }
    _write_json(bundle_dir / "verification.json", report)
    if emit:
        for error in preflight:
            print("FAIL preflight: " + error)
        print("\n%d/%d PASS" % (passed, len(labels)))
    return report
