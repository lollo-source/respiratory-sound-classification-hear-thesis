#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.io_utils import read_csv, read_json, root_path, sha256_file
from src.protocol import BINARY_CLASSES, COARSE_CLASSES, CHALLENGE_CLASSES, assert_exclusion_contract
from src.reproduction import LABELS

ABSOLUTE_TOLERANCE = 1e-12


def compare(actual, expected, location="root"):
    errors = []
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > ABSOLUTE_TOLERANCE:
            errors.append("%s: %r != %r" % (location, actual, expected))
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            errors.append("%s: dictionary keys differ" % location)
        else:
            for key in sorted(expected):
                errors.extend(compare(actual[key], expected[key], location + "." + key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            errors.append("%s: list length differs" % location)
        else:
            for index, value in enumerate(expected):
                errors.extend(compare(actual[index], value, "%s[%d]" % (location, index)))
    elif actual != expected:
        errors.append("%s: %r != %r" % (location, actual, expected))
    return errors


def rounded(value, decimals):
    if isinstance(value, float):
        return format(value, ".%df" % decimals)
    if isinstance(value, list):
        return [rounded(item, decimals) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item, decimals) for key, item in value.items()}
    return value


def source_integrity_scan(expected):
    errors = []
    targets = []
    for directory in (root_path("src"), root_path("scripts")):
        for name in os.listdir(directory):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                if os.path.basename(path) not in ("verify_against_thesis.py",):
                    targets.append(path)
    forbidden_names = ("reference_results", "expected_values.json")
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
    for path in targets:
        with open(path) as handle:
            text = handle.read()
        for forbidden in forbidden_names:
            if forbidden in text:
                errors.append("forbidden reference access in %s" % os.path.relpath(path, ROOT))
        for value in precise_values:
            if value in text:
                errors.append("expected metric literal %s embedded in %s" % (value, os.path.relpath(path, ROOT)))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare independently generated outputs to frozen thesis references")
    parser.add_argument("--level", choices=("A", "B", "C"), required=True)
    args = parser.parse_args(argv)
    if args.level != "C":
        print("Level %s verification is not yet certified." % args.level, file=sys.stderr)
        return 2

    expected = read_json(root_path("reference_results", "expected_values.json"))
    provenance = read_json(root_path("manifests", "ARTIFACTS.json"))
    preflight = []
    for relative, metadata in sorted(provenance["artifacts"].items()):
        actual = sha256_file(root_path(*relative.split("/")))
        if actual != metadata["packaged_sha256"]:
            preflight.append("checksum mismatch: " + relative)
    bio = read_csv(root_path("manifests", "biocas_records.csv.gz"))
    try:
        assert_exclusion_contract(bio)
    except AssertionError:
        preflight.append("BioCAS exclusion contract mismatch")
    progression = read_csv(root_path("artifacts", "predictions", "progression.csv.gz"))
    for row in progression:
        required = BINARY_CLASSES if row["task"] == "binary" else COARSE_CLASSES
        if json.loads(row["class_order"]) != required:
            preflight.append("progression class-order mismatch")
            break
    challenge = read_csv(root_path("artifacts", "predictions", "challenge.csv.gz"))
    for row in challenge:
        if json.loads(row["class_order"]) != CHALLENGE_CLASSES[row["task"]]:
            preflight.append("Challenge class-order mismatch")
            break
    preflight.extend(source_integrity_scan(expected))
    if preflight:
        for error in preflight:
            print("FAIL preflight: " + error)
        return 1

    passed = 0
    for label in LABELS:
        filename = label.replace("tab:", "").replace(":", "_") + ".json"
        actual = read_json(root_path("artifacts", "generated", "level_c", filename))
        frozen = expected["tables"][label]
        errors = compare(actual, frozen, label)
        decimals = int(expected["display_decimals"].get(label, 4))
        if rounded(actual, decimals) != rounded(frozen, decimals):
            errors.append(label + ": thesis display rounding differs")
        if errors:
            print("FAIL " + label)
            for error in errors[:5]:
                print("  " + error)
        else:
            print("PASS " + label)
            passed += 1
    print("\n%d/17 PASS" % passed)
    return 0 if passed == 17 else 1


if __name__ == "__main__":
    raise SystemExit(main())
