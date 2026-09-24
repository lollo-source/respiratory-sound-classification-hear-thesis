#!/usr/bin/env python
"""Public Quick Verification entry point."""

from __future__ import print_function

import os
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.progress import StageProgress
from src.reproduction import LABELS
from src.results_output import create_bundle, prepare_temporary_slot, publish_slot, render_results_markdown
from src.results_verification import verify_bundle


STEPS = (
    ("progression", "dataset statistics and progression", "reproduce_progression.py"),
    ("opera", "OPERA comparison", "reproduce_opera_comparison.py"),
    ("challenge", "BioCAS Challenge", "reproduce_challenge.py"),
    ("bootstrap", "bootstrap tables", "reproduce_bootstrap.py"),
)
FINAL_STEPS = (
    ("assembly", "assembling thesis results"),
    ("verification", "verifying thesis results"),
    ("publication", "publishing results bundle"),
)
TABLE_COUNT = 17


def prepare_fresh_calculation_output(workspace):
    """Create an empty table-output directory inside the isolated workspace."""
    generated = Path(workspace) / "artifacts" / "generated" / "level_c"
    if generated.exists():
        shutil.rmtree(str(generated))
    generated.mkdir(parents=True)
    return generated


def require_fresh_tables(generated):
    """Reject missing or unexpected outputs before public bundle assembly."""
    generated = Path(generated)
    expected = {
        label.replace("tab:", "").replace(":", "_") + ".json"
        for label in LABELS
    }
    actual = {entry.name for entry in generated.iterdir()}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing freshly generated tables: %s" % ", ".join(missing))
        if unexpected:
            details.append("unexpected calculation outputs: %s" % ", ".join(unexpected))
        raise RuntimeError("; ".join(details))
    return generated


def _workspace_ignore(directory, names):
    ignored = {
        name for name in names
        if name in (".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov")
        or name.endswith((".pyc", ".log", ".tmp"))
        or name == ".coverage"
    }
    if os.path.abspath(directory) == ROOT:
        ignored.update(name for name in names if name in (
            ".venv", "venv", "env", "data", "datasets", "models", "checkpoints", "outputs", "results", "runs", "logs"
        ))
    if os.path.relpath(directory, ROOT) == os.path.join("artifacts", "generated"):
        ignored.update(name for name in names if name == "full_reproduction")
    return ignored


def run(workspace, script, *arguments):
    command = [sys.executable, os.path.join(workspace, "scripts", script)]
    command.extend(arguments)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call(command, cwd=workspace, env=environment)


def main():
    stages = [(key, label) for key, label, _script in STEPS] + list(FINAL_STEPS)
    progress = StageProgress("Quick Verification", stages)
    temporary_root = tempfile.mkdtemp(prefix="thesis-quick-verification-")
    workspace = os.path.join(temporary_root, "release")
    results_root = Path(ROOT) / "results"
    bundle = None
    current_stage = None
    succeeded = False
    try:
        shutil.copytree(ROOT, workspace, symlinks=True, ignore=_workspace_ignore)
        generated = prepare_fresh_calculation_output(workspace)
        for key, _label, script in STEPS:
            progress.start(key)
            status = run(workspace, script, "--level", "C")
            if status:
                progress.fail(key, "subprocess exited with status %d" % status)
                return status
            progress.complete(key)
            if key == "bootstrap":
                print(
                    "Quick Verification calculations complete: %d independently computed thesis tables" % TABLE_COUNT,
                    flush=True,
                )
        require_fresh_tables(generated)
        current_stage = "assembly"
        progress.start(current_stage)
        bundle = prepare_temporary_slot(results_root, "quick")
        tables = {}
        for label in LABELS:
            filename = label.replace("tab:", "").replace(":", "_") + ".json"
            tables[label] = json.loads((generated / filename).read_text(encoding="utf-8"))
        artifact_manifest = json.loads(
            (Path(workspace) / "manifests" / "ARTIFACTS.json").read_text(encoding="utf-8")
        )
        source_artifacts = {
            relative: Path(workspace) / relative
            for relative in artifact_manifest["artifacts"]
        }
        create_bundle(
            bundle,
            tables,
            mode="quick",
            applicable_count=17,
            source_workflows=["Quick Verification from packaged public inputs"],
            source_artifacts=source_artifacts,
            release_root=ROOT,
            extra_manifest={
                "calculation_workspace": "isolated temporary public repository copy",
                "tracked_release_files_modified": False,
            },
        )
        progress.complete(current_stage)

        current_stage = "verification"
        progress.start(current_stage)
        report = verify_bundle(bundle, ROOT, mode="quick", emit=True)
        if report["overall_status"] != "PASS":
            progress.fail("verification", "%d/%d tables passed" % (report["passed_table_count"], TABLE_COUNT))
            return 1
        render_results_markdown(bundle)
        progress.complete(current_stage)

        current_stage = "publication"
        progress.start(current_stage)
        destination = publish_slot(results_root, "quick")
        progress.complete(current_stage)
        progress.finish()
        succeeded = True
        print("Quick Verification complete")
        print("17/17 thesis tables reproduced")
        print("17/17 verification PASS")
        print("\nResults:\n%s" % os.path.relpath(destination / "RESULTS.md", ROOT))
        return 0
    except Exception as error:
        if current_stage:
            progress.fail(current_stage, str(error))
        else:
            sys.stderr.write("Quick Verification stopped: %s\n" % error)
        return 1
    finally:
        if succeeded:
            shutil.rmtree(temporary_root)
        else:
            sys.stderr.write("Quick Verification workspace retained for diagnostics: %s\n" % temporary_root)
            if bundle is not None and bundle.exists():
                sys.stderr.write("Unpublished results retained for diagnostics: %s\n" % bundle)
            sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
