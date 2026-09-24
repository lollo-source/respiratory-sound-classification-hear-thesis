#!/usr/bin/env python
"""Public entry point for the certified Full Reproduction workflows."""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


SUPPORTED_STAGES = {
    "all": ("all",),
    "hear-harmonised": ("preflight", "first-batch", "subset", "extract", "downstream", "all"),
    "hear-challenge": ("preflight", "first-batch", "subset", "extract", "downstream", "published", "all"),
    "biocas-challenge": ("preflight", "first-batch", "subset", "extract", "downstream", "published", "all"),
    "opera": ("preflight", "first-batch", "subset", "extract", "downstream", "all"),
}

STAGE_LABELS = {
    "preflight": "preflight",
    "first-batch": "first-batch validation",
    "subset": "subset/repeatability validation",
    "extract": "extraction",
    "downstream": "downstream",
    "published": "published comparison",
}

WORKFLOW_TITLES = {
    "all": "Complete Full Reproduction",
    "hear-harmonised": "HeAR harmonised Full Reproduction",
    "hear-challenge": "HeAR Challenge Full Reproduction",
    "biocas-challenge": "HeAR Challenge Full Reproduction",
    "opera": "OPERA Full Reproduction",
}


def parser():
    result = argparse.ArgumentParser(
        description="Full Reproduction for harmonised HeAR, official BioCAS Challenge, and OPERA workflows"
    )
    result.add_argument(
        "--workflow",
        required=True,
        choices=("all", "hear-harmonised", "hear-challenge", "biocas-challenge", "opera"),
    )
    result.add_argument(
        "--stage",
        choices=("preflight", "first-batch", "subset", "extract", "downstream", "published", "all"),
        default="all",
    )
    result.add_argument(
        "--datasets",
        nargs="+",
        choices=("hf_lung", "biocas_harmonised"),
        default=("hf_lung", "biocas_harmonised"),
    )
    result.add_argument("--hf-lung-root")
    result.add_argument("--sprsound-root")
    result.add_argument("--hear-repo")
    result.add_argument("--hear-model-path")
    result.add_argument("--opera-root")
    result.add_argument("--opera-checkpoint-root")
    result.add_argument("--hear-metrics", help="Certified fresh harmonised HeAR metrics.json used only for comparison")
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "artifacts", "generated", "full_reproduction"),
    )
    result.add_argument("--resume", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage not in SUPPORTED_STAGES[args.workflow]:
        print(
            "unsupported workflow/stage combination: --workflow %s does not support --stage %s"
            % (args.workflow, args.stage),
            file=sys.stderr,
        )
        return 2
    # The harmonised replay was audited with one BLAS thread. The separately
    # audited Challenge source intentionally retains its original unconstrained
    # BLAS execution and records the observed numerical environment.
    if args.workflow == "all":
        if args.stage != "all":
            print("--workflow all requires --stage all", file=sys.stderr)
            return 2
        missing = [
            flag for flag, value in (
                ("--hf-lung-root", args.hf_lung_root),
                ("--sprsound-root", args.sprsound_root),
                ("--hear-repo", args.hear_repo),
                ("--hear-model-path", args.hear_model_path),
                ("--opera-root", args.opera_root),
                ("--opera-checkpoint-root", args.opera_checkpoint_root),
            ) if not value
        ]
        if missing:
            print("--workflow all requires: %s" % ", ".join(missing), file=sys.stderr)
            return 2
        return run_all(args)
    if args.workflow == "hear-harmonised":
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
    if args.workflow == "opera" and (not args.opera_root or not args.opera_checkpoint_root):
        print("--opera-root and --opera-checkpoint-root are required for OPERA. No fallback was run.", file=sys.stderr)
        return 2
    if args.workflow != "opera" and args.stage not in ("preflight", "downstream", "published") and (not args.hear_repo or not args.hear_model_path):
        print("--hear-repo and --hear-model-path are required for inference stages", file=sys.stderr)
        return 2
    from src.progress import StageProgress
    requested = (
        [stage for stage in SUPPORTED_STAGES[args.workflow] if stage != "all"]
        if args.stage == "all"
        else [args.stage]
    )
    args.progress = StageProgress(
        WORKFLOW_TITLES[args.workflow],
        [(stage, STAGE_LABELS[stage]) for stage in requested],
    )
    try:
        if args.workflow == "opera":
            from src.full_reproduction.opera_pipeline import run
        elif args.workflow in ("hear-challenge", "biocas-challenge"):
            from src.full_reproduction.challenge_pipeline import run
        else:
            from src.full_reproduction.pipeline import run
        result = run(args, ROOT)
    except Exception as error:
        print("Full Reproduction stopped: %s" % error, file=sys.stderr)
        return 1
    args.progress.finish()
    print(json.dumps({"status": result["status"], "scope": result["scope"], "stage": result["stage"]}, indent=2))
    return 0


def _isolated_workflow_command(args, workflow, script_path=None):
    """Build one standalone workflow command for the current interpreter."""
    script_path = script_path or os.path.abspath(__file__)
    command = [sys.executable, script_path, "--workflow", workflow, "--stage", "all"]

    def option(flag, value):
        if value is not None:
            command.extend([flag, str(value)])

    option("--output-dir", args.output_dir)
    option("--device", args.device)
    option("--sprsound-root", args.sprsound_root)
    if workflow == "hear-harmonised":
        command.append("--datasets")
        command.extend(str(dataset) for dataset in args.datasets)
        option("--hf-lung-root", args.hf_lung_root)
        option("--hear-repo", args.hear_repo)
        option("--hear-model-path", args.hear_model_path)
    elif workflow == "hear-challenge":
        option("--hear-repo", args.hear_repo)
        option("--hear-model-path", args.hear_model_path)
    elif workflow == "opera":
        option("--opera-root", args.opera_root)
        option("--opera-checkpoint-root", args.opera_checkpoint_root)
        option(
            "--hear-metrics",
            os.path.join(
                os.path.abspath(os.path.expanduser(args.output_dir)),
                "datasets",
                "biocas_harmonised",
                "downstream",
                "metrics.json",
            ),
        )
    if args.resume:
        command.append("--resume")
    return command


def run_isolated_workflows(args, runner=None, script_path=None):
    """Run each certified top-level workflow in a fresh Python process."""
    runner = runner or subprocess.run
    for workflow in ("hear-harmonised", "hear-challenge", "opera"):
        command = _isolated_workflow_command(args, workflow, script_path=script_path)
        # Preserve the caller's working-directory semantics for any relative
        # public paths while still using an absolute entry-point path.
        completed = runner(command, cwd=os.getcwd())
        if completed.returncode != 0:
            print(
                "Full Reproduction stopped: isolated %s workflow exited with status %d"
                % (workflow, completed.returncode),
                file=sys.stderr,
            )
            return completed.returncode
    return 0


def run_all(args):
    """Run isolated certified workflows, then assemble and publish one Full bundle."""
    from pathlib import Path

    from src.full_results import QUICK_ONLY_TABLES, assemble_full_tables
    from src.progress import StageProgress
    from src.results_output import (
        compare_with_quick,
        create_bundle,
        prepare_temporary_slot,
        publish_slot,
        render_results_markdown,
    )
    from src.results_verification import verify_bundle

    output_root = Path(args.output_dir).expanduser().resolve()
    try:
        workflow_status = run_isolated_workflows(args)
        if workflow_status != 0:
            return workflow_status

        final_progress = StageProgress(
            "Full results output",
            [
                ("assembly", "assembling thesis results"),
                ("verification", "verifying thesis results"),
                ("publication", "publishing results bundle"),
            ],
        )
        results_root = Path(ROOT) / "results"
        final_progress.start("assembly")
        temporary = prepare_temporary_slot(results_root, "full")
        tables, provenance, source_artifacts = assemble_full_tables(output_root, ROOT)
        comparison = compare_with_quick(tables, results_root / "quick")
        create_bundle(
            temporary,
            tables,
            mode="full",
            applicable_count=len(tables),
            source_workflows=["HeAR harmonised", "HeAR official BioCAS Challenge", "OPERA CE/CT/GT"],
            source_artifacts=source_artifacts,
            release_root=ROOT,
            quick_only_tables=QUICK_ONLY_TABLES,
            extra_manifest=provenance,
            quick_comparison=comparison,
        )
        final_progress.complete("assembly")

        final_progress.start("verification")
        report = verify_bundle(temporary, ROOT, mode="full", emit=True)
        if report["overall_status"] != "PASS":
            final_progress.fail(
                "verification",
                "%d/%d applicable tables passed" % (report["passed_table_count"], len(tables)),
            )
            return 1
        render_results_markdown(temporary)
        final_progress.complete("verification")

        final_progress.start("publication")
        destination = publish_slot(results_root, "full")
        final_progress.complete("publication")
        final_progress.finish()
        print("Full Reproduction complete")
        print("%d/17 applicable thesis tables reproduced" % len(tables))
        print("%d/%d verification PASS" % (len(tables), len(tables)))
        print("\nResults:\n%s" % os.path.relpath(destination / "RESULTS.md", ROOT))
        return 0
    except Exception as error:
        print("Full Reproduction stopped: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
