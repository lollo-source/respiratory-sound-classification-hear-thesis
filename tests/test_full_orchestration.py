import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "full_reproduce.py"
SPEC = importlib.util.spec_from_file_location("public_full_reproduce", str(SCRIPT_PATH))
FULL_REPRODUCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL_REPRODUCE)


def arguments(output_dir):
    return argparse.Namespace(
        workflow="all",
        stage="all",
        datasets=("hf_lung", "biocas_harmonised"),
        hf_lung_root="/data/hf lung",
        sprsound_root="/data/sprsound",
        hear_repo="/models/hear repo",
        hear_model_path="/models/hear/model.pt",
        opera_root="/models/opera source",
        opera_checkpoint_root="/models/opera checkpoints",
        hear_metrics=None,
        device="cuda:0",
        output_dir=str(output_dir),
        resume=True,
    )


class Completed(object):
    def __init__(self, returncode):
        self.returncode = returncode


class FullOrchestrationIsolationTests(unittest.TestCase):
    def test_run_all_delegates_to_isolated_workflow_dispatch(self):
        args = arguments("/tmp/output")
        with mock.patch.object(FULL_REPRODUCE, "run_isolated_workflows", return_value=9) as isolated:
            status = FULL_REPRODUCE.run_all(args)
        self.assertEqual(status, 9)
        isolated.assert_called_once_with(args)

    def test_all_dispatches_three_fresh_python_processes(self):
        calls = []

        def runner(command, cwd):
            calls.append((command, cwd))
            return Completed(0)

        status = FULL_REPRODUCE.run_isolated_workflows(arguments("/tmp/output"), runner=runner)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [command[command.index("--workflow") + 1] for command, _cwd in calls],
            ["hear-harmonised", "hear-challenge", "opera"],
        )
        for command, cwd in calls:
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(Path(command[1]).resolve(), SCRIPT_PATH.resolve())
            self.assertEqual(cwd, os.getcwd())
            self.assertEqual(command[command.index("--stage") + 1], "all")

    def test_runtime_environment_mutation_cannot_leak_between_workflows(self):
        with tempfile.TemporaryDirectory() as temporary:
            probe = Path(temporary) / "probe.py"
            probe.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    import os
                    from pathlib import Path

                    parser = argparse.ArgumentParser(add_help=False)
                    parser.add_argument("--workflow")
                    parser.add_argument("--output-dir")
                    args, _unknown = parser.parse_known_args()
                    root = Path(args.output_dir)
                    root.mkdir(parents=True, exist_ok=True)
                    (root / (args.workflow + ".json")).write_text(json.dumps({
                        "sentinel": os.environ.get("FULL_REPRODUCTION_ISOLATION_TEST")
                    }))
                    if args.workflow == "hear-harmonised":
                        os.environ["FULL_REPRODUCTION_ISOLATION_TEST"] = "mutated-in-child"
                    """
                ),
                encoding="utf-8",
            )
            output = Path(temporary) / "output"
            with mock.patch.dict(os.environ, {"FULL_REPRODUCTION_ISOLATION_TEST": "parent"}):
                status = FULL_REPRODUCE.run_isolated_workflows(
                    arguments(output), script_path=str(probe)
                )
            self.assertEqual(status, 0)
            for workflow in ("hear-harmonised", "hear-challenge", "opera"):
                payload = json.loads((output / (workflow + ".json")).read_text(encoding="utf-8"))
                self.assertEqual(payload["sentinel"], "parent")

    def test_child_nonzero_exit_status_propagates_and_stops_dispatch(self):
        calls = []

        def runner(command, cwd):
            workflow = command[command.index("--workflow") + 1]
            calls.append(workflow)
            return Completed(9 if workflow == "hear-challenge" else 0)

        status = FULL_REPRODUCE.run_isolated_workflows(arguments("/tmp/output"), runner=runner)
        self.assertEqual(status, 9)
        self.assertEqual(calls, ["hear-harmonised", "hear-challenge"])

    def test_public_arguments_are_forwarded_to_relevant_children(self):
        args = arguments("/tmp/full output")
        harmonised = FULL_REPRODUCE._isolated_workflow_command(args, "hear-harmonised")
        challenge = FULL_REPRODUCE._isolated_workflow_command(args, "hear-challenge")
        opera = FULL_REPRODUCE._isolated_workflow_command(args, "opera")

        self.assertEqual(harmonised[harmonised.index("--datasets") + 1:], [
            "hf_lung", "biocas_harmonised", "--hf-lung-root", "/data/hf lung",
            "--hear-repo", "/models/hear repo", "--hear-model-path", "/models/hear/model.pt", "--resume",
        ])
        self.assertIn("/data/sprsound", challenge)
        self.assertIn("/models/hear repo", challenge)
        self.assertIn("/models/hear/model.pt", challenge)
        self.assertNotIn("--hf-lung-root", challenge)
        self.assertIn("/models/opera source", opera)
        self.assertIn("/models/opera checkpoints", opera)
        metrics = opera[opera.index("--hear-metrics") + 1]
        self.assertEqual(
            metrics,
            "/tmp/full output/datasets/biocas_harmonised/downstream/metrics.json",
        )
        for command in (harmonised, challenge, opera):
            self.assertEqual(command[command.index("--output-dir") + 1], "/tmp/full output")
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertIn("--resume", command)

    def test_standalone_workflow_remains_directly_reachable(self):
        result = {"status": "complete", "scope": "test", "stage": "preflight"}
        with mock.patch.object(FULL_REPRODUCE, "run_isolated_workflows") as isolated:
            with mock.patch("src.full_reproduction.challenge_pipeline.run", return_value=result) as standalone:
                status = FULL_REPRODUCE.main([
                    "--workflow", "hear-challenge", "--stage", "preflight"
                ])
        self.assertEqual(status, 0)
        isolated.assert_not_called()
        standalone.assert_called_once()


if __name__ == "__main__":
    unittest.main()
