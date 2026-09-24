import hashlib
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def release_inventory():
    transient_directories = {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "results",
    }
    result = {}
    for directory, directories, filenames in os.walk(ROOT):
        directories[:] = sorted(name for name in directories if name not in transient_directories)
        for filename in sorted(filenames):
            if filename.endswith((".pyc", ".log", ".tmp")) or filename == ".coverage":
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, ROOT)
            if os.path.islink(path):
                result[relative] = "symlink:" + os.readlink(path)
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[relative] = digest.hexdigest()
    return result


class PublicWorkflowRegression(unittest.TestCase):
    def test_independent_calculation_matches_all_frozen_tables(self):
        process = subprocess.run([sys.executable, "scripts/quick_verify.py"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("17/17 PASS", process.stdout)
        self.assertIn("17/17 thesis tables reproduced", process.stdout)
        self.assertIn("results/quick/RESULTS.md", process.stdout)
        self.assertIn("Quick Verification", process.stderr)
        self.assertIn("-> [1/7] dataset statistics and progression", process.stderr)
        self.assertIn("OK [7/7] publishing results bundle", process.stderr)
        result_root = os.path.join(ROOT, "results", "quick")
        self.assertTrue(os.path.isfile(os.path.join(result_root, "RESULTS.md")))
        self.assertEqual(len(os.listdir(os.path.join(result_root, "tables"))), 17)
        self.assertEqual(len(os.listdir(os.path.join(result_root, "machine_readable"))), 17)

    def test_quick_verification_does_not_modify_release_files(self):
        before = release_inventory()
        process = subprocess.run([sys.executable, "scripts/quick_verify.py"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        after = release_inventory()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("17/17 PASS", process.stdout)
        self.assertEqual(before, after)

    def test_opera_requires_explicit_public_setup_without_fallback(self):
        process = subprocess.run([sys.executable, "scripts/full_reproduce.py", "--workflow", "opera"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn("--opera-root and --opera-checkpoint-root are required", process.stderr)
        self.assertIn("No fallback", process.stderr)

    def test_published_stage_is_rejected_for_workflows_without_it(self):
        for workflow in ("opera", "hear-harmonised"):
            with self.subTest(workflow=workflow):
                process = subprocess.run(
                    [sys.executable, "scripts/full_reproduce.py", "--workflow", workflow, "--stage", "published"],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                self.assertEqual(process.returncode, 2)
                self.assertIn("unsupported workflow/stage combination", process.stderr)
                self.assertIn("--workflow %s" % workflow, process.stderr)
                self.assertIn("--stage published", process.stderr)
                self.assertNotIn('"status": "complete"', process.stdout)

    def test_complete_orchestrator_rejects_partial_stage_without_running(self):
        process = subprocess.run(
            [sys.executable, "scripts/full_reproduce.py", "--workflow", "all", "--stage", "preflight"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("unsupported workflow/stage combination", process.stderr)
        self.assertNotIn('"status": "complete"', process.stdout)


if __name__ == "__main__":
    unittest.main()
