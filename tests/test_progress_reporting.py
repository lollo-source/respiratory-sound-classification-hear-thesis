import io
import unittest

from src.progress import LoopProgress, StageProgress, cache_reused, downstream_progress, run_stage


class ProgressReportingTests(unittest.TestCase):
    def test_noninteractive_loop_uses_sparse_milestones(self):
        stream = io.StringIO()
        progress = LoopProgress("dataset extraction", 100, stream=stream, clock=lambda: 0.0)
        for completed in range(1, 101):
            progress.update(completed)
        progress.complete()
        output = stream.getvalue()
        self.assertIn("dataset extraction: 0/100 (0.0%)", output)
        self.assertIn("dataset extraction: 100/100 (100.0%)", output)
        self.assertIn("OK dataset extraction — 100/100 complete", output)
        self.assertLess(len(output.splitlines()), 20)
        self.assertNotIn("\r", output)

    def test_cache_reuse_is_labelled_explicitly(self):
        stream = io.StringIO()
        cache_reused("HF Lung extraction", stream=stream)
        self.assertEqual(stream.getvalue(), "OK HF Lung extraction — validated cache reused\n")

    def test_stage_reporting_preserves_return_value(self):
        stream = io.StringIO()
        progress = StageProgress("Workflow", [("extract", "extraction")], stream=stream, clock=lambda: 0.0)
        marker = object()
        self.assertIs(run_stage(progress, "extract", lambda: marker), marker)
        progress.finish()
        output = stream.getvalue()
        self.assertIn("Requested stages (1): extraction", output)
        self.assertIn("OK [1/1] extraction — complete", output)
        self.assertNotIn("preflight", output)

    def test_long_stage_summary_wraps_only_between_complete_labels(self):
        stream = io.StringIO()
        stages = [
            ("progression", "dataset statistics and progression"),
            ("opera", "OPERA comparison"),
            ("challenge", "BioCAS Challenge"),
            ("bootstrap", "bootstrap tables"),
            ("assembly", "assembling thesis results"),
            ("verification", "verifying thesis results"),
            ("publication", "publishing results bundle"),
        ]
        StageProgress("Quick Verification", stages, stream=stream, clock=lambda: 0.0)
        summary_lines = stream.getvalue().splitlines()[1:]
        self.assertIn("assembling thesis results", stream.getvalue())
        self.assertTrue(all(len(line) <= 100 for line in summary_lines))

    def test_downstream_progress_is_stderr_style_and_coarse(self):
        stream = io.StringIO()
        downstream_progress("BioCAS 2022 — binary task: final evaluation", stream=stream)
        self.assertEqual(stream.getvalue(), "   Downstream: BioCAS 2022 — binary task: final evaluation\n")

    def test_downstream_wait_message_preserves_scientific_return_value(self):
        stream = io.StringIO()
        progress = StageProgress("Workflow", [("downstream", "downstream")], stream=stream, clock=lambda: 0.0)
        scientific_result = {"score": 0.75, "predictions": ["A", "B"]}
        result = run_stage(progress, "downstream", lambda: scientific_result)
        self.assertIs(result, scientific_result)
        self.assertIn("Fitting, cross-validation and evaluation are running", stream.getvalue())
        self.assertIn("coarse progress appears at major operation boundaries", stream.getvalue())

    def test_stage_reporting_reraises_failures(self):
        stream = io.StringIO()
        progress = StageProgress("Workflow", [("downstream", "downstream")], stream=stream, clock=lambda: 0.0)

        def fail():
            raise RuntimeError("underlying diagnostic")

        with self.assertRaisesRegex(RuntimeError, "underlying diagnostic"):
            run_stage(progress, "downstream", fail)
        self.assertIn("FAIL [1/1] downstream — underlying diagnostic", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
