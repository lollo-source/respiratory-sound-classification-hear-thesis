import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.quick_verify import prepare_fresh_calculation_output, require_fresh_tables
from src.full_reproduction.challenge_data import prepare_protocol, public_record_frame as challenge_record_frame
from src.full_reproduction.datasets import public_record_frame
from src.full_results import (
    FULL_TABLE_LABELS,
    QUICK_ONLY_TABLES,
    challenge_tables,
    harmonised_tables,
    opera_comparison_table,
)
from src.io_utils import read_csv
from src.reproduction import LABELS
from src.results_output import (
    compare_with_quick,
    create_bundle,
    prepare_temporary_slot,
    publish_slot,
    render_results_markdown,
    sha256_file,
    table_filename,
)
from src.results_verification import _source_integrity_scan, verify_bundle


ROOT = Path(__file__).resolve().parents[1]


class ResultsBundleTests(unittest.TestCase):
    def test_reference_blind_bundle_contains_all_quick_tables_and_formats(self):
        tables = {
            label: json.loads(
                (ROOT / "artifacts" / "generated" / "level_c" / (table_filename(label) + ".json")).read_text()
            )
            for label in LABELS
        }
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, tables, "quick", 17, ["fixture"], {}, ROOT)
            markdown = (bundle / "RESULTS.md").read_text(encoding="utf-8")
            self.assertEqual(len(list((bundle / "tables").glob("*.csv"))), 17)
            self.assertEqual(len(list((bundle / "machine_readable").glob("*.json"))), 17)
            for label in LABELS:
                self.assertIn(label, markdown)

    def test_explicit_bundle_verifier_writes_machine_report(self):
        tables = {
            label: json.loads(
                (ROOT / "artifacts" / "generated" / "level_c" / (table_filename(label) + ".json")).read_text()
            )
            for label in LABELS
        }
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, tables, "quick", 17, ["fixture"], {}, ROOT)
            report = verify_bundle(bundle, ROOT, "quick", emit=False)
            self.assertEqual(report["overall_status"], "PASS")
            self.assertEqual(report["passed_table_count"], 17)
            self.assertTrue((bundle / "verification.json").is_file())

    def test_transaction_replaces_only_requested_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "quick").mkdir()
            (root / "quick" / "old.txt").write_text("old")
            (root / "full").mkdir()
            (root / "full" / "keep.txt").write_text("keep")
            temporary = prepare_temporary_slot(root, "quick")
            (temporary / "new.txt").write_text("new")
            publish_slot(root, "quick")
            self.assertTrue((root / "quick" / "new.txt").is_file())
            self.assertFalse((root / "quick" / "old.txt").exists())
            self.assertEqual((root / "full" / "keep.txt").read_text(), "keep")

    def test_failed_publication_preserves_previous_valid_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "quick").mkdir()
            (root / "quick" / "valid.txt").write_text("valid")
            with self.assertRaises(RuntimeError):
                publish_slot(root, "quick")
            self.assertEqual((root / "quick" / "valid.txt").read_text(), "valid")

    def test_missing_fresh_table_never_falls_back_to_protected_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            stale = workspace / "artifacts" / "generated" / "level_c"
            stale.mkdir(parents=True)
            for label in LABELS:
                (stale / (table_filename(label) + ".json")).write_text("stale", encoding="utf-8")
            previous = root / "results" / "quick"
            previous.mkdir(parents=True)
            (previous / "valid.txt").write_text("valid", encoding="utf-8")

            generated = prepare_fresh_calculation_output(workspace)
            self.assertEqual(list(generated.iterdir()), [])
            for label in LABELS[:-1]:
                (generated / (table_filename(label) + ".json")).write_text("fresh", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing freshly generated tables"):
                require_fresh_tables(generated)
            self.assertEqual((previous / "valid.txt").read_text(encoding="utf-8"), "valid")

    def test_integrity_scan_recurses_into_nested_full_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "src" / "full_reproduction"
            nested.mkdir(parents=True)
            (root / "scripts").mkdir()
            (nested / "forbidden.py").write_text(
                'path = "reference_results/expected_values.json"\n', encoding="utf-8"
            )
            errors = _source_integrity_scan(root, {"tables": {}})
            self.assertTrue(any("src/full_reproduction/forbidden.py" in error for error in errors))

    def test_results_directory_is_gitignored(self):
        rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("results/", rules)


class ResultsMarkdownPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = {
            label: json.loads(
                (ROOT / "artifacts" / "generated" / "level_c" / (table_filename(label) + ".json")).read_text()
            )
            for label in LABELS
        }

    def test_thesis_precision_labels_and_explicit_column_order(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            markdown = (bundle / "RESULTS.md").read_text(encoding="utf-8")
            self.assertRegex(markdown, r"\| Dataset\s+\| Evaluation\s+\| Task\s+\|\s+BA \|\s+Macro-F1 \|\s+Acc\. \|\s+N \|")
            self.assertRegex(markdown, r"\| BioCAS 2022\s+\| Test 2022\s+\| Binary\s+\| 0\.6307 \|\s+0\.6298 \| 0\.6685 \|\s+724 \|")
            self.assertNotIn("0.6306711018140667", markdown)
            self.assertRegex(markdown, r"\| Split\s+\| Device\s+\|\s+Count \|\s+Percentage \(%\) \|")
            self.assertRegex(markdown, r"\| Train\s+\| HF-Type-1\s+\|\s+4555 \|\s+58\.3 \|")

    def test_wide_comparisons_are_split_into_metric_subtables(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            markdown = (bundle / "RESULTS.md").read_text(encoding="utf-8")
            self.assertGreaterEqual(markdown.count("### Balanced Accuracy"), 3)
            self.assertGreaterEqual(markdown.count("### Macro-F1"), 3)
            self.assertGreaterEqual(markdown.count("### Accuracy"), 3)
            self.assertRegex(markdown, r"\| Dataset\s+\| Evaluation\s+\| Task\s+\|\s+Single \|\s+Dual \|\s+Δ BA \|")

    def test_markdown_tables_are_valid_aligned_and_no_wider_than_eight_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            lines = (bundle / "RESULTS.md").read_text(encoding="utf-8").splitlines()
            table_lines = [line for line in lines if line.startswith("|")]
            self.assertTrue(table_lines)
            for line in table_lines:
                self.assertEqual(line[0], "|")
                self.assertEqual(line[-1], "|")
                self.assertLessEqual(line.count("|") - 1, 8, line)
            separators = [line for line in table_lines if set(line.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")) == set()]
            self.assertTrue(separators)
            self.assertTrue(any("---:" in line for line in separators))

    def test_rerender_changes_only_markdown_not_csv_or_json(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            machine_paths = sorted((bundle / "tables").glob("*.csv")) + sorted(
                (bundle / "machine_readable").glob("*.json")
            )
            before = {str(path.relative_to(bundle)): sha256_file(path) for path in machine_paths}
            render_results_markdown(bundle, verification_status="PASS")
            after = {str(path.relative_to(bundle)): sha256_file(path) for path in machine_paths}
            self.assertEqual(before, after)
            csv_text = (bundle / "tables" / "single_window_results.csv").read_text()
            json_text = (bundle / "machine_readable" / "single_window_results.json").read_text()
            self.assertIn("0.6306711018140667", csv_text)
            self.assertIn("0.6306711018140667", json_text)

    def test_quick_and_full_use_identical_table_formatter(self):
        table = {"tab:single_window_results": self.tables["tab:single_window_results"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quick = root / "quick"
            full = root / "full"
            create_bundle(quick, table, "quick", 1, ["fixture"], {}, ROOT)
            create_bundle(full, table, "full", 1, ["fixture"], {}, ROOT)

            def rendered_table(path):
                text = (path / "RESULTS.md").read_text(encoding="utf-8")
                return text.split("## Single Window results", 1)[1].split("## Verification", 1)[0]

            self.assertEqual(rendered_table(quick), rendered_table(full))

    def test_quick_full_display_comparison_accepts_equal_text_fields(self):
        label = "tab:single_window_results"
        full_tables = {
            label: {
                "label": label,
                "rows": [{"dataset": "HF Lung", "balanced_accuracy": 0.12341}],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            quick = Path(directory) / "quick"
            (quick / "machine_readable").mkdir(parents=True)
            (quick / "manifest.json").write_text(json.dumps({
                "mode": "quick",
                "generated_table_count": 17,
            }), encoding="utf-8")
            (quick / "machine_readable" / "single_window_results.json").write_text(
                json.dumps({
                    "label": label,
                    "rows": [{"dataset": "HF Lung", "balanced_accuracy": 0.12342}],
                }),
                encoding="utf-8",
            )

            comparison = compare_with_quick(full_tables, quick)

            self.assertTrue(comparison["display_consistent"])
            self.assertEqual(comparison["comparable_table_count"], 1)
            self.assertAlmostEqual(comparison["maximum_absolute_difference"], 1e-5)

            quick_path = quick / "machine_readable" / "single_window_results.json"
            quick_payload = json.loads(quick_path.read_text(encoding="utf-8"))
            quick_payload["rows"][0]["dataset"] = "Different dataset"
            quick_path.write_text(json.dumps(quick_payload), encoding="utf-8")
            comparison = compare_with_quick(full_tables, quick)
            self.assertFalse(comparison["display_consistent"])

    def test_generated_source_uses_ordinary_gfm_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            markdown = (bundle / "RESULTS.md").read_text(encoding="utf-8")
            self.assertTrue(markdown.startswith("# Reproduced thesis results\n"))
            self.assertIn("\n## Single Window results\n", markdown)
            self.assertIn("\n### Balanced Accuracy\n", markdown)
            self.assertIn("Thesis label: `tab:single_window_results`", markdown)
            self.assertIn("Mode: **Quick Verification**", markdown)
            self.assertNotIn("\\|", markdown)
            self.assertNotIn("**##", markdown)
            self.assertNotIn("\\`", markdown)
            self.assertNotIn("**\\*\\*", markdown)
            self.assertNotIn("## Machine-readable equivalents", markdown)
            self.assertNotIn("[CSV](tables/", markdown)
            self.assertIn("\n## Verification\n", markdown)

    def test_every_gfm_table_is_one_contiguous_block(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            lines = (bundle / "RESULTS.md").read_text(encoding="utf-8").splitlines()
            blocks = []
            index = 0
            while index < len(lines):
                if not lines[index].startswith("|"):
                    index += 1
                    continue
                start = index
                while index < len(lines) and lines[index].startswith("|"):
                    index += 1
                blocks.append(lines[start:index])
            self.assertEqual(len(blocks), 23)
            for block in blocks:
                self.assertGreaterEqual(len(block), 3, block)
                self.assertRegex(block[1], r"^\|(?:\s*:?-+:?\s*\|)+$")
                column_count = block[0].count("|")
                self.assertTrue(all(row.count("|") == column_count for row in block), block)

    def test_raw_markdown_cells_are_visually_column_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            lines = (bundle / "RESULTS.md").read_text(encoding="utf-8").splitlines()
            blocks = []
            index = 0
            while index < len(lines):
                if not lines[index].startswith("|"):
                    index += 1
                    continue
                start = index
                while index < len(lines) and lines[index].startswith("|"):
                    index += 1
                blocks.append(lines[start:index])
            for block in blocks:
                cell_widths = [len(cell) for cell in block[0].split("|")[1:-1]]
                for row in block[1:]:
                    self.assertEqual([len(cell) for cell in row.split("|")[1:-1]], cell_widths, row)

    def test_report_ends_with_concise_verification_not_file_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            verification = {
                "overall_status": "PASS",
                "passed_table_count": 17,
                "checked_table_count": 17,
            }
            (bundle / "verification.json").write_text(json.dumps(verification), encoding="utf-8")
            render_results_markdown(bundle)
            markdown = (bundle / "RESULTS.md").read_text(encoding="utf-8")
            ending = markdown.split("## Verification", 1)[1]
            self.assertIn("All reproduced thesis tables passed post-computation verification.", ending)
            self.assertIn("**17/17 PASS**", ending)
            self.assertIn("Full-precision CSV/JSON outputs", ending)
            self.assertNotIn("machine_readable/", ending)
            self.assertNotIn("tables/", ending)

    def test_challenge_placement_note_is_once_above_compact_human_table(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "quick"
            create_bundle(bundle, self.tables, "quick", 17, ["fixture"], {}, ROOT)
            markdown = (bundle / "RESULTS.md").read_text(encoding="utf-8")
            start = markdown.index("## BioCAS Challenge retrospective placements")
            end = markdown.index("\n## ", start + 3)
            section = markdown[start:end]
            note = (
                "Positions are retrospective estimates within the published comparison subsets "
                "and do not represent official Challenge submissions."
            )
            self.assertEqual(section.count(note), 1)
            self.assertRegex(
                section,
                r"\| Evaluation\s+\| Task \| Published table\s+\|\s+Score \| Position \| Comparison N \|",
            )
            self.assertNotIn("| Note", section)
            self.assertNotIn("retrospective estimate, not an official submission", section)

            csv_text = (bundle / "tables" / "biocas_challenge_placements.csv").read_text(encoding="utf-8")
            json_payload = json.loads(
                (bundle / "machine_readable" / "biocas_challenge_placements.json").read_text(encoding="utf-8")
            )
            self.assertIn("status", csv_text.splitlines()[0])
            self.assertTrue(all("status" in row for row in json_payload["rows"]))


class FullResultsAssemblyTests(unittest.TestCase):
    def test_full_coverage_and_quick_only_exclusions_are_honest(self):
        self.assertEqual(len(FULL_TABLE_LABELS), 14)
        self.assertEqual(QUICK_ONLY_TABLES, [
            "tab:single_window_event_capture",
            "tab:bootstrap_harmonised",
            "tab:bootstrap_challenge_readouts",
        ])

    def test_hf_device_is_preserved_as_public_full_provenance(self):
        frame = pd.DataFrame([{
            "record_id": "hf_record_1", "split": "train", "group_id": "hf_group_1",
            "device": "Littmann", "binary_label": "Normal", "coarse_label": "Normal", "fold_id": 0,
        }])
        public = public_record_frame(frame)
        self.assertEqual(public.columns.tolist(), [
            "record_id", "split", "group_id", "device", "binary_label", "coarse_label", "fold_id",
        ])
        self.assertEqual(public.iloc[0]["device"], "Littmann")

    def test_challenge_placement_assembly_is_table_aware_and_six_rows(self):
        source = pd.read_csv(ROOT / "manifests" / "biocas_records.csv.gz", dtype=str)
        records = prepare_protocol(source)
        record_rows = []
        for dataset in ("biocas2022", "biocas2023"):
            record_rows.extend(
                challenge_record_frame(records[records["dataset"] == dataset]).to_dict(orient="records")
            )
        predictions = read_csv(str(ROOT / "artifacts" / "predictions" / "challenge.csv.gz"))
        published = read_csv(str(ROOT / "published_challenge_reference" / "published_biocas_task2_scores.csv"))
        tables = challenge_tables(record_rows, predictions, published)
        placements = tables["tab:biocas_challenge_placements"]["rows"]
        self.assertEqual(len(placements), 6)
        self.assertEqual(
            {(row["evaluation"], row["task"], row["published_table"]) for row in placements},
            {
                ("biocas2022", "T2-1", "Table II"), ("biocas2022", "T2-2", "Table II"),
                ("biocas2023", "T2-1", "Table II"), ("biocas2023", "T2-2", "Table II"),
                ("biocas2023", "T2-1", "Table III"), ("biocas2023", "T2-2", "Table III"),
            },
        )

    def test_all_14_full_tables_match_certified_output_schemas_on_fixtures(self):
        hf = read_csv(str(ROOT / "manifests" / "hf_lung_records.csv.gz"))
        bio = [
            row for row in read_csv(str(ROOT / "manifests" / "biocas_records.csv.gz"))
            if row["inclusion_status"] == "included"
        ]
        progression = read_csv(str(ROOT / "artifacts" / "predictions" / "progression.csv.gz"))
        tables = harmonised_tables(hf, bio, progression)

        source = pd.read_csv(ROOT / "manifests" / "biocas_records.csv.gz", dtype=str)
        records = prepare_protocol(source)
        record_rows = []
        for dataset in ("biocas2022", "biocas2023"):
            record_rows.extend(
                challenge_record_frame(records[records["dataset"] == dataset]).to_dict(orient="records")
            )
        challenge_predictions = read_csv(str(ROOT / "artifacts" / "predictions" / "challenge.csv.gz"))
        for row in challenge_predictions:
            row["evaluation"] = row["evaluation"].replace("biocas", "BioCAS")
        published = read_csv(str(ROOT / "published_challenge_reference" / "published_biocas_task2_scores.csv"))
        tables.update(challenge_tables(record_rows, challenge_predictions, published))
        opera_payload = json.loads(
            (ROOT / "artifacts" / "generated" / "level_c" / "hear_opera_comparison.json").read_text()
        )
        opera = opera_comparison_table(opera_payload)
        tables[opera["label"]] = opera

        tables = {label: tables[label] for label in FULL_TABLE_LABELS}
        self.assertEqual(list(tables), FULL_TABLE_LABELS)
        for label in FULL_TABLE_LABELS:
            expected = json.loads(
                (ROOT / "artifacts" / "generated" / "level_c" / (table_filename(label) + ".json")).read_text()
            )
            self.assertEqual(tables[label], expected, label)

    def test_assemblers_are_reference_blind_and_only_verifiers_read_frozen_values(self):
        for module in ("results_output.py", "full_results.py"):
            source = (ROOT / "src" / module).read_text(encoding="utf-8")
            self.assertNotIn("reference" + "_results", source)
            self.assertNotIn("expected" + "_values", source)
        readers = []
        for directory in (ROOT / "src", ROOT / "scripts"):
            for path in directory.glob("*.py"):
                source = path.read_text(encoding="utf-8")
                if "reference" + "_results" in source or "expected" + "_values" in source:
                    readers.append(path.name)
        self.assertTrue(readers)
        self.assertTrue(all("verif" in name for name in readers), readers)


class ProtectedBaselineTests(unittest.TestCase):
    def test_all_77_protected_files_remain_byte_identical(self):
        payload = json.loads((ROOT / "reference_results" / "LEVEL_C_BASELINE_SHA256.json").read_text())
        self.assertEqual(payload["file_count"], 77)
        for relative, metadata in payload["files"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(path.stat().st_size, metadata["bytes"], relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"], relative)


if __name__ == "__main__":
    unittest.main()
