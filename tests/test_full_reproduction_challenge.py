import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.full_reproduction.challenge_data import EXPECTED_COUNTS, TASKS, prepare_protocol
from src.full_reproduction.challenge_downstream import (
    FUSION_ALPHA_CLS, TEMPERATURES, aggregate_q75, aggregate_sdp, official_metrics, select_incumbent,
)


ROOT = Path(__file__).resolve().parents[1]


class ChallengeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = prepare_protocol(pd.read_csv(ROOT / "manifests" / "biocas_records.csv.gz", dtype=str))

    def test_inclusion_mapping_order_and_counts(self):
        self.assertEqual(len(self.records), 3554)
        self.assertEqual(self.records["official_split"].value_counts().to_dict(), EXPECTED_COUNTS)
        self.assertEqual(self.records.iloc[0].official_split, "train2022")
        for spec in TASKS.values():
            self.assertFalse(set(self.records[spec["column"]]) - set(spec["classes"]))

    def test_deterministic_grouped_folds_without_leakage(self):
        again = prepare_protocol(pd.read_csv(ROOT / "manifests" / "biocas_records.csv.gz", dtype=str))
        for spec in TASKS.values():
            np.testing.assert_array_equal(self.records[spec["fold_column"]], again[spec["fold_column"]])
            train = self.records[self.records[spec["fold_column"]].astype(int) >= 0]
            self.assertTrue((train.groupby("group_id")[spec["fold_column"]].nunique() == 1).all())


class ChallengeScientificCoreTests(unittest.TestCase):
    def test_sdp_positive_distance_and_float_contract(self):
        values = np.asarray([[0.0], [1.0], [10.0]], dtype=np.float32)
        result = aggregate_sdp(values, [np.arange(3)], 1.0)
        self.assertEqual(result.dtype, np.float32)
        self.assertGreater(float(result[0, 0]), float(values.mean()))

    def test_q75_exact_numpy_linear(self):
        values = np.arange(20, dtype=np.float32).reshape(5, 4)
        got = aggregate_q75(values, [np.arange(5)])[0]
        expected = np.quantile(values, 0.75, axis=0, method="linear").astype(np.float32)
        np.testing.assert_array_equal(got, expected)

    def test_strict_replacement_tolerance(self):
        scores = {value: 0.5 for value in TEMPERATURES}
        scores[2.0] = 0.5001
        scores[5.0] = 0.50010000001
        self.assertEqual(select_incumbent(scores), 5.0)
        scores[10.0] = 0.5002001
        self.assertEqual(select_incumbent(scores), 10.0)

    def test_official_score_excludes_poor_quality_denominator(self):
        truth = ["Normal", "Normal", "CAS", "DAS", "Poor Quality"]
        pred = ["Normal", "CAS", "CAS", "Normal", "Normal"]
        got = official_metrics(truth, pred, ["CAS", "DAS", "CAS & DAS"])
        self.assertEqual(got["SP"], 0.5); self.assertEqual(got["SE"], 0.5)
        self.assertEqual(got["AS"], 0.5); self.assertEqual(got["HS"], 0.5); self.assertEqual(got["Score"], 0.5)

    def test_class_order_and_fixed_fusion(self):
        self.assertEqual(FUSION_ALPHA_CLS, 0.5)
        cls = np.asarray([[0.9, 0.1]]); tg = np.asarray([[0.1, 0.9]])
        np.testing.assert_array_equal(FUSION_ALPHA_CLS * cls + (1-FUSION_ALPHA_CLS) * tg, [[0.5, 0.5]])
        self.assertEqual(TASKS["T2-2"]["classes"], ["Normal", "CAS", "DAS", "CAS & DAS", "Poor Quality"])

    def test_calculation_modules_have_no_frozen_inputs_or_expected_temperatures(self):
        names = ("challenge_data.py", "challenge_features.py", "challenge_downstream.py")
        forbidden = ("reference" + "_results", "expected" + "_values", "artifacts/" + "predictions",
                     "published_challenge_" + "reference", "level_" + "b", "T2-1\": 100", "T2-2\": 50")
        for name in names:
            text = (ROOT / "src" / "full_reproduction" / name).read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
