import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.challenge_score import official_challenge_metrics
from src.downstream import fuse_probabilities
from src.pooling import arithmetic_mean, dimensionwise_q75, sdp
from src.protocol import assert_exclusion_contract
from src.io_utils import read_csv, root_path
from src.temporal_grid import temporal_grid_readout
from src.windowing import complete_record_windows, max_energy_crop


class WindowingTests(unittest.TestCase):
    def test_exact_two_seconds_is_one_window(self):
        self.assertEqual(complete_record_windows(np.ones(32000)).shape, (1, 32000))

    def test_exact_four_seconds_is_two_windows(self):
        self.assertEqual(complete_record_windows(np.ones(64000)).shape, (2, 32000))

    def test_partial_window_is_right_padded(self):
        windows = complete_record_windows(np.arange(32001))
        self.assertEqual(windows.shape, (2, 32000))
        self.assertEqual(windows[1, 0], 32000)
        self.assertTrue(np.all(windows[1, 1:] == 0))

    def test_max_energy_tie_is_earliest(self):
        crop, start = max_energy_crop(np.ones(32002), 32000)
        self.assertEqual(start, 0)
        self.assertEqual(len(crop), 32000)


class TemporalGridTests(unittest.TestCase):
    def test_shape_axis_population_std_and_size(self):
        tokens = np.arange(96 * 1024, dtype=np.float64).reshape(96, 1024)
        output = temporal_grid_readout(tokens)
        grid = tokens.reshape(12, 8, 1024)
        self.assertEqual(grid.shape, (12, 8, 1024))
        self.assertEqual(output.shape, (24576,))
        np.testing.assert_allclose(output[:12288], grid.mean(axis=1).reshape(-1))
        np.testing.assert_allclose(output[12288:], grid.std(axis=1, ddof=0).reshape(-1))


class PoolingFusionTests(unittest.TestCase):
    def test_mean(self):
        np.testing.assert_allclose(arithmetic_mean([[1, 2], [3, 4]]), [2, 3])

    def test_sdp_positive_deviation_weight(self):
        values = np.array([[0.0], [1.0], [4.0]])
        centre = values.mean(axis=0)
        distances = np.abs(values[:, 0] - centre[0])
        logits = distances
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        self.assertGreater(weights[np.argmax(distances)], weights[np.argmin(distances)])
        np.testing.assert_allclose(sdp(values, 1.0), np.sum(values * weights[:, None], axis=0))

    def test_q75_numpy_linear(self):
        values = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        np.testing.assert_allclose(dimensionwise_q75(values), np.percentile(values, 75, axis=0, interpolation="linear"))

    def test_fusion_convention_and_half_average(self):
        cls = np.array([[0.8, 0.2]])
        grid = np.array([[0.2, 0.8]])
        np.testing.assert_allclose(fuse_probabilities(cls, grid, 0.75), 0.75 * cls + 0.25 * grid)
        np.testing.assert_allclose(fuse_probabilities(cls, grid, 0.5), (cls + grid) / 2.0)


class ChallengeProtocolTests(unittest.TestCase):
    def test_poor_quality_excluded_from_denominators_but_prediction_is_valid(self):
        truth = ["Normal", "Adventitious", "Poor Quality"]
        first = official_challenge_metrics(truth, ["Normal", "Adventitious", "Poor Quality"], ["Adventitious"])
        second = official_challenge_metrics(truth, ["Normal", "Adventitious", "Normal"], ["Adventitious"])
        self.assertEqual(first, second)
        self.assertEqual(first, {"SE": 1.0, "SP": 1.0, "AS": 1.0, "HS": 1.0, "Score": 1.0})

    def test_components(self):
        values = official_challenge_metrics(["Normal", "Normal", "Adventitious", "Adventitious"], ["Normal", "Adventitious", "Adventitious", "Normal"], ["Adventitious"])
        self.assertEqual(values, {"SE": 0.5, "SP": 0.5, "AS": 0.5, "HS": 0.5, "Score": 0.5})

    def test_exclusion_contract(self):
        assert_exclusion_contract(read_csv(root_path("manifests", "biocas_records.csv.gz")))


if __name__ == "__main__":
    unittest.main()

