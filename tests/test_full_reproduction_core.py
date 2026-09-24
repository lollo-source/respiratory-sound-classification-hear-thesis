import inspect
import hashlib
import json
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from src.full_reproduction.audio import (
    canonical_resample,
    complete_record_windows,
    hf_fixed_windows,
    maximum_energy_window,
    read_wav_float32,
)
from src.full_reproduction.datasets import _unique_named_files
from src.full_reproduction.downstream import _oof_probabilities, select_alpha
from src.full_reproduction.features import _cache_complete
from src.full_reproduction.hear import temporal_grid_readout


class CanonicalAudioTests(unittest.TestCase):
    def test_pcm_decode_and_channel_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes(struct.pack("<hhhh", 32767, -32768, 16384, 16384))
            rate, values = read_wav_float32(path)
        self.assertEqual(rate, 8000)
        expected = np.array([(-1.0 / 32768.0) / 2.0, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(values, expected)

    def test_polyphase_resampling_contract(self):
        values = np.linspace(-1.0, 1.0, 8001, dtype=np.float32)
        actual = canonical_resample(values, 8000, 16000)
        expected = resample_poly(values, 2, 1).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_segmentation_and_padding(self):
        values = np.arange(70000, dtype=np.float32)
        complete = complete_record_windows(values)
        self.assertEqual(len(complete), 3)
        self.assertEqual(complete[-1][5999], 69999)
        self.assertEqual(complete[-1][6000], 0)
        fixed = hf_fixed_windows(np.ones(15 * 16000, dtype=np.float32))
        self.assertEqual(len(fixed), 8)
        self.assertTrue(np.all(fixed[-1][:16000] == 1))
        self.assertTrue(np.all(fixed[-1][16000:] == 0))

    def test_max_energy_tie_is_earliest(self):
        values = np.zeros(40000, dtype=np.float32)
        values[100:200] = 1
        values[7000:7100] = 1
        _window, start = maximum_energy_window(values, 1000)
        self.assertEqual(start, 0)


class RepresentationAndAdapterTests(unittest.TestCase):
    def test_temporal_grid_population_statistics(self):
        tokens = np.arange(96 * 1024, dtype=np.float32).reshape(96, 1024)
        output = temporal_grid_readout(tokens)
        grid = tokens.reshape(12, 8, 1024)
        expected = np.concatenate((grid.mean(axis=1).reshape(-1), grid.std(axis=1, ddof=0).reshape(-1)))
        self.assertEqual(output.shape, (24576,))
        np.testing.assert_array_equal(output, expected.astype(np.float32))

    def test_file_inventory_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("c.wav", "a.wav", "b.wav"):
                (root / name).write_bytes(b"")
            first = [path.name for path in _unique_named_files(root, ".wav")]
            second = [path.name for path in _unique_named_files(root, ".wav")]
        self.assertEqual(first, ["a.wav", "b.wav", "c.wav"])
        self.assertEqual(first, second)


class TrainingOnlySelectionTests(unittest.TestCase):
    def test_oof_ignores_evaluation_rows(self):
        rows = []
        for fold in range(5):
            for label, value in (("Normal", 0.0), ("Adventitious", 1.0)):
                rows.append({"record_id": "r%d_%s" % (fold, label), "group_id": "g%d_%s" % (fold, label), "split": "train", "fold_id": fold, "binary_label": label, "coarse_label": "Normal"})
        rows.extend([
            {"record_id": "eval1", "group_id": "eg1", "split": "test", "fold_id": -1, "binary_label": "Normal", "coarse_label": "Normal"},
            {"record_id": "eval2", "group_id": "eg2", "split": "test", "fold_id": -1, "binary_label": "Adventitious", "coarse_label": "Normal"},
        ])
        records = pd.DataFrame(rows)
        features = np.arange(len(records) * 2, dtype=np.float64).reshape(len(records), 2)
        indices, truth, probability = _oof_probabilities(features, records, "binary")
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(truth), 10)
        self.assertEqual(probability.shape, (10, 2))

    def test_alpha_tie_break_is_declared_and_deterministic(self):
        candidates = [
            {"alpha": 0.4, "balanced_accuracy": 0.8, "macro_f1": 0.7, "accuracy": 0.7},
            {"alpha": 0.6, "balanced_accuracy": 0.8, "macro_f1": 0.7, "accuracy": 0.7},
        ]
        selected, _ranked = select_alpha(candidates)
        self.assertEqual(selected, 0.4)


class ScientificSeparationTests(unittest.TestCase):
    def test_calculation_sources_have_no_frozen_input_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        sources = list((root / "src" / "full_reproduction").glob("*.py")) + [root / "scripts" / "full_reproduce.py"]
        forbidden = (
            "reference" + "_results",
            "expected" + "_values",
            "artifacts/" + "predictions",
            "historical_feature_cache/" + "private_assets",
        )
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, "%s contains %s" % (path, marker))


class ResumeIntegrityTests(unittest.TestCase):
    def test_resume_requires_file_and_protocol_hashes(self):
        names = (
            "cls_windows.npy",
            "temporal_grid_windows.npy",
            "cls_record_mean.npy",
            "temporal_grid_record_mean.npy",
            "single_window_cls.npy",
            "window_metadata.csv.gz",
            "single_window_metadata.csv.gz",
        )
        model = {
            "model_id": "model",
            "model_revision": "revision",
            "weights_sha256": "weights",
            "config_sha256": "config",
            "preprocessing_sha256": "preprocess",
        }
        contract = {"source_manifest_sha256": "source", "record_manifest_sha256": "records"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for index, name in enumerate(names):
                payload = ("file-%d" % index).encode("ascii")
                (root / name).write_bytes(payload)
                files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            manifest = {
                "status": "complete",
                "dataset": "hf_lung",
                "windows": 8,
                "records": 1,
                "batch_size": 32,
                "global_record_major_stream": True,
                "same_forward_cls_and_temporal_grid": True,
                "record_aggregation": "arithmetic_mean",
                "source_manifest_sha256": "source",
                "record_manifest_sha256": "records",
                "model": model,
                "files": files,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(_cache_complete(root, 8, 1, "hf_lung", model, contract))
            changed = dict(contract, record_manifest_sha256="different")
            self.assertFalse(_cache_complete(root, 8, 1, "hf_lung", model, changed))
            (root / names[0]).write_bytes(b"tampered")
            self.assertFalse(_cache_complete(root, 8, 1, "hf_lung", model, contract))


if __name__ == "__main__":
    unittest.main()
