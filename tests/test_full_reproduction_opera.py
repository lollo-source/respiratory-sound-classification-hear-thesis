import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.full_reproduction import datasets
from src.full_reproduction import opera
from src.full_reproduction import opera_downstream


class OperaIdentityTests(unittest.TestCase):
    def test_model_checkpoint_mapping_and_dimensions_are_pinned(self):
        expected = {
            "operaCE": ("OPERA-CE", "encoder-operaCE.ckpt", 1280),
            "operaCT": ("OPERA-CT", "encoder-operaCT.ckpt", 768),
            "operaGT": ("OPERA-GT", "encoder-operaGT.ckpt", 384),
        }
        self.assertEqual(set(opera.MODEL_SPECS), set(expected))
        for name, (method, checkpoint, dimension) in expected.items():
            spec = opera.MODEL_SPECS[name]
            self.assertEqual((spec["method"], spec["checkpoint"], spec["dimension"]),
                             (method, checkpoint, dimension))
            self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(opera.OPERA_REVISION, r"^[0-9a-f]{40}$")

    def test_native_preprocessing_routes_ce_and_gt_separately(self):
        calls = []
        utility = types.ModuleType("src.util")

        def entire(root, stem, **kwargs):
            calls.append(("entire", root, stem, kwargs))
            return np.zeros((12, 64), dtype=np.float32)

        def split(root, stem, **kwargs):
            calls.append(("split", root, stem, kwargs))
            return [np.zeros((16, 64), dtype=np.float32)]

        utility.get_entire_signal_librosa = entire
        utility.get_split_signal_librosa = split
        with mock.patch.object(opera, "_add_source"), mock.patch.dict(sys.modules, {"src.util": utility}):
            opera.spectrograms("/authorised/example.wav", "operaCE", "/explicit/opera", 8.0)
            opera.spectrograms("/authorised/example.wav", "operaGT", "/explicit/opera", 8.0)
        self.assertEqual(calls[0], ("entire", "", "/authorised/example",
                                    {"spectrogram": True, "input_sec": 8.0, "pad": True}))
        self.assertEqual(calls[1], ("split", "", "/authorised/example",
                                    {"spectrogram": True, "input_sec": 8.0}))

    def test_gt_record_representation_is_mean_of_native_splits(self):
        import torch

        class Model:
            def forward_feature(self, tensor):
                values = tensor[:, 0, 0].reshape(-1, 1)
                return values.repeat(1, opera.MODEL_SPECS["operaGT"]["dimension"])

        records = pd.DataFrame([{"record_id": "bio_record_00001", "audio_path": "/x.wav"}])
        splits = [np.ones((16, 64), dtype=np.float32), np.full((16, 64), 3, dtype=np.float32)]
        with mock.patch.object(opera, "spectrograms", return_value=splits):
            actual = opera.infer_records(Model(), "operaGT", records, "/explicit/opera", torch.device("cpu"))
        self.assertEqual(actual.shape, (1, 384))
        np.testing.assert_array_equal(actual, np.full((1, 384), 2, dtype=np.float32))


class OperaDatasetTests(unittest.TestCase):
    def test_adapter_preserves_public_manifest_order_and_pseudonyms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locations = [
                root / "BioCAS2022" / "train2022_wav",
                root / "BioCAS2022" / "test2022_wav",
                root / "BioCAS2023" / "test2023_wav",
            ]
            for location, stem in zip(locations, ("z", "a", "m")):
                location.mkdir(parents=True)
                (location / (stem + ".wav")).write_bytes(b"")
            # Sorted stems a,m,z map to public pseudonyms 1,2,3. The manifest
            # deliberately uses a different row order, which the adapter keeps.
            manifest = pd.DataFrame([
                {"record_id": "bio_record_00003", "split": "train2022", "patient_group": "bio_group_003",
                 "binary_label": "Normal", "coarse_label": "Normal", "inclusion_status": "included"},
                {"record_id": "bio_record_00001", "split": "test2022", "patient_group": "bio_group_001",
                 "binary_label": "Adventitious", "coarse_label": "CAS only", "inclusion_status": "included"},
                {"record_id": "bio_record_00002", "split": "test2023", "patient_group": "bio_group_002",
                 "binary_label": "Normal", "coarse_label": "Normal", "inclusion_status": "excluded"},
            ])
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            with mock.patch.object(datasets, "BIO_ALL_RECORDS", 3), \
                 mock.patch.object(datasets, "BIO_INCLUDED_RECORDS", 2), \
                 mock.patch.object(datasets, "_folds", side_effect=lambda frame, *_: frame.assign(fold_id=-1)):
                first = datasets.load_biocas(root, manifest_path)
                second = datasets.load_biocas(root, manifest_path)
        self.assertEqual(first.record_id.tolist(), ["bio_record_00003", "bio_record_00001"])
        self.assertEqual(first.record_id.tolist(), second.record_id.tolist())
        self.assertNotIn("audio_path", datasets.public_record_frame(first).columns)
        self.assertTrue(all(value.startswith("bio_record_") for value in first.record_id))


class OperaDownstreamTests(unittest.TestCase):
    def test_class_order_and_classifier_protocol(self):
        self.assertEqual(opera_downstream.TASKS["binary"]["classes"], ["Normal", "Adventitious"])
        self.assertEqual(opera_downstream.TASKS["coarse"]["classes"],
                         ["Normal", "CAS only", "DAS only", "CAS and DAS"])
        pipeline = opera_downstream.make_classifier()
        classifier = pipeline.named_steps["classifier"]
        self.assertEqual(classifier.C, 1.0)
        self.assertEqual(classifier.penalty, "l2")
        self.assertEqual(classifier.solver, "lbfgs")
        self.assertEqual(classifier.max_iter, 5000)
        self.assertIsNone(classifier.class_weight)

    def test_metric_calculation(self):
        truth = np.array(["Normal", "Normal", "Adventitious", "Adventitious"])
        prediction = np.array(["Normal", "Adventitious", "Adventitious", "Adventitious"])
        result = opera_downstream.metrics(truth, prediction, ["Normal", "Adventitious"])
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2.0 / 3.0 + 0.8) / 2.0)

    def test_fit_receives_training_rows_only(self):
        records = pd.DataFrame([
            {"record_id": "r0", "split": "train2022", "binary_label": "Normal", "coarse_label": "Normal"},
            {"record_id": "r1", "split": "train2022", "binary_label": "Adventitious", "coarse_label": "CAS only"},
            {"record_id": "r2", "split": "train2022", "binary_label": "Adventitious", "coarse_label": "DAS only"},
            {"record_id": "r3", "split": "train2022", "binary_label": "Adventitious", "coarse_label": "CAS and DAS"},
            {"record_id": "r4", "split": "test2022", "binary_label": "Normal", "coarse_label": "Normal"},
            {"record_id": "r5", "split": "test2023", "binary_label": "Adventitious", "coarse_label": "CAS only"},
        ])
        fitted_rows = []

        class Spy:
            def __init__(self):
                self.named_steps = {"classifier": self}
                self.classes_ = None

            def fit(self, values, labels):
                fitted_rows.append((values.copy(), labels.copy()))
                self.classes_ = np.unique(labels)
                return self

            def predict_proba(self, values):
                return np.full((len(values), len(self.classes_)), 1.0 / len(self.classes_))

        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.npy"
            values = np.arange(len(records) * 384, dtype=np.float32).reshape(len(records), 384)
            np.save(feature_path, values)
            with mock.patch.object(opera_downstream, "make_classifier", side_effect=Spy):
                opera_downstream.run_downstream(records, {"operaGT": feature_path}, Path(directory) / "out")
        self.assertEqual(len(fitted_rows), 2)
        for fitted_values, _labels in fitted_rows:
            self.assertEqual(len(fitted_values), 4)
            np.testing.assert_array_equal(fitted_values, values[:4])


class OperaScientificSeparationTests(unittest.TestCase):
    def test_opera_calculation_has_no_reference_cache_or_private_fallback(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "src" / "full_reproduction" / "opera.py",
            root / "src" / "full_reproduction" / "opera_pipeline.py",
            root / "src" / "full_reproduction" / "opera_downstream.py",
        ]
        forbidden = (
            "reference" + "_results", "expected" + "_values", "artifacts/" + "predictions",
            "data/" + "embeddings", "/" + "nas/", "/" + "home/", "old final OPERA",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, source, "%s contains forbidden marker %s" % (path, marker))

    def test_downstream_reads_only_explicit_fresh_feature_paths(self):
        source = inspect.getsource(opera_downstream.run_downstream)
        self.assertIn("feature_paths.items()", source)
        self.assertNotIn("glob(", source)
        self.assertNotIn("rglob(", source)


if __name__ == "__main__":
    unittest.main()
