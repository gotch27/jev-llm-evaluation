"""CPU regression checks for calibration and training orchestration, without checkpoints."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from laya.common import QTYPES

EXPERIMENT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load_script("train_sentiment")
calibration = load_script("calibrate_sentiment")


class TargetAndCalibrationTests(unittest.TestCase):
    def test_training_rejects_soft_or_nonfinite_targets_with_correct_argmax(self):
        for target in ([0.8, 0.1, 0.1], [1.0, np.nan, 0.0]):
            with self.subTest(target=target), self.assertRaises(AssertionError):
                training.validation_metrics(
                    [(QTYPES["choice"], [3.0, 1.0, 0.0], target, 3)], ["negative"]
                )

    def test_temperature_fits_real_laya_scalar_and_preserves_labels_and_logits(self):
        records, labels = [], []
        for index in range(60):
            gold = index % 3
            predicted = gold if index % 4 else (gold + 1) % 3
            records.append(
                (QTYPES["choice"], np.eye(3)[predicted] * 8, np.eye(3)[gold], 3)
            )
            labels.append(training.LABELS[gold])
        original = np.asarray([record[1].copy() for record in records])
        fitted, temperature, before, after, logits, _ = calibration.fit_and_measure(
            records, labels
        )
        self.assertGreater(temperature, 1.0)
        self.assertLessEqual(temperature, 5.0)
        self.assertEqual(fitted["temperature_by_options"], {})
        self.assertEqual(sum(fitted["n_by_bucket"].values()), 60)
        self.assertEqual(before["accuracy"], after["accuracy"])
        self.assertLess(after["loss"], before["loss"])
        np.testing.assert_array_equal(logits, original)
        np.testing.assert_array_equal([record[1] for record in records], original)

    def test_calibration_rejects_incorrect_target_order_before_fitting(self):
        with patch.object(calibration, "fit_temperature_map") as fit:
            with self.assertRaises(AssertionError):
                calibration.fit_and_measure(
                    [(QTYPES["choice"], [3.0, 1.0, 0.0], [1.0, 0.0, 0.0], 3)],
                    ["positive"],
                )
            fit.assert_not_called()


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        self.head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.encoder.weight.fill_(1)
            self.head.weight.zero_()

    def to(self, *args, **kwargs):
        return self  # Keep orchestration tests on CPU while mocking the CUDA boundary.


class TrainingFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        questions = (EXPERIMENT / "questions.json").read_bytes()
        Path("questions.json").write_bytes(questions)
        files, contents = {}, {}
        for split, count in (("train", 6849), ("validation", 856)):
            rows = [
                {
                    "split": split,
                    "inputs": {"text_mk": f"{split}:{index}"},
                    "target": {"label": training.LABELS[index % 3]},
                    "metadata": {"duplicate_group": f"{split}:{index}"},
                }
                for index in range(count)
            ]
            data = "".join(json.dumps(row) + "\n" for row in rows).encode()
            contents[split] = data
            files[split] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": count}
        manifest = {
            "reference_schema": {"type": "choice", "labels": training.LABELS},
            "files": files,
        }
        canonical = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        self.dataset_version = hashlib.sha256(canonical).hexdigest()
        manifest["version"] = self.dataset_version
        release = (
            Path("datasets/releases/macedonian-tweet-sentiment") / self.dataset_version
        )
        release.mkdir(parents=True)
        for split, data in contents.items():
            (release / f"{split}.jsonl").write_bytes(data)
        training.atomic_json(release / "manifest.json", manifest)
        # There are deliberately no calibration/test files in this fixture.
        self.checkpoint = self.root / "laya-yelp/models/laya-english"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "model.safetensors").write_bytes(b"synthetic base weights")
        baseline = Path("outputs/english-base-validation")
        baseline.mkdir(parents=True)
        training.atomic_json(
            baseline / "report.json",
            {
                "weights_sha256": training.file_hash(
                    self.checkpoint / "model.safetensors"
                ),
                "questions_sha256": training.file_hash("questions.json"),
                "dataset_version": self.dataset_version,
                "metrics": {},
            },
        )

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary.cleanup()

    def run_training(self, frozen):
        saved = {}

        def load(path):
            model = ToyModel()
            if path in saved:
                model.load_state_dict(saved[path][0])
            return (
                model,
                None,
                {
                    "temperature": [2, 2, 2],
                    "temperature_by_options": {"old": 3},
                    "binning_map": {"old": []},
                    "lang_temperatures": {"mk": 3},
                },
            )

        def save(model, tokenizer, config, path):
            Path(path).mkdir(parents=True)
            saved[str(Path(path).resolve())] = (
                {name: value.clone() for name, value in model.state_dict().items()},
                config,
            )

        def predict(model, tokenizer, items, *args, **kwargs):
            model.eval()  # The real library does this; the callback must restore train mode.
            stage = int(model.head.weight.item())
            records = []
            for index, item in enumerate(items):
                self.assertTrue(item["state"].startswith("validation:"))
                gold = training.LABELS.index(item["expected"]["sentiment"])
                predicted = gold if stage == 2 or index % 2 else (gold + 1) % 3
                records.append(
                    (QTYPES["choice"], np.eye(3)[predicted] * 4, np.eye(3)[gold], 3)
                )
            return records

        def train(model, tokenizer, items, config, *args, on_epoch_end):
            self.assertTrue(all(item["state"].startswith("train:") for item in items))
            self.assertEqual(len(items), 6849)
            self.assertEqual(config.freeze_encoder, frozen)
            self.assertEqual(config.loss, "soft-ce")
            self.assertFalse(config.amp)
            self.assertEqual(config.calib_frac, 0)
            model.train()
            if frozen:
                model.encoder.requires_grad_(False).eval()
            for epoch in range(config.epochs):
                with torch.no_grad():
                    model.head.weight.fill_(epoch + 1)
                    if not frozen:
                        model.encoder.weight.add_(1)
                on_epoch_end(epoch, 0.5)
                self.assertTrue(model.training)
                self.assertEqual(model.encoder.training, not frozen)
            model.eval()

        argv = [
            "train_sentiment.py",
            "--model",
            "english",
            "--epochs",
            "3",
            "--output",
            "outputs/run",
        ]
        if frozen:
            argv.append("--freeze-encoder")
        with contextlib.ExitStack() as stack:
            for name, value in (
                ("is_available", True),
                ("reset_peak_memory_stats", None),
                ("empty_cache", None),
                ("max_memory_allocated", 0),
            ):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            stack.enter_context(patch.object(training, "VERSION", self.dataset_version))
            stack.enter_context(patch.object(Path, "home", return_value=self.root))
            stack.enter_context(
                patch.object(training, "load_checkpoint", side_effect=load)
            )
            stack.enter_context(
                patch.object(training, "save_checkpoint", side_effect=save)
            )
            stack.enter_context(
                patch.object(training, "train_model", side_effect=train)
            )
            stack.enter_context(
                patch.object(training, "calibration_records", side_effect=predict)
            )
            stack.enter_context(
                patch.object(
                    training,
                    "items_from_rows",
                    side_effect=lambda tok, rows, *args: (rows, {}),
                )
            )
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            training.main()
            with self.assertRaises(FileExistsError):
                training.main()
        report = json.loads(Path("outputs/run/report.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["best_epoch"], 2)
        self.assertEqual(len(report["epochs"]), 3)
        self.assertEqual(report["saved_checkpoint_validation"]["macro_f1"], 1)
        self.assertEqual(report["encoder_parameters_changed"], not frozen)
        _, config = saved[report["best_checkpoint"]]
        self.assertEqual(config["temperature"], [1.0, 1.0, 1.0])
        self.assertEqual(config["option_layout"], "sequential")
        self.assertTrue(config["fine_tuned"])
        for key in ("temperature_by_options", "binning_map", "lang_temperatures"):
            self.assertNotIn(key, config)
        self.assertEqual(
            (Path(report["best_checkpoint"]) / "questions.json").read_bytes(),
            Path("questions.json").read_bytes(),
        )

    def test_full_training_selects_validation_epoch_and_reloads_it(self):
        self.run_training(frozen=False)

    def test_frozen_training_restores_encoder_eval_mode_after_validation(self):
        self.run_training(frozen=True)
