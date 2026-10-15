"""CPU checks for metrics and the fixed-artifact test protocol; GPU calls are mocked."""

import contextlib
import csv
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
from laya.common import QTYPES, temp_bucket

SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_sentiment_test.py"
SPEC = importlib.util.spec_from_file_location("sentiment_test_script", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
REVIEW_SPEC = importlib.util.spec_from_file_location(
    "sentiment_review_script", SCRIPT.with_name("review_sentiment_results.py")
)
review = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(review)


class MetricsTests(unittest.TestCase):
    def test_confusion_macro_f1_and_scalar_invariance(self):
        gold, predicted = [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2]
        records = [
            (QTYPES["choice"], np.eye(3)[pred] * 6, np.eye(3)[label], 3)
            for label, pred in zip(gold, predicted)
        ]
        labels = [evaluation.LABELS[index] for index in gold]
        raw, _, _, p_raw = evaluation.score_records(records, labels)
        calibrated, _, _, p_cal = evaluation.score_records(records, labels, 2)
        self.assertEqual(raw["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(raw["accuracy"], 4 / 6)
        self.assertAlmostEqual(raw["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)
        self.assertEqual(raw["macro_f1"], calibrated["macro_f1"])
        self.assertEqual(raw["accuracy"], calibrated["accuracy"])
        np.testing.assert_array_equal(p_raw.argmax(axis=1), p_cal.argmax(axis=1))
        self.assertLess(p_cal.max(), p_raw.max())

    def test_rejects_wrong_labels_and_nonfinite_logits(self):
        records = [(QTYPES["choice"], np.asarray([1.0, 2.0, 3.0]), np.eye(3)[0], 3)]
        with self.assertRaises(AssertionError):
            evaluation.score_records(records, ["positive"])
        records[0][1][0] = np.nan
        with self.assertRaises(AssertionError):
            evaluation.score_records(records, ["negative"])


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    def to(self, *args, **kwargs):
        return self


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_cwd, self.old_version = Path.cwd(), evaluation.VERSION
        os.chdir(self.root)
        rows = []
        for index, label in enumerate(
            ["negative"] * 508 + ["neutral"] * 200 + ["positive"] * 431
        ):
            rows.append(
                {
                    "id": f"synthetic:test:{index}",
                    "split": "test",
                    "inputs": {
                        "text_mk": f"Тест објава {index}"
                        if index
                        else "Тест </script><img src=x>"
                    },
                    "target": {"type": "choice", "label": label},
                    "metadata": {"source_split": "test"},
                }
            )
        data = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode()
        manifest = {
            "reference_schema": {"type": "choice", "labels": evaluation.LABELS},
            "files": {
                "test": {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)},
                "calibration": {"sha256": "a" * 64, "rows": 856},
            },
        }
        canonical = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        evaluation.VERSION = hashlib.sha256(canonical).hexdigest()
        manifest["version"] = evaluation.VERSION
        release = (
            Path("datasets/releases/macedonian-tweet-sentiment") / evaluation.VERSION
        )
        release.mkdir(parents=True)
        (release / "test.jsonl").write_bytes(data)
        evaluation.atomic_json(release / "manifest.json", manifest)
        # No training, validation, or calibration data files exist in this fixture.
        questions = {
            "sentiment": {
                "type": "choice",
                "instructions": "Sentiment?",
                "criteria": dict(zip(evaluation.LABELS, evaluation.LABELS)),
            }
        }
        evaluation.atomic_json("questions.json", questions)
        self.selection = {
            "dataset_version": evaluation.VERSION,
            "max_len": 512,
            "head_max_len": 192,
            "questions_sha256": evaluation.file_hash("questions.json"),
            "models": {},
        }
        for alias in evaluation.MODELS:
            checkpoint = Path("models") / alias
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(alias.encode())
            evaluation.atomic_json(
                checkpoint / "rl_agent_config.json", {"max_len": 512}
            )
            self.selection["models"][alias] = {
                "checkpoint": str(checkpoint.resolve()),
                "weights_sha256": evaluation.file_hash(
                    checkpoint / "model.safetensors"
                ),
                "config_sha256": evaluation.file_hash(
                    checkpoint / "rl_agent_config.json"
                ),
                "best_epoch": None if alias.endswith("base") else 5,
            }
        evaluation.atomic_json("selected-models.json", self.selection)
        for alias, selected in self.selection["models"].items():
            folder = Path("outputs/calibration") / alias
            folder.mkdir(parents=True)
            if alias.endswith("base"):
                (folder / "calibration.json").write_text(
                    "invalid base calibration must be ignored"
                )
                continue
            payload = {
                "version": 2,
                "temperature": [2.0, 1.0, 1.0],
                "temperature_by_options": {},
            }
            evaluation.atomic_json(folder / "calibration.json", payload)
            report = dict(selected)
            report.update(
                status="complete",
                model=alias,
                split="calibration",
                examples=856,
                dataset_version=evaluation.VERSION,
                calibration_data_sha256="a" * 64,
                selection_sha256=evaluation.file_hash("selected-models.json"),
                questions_sha256=self.selection["questions_sha256"],
                selected_epoch=selected["best_epoch"],
                max_len=512,
                head_max_len=192,
                option_layout="sequential",
                label_order=evaluation.LABELS,
                raw_temperature=1.0,
                inference_precision="FP32",
                versions={"laya": "0.3.29"},
                checkpoint_weights_and_config_unchanged=True,
                calibration_sha256=evaluation.file_hash(folder / "calibration.json"),
                temperature=payload["temperature"],
                temperature_by_options={},
                n_by_bucket={temp_bucket(QTYPES["choice"], 3): 856},
                effective_choice_temperature=2.0,
            )
            evaluation.atomic_json(folder / "report.json", report)

    def tearDown(self):
        os.chdir(self.old_cwd)
        evaluation.VERSION = self.old_version
        self.temporary.cleanup()

    def test_mixed_checkpoint_calibration_is_rejected_before_inference(self):
        path = Path("outputs/calibration/english-full/report.json")
        report = json.loads(path.read_text())
        report["weights_sha256"] = self.selection["models"]["multilingual-full"][
            "weights_sha256"
        ]
        evaluation.atomic_json(path, report)
        with self.assertRaises(AssertionError):
            evaluation.load_protocol(
                Path("selected-models.json"), Path("outputs/calibration")
            )

    def test_all_models_raw_and_four_calibrated_without_other_cohort_reads(self):
        models = []

        def load(path):
            model = ToyModel()
            models.append(model)
            return model, None, {}

        def predict(model, tokenizer, items, *args, **kwargs):
            records = []
            for item in items:
                gold = evaluation.LABELS.index(item["expected"]["sentiment"])
                logits = np.eye(3, dtype=np.float32)[gold] * 4
                records.append(
                    (QTYPES["choice"], logits, np.eye(3, dtype=np.float32)[gold], 3)
                )
            return records

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(torch.cuda, "is_available", return_value=True)
            )
            stack.enter_context(
                patch.object(torch.cuda, "get_device_name", return_value="mock CUDA")
            )
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            stack.enter_context(patch.object(torch.cuda, "empty_cache"))
            stack.enter_context(
                patch.object(evaluation, "load_checkpoint", side_effect=load)
            )
            stack.enter_context(
                patch.object(
                    evaluation,
                    "items_from_rows",
                    side_effect=lambda tok, rows, *args: (rows, {}),
                )
            )
            stack.enter_context(
                patch.object(evaluation, "calibration_records", side_effect=predict)
            )
            stack.enter_context(patch.object(sys, "argv", [str(SCRIPT)]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            evaluation.main()
            with self.assertRaises(FileExistsError):
                evaluation.main()
        summary = json.loads(Path("outputs/final-test/summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["examples"], 1139)
        self.assertEqual(len(summary["models"]), 6)
        self.assertAlmostEqual(summary["majority_baseline"]["accuracy"], 508 / 1139)
        with Path("outputs/final-test/summary.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 10)
        for alias, result in summary["models"].items():
            self.assertEqual(result["raw"]["accuracy"], 1.0)
            self.assertEqual("calibrated" in result, not alias.endswith("base"))
            if "calibrated" in result:
                self.assertEqual(
                    result["raw"]["macro_f1"], result["calibrated"]["macro_f1"]
                )
                # A fixed temperature can worsen test NLL. Never refit it on test.
                self.assertGreater(result["calibrated"]["loss"], result["raw"]["loss"])
            with np.load(
                Path("outputs/final-test") / alias / "logits.npz", allow_pickle=False
            ) as arrays:
                self.assertEqual(arrays["logits"].shape, (1139, 3))
                self.assertEqual(arrays["label_order"].tolist(), evaluation.LABELS)
        self.assertTrue(
            all(
                torch.equal(model.weight.detach(), torch.tensor([1.0, 2.0, 3.0]))
                for model in models
            )
        )
        source_hash = evaluation.file_hash("outputs/final-test/summary.json")
        with contextlib.redirect_stdout(io.StringIO()):
            review.build_review(Path("outputs/final-test"), Path("review"))
        self.assertEqual(
            source_hash, evaluation.file_hash("outputs/final-test/summary.json")
        )
        html = Path("review/review.html").read_text(encoding="utf-8")
        self.assertNotIn("</script><img src=x>", html)
        embedded = html.split('<script type="application/json" id="data">', 1)[1].split(
            "</script>", 1
        )[0]
        self.assertEqual(
            json.loads(embedded)["rows"][0]["text"], "Тест </script><img src=x>"
        )
        corrupted = Path("outputs/final-test/english-base/predictions.jsonl")
        corrupted.write_text(corrupted.read_text() + "\n", encoding="utf-8")
        with self.assertRaises(AssertionError):
            review.build_review(Path("outputs/final-test"), Path("bad-review"))
        self.assertFalse(Path("bad-review").exists())


if __name__ == "__main__":
    unittest.main()
