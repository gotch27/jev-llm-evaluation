"""Evaluate the six fixed checkpoints and four fixed calibration maps on test only."""

import argparse
import csv
import gc
import hashlib
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, temp_bucket
from laya.train import (
    calibration_records,
    evaluate_records,
    items_from_rows,
    load_checkpoint,
)

VERSION = "e77449ed57a6a039a6b3a9fb21041b006307d7996d3eae5ed9177c40de528afc"
LABELS = ["negative", "neutral", "positive"]
MAX_LEN, HEAD_MAX_LEN = 512, 192
MODELS = [
    f"{model}-{mode}"
    for model in ("english", "multilingual")
    for mode in ("base", "full", "frozen")
]

if not __debug__:
    raise RuntimeError(
        "Run without -O or PYTHONOPTIMIZE; integrity checks must stay enabled."
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_checkpoint(selected):
    checkpoint = Path(selected["checkpoint"])
    assert file_hash(checkpoint / "model.safetensors") == selected["weights_sha256"], (
        "Checkpoint weights changed"
    )
    assert (
        file_hash(checkpoint / "rl_agent_config.json") == selected["config_sha256"]
    ), "Checkpoint config changed"
    return checkpoint


def load_protocol(selection_path, calibration_root):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["dataset_version"] == VERSION
    assert selection["max_len"] == MAX_LEN and selection["head_max_len"] == HEAD_MAX_LEN
    assert set(selection["models"]) == set(MODELS)
    assert file_hash("questions.json") == selection["questions_sha256"]
    questions = json.loads(Path("questions.json").read_text(encoding="utf-8"))
    assert set(questions) == {"sentiment"}
    assert questions["sentiment"]["type"] == "choice"
    assert list(questions["sentiment"]["criteria"]) == LABELS

    root = Path("datasets/releases/macedonian-tweet-sentiment") / VERSION
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    canonical = json.dumps(
        {key: value for key, value in manifest.items() if key != "version"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert manifest["version"] == hashlib.sha256(canonical).hexdigest() == VERSION
    assert manifest["reference_schema"] == {"type": "choice", "labels": LABELS}
    calibrations = {}
    for alias in MODELS:
        selected = selection["models"][alias]
        checkpoint = verify_checkpoint(selected)
        if alias.endswith("-base"):
            assert selected["best_epoch"] is None
            # Deliberately do not load any base-model calibration files.
            continue
        folder = calibration_root / alias
        report_path, payload_path = folder / "report.json", folder / "calibration.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert report["status"] == "complete" and report["model"] == alias
        assert report["split"] == "calibration" and report["examples"] == 856
        assert report["dataset_version"] == VERSION
        assert (
            report["calibration_data_sha256"]
            == manifest["files"]["calibration"]["sha256"]
        )
        assert report["selection_sha256"] == file_hash(selection_path)
        assert report["questions_sha256"] == selection["questions_sha256"]
        assert report["weights_sha256"] == selected["weights_sha256"]
        assert report["config_sha256"] == selected["config_sha256"]
        assert Path(report["checkpoint"]).resolve() == checkpoint.resolve()
        assert report["selected_epoch"] == selected["best_epoch"]
        assert report["max_len"] == MAX_LEN and report["head_max_len"] == HEAD_MAX_LEN
        assert (
            report["option_layout"] == "sequential" and report["label_order"] == LABELS
        )
        assert (
            report["raw_temperature"] == 1.0 and report["inference_precision"] == "FP32"
        )
        assert report["versions"]["laya"] == "0.3.29"
        assert report["checkpoint_weights_and_config_unchanged"]
        assert file_hash(payload_path) == report["calibration_sha256"]
        assert payload["temperature"] == report["temperature"]
        assert (
            payload["temperature_by_options"] == report["temperature_by_options"] == {}
        )
        assert len(payload["temperature"]) == 3
        assert report["n_by_bucket"] == {temp_bucket(QTYPES["choice"], 3): 856}
        assert not payload.get("binning_map"), (
            "This protocol uses temperature scaling only"
        )
        temperature = float(payload["temperature"][QTYPES["choice"]])
        assert np.isfinite(temperature) and TEMP_MIN <= temperature <= TEMP_MAX
        assert temperature == report["effective_choice_temperature"]
        calibrations[alias] = {
            "temperature": temperature,
            "payload_path": str(payload_path.resolve()),
            "payload_sha256": file_hash(payload_path),
            "report_path": str(report_path.resolve()),
            "report_sha256": file_hash(report_path),
        }
    return selection, questions, manifest, root, calibrations


def load_test(root, manifest):
    path = root / "test.jsonl"
    assert file_hash(path) == manifest["files"]["test"]["sha256"]
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == manifest["files"]["test"]["rows"] == 1139
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(
        row["split"] == "test" and row["metadata"]["source_split"] == "test"
        for row in rows
    )
    assert all(
        row["target"]["type"] == "choice" and row["target"]["label"] in LABELS
        for row in rows
    )
    return rows, path


def score_records(records, expected_labels, temperature=1.0):
    assert len(records) == len(expected_labels) > 0
    assert all(qtype == QTYPES["choice"] and k == 3 for qtype, _, _, k in records)
    assert np.isfinite(temperature) and temperature > 0
    logits = np.asarray([record[1] for record in records], dtype=np.float64)
    targets = np.asarray([record[2] for record in records], dtype=np.float64)
    assert logits.shape == targets.shape == (len(records), 3)
    assert np.isfinite(logits).all() and np.isfinite(targets).all()
    gold = np.asarray([LABELS.index(label) for label in expected_labels])
    assert np.array_equal(targets, np.eye(3)[gold]), "Incorrect target mapping/order"
    z = logits / temperature
    z -= z.max(axis=1, keepdims=True)
    probabilities = np.exp(z)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    assert np.array_equal(predicted, logits.argmax(axis=1)), (
        "Scaling changed predictions"
    )
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (gold, predicted), 1)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2 * confusion.diagonal(), denominator, out=np.zeros(3), where=denominator > 0
    )
    metrics = evaluate_records(records, temperature=temperature)
    metrics.update(
        accuracy=float((predicted == gold).mean()),
        macro_f1=float(f1.mean()),
        f1_by_class={label: float(f1[index]) for index, label in enumerate(LABELS)},
        predicted_counts={
            label: int((predicted == index).sum()) for index, label in enumerate(LABELS)
        },
        gold_counts={
            label: int((gold == index).sum()) for index, label in enumerate(LABELS)
        },
        confusion_matrix=confusion.tolist(),
        confusion_label_order=LABELS,
        confusion_rows="actual",
        confusion_columns="predicted",
        temperature=float(temperature),
    )
    return metrics, logits, targets, probabilities


def write_summary(output, report):
    atomic_json(output / "summary.json", report)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "variant",
                "selected_epoch",
                "temperature",
                "accuracy",
                "macro_f1",
                "loss",
                "ece",
                "brier",
            ],
        )
        writer.writeheader()
        for alias, result in report["models"].items():
            for variant in ("raw", "calibrated"):
                if variant in result:
                    metrics = result[variant]
                    writer.writerow(
                        dict(
                            model=alias,
                            variant=variant,
                            selected_epoch=result["selected_epoch"],
                            **{
                                name: metrics[name]
                                for name in (
                                    "temperature",
                                    "accuracy",
                                    "macro_f1",
                                    "loss",
                                    "ece",
                                    "brier",
                                )
                            },
                        )
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("selected-models.json"))
    parser.add_argument(
        "--calibration-root", type=Path, default=Path("outputs/calibration")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/final-test"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    assert args.batch_size > 0 and torch.cuda.is_available(), (
        "A working CUDA GPU is required"
    )
    assert version("laya") == "0.3.29", (
        "Recheck the evaluation protocol before changing Laya versions"
    )
    selection, questions, manifest, root, calibrations = load_protocol(
        args.selection, args.calibration_root
    )
    rows, data_path = load_test(root, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    counts = Counter(row["target"]["label"] for row in rows)
    # Negative was the majority class in the training cohort. Fix it before test;
    # do not choose this baseline's label by inspecting test frequencies.
    majority = "negative"
    majority_records = [
        (
            QTYPES["choice"],
            np.eye(3)[LABELS.index(majority)],
            np.eye(3)[LABELS.index(row["target"]["label"])],
            3,
        )
        for row in rows
    ]
    majority_metrics, _, _, _ = score_records(
        majority_records, [row["target"]["label"] for row in rows]
    )
    report = {
        "status": "running",
        "split": "test",
        "examples": len(rows),
        "dataset_version": VERSION,
        "test_data_sha256": file_hash(data_path),
        "questions_sha256": selection["questions_sha256"],
        "selection_sha256": file_hash(args.selection),
        "script_sha256": file_hash(__file__),
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "label_order": LABELS,
        "option_layout": "sequential",
        "inference_precision": "FP32",
        "batch_size": args.batch_size,
        "versions": {
            name: version(name) for name in ("laya", "torch", "transformers", "numpy")
        },
        "gpu": torch.cuda.get_device_name(0),
        "protocol": "Six fixed checkpoints at T=1; four fine-tuned checkpoints additionally use previously fitted calibration. No fitting or test-based selection.",
        "majority_baseline": {
            "label": majority,
            "label_source": "training-cohort majority, fixed before test",
            "accuracy": majority_metrics["accuracy"],
            "macro_f1": majority_metrics["macro_f1"],
            "gold_counts": dict(counts),
        },
        "models": {},
    }
    started = time.perf_counter()
    atomic_json(args.output / "report.json", report)
    try:
        for alias in MODELS:
            report["current_model"] = alias
            atomic_json(args.output / "report.json", report)
            selected = selection["models"][alias]
            checkpoint = verify_checkpoint(selected)
            print(f"\nEvaluating {alias}: {checkpoint}", flush=True)
            model, tokenizer, _ = load_checkpoint(str(checkpoint))
            # Gold labels create evaluation targets only; the model sees text + the shared question/options.
            formatted = [
                {
                    "state": row["inputs"]["text_mk"],
                    "questions": questions,
                    "expected": {"sentiment": row["target"]["label"]},
                }
                for row in rows
            ]
            items, skipped = items_from_rows(
                tokenizer, formatted, MAX_LEN, HEAD_MAX_LEN
            )
            assert not skipped and len(items) == len(rows), (
                f"Rejected test examples: {skipped}"
            )
            device = torch.device("cuda:0")
            model = model.float().to(device).eval()
            model.requires_grad_(False)
            records = []
            inference_started = time.perf_counter()
            with torch.inference_mode():
                for offset in range(0, len(items), 100):
                    records.extend(
                        calibration_records(
                            model,
                            tokenizer,
                            items[offset : offset + 100],
                            device,
                            MAX_LEN,
                            HEAD_MAX_LEN,
                            batch_size=args.batch_size,
                            parallel=False,
                        )
                    )
                    print(f"Evaluated {len(records)}/{len(rows)}", flush=True)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_started
            del model
            gc.collect()
            torch.cuda.empty_cache()
            expected = [row["target"]["label"] for row in rows]
            raw, logits, targets, probabilities = score_records(records, expected)
            result = {
                "model": alias,
                "checkpoint": str(checkpoint.resolve()),
                "weights_sha256": selected["weights_sha256"],
                "config_sha256": selected["config_sha256"],
                "selected_epoch": selected["best_epoch"],
                "raw": raw,
                "inference_seconds": inference_seconds,
            }
            calibrated_probabilities = None
            if alias in calibrations:
                calibration = calibrations[alias]
                calibrated, _, _, calibrated_probabilities = score_records(
                    records, expected, calibration["temperature"]
                )
                assert calibrated["confusion_matrix"] == raw["confusion_matrix"]
                assert (
                    calibrated["accuracy"] == raw["accuracy"]
                    and calibrated["macro_f1"] == raw["macro_f1"]
                )
                result.update(calibrated=calibrated, calibration=calibration)
                assert (
                    file_hash(calibration["payload_path"])
                    == calibration["payload_sha256"]
                )
                assert (
                    file_hash(calibration["report_path"])
                    == calibration["report_sha256"]
                )
            verify_checkpoint(selected)
            destination = args.output / alias
            destination.mkdir()
            np.savez_compressed(
                destination / "logits.npz",
                ids=np.asarray([row["id"] for row in rows]),
                label_order=np.asarray(LABELS),
                logits=logits,
                targets=targets,
            )
            with (destination / "predictions.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for index, row in enumerate(rows):
                    prediction = {
                        "id": row["id"],
                        "split": "test",
                        "text_mk": row["inputs"]["text_mk"],
                        "gold": expected[index],
                        "predicted": LABELS[int(logits[index].argmax())],
                        "probabilities_raw": probabilities[index].tolist(),
                    }
                    if calibrated_probabilities is not None:
                        prediction["probabilities_calibrated"] = (
                            calibrated_probabilities[index].tolist()
                        )
                    stream.write(
                        json.dumps(prediction, ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
            result.update(
                logits_sha256=file_hash(destination / "logits.npz"),
                predictions_sha256=file_hash(destination / "predictions.jsonl"),
                checkpoint_weights_and_config_unchanged=True,
            )
            atomic_json(destination / "report.json", result)
            report["models"][alias] = result
            report["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(args.output / "report.json", report)
            print(
                f"{alias}: test accuracy={raw['accuracy']:.1%}; macro-F1={raw['macro_f1']:.4f}",
                flush=True,
            )
            if "calibrated" in result:
                print(
                    f"  Confidence metrics before -> after calibration: NLL {raw['loss']:.4f} -> {calibrated['loss']:.4f}; ECE {raw['ece']:.4f} -> {calibrated['ece']:.4f}; Brier {raw['brier']:.4f} -> {calibrated['brier']:.4f}",
                    flush=True,
                )
        assert file_hash(args.selection) == report["selection_sha256"]
        assert file_hash("questions.json") == report["questions_sha256"]
        assert file_hash(data_path) == report["test_data_sha256"]
        report.pop("current_model", None)
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        write_summary(args.output, report)
        atomic_json(args.output / "report.json", report)
        print("\nFINAL TEST COMPLETE")
        print(
            f"Training-majority-class baseline ({majority}): accuracy={majority_metrics['accuracy']:.1%}; macro-F1={majority_metrics['macro_f1']:.4f}"
        )
        for alias, result in report["models"].items():
            print(
                f"{alias}: accuracy={result['raw']['accuracy']:.1%}; macro-F1={result['raw']['macro_f1']:.4f}; epoch={result['selected_epoch']}"
            )
        print(
            "Saved:", args.output / "summary.csv", "and", args.output / "summary.json"
        )
        print(
            "No training, calibration fitting, or checkpoint selection performed. Bases evaluated at T=1 only."
        )
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_json(args.output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
