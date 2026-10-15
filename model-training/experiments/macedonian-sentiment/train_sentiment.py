"""Fine-tune Laya on the frozen Macedonian sentiment training split only."""

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from laya.common import QTYPES
from laya.train import (
    TrainConfig,
    calibration_records,
    evaluate_records,
    items_from_rows,
    load_checkpoint,
    save_checkpoint,
    train_model,
)

VERSION = "e77449ed57a6a039a6b3a9fb21041b006307d7996d3eae5ed9177c40de528afc"
MAX_LEN, HEAD_MAX_LEN = 512, 192
LABELS = ["negative", "neutral", "positive"]

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


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validation_metrics(records, expected_labels):
    assert len(records) == len(expected_labels) > 0
    assert all(qtype == QTYPES["choice"] and k == 3 for qtype, _, _, k in records)
    logits = np.asarray([record[1] for record in records], dtype=np.float64)
    targets = np.asarray([record[2] for record in records], dtype=np.float64)
    assert logits.shape == targets.shape == (len(records), 3)
    assert np.isfinite(logits).all() and np.isfinite(targets).all()
    gold = np.asarray([LABELS.index(label) for label in expected_labels])
    assert np.array_equal(targets, np.eye(3)[gold]), "Incorrect target mapping/order"
    predicted = logits.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (gold, predicted), 1)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2 * confusion.diagonal(), denominator, out=np.zeros(3), where=denominator > 0
    )
    metrics = evaluate_records(records, temperature=1.0)
    metrics.update(
        macro_f1=float(f1.mean()),
        f1_by_class={label: float(f1[i]) for i, label in enumerate(LABELS)},
        predicted_counts={
            label: int((predicted == i).sum()) for i, label in enumerate(LABELS)
        },
        confusion_matrix=confusion.tolist(),
        confusion_label_order=LABELS,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["english", "multilingual"], required=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert args.epochs > 0 and torch.cuda.is_available()
    assert version("laya") == "0.3.29", (
        "Recheck the training protocol before changing Laya versions"
    )
    device = torch.device("cuda:0")
    root = Path("datasets/releases/macedonian-tweet-sentiment") / VERSION
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = {key: value for key, value in manifest.items() if key != "version"}
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert manifest["version"] == hashlib.sha256(canonical).hexdigest() == VERSION
    assert manifest["reference_schema"] == {"type": "choice", "labels": LABELS}
    question_bytes = Path("questions.json").read_bytes()
    question_hash = hashlib.sha256(question_bytes).hexdigest()
    questions = json.loads(question_bytes)
    assert set(questions) == {"sentiment"}
    assert questions["sentiment"]["type"] == "choice"
    assert list(questions["sentiment"]["criteria"]) == LABELS
    rows = {}
    for split, expected_count in (("train", 6849), ("validation", 856)):
        path = root / f"{split}.jsonl"
        assert file_hash(path) == manifest["files"][split]["sha256"]
        rows[split] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows[split]) == expected_count
        assert all(row["split"] == split for row in rows[split])
    train_groups = {row["metadata"]["duplicate_group"] for row in rows["train"]}
    validation_groups = {
        row["metadata"]["duplicate_group"] for row in rows["validation"]
    }
    assert not train_groups & validation_groups

    checkpoint = Path.home() / "laya-yelp/models" / f"laya-{args.model}"
    weights_hash = file_hash(checkpoint / "model.safetensors")
    baseline = json.loads(
        (Path("outputs") / f"{args.model}-base-validation/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["weights_sha256"] == weights_hash
    assert baseline["questions_sha256"] == question_hash
    assert baseline["dataset_version"] == VERSION
    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=4,
        grad_accum=4,
        encoder_lr=2.5e-5,
        head_lr=1e-4,
        loss="soft-ce",
        freeze_encoder=args.freeze_encoder,
        amp=False,
        gradient_checkpointing=True,
        shuffle_options=(),
        label_smoothing=0.0,
        calib_frac=0.0,
        calib_max=0,
        seed=42,
        log_every=250,
    )
    config.validate()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "model": args.model,
        "source_checkpoint": str(checkpoint),
        "source_weights_sha256": weights_hash,
        "dataset_version": VERSION,
        "questions_sha256": question_hash,
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "config": asdict(config),
        "temperature": 1.0,
        "precision": "FP32",
        "checkpoint_export_precision": "FP16",
        "selection": "Highest validation macro-F1; tie: lowest cross-entropy",
        "baseline_validation": baseline["metrics"],
        "versions": {
            name: version(name) for name in ("laya", "torch", "transformers", "numpy")
        },
        "script_sha256": file_hash(__file__),
        "epochs": [],
    }
    started = time.perf_counter()
    atomic_json(output / "report.json", report)
    try:
        print("Loading original checkpoint:", checkpoint, flush=True)
        model, tokenizer, cfg = load_checkpoint(str(checkpoint))
        items = {}
        for split, split_rows in rows.items():
            formatted = [
                {
                    "state": row["inputs"]["text_mk"],
                    "questions": questions,
                    "expected": {"sentiment": row["target"]["label"]},
                }
                for row in split_rows
            ]
            items[split], skipped = items_from_rows(
                tokenizer, formatted, MAX_LEN, HEAD_MAX_LEN
            )
            assert not skipped and len(items[split]) == len(split_rows), (
                f"{split}: {skipped}"
            )
        encoder_before = parameter_hash(model.encoder.named_parameters())
        model = model.to(device)
        torch.cuda.reset_peak_memory_stats()
        expected = [row["target"]["label"] for row in rows["validation"]]

        def measure(current_model, current_tokenizer):
            was_training = current_model.training
            try:
                records = []
                with torch.inference_mode():
                    for offset in range(0, len(items["validation"]), 200):
                        records.extend(
                            calibration_records(
                                current_model,
                                current_tokenizer,
                                items["validation"][offset : offset + 200],
                                device,
                                MAX_LEN,
                                HEAD_MAX_LEN,
                                batch_size=8,
                            )
                        )
                return validation_metrics(records, expected)
            finally:
                current_model.train(was_training)
                if args.freeze_encoder:
                    current_model.encoder.eval()

        def on_epoch_end(epoch_index, mean_loss):
            assert np.isfinite(mean_loss), "Non-finite training loss"
            epoch = epoch_index + 1
            metrics = measure(model, tokenizer)
            key = (metrics["macro_f1"], -metrics["loss"])
            better = "best_key" not in report or key > tuple(report["best_key"])
            entry = {"epoch": epoch, "training_loss": mean_loss, "validation": metrics}
            if better:
                destination = output / f"epoch-{epoch}"
                saved = dict(
                    cfg,
                    max_len=MAX_LEN,
                    head_max_len=HEAD_MAX_LEN,
                    fine_tuned=True,
                    temperature=[1.0, 1.0, 1.0],
                    option_layout="sequential",
                )
                for name in (
                    "temperature_by_options",
                    "binning_map",
                    "lang_temperatures",
                ):
                    saved.pop(name, None)
                saved["training"] = {
                    "sentiment": {
                        "config": asdict(config),
                        "dataset_version": VERSION,
                        "questions_sha256": question_hash,
                        "epoch": epoch,
                    }
                }
                save_checkpoint(model, tokenizer, saved, str(destination))
                (destination / "questions.json").write_bytes(question_bytes)
                report.update(
                    best_key=list(key),
                    best_epoch=epoch,
                    best_checkpoint=str(destination.resolve()),
                )
                entry["checkpoint"] = str(destination.resolve())
            report["epochs"].append(entry)
            report["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(output / "report.json", report)
            print(
                f"Epoch {epoch}: validation macro-F1={metrics['macro_f1']:.4f}; "
                f"accuracy={metrics['accuracy']:.1%}; best epoch={report['best_epoch']}",
                flush=True,
            )
            print("  Predicted counts:", metrics["predicted_counts"], flush=True)

        print(
            f"Training {len(items['train'])} tweets for {args.epochs} epochs; "
            f"encoder frozen={args.freeze_encoder}",
            flush=True,
        )
        train_model(
            model,
            tokenizer,
            items["train"],
            config,
            device,
            MAX_LEN,
            HEAD_MAX_LEN,
            on_epoch_end=on_epoch_end,
        )
        encoder_after = parameter_hash(model.encoder.named_parameters())
        assert (encoder_before == encoder_after) == args.freeze_encoder
        report["encoder_parameters_changed"] = encoder_before != encoder_after
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        best_model, best_tokenizer, _ = load_checkpoint(report["best_checkpoint"])
        best_model = best_model.to(device)
        report["saved_checkpoint_validation"] = measure(best_model, best_tokenizer)
        report["status"] = "complete"
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output / "report.json", report)
        print("\nTRAINING COMPLETE")
        print("Best epoch:", report["best_epoch"])
        print("Best checkpoint:", report["best_checkpoint"])
        print(
            "Saved-checkpoint validation macro-F1:",
            f"{report['saved_checkpoint_validation']['macro_f1']:.4f}",
        )
        print("Details:", output / "report.json")
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
