"""Fit a separate Laya temperature file on the frozen calibration cohort only."""

import argparse
import gc
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from laya.calibrate import calibration_payload, fit_temperature_map
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


def load_inputs(selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["dataset_version"] == VERSION
    assert selection["max_len"] == MAX_LEN and selection["head_max_len"] == HEAD_MAX_LEN
    assert set(selection["models"]) == set(MODELS)
    questions_path = Path("questions.json")
    assert file_hash(questions_path) == selection["questions_sha256"]
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
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
    # Read calibration text/labels only. Do not open train, validation, or test files.
    data_path = root / "calibration.jsonl"
    assert file_hash(data_path) == manifest["files"]["calibration"]["sha256"]
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == manifest["files"]["calibration"]["rows"] == 856
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["split"] == "calibration" for row in rows)
    assert all(
        row["target"]["type"] == "choice" and row["target"]["label"] in LABELS
        for row in rows
    )
    return selection, questions, rows, data_path


def verify_checkpoint(selected):
    checkpoint = Path(selected["checkpoint"])
    assert file_hash(checkpoint / "model.safetensors") == selected["weights_sha256"], (
        "Checkpoint weights changed"
    )
    assert (
        file_hash(checkpoint / "rl_agent_config.json") == selected["config_sha256"]
    ), "Checkpoint config changed"
    return checkpoint


def fit_and_measure(records, expected_labels):
    assert len(records) == len(expected_labels) > 0
    assert all(qtype == QTYPES["choice"] and k == 3 for qtype, _, _, k in records)
    logits = np.asarray([record[1] for record in records], dtype=np.float64)
    targets = np.asarray([record[2] for record in records], dtype=np.float64)
    assert logits.shape == targets.shape == (len(records), 3)
    assert np.isfinite(logits).all() and np.isfinite(targets).all()
    gold = np.asarray([LABELS.index(label) for label in expected_labels])
    assert np.array_equal(targets, np.eye(3)[gold]), "Target order/mapping changed"

    # Outside inference_mode: the fitter differentiates its CPU temperature scalar,
    # never the model's weights. Fit all calibration examples, without another split.
    fitted = fit_temperature_map(records, compute_ece=False)
    bucket = temp_bucket(QTYPES["choice"], 3)
    effective_temperature = float(
        fitted["temperature_by_options"].get(
            bucket, fitted["temperature"][QTYPES["choice"]]
        )
    )
    assert (
        np.isfinite(effective_temperature)
        and TEMP_MIN <= effective_temperature <= TEMP_MAX
    )
    assert sum(fitted["n_by_bucket"].values()) == len(records)
    before = evaluate_records(records, temperature=1.0)
    after = evaluate_records(
        records, fitted["temperature"], fitted["temperature_by_options"]
    )
    assert np.array_equal(
        logits.argmax(axis=1), (logits / effective_temperature).argmax(axis=1)
    )
    assert before["accuracy"] == after["accuracy"], (
        "Temperature must preserve class predictions"
    )
    return fitted, effective_temperature, before, after, logits, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--selection", type=Path, default=Path("selected-models.json"))
    parser.add_argument(
        "--output", type=Path, help="Default: outputs/calibration/<model>; must be new"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    assert args.batch_size > 0 and torch.cuda.is_available(), (
        "A working CUDA GPU is required"
    )
    assert version("laya") == "0.3.29", (
        "Recheck the fitting protocol before changing Laya versions"
    )
    selection, questions, rows, data_path = load_inputs(args.selection)
    selected = selection["models"][args.model]
    checkpoint = verify_checkpoint(selected)
    output = args.output or Path("outputs/calibration") / args.model
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "model": args.model,
        "split": "calibration",
        "examples": len(rows),
        "dataset_version": VERSION,
        "calibration_data_sha256": file_hash(data_path),
        "questions_sha256": selection["questions_sha256"],
        "selection_sha256": file_hash(args.selection),
        "checkpoint": str(checkpoint.resolve()),
        "weights_sha256": selected["weights_sha256"],
        "config_sha256": selected["config_sha256"],
        "selected_epoch": selected["best_epoch"],
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "label_order": LABELS,
        "option_layout": "sequential",
        "raw_temperature": 1.0,
        "inference_precision": "FP32",
        "method": "Laya fit_temperature_map, compute_ece=False; temperature scaling only",
        "temperature_bounds": [TEMP_MIN, TEMP_MAX],
        "batch_size": args.batch_size,
        "versions": {
            name: version(name) for name in ("laya", "torch", "transformers", "numpy")
        },
        "script_sha256": file_hash(__file__),
        "diagnostic_scope": "Before/after metrics are on the fitting cohort, not an independent evaluation.",
    }
    started = time.perf_counter()
    atomic_json(output / "report.json", report)
    try:
        print("Loading selected checkpoint:", checkpoint, flush=True)
        model, tokenizer, cfg = load_checkpoint(str(checkpoint))
        formatted = [
            {
                "state": row["inputs"]["text_mk"],
                "questions": questions,
                "expected": {"sentiment": row["target"]["label"]},
            }
            for row in rows
        ]
        items, skipped = items_from_rows(tokenizer, formatted, MAX_LEN, HEAD_MAX_LEN)
        assert not skipped and len(items) == len(rows), f"Rejected examples: {skipped}"
        device = torch.device("cuda:0")
        model = model.float().to(device).eval()
        # No gradients or model optimizer are created during prediction.
        model.requires_grad_(False)
        records = []
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
                print(
                    f"Collected {len(records)}/{len(rows)} calibration predictions",
                    flush=True,
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()

        fitted, temperature, before, after, logits, targets = fit_and_measure(
            records, [row["target"]["label"] for row in rows]
        )
        # Store the standard Laya sidecar, without rewriting the model or its config.
        payload = calibration_payload(
            fitted["temperature"],
            fitted["temperature_by_options"],
            model_id_or_path=str(checkpoint.resolve()),
            config=cfg,
        )
        atomic_json(output / "calibration.json", payload)
        np.savez_compressed(
            output / "logits.npz",
            ids=np.asarray([row["id"] for row in rows]),
            label_order=np.asarray(LABELS),
            logits=logits,
            targets=targets,
        )
        verify_checkpoint(selected)
        assert file_hash(args.selection) == report["selection_sha256"], (
            "Selection changed during fitting"
        )
        report.update(
            status="complete",
            effective_choice_temperature=temperature,
            temperature=fitted["temperature"],
            temperature_by_options=fitted["temperature_by_options"],
            n_by_bucket=fitted["n_by_bucket"],
            temperature_at_bound=temperature in (TEMP_MIN, TEMP_MAX),
            calibration_set_fit_diagnostics={"before": before, "after": after},
            checkpoint_weights_and_config_unchanged=True,
            calibration_sha256=file_hash(output / "calibration.json"),
            logits_sha256=file_hash(output / "logits.npz"),
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_json(output / "report.json", report)
        print("\nCALIBRATION COMPLETE")
        print("Model:", args.model)
        print(f"Temperature: {temperature:.4f} (Laya bounds {TEMP_MIN}–{TEMP_MAX})")
        for title, key in (
            ("Cross-entropy", "loss"),
            ("ECE", "ece"),
            ("Brier score", "brier"),
        ):
            print(f"Fitting-cohort {title}: {before[key]:.4f} -> {after[key]:.4f}")
        print(f"Fitting-cohort accuracy unchanged: {after['accuracy']:.1%}")
        if report["temperature_at_bound"]:
            print(
                "Note: temperature reached a Laya bound; record this limitation for the thesis."
            )
        print("Saved:", output / "calibration.json")
        print(
            "Checkpoint weights/config unchanged. No training or test evaluation performed."
        )
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
