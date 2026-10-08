"""Offline integration checks; no real Gateway requests or credentials."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import duckdb
import pipeline as p

MODEL = {"id": "test/model", "supported_parameters": ["temperature", "response_format"],
         "pricing": {"input": "0.0000001", "output": "0.0000005"}}


def fixture_sources(root):
    """Real Parquet files with deliberate train/test duplicates and contradictory labels."""
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    c = duckdb.connect()
    c.execute("CREATE TABLE reviews(label INTEGER,text VARCHAR)")
    for split in ("train", "test"):
        c.execute("DELETE FROM reviews")
        for label in range(5):
            for index in range(20 if split == "train" else 4):
                text = f"This {split} customer review {label}-{index} discusses food, service and prices. "
                if index == 0:
                    text += "\\nThe café was comfortable; preserve these Unicode characters."
                c.execute("INSERT INTO reviews VALUES (?,?)", [label, text])
        shared = "Shared original test review should never enter training. " * 2
        # Boundary tabs/newlines must not hide an original test duplicate from the guard.
        if split == "test":
            shared = "\n\t" + shared + "\t\n"
        c.execute("INSERT INTO reviews VALUES (0,?)", [shared])
        if split == "train":
            c.execute("INSERT INTO reviews VALUES (1,?)", ["Contradictory duplicate must be excluded from the dataset. " * 2])
            c.execute("INSERT INTO reviews VALUES (2,?)", ["CONTRADICTORY  DUPLICATE must be excluded from the dataset. " * 2])
            # Same-rating case/whitespace duplicate; only one representative may be chosen.
            c.execute("INSERT INTO reviews SELECT label,upper(text) FROM reviews WHERE label=4 LIMIT 1")
        path = root / f"{split}.parquet"
        c.execute("COPY reviews TO ? (FORMAT PARQUET)", [str(path)])
        sources[split] = {"path": str(path), "sha256": p.file_hash(path)}
    c.close()
    return sources


def make_sample(root):
    rows, audit = p.build_sample(root, fixture_sources(root / "fixtures"), 50, 42, 40, 3000)
    manifest = {"revision": p.REVISION, "source_sha256": p.engine.digest(p.engine.encoded(rows)),
                "rows": len(rows), "labels": [{"label": i, "stars": i + 1} for i in range(5)]}
    return rows, audit, manifest


def response(payload, missing=False):
    inputs = json.loads(payload["messages"][1]["content"])["items"]
    translations = [{"id": item["id"], "text_mk": "Ова е превод на рецензијата за храната, услугата и цените."}
                    for item in inputs]
    if missing:
        translations.pop()
    return json.dumps({"id": "offline-response", "model": "test/provider",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"translations": translations})}}]}).encode()


class PipelineTests(unittest.TestCase):
    def baseline_with_flags(self, root):
        rows, _, manifest = make_sample(root)
        store = p.engine.Store(root / "cache/translations.sqlite3")
        profile = p.engine.profile_for(MODEL)
        store.save_batch(profile, MODEL, rows, ["Оригинален македонски превод." for _ in rows], {})
        with store.db:
            store.db.execute("UPDATE translations SET flags='[]' WHERE profile=?", [profile])
            for row, flags in zip(rows, (["check_numbers"], ["no_cyrillic"], ["check_numbers", "check_possible_omission"])):
                store.db.execute("UPDATE translations SET flags=? WHERE profile=? AND source_hash=?",
                                 [json.dumps(flags), profile, row["duplicate_group"]])
        _, original = p.export(root, rows, manifest, MODEL, store)
        store.close()
        return rows, manifest, profile, original

    def test_selective_retranslation_preserves_originals_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, manifest, base_profile, original = self.baseline_with_flags(root)
            before = (original / "all.jsonl").read_bytes()
            calls = []
            def fake_request(url, payload, key, timeout):
                calls.append(payload)
                return response(payload)
            with patch.object(p, "prepare", return_value=(rows, root, manifest)), \
                    patch.object(p, "get_key", return_value="fake"), \
                    patch.object(p, "get_model", return_value=MODEL), \
                    patch.object(p.engine, "request", side_effect=fake_request):
                command = ["--root", str(root), "retranslate", "--total", "50", "--exclude-flag", "check_numbers", "--workers", "1"]
                self.assertEqual(p.main(command + ["--dry-run"]), 0)
                self.assertEqual(len(calls), 0)
                self.assertEqual(p.main(command), 0)
                first_count = len(calls)
                self.assertEqual(first_count, 1)
                self.assertEqual(p.main(command), 0)
                self.assertEqual(len(calls), first_count)
                self.assertEqual(p.main(["--root", str(root), "status", "--total", "50"]), 0)
            combined = Path(json.loads((root / "translated/latest.json").read_text())["directory"])
            self.assertNotEqual(combined, original)
            self.assertEqual((original / "all.jsonl").read_bytes(), before)
            result = {r["id"]: r for r in p.engine.read_jsonl(combined / "all.jsonl")}
            self.assertEqual(len(result), 50)
            self.assertEqual(result[rows[0]["id"]]["text_mk"], "Оригинален македонски превод.")
            self.assertEqual(result[rows[0]["id"]]["translation_profile"], base_profile)
            for row in rows[1:3]:
                self.assertNotEqual(result[row["id"]]["translation_profile"], base_profile)
            self.assertEqual(p.engine.PROMPT, p.ORIGINAL_PROMPT)
            summary = json.loads((combined / "summary.json").read_text())
            self.assertTrue(summary["retranslation"]["complete"])
            self.assertEqual(summary["retranslation"]["completed_rows"], 2)
            self.assertEqual(len(p.engine.read_jsonl(combined / "retranslation-pending.jsonl")), 0)
            self.assertIn(p.RETRY_INSTRUCTIONS, calls[0]["messages"][0]["content"])

    def test_failed_retranslation_keeps_baseline_dataset_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, manifest, base_profile, original = self.baseline_with_flags(root)
            before = (original / "all.jsonl").read_bytes()
            with patch.object(p, "prepare", return_value=(rows, root, manifest)), \
                    patch.object(p, "get_key", return_value="fake"), \
                    patch.object(p, "get_model", return_value=MODEL), \
                    patch.object(p.engine, "request", side_effect=p.engine.FatalError("synthetic auth failure")):
                self.assertEqual(p.main(["--root", str(root), "retranslate", "--total", "50", "--workers", "1"]), 1)
            combined = Path(json.loads((root / "translated/latest.json").read_text())["directory"])
            summary = json.loads((combined / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["retranslation"]["complete"])
            self.assertEqual(summary["retranslation"]["missing_rows"], 2)
            self.assertEqual((original / "all.jsonl").read_bytes(), before)
            self.assertEqual(len(p.engine.read_jsonl(combined / "all.jsonl")), 50)

    def test_balanced_splits_duplicate_exclusions_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = fixture_sources(root / "fixtures")
            rows, audit = p.build_sample(root, sources, 50, 42, 40, 3000)
            p.validate_sample(rows, 50, p.split_sizes(50))
            counts = Counter((r["split"], r["stars"]) for r in rows)
            for star in range(1, 6):
                self.assertEqual(counts["train", star], 7)
                for split in ("validation", "calibration", "test"):
                    self.assertEqual(counts[split, star], 1)
            self.assertFalse(any("contradictory" in r["text_en"].lower() for r in rows))
            self.assertFalse(any(r["source_split"] == "train" and "Shared original test" in r["text_en"] for r in rows))
            self.assertEqual(audit["conflicting_label_groups"], 1)
            self.assertEqual(audit["original_cross_split_groups"], 1)
            self.assertEqual(rows, p.build_sample(root, sources, 50, 42, 40, 3000)[0])

    def test_unicode_and_paragraph_decoding_preserve_complete_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = fixture_sources(root / "fixtures")
            rows, _ = p.build_sample(root, sources, 50, 42, 40, 3000)
            originals = {}
            c = duckdb.connect()
            for split in ("train", "test"):
                originals[split] = dict(c.execute("SELECT file_row_number,text FROM read_parquet(?,file_row_number=true)",
                                                 [sources[split]["path"]]).fetchall())
            c.close()
            for row in rows:
                self.assertEqual(row["text_en"], originals[row["source_split"]][row["source_index"]].replace("\\n", "\n").strip())
            self.assertTrue(any("café" in row["text_en"] and "\n" in row["text_en"] for row in rows))

    def test_prepare_cache_and_checksum_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = fixture_sources(root / "fixtures")
            with patch.object(p, "download", return_value=sources):
                rows, dest, manifest = p.prepare(root, total=50)
            with patch.object(p, "download", side_effect=AssertionError("unexpected download")):
                self.assertEqual(p.prepare(root, total=50)[0], rows)
            (dest / "all.jsonl").write_text("damaged")
            with self.assertRaisesRegex(p.engine.FatalError, "checksum"):
                p.prepare(root, total=50)

    def test_insufficient_data_and_invalid_size_fail_explicitly(self):
        with self.assertRaises(p.engine.FatalError):
            p.split_sizes(10001)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = fixture_sources(root / "fixtures")
            with self.assertRaisesRegex(p.engine.FatalError, "Only"):
                p.build_sample(root, sources, 1000, 42, 40, 3000)

    def test_pilot_is_balanced_and_uses_training_only(self):
        with tempfile.TemporaryDirectory() as folder:
            rows, _, _ = make_sample(Path(folder))
            pilot = p.select_pilot(rows, 25)
            self.assertEqual(len(pilot), 25)
            self.assertEqual(Counter(r["stars"] for r in pilot), {i: 5 for i in range(1, 6)})
            self.assertEqual({r["split"] for r in pilot}, {"train"})
            self.assertEqual(pilot, p.select_pilot(rows, 25))
            with self.assertRaises(p.engine.FatalError):
                p.select_pilot(rows, 26)

    def test_cli_retry_resume_and_export_completeness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, _, manifest = make_sample(root)
            calls = []

            def fake_request(url, payload, key, timeout):
                self.assertEqual(url, p.engine.GATEWAY + "/chat/completions")
                self.assertEqual(key, "fake-key")
                calls.append(payload)
                # First response is billed but invalid: retry and account for both.
                return response(payload, missing=len(calls) == 1)

            with patch.object(p, "prepare", return_value=(rows, root, manifest)), \
                    patch.object(p, "get_key", return_value="fake-key"), \
                    patch.object(p, "get_model", return_value=MODEL), \
                    patch.object(p.engine, "request", side_effect=fake_request), \
                    patch.object(p.engine.random, "random", return_value=0):
                command = ["--root", str(root), "translate", "--total", "50", "--limit", "25", "--workers", "1"]
                self.assertEqual(p.main(command), 0)
                first_count = len(calls)
                self.assertEqual(p.main(command), 0)
                self.assertEqual(first_count, len(calls))
                full_command = ["--root", str(root), "translate", "--total", "50", "--workers", "1"]
                self.assertEqual(p.main(full_command), 0)
                full_count = len(calls)
                self.assertEqual(p.main(full_command), 0)
                self.assertEqual(full_count, len(calls))
            destination = Path(json.loads((root / "translated" / "latest.json").read_text())["directory"])
            summary = json.loads((destination / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["translated_rows"], 50)
            self.assertEqual(summary["http_attempts"], len(calls))
            self.assertEqual(summary["reported_prompt_tokens"], len(calls) * 10)
            self.assertFalse((destination / "all.partial.jsonl").exists())
            self.assertEqual(len(p.engine.read_jsonl(destination / "all.jsonl")), 50)
            self.assertEqual(len(p.engine.read_jsonl(destination / "pending.jsonl")), 0)
            for payload in calls:
                self.assertEqual(payload["messages"][0]["content"], p.engine.PROMPT)
                for item in json.loads(payload["messages"][1]["content"])["items"]:
                    self.assertEqual(set(item), {"id", "text"})

    def test_fatal_error_saves_partial_export_and_stops_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, _, manifest = make_sample(root)
            with patch.object(p, "prepare", return_value=(rows, root, manifest)), \
                    patch.object(p, "get_key", return_value="fake-key"), \
                    patch.object(p, "get_model", return_value=MODEL), \
                    patch.object(p.engine, "request", side_effect=p.engine.FatalError("synthetic auth failure")) as request:
                code = p.main(["--root", str(root), "translate", "--total", "50", "--workers", "1"])
            self.assertEqual(code, 1)
            self.assertEqual(request.call_count, 1)
            dest = Path(json.loads((root / "translated" / "latest.json").read_text())["directory"])
            summary = json.loads((dest / "summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["missing_rows"], 50)
            self.assertTrue((dest / "all.partial.jsonl").exists())
            self.assertFalse((dest / "all.jsonl").exists())

    def test_download_retries_transient_failure_and_rejects_corruption(self):
        import io
        class Response(io.BytesIO):
            pass
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.bin"
            raw = b"synthetic parquet content"
            with patch.object(p.urllib.request, "urlopen", side_effect=[TimeoutError(), Response(raw)]), \
                    patch.object(p.time, "sleep"):
                p.download_file("https://offline.invalid/source", path, p.engine.digest(raw), len(raw), attempts=2)
            self.assertEqual(path.read_bytes(), raw)
            with patch.object(p.urllib.request, "urlopen", side_effect=AssertionError("unexpected request")):
                p.download_file("https://offline.invalid/source", path, p.engine.digest(raw), len(raw))
            path.unlink()
            with patch.object(p.urllib.request, "urlopen", return_value=Response(b"corrupt")):
                with self.assertRaises(p.engine.RetryableError):
                    p.download_file("https://offline.invalid/source", path, p.engine.digest(raw), len(raw), attempts=1)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_yelp_cache_profile_differs_from_snips_and_model_changes(self):
        original_prompt = p.engine.PROMPT
        profile = p.engine.profile_for(MODEL)
        try:
            p.engine.PROMPT = "SNIPS intent translation"
            self.assertNotEqual(profile, p.engine.profile_for(MODEL))
        finally:
            p.engine.PROMPT = original_prompt
        self.assertNotEqual(profile, p.engine.profile_for({**MODEL, "id": "test/other"}))


if __name__ == "__main__":
    unittest.main()
