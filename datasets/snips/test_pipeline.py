"""Offline integration tests: fake HTTP Gateway, real SQLite and export files. No paid API calls."""
import argparse
import contextlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pipeline as p


def model(name=p.DEFAULT_MODEL, temperature=False):
    return {"id": name, "supported_parameters": ["max_tokens", "response_format", "reasoning"] +
            (["temperature"] if temperature else []),
            "reasoning_options": [{"type": "effort", "values": ["none", "low"]}],
            "pricing": {"input": "0.0000001", "output": "0.0000005"}}


def source_row(index, text, split="train", intent="PlayMusic"):
    return {"id": str(index), "text_en": text, "split": split, "intent": intent,
            "label": p.INTENTS.index(intent), "duplicate_group": p.digest(text),
            "source_path": "test-fixture.json", "source_index": index}


@contextlib.contextmanager
def fake_gateway(actions=()):
    requests, remaining = [], list(actions)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            items = json.loads(payload["messages"][1]["content"])["items"]
            action = remaining.pop(0) if remaining else "valid"
            if isinstance(action, int):
                self.send_response(action)
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(b'{"error":"fake transient or credential error"}')
                return
            translations = [{"id": item["id"], "text_mk": "Пушти " + item["text"]} for item in items]
            if action == "missing":
                translations.pop()
            if action == "duplicate" and len(translations) > 1:
                translations[1]["id"] = translations[0]["id"]
            response = {"id": "fake-response", "model": payload["model"],
                        "choices": [{"finish_reason": "length" if action == "truncated" else "stop",
                                     "message": {"content": "not JSON" if action == "malformed" else
                                                 p.encoded({"translations": translations})}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
            raw = p.encoded(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def arguments(url, **overrides):
    values = {"base_url": url, "batch_size": 2, "workers": 1, "retries": 3,
              "timeout": 5, "max_tokens": 8192, "backoff": 0, "backoff_max": 0,
              "per_intent": None}
    values.update(overrides)
    return argparse.Namespace(**values)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = p.Store(self.root / "cache" / "test.sqlite3")
        self.model = model()
        self.rows = [source_row(0, "play 2 songs"), source_row(1, "play a song"),
                     source_row(2, "play 2 songs", "heldout")]

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def run_job(self, url, **options):
        return p.run_translation(arguments(url, **options), self.rows, self.model, "fake-key",
                                 self.store, threading.Event())

    @patch("pipeline.random.random", return_value=0)
    def test_transient_http_errors_retry_and_duplicate_text_translates_once(self, _):
        with fake_gateway([429, 503, "valid"]) as (url, requests):
            self.assertTrue(self.run_job(url))
            self.assertEqual(len(requests), 3)
            items = json.loads(requests[-1]["messages"][1]["content"])["items"]
            self.assertEqual(len(items), 2)  # Three original rows, only two unique texts.
            self.assertEqual(set(items[0]), {"id", "text"})  # No gold-label leakage.
            self.assertNotIn("temperature", requests[-1])  # Claude does not accept it.
            self.assertEqual(requests[-1]["reasoning"], {"effort": "none"})
            stats = self.store.stats(p.profile_for(self.model))
            self.assertEqual(stats["http_attempts"], 3)
            self.assertEqual(stats["attempts_without_usage"], 2)

    @patch("pipeline.random.random", return_value=0)
    def test_invalid_outputs_retry_without_saving_partial_batch(self, _):
        for action in ("missing", "duplicate", "malformed", "truncated"):
            with self.subTest(action=action), fake_gateway([action, "valid"]) as (url, requests):
                batch = [source_row(10, action + " A"), source_row(11, action + " B")]
                self.assertTrue(p.translate_batch(batch, 1, arguments(url), self.model, "fake",
                                                  self.store, threading.Event()))
                self.assertEqual(len(requests), 2)
        stats = self.store.stats(p.profile_for(self.model))
        self.assertEqual(stats["reported_prompt_tokens"], 800)  # Rejected outputs count too.
        self.assertAlmostEqual(stats["base_list_price_estimate_usd"], 8 * (100e-7 + 50 * 5e-7))

    def test_auth_error_fails_fast_and_stops_scheduling(self):
        with fake_gateway([401]) as (url, requests):
            with self.assertRaises(p.FatalError):
                self.run_job(url, batch_size=1)
            self.assertEqual(len(requests), 1)
            self.assertEqual(self.store.entries(p.profile_for(self.model)), {})

    @patch("pipeline.random.random", return_value=0)
    def test_failed_batch_can_resume_then_no_additional_paid_calls(self, _):
        with fake_gateway(["missing", "missing", "missing"]) as (url, requests):
            self.assertFalse(self.run_job(url))
            self.assertEqual(len(requests), 3)
            self.assertFalse(self.store.entries(p.profile_for(self.model)))
        with fake_gateway() as (url, requests):
            self.assertTrue(self.run_job(url))
            self.assertEqual(len(requests), 1)
            self.assertTrue(self.run_job(url))
            self.assertEqual(len(requests), 1)

    def test_separate_models_do_not_share_translation_cache(self):
        with fake_gateway() as (url, requests):
            self.assertTrue(self.run_job(url))
            second = model("openai/gpt-6-luna", temperature=True)
            self.assertTrue(p.run_translation(arguments(url), self.rows, second, "fake",
                                              self.store, threading.Event()))
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[-1]["temperature"], 0)
            self.assertNotEqual(p.profile_for(self.model), p.profile_for(second))

    def test_partial_export_cannot_masquerade_as_complete_then_full_export_keeps_ids(self):
        p.write_jsonl(self.root / "source" / "all.jsonl", self.rows)
        summary, dest = p.export(self.root, self.rows, self.model, self.store)
        self.assertFalse(summary["complete"])
        self.assertTrue((dest / "all.partial.jsonl").exists())
        self.assertFalse((dest / "all.jsonl").exists())
        with fake_gateway() as (url, _):
            self.assertTrue(self.run_job(url))
        summary, dest = p.export(self.root, self.rows, self.model, self.store)
        self.assertTrue(summary["complete"])
        exported = p.read_jsonl(dest / "all.jsonl")
        self.assertEqual([r["id"] for r in exported], [r["id"] for r in self.rows])
        self.assertEqual([r["label"] for r in exported], [r["label"] for r in self.rows])
        self.assertEqual(exported[0]["text_mk"], exported[2]["text_mk"])
        self.assertFalse((dest / "all.partial.jsonl").exists())

    def test_stopped_job_does_not_send_requests(self):
        event = threading.Event()
        event.set()
        with fake_gateway() as (url, requests):
            self.assertFalse(p.run_translation(arguments(url), self.rows, self.model, "fake",
                                               self.store, event))
            self.assertEqual(requests, [])

    def test_review_flags_are_heuristics_and_check_numbers(self):
        self.assertIn("check_numbers", p.review_flags("give 4 stars", "Дај 5 ѕвезди"))
        self.assertIn("no_cyrillic", p.review_flags("play The Beatles", "play The Beatles"))
        self.assertEqual(p.review_flags("give 4 stars", "Дај 4 ѕвезди"), [])
        self.assertNotIn("check_numbers", p.review_flags("rate 4.5", "Оцени со 4,5"))

    def test_retry_after_supports_seconds_and_http_dates(self):
        self.assertEqual(p.retry_after_seconds("30"), 30)
        self.assertEqual(p.retry_after_seconds("invalid"), 0)
        with patch("pipeline.time.time", return_value=0):
            self.assertEqual(p.retry_after_seconds("Thu, 01 Jan 1970 00:01:00 GMT"), 60)

    def test_comparison_writes_unicode_and_model_columns(self):
        with fake_gateway() as (url, _):
            self.assertTrue(self.run_job(url))
        p.comparison(self.root, p.unique_rows(self.rows), [self.model], self.store)
        data = (self.root / "comparison" / "review.csv").read_text(encoding="utf-8-sig")
        self.assertIn("Пушти", data)
        self.assertIn(p.DEFAULT_MODEL, data)

    def test_source_integrity_detects_modified_file(self):
        p.write_jsonl(self.root / "source" / "all.jsonl", self.rows)
        p.write_json(self.root / "source" / "manifest.json", {
            "revision": p.REVISION, "rows": len(self.rows),
            "normalized_sha256": p.digest((self.root / "source" / "all.jsonl").read_bytes())})
        self.assertEqual(p.load_source(self.root), self.rows)
        with (self.root / "source" / "all.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaises(p.FatalError):
            p.load_source(self.root)

    def test_legacy_cesu8_preserves_playlist_emoji(self):
        raw = b'{"text":"Pop Punk Perfection \xed\xa0\xbc\xed\xbd\x95"}'
        data, encoding = p.decode_source(raw)
        self.assertEqual(data["text"], "Pop Punk Perfection 🍕")
        self.assertIn("CESU-8", encoding)
        self.assertEqual(p.decode_source('{"text":"Македонски 🍕"}'.encode())[0]["text"], "Македонски 🍕")
        with self.assertRaises(UnicodeError):
            p.decode_source(b'{"text":"unpaired \xed\xa0\xbc"}')

    def test_balanced_pilot_covers_each_intent_and_is_stable(self):
        rows = [source_row(i * 10 + j, f"{intent} example {j}", intent=intent)
                for i, intent in enumerate(p.INTENTS) for j in range(5)]
        selected = p.unique_rows(rows, per_intent=2)
        self.assertEqual(len(selected), 14)
        for intent in p.INTENTS:
            self.assertEqual(sum(r["intent"] == intent for r in selected), 2)
        self.assertEqual(selected, p.unique_rows(rows, per_intent=2))

    @patch("pipeline.random.random", return_value=0)
    def test_network_failure_retries_and_records_unknown_usage(self, _):
        real_request = p.request
        attempts = []
        def unreliable(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise p.RetryableError("Network error (timeout)")
            return real_request(*args, **kwargs)
        with fake_gateway() as (url, requests), patch("pipeline.request", side_effect=unreliable):
            self.assertTrue(self.run_job(url))
            self.assertEqual(len(requests), 1)
            self.assertEqual(self.store.stats(p.profile_for(self.model))["attempts_without_usage"], 1)

    def test_cli_compare_pilot_then_full_translation_reuses_cache(self):
        p.write_jsonl(self.root / "source" / "all.jsonl", self.rows)
        p.write_json(self.root / "source" / "manifest.json", {
            "revision": p.REVISION, "rows": len(self.rows),
            "normalized_sha256": p.digest((self.root / "source" / "all.jsonl").read_bytes())})
        second = model("openai/gpt-6-luna", temperature=True)
        models = {m["id"]: m for m in (self.model, second)}
        with fake_gateway() as (url, requests), patch("pipeline.catalog", return_value=models), \
                patch("pipeline.load_key", return_value="fake"), patch("pipeline.signal.signal"):
            prefix = ["--root", str(self.root)]
            code = p.main(prefix + ["compare", "--models", self.model["id"], second["id"],
                                    "--per-intent", "1", "--base-url", url, "--workers", "1"])
            self.assertEqual(code, 0)
            self.assertEqual(len(requests), 2)
            self.assertTrue((self.root / "comparison" / "review.csv").exists())
            code = p.main(prefix + ["translate", "--base-url", url, "--workers", "1"])
            self.assertEqual(code, 0)
            self.assertEqual(len(requests), 3)  # Only the one missing Haiku translation is requested.
            dest = self.root / "translated" / self.model["id"].replace("/", "--") / p.profile_for(self.model)[:12]
            self.assertTrue(json.loads((dest / "summary.json").read_text())["complete"])
            self.assertEqual(len(p.read_jsonl(dest / "all.jsonl")), 3)


if __name__ == "__main__":
    unittest.main()
