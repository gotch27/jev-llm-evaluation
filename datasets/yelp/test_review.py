"""Offline HTML snapshot checks using synthetic reviews, without dataset mutations."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import build_review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "review").mkdir()
        self.template = (build_review.ROOT / "review/template.html").read_text(encoding="utf-8")
        (self.root / "review/template.html").write_text(self.template, encoding="utf-8")
        self.source = self.root / "translated/synthetic"
        self.source.mkdir(parents=True)
        self.summary = {
            "profile": "synthetic-profile",
            "source_manifest": {"source_sha256": "synthetic-source"},
            "updated_at": "2026-10-08T00:00:00+00:00",
            "translated_rows": 1,
            "missing_rows": 0,
            "model": "synthetic/model",
        }
        (self.source / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        (self.root / "translated/latest.json").write_text(
            json.dumps({"directory": str(self.source)}), encoding="utf-8"
        )

    def write_reviews(self, rows):
        (self.source / "review.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_review.build(self.root)

    def payload(self, output):
        html = output.read_text(encoding="utf-8")
        marker = '<script id="review-data" type="application/json">'
        embedded = html.split(marker, 1)[1].split("</script>", 1)[0]
        return html, embedded, json.loads(embedded)

    def test_untrusted_review_text_cannot_close_data_script(self):
        row = {
            "id": "yelp-train-1",
            "split": "train",
            "stars": 3,
            "text_en": '</script><script>alert("source")</script> & < >\u2028\u2029',
            "text_mk": 'Македонски </script><img src=x onerror="alert(1)">',
            "review_flags": ["check_numbers"],
            "duplicate_group": "source-hash",
        }
        self.write_reviews([row])
        before = (self.source / "review.jsonl").read_bytes()
        output = self.build()
        html, embedded, payload = self.payload(output)
        self.assertEqual(payload["records"], [row])
        self.assertEqual(payload["flag_counts"], {"check_numbers": 1})
        self.assertEqual(payload["source_sha256"], "synthetic-source")
        self.assertNotIn("<", embedded)
        self.assertNotIn("&", embedded)
        self.assertNotIn("\u2028", embedded)
        self.assertNotIn("\u2029", embedded)
        self.assertEqual(html.count("</script>"), self.template.count("</script>"))
        self.assertNotIn("innerHTML", html)
        self.assertEqual((self.source / "review.jsonl").read_bytes(), before)
        self.assertEqual(self.build().read_bytes(), output.read_bytes())

    def test_empty_flagged_export_is_a_valid_snapshot(self):
        self.write_reviews([])
        _, _, payload = self.payload(self.build())
        self.assertEqual(payload["records"], [])
        self.assertEqual(payload["flag_counts"], {})
        self.assertEqual(len(payload["fingerprint"]), 64)

    def test_duplicate_ids_fail_before_overwriting_html(self):
        self.write_reviews([])
        output = self.build()
        before = output.read_bytes()
        row = {
            "id": "yelp-train-1", "split": "train", "stars": 3,
            "text_en": "source", "text_mk": "Превод", "review_flags": ["unchanged"],
            "duplicate_group": "source-hash",
        }
        self.write_reviews([row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate source IDs"):
            self.build()
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
