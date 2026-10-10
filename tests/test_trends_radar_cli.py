"""Radar CLI contract tests: stdout modes and exit codes. Sources are stubbed."""
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends import cli
from trends.models import SourceUnreachableError
from trends.radar.formats import tag_item


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def trend_item(title, source="reddit", metrics=None):
    item = {
        "id": f"cli:{title}",
        "source": source,
        "title": title,
        "url": f"https://example.com/{title}",
        "author": None,
        "published_at": None,
        "metrics": metrics or {},
        "tags": [],
        "thumbnail_url": None,
        "fetched_at": "2026-10-01T00:00:00Z",
    }
    item["tags"] = tag_item(item)
    return item


ITEMS = [
    trend_item("Wind particles over the Atlantic", metrics={"score": 9000}),
    trend_item("More wind particles", metrics={"score": 1000}),
    trend_item("Rotating globe of flights", source="youtube", metrics={}),
    trend_item("Another rotating globe", source="youtube", metrics={}),
]


class TestRadarCli(unittest.TestCase):
    def _all_ok(self):
        return (
            mock.patch.object(cli, "fetch_reddit", return_value=ITEMS[:2]),
            mock.patch.object(cli, "fetch_tiktok_videos", return_value=[]),
            mock.patch.object(
                cli, "fetch_youtube_videos", return_value=ITEMS[2:]
            ),
        )

    def test_radar_markdown_all_ok_exit_zero(self):
        patches = self._all_ok()
        for p in patches:
            p.start()
        try:
            code, out, _ = run_cli(["radar", "--markdown"])
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(code, 0)
        self.assertIn("# 🛰️ Map-Viz Trend Radar", out)
        self.assertIn("particle-flow", out)
        self.assertIn("Inspiration, not replication", out)
        self.assertIn("| reddit | ✅ ok |", out)

    def test_radar_partial_failure_exit_two_still_renders(self):
        with mock.patch.object(
            cli, "fetch_reddit",
            side_effect=SourceUnreachableError("reddit: HTTP 403"),
        ), mock.patch.object(
            cli, "fetch_tiktok_videos",
            side_effect=SourceUnreachableError("tiktok: no video entities"),
        ), mock.patch.object(
            cli, "fetch_youtube_videos", return_value=ITEMS[2:]
        ):
            code, out, _ = run_cli(["radar", "--markdown"])
        self.assertEqual(code, 2)
        # digest renders from what responded, with honest unavailable notes
        self.assertIn("# 🛰️ Map-Viz Trend Radar", out)
        self.assertIn("❌ unavailable", out)
        self.assertIn("reddit: HTTP 403", out)
        self.assertIn("globe-spin", out)

    def test_radar_all_failed_exit_two_empty_signals(self):
        with mock.patch.object(
            cli, "fetch_reddit", side_effect=SourceUnreachableError("down")
        ), mock.patch.object(
            cli, "fetch_tiktok_videos", side_effect=SourceUnreachableError("down")
        ), mock.patch.object(
            cli, "fetch_youtube_videos", side_effect=SourceUnreachableError("down")
        ):
            code, out, _ = run_cli(["radar"])
        self.assertEqual(code, 2)
        self.assertIn("No verified trend items this pull", out)

    def test_radar_json_emits_signals_coverage_priors(self):
        patches = self._all_ok()
        for p in patches:
            p.start()
        try:
            code, out, _ = run_cli(["radar", "--json"])
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            set(payload.keys()), {"ok", "fetched_at", "signals", "coverage", "priors"}
        )
        self.assertEqual(
            set(payload["priors"].keys()),
            {"version", "generated_at", "window_days", "priors", "notes"},
        )
        self.assertEqual(len(payload["coverage"]), 3)

    def test_radar_json_partial_failure_ok_false_exit_two(self):
        with mock.patch.object(
            cli, "fetch_reddit", side_effect=SourceUnreachableError("down")
        ), mock.patch.object(
            cli, "fetch_tiktok_videos", return_value=[]
        ), mock.patch.object(
            cli, "fetch_youtube_videos", return_value=[]
        ):
            code, out, _ = run_cli(["radar", "--json"])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])

    def test_radar_unexpected_error_exit_one(self):
        with mock.patch.object(
            cli, "fetch_reddit", side_effect=RuntimeError("boom")
        ):
            code, out, err = run_cli(["radar", "--markdown"])
        self.assertEqual(code, 1)
        self.assertIn("unexpected error", err)

    def test_radar_flags_passed_through(self):
        seen = {}

        def _fake_reddit(subreddits, limit):
            seen["subreddits"] = subreddits
            seen["limit"] = limit
            return []

        def _fake_yt(query, max_results):
            seen["query"] = query
            seen["max_results"] = max_results
            return []

        with mock.patch.object(cli, "fetch_reddit", side_effect=_fake_reddit), \
            mock.patch.object(cli, "fetch_tiktok_videos", return_value=[]), \
            mock.patch.object(cli, "fetch_youtube_videos", side_effect=_fake_yt):
            code, _, _ = run_cli(
                ["radar", "--markdown", "--subreddits", "MapPorn",
                 "--query", "globe", "--limit", "10"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(seen["subreddits"], ("MapPorn",))
        self.assertEqual(seen["limit"], 10)
        self.assertEqual(seen["query"], "globe")
        self.assertEqual(seen["max_results"], 10)


if __name__ == "__main__":
    unittest.main()
