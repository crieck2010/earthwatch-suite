"""CLI contract tests: stdout modes and exit codes. Fetch is monkeypatched."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends import cli
from trends.models import SongRank, SourceUnreachableError

FETCHED_AT = "2026-09-29T03:00:00Z"


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def test_digest_success_exit_zero(self):
        songs = [
            SongRank(1, "Midnight Piano", "Elena Voss", 2500000, "up",
                     "tiktok-creative-center", FETCHED_AT)
        ]
        with mock.patch.object(cli, "fetch_trending_songs", return_value=songs):
            code, out, _ = run_cli(["digest", "--markdown"])
        self.assertEqual(code, 0)
        self.assertIn("# Weekly Trending Sounds", out)
        self.assertIn("Midnight Piano", out)

    def test_digest_unavailable_exit_two_with_notice_on_stdout(self):
        with mock.patch.object(
            cli, "fetch_trending_songs",
            side_effect=SourceUnreachableError("no endpoint responded"),
        ):
            code, out, _ = run_cli(["digest", "--markdown"])
        self.assertEqual(code, 2)
        self.assertIn("Trending data unavailable this week", out)

    def test_fetch_json_success(self):
        songs = [SongRank(1, "Midnight Piano", "Elena Voss", 7, "up",
                          "tiktok-creative-center", FETCHED_AT)]
        with mock.patch.object(cli, "fetch_trending_songs", return_value=songs):
            code, out, _ = run_cli(["fetch", "--json"])
        self.assertEqual(code, 0)
        import json
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["songs"][0]["title"], "Midnight Piano")

    def test_fetch_json_failure_reports_error(self):
        with mock.patch.object(
            cli, "fetch_trending_songs",
            side_effect=SourceUnreachableError("down"),
        ):
            code, out, _ = run_cli(["fetch", "--json"])
        self.assertEqual(code, 2)
        import json
        payload = json.loads(out)
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
