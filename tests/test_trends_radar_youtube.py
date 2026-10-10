"""YouTube source tests: fixture-based, no network. urlopen is stubbed."""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.models import (
    CredentialsMissingError,
    SourceUnreachableError,
    SourceUnparsableError,
)
from trends.radar.sources import youtube
from trends.radar.sources._http import clear_cache

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    with open(os.path.join(FIXTURES, name), "rb") as fh:
        return fh.read()


class FakeResp:
    def __init__(self, body: bytes):
        self._body = body
        self.status = 200

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _no_key_env():
    env = dict(os.environ)
    env.pop("YOUTUBE_API_KEY", None)
    return mock.patch.dict(os.environ, env, clear=True)


class TestYoutubeSource(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_missing_key_raises_before_any_request(self):
        requested = []

        def _boom(request, timeout=None):
            requested.append(request.full_url)
            return FakeResp(b"{}")

        with _no_key_env(), mock.patch("urllib.request.urlopen", _boom):
            with self.assertRaises(CredentialsMissingError) as ctx:
                youtube.fetch_youtube_videos("map data visualization")
        message = str(ctx.exception)
        # message must say exactly how to get a key and that nothing was fetched
        self.assertIn("YouTube Data API v3", message)
        self.assertIn("console.cloud.google.com", message)
        self.assertIn("YOUTUBE_API_KEY", message)
        self.assertIn("Nothing was fetched", message)
        self.assertEqual(requested, [])  # no request was ever made

    def test_credentials_missing_is_a_trends_error(self):
        from trends.models import TrendsError

        self.assertTrue(issubclass(CredentialsMissingError, TrendsError))

    def test_env_key_is_accepted(self):
        with mock.patch.dict(os.environ, {"YOUTUBE_API_KEY": "ENVKEY"}), mock.patch(
            "urllib.request.urlopen",
            lambda request, timeout=None: FakeResp(
                _fixture("radar_youtube_sample.json")
            ),
        ):
            items = youtube.fetch_youtube_videos("map data visualization")
        self.assertEqual(len(items), 2)

    def test_parses_search_response(self):
        def _fake(request, timeout=None):
            self.assertIn("key=TESTKEY", request.full_url)
            self.assertIn("type=video", request.full_url)
            self.assertIn("order=viewCount", request.full_url)
            return FakeResp(_fixture("radar_youtube_sample.json"))

        with _no_key_env(), mock.patch("urllib.request.urlopen", _fake):
            items = youtube.fetch_youtube_videos(
                "map data visualization", api_key="TESTKEY"
            )
        # channel result and "Private video" are skipped, not fabricated
        self.assertEqual(len(items), 2)

        first = items[0]
        self.assertEqual(first["source"], "youtube")
        self.assertEqual(first["id"], "youtube:dQw4w9WgXcQ")
        self.assertEqual(
            first["url"], "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        self.assertEqual(first["author"], "GeoViz Studio")
        self.assertEqual(first["published_at"], "2026-09-28T14:00:00Z")
        # search.list carries no statistics: only keys the source provides
        self.assertEqual(first["metrics"], {})
        self.assertEqual(
            first["thumbnail_url"],
            "https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg",
        )
        self.assertIn("3d-extrusion", first["tags"])
        self.assertIn("timelapse", items[1]["tags"])

    def test_api_error_payload_raises_unreachable(self):
        body = json.dumps(
            {"error": {"code": 403, "message": "quotaExceeded"}}
        ).encode()
        with _no_key_env(), mock.patch(
            "urllib.request.urlopen", lambda r, timeout=None: FakeResp(body)
        ):
            with self.assertRaises(SourceUnreachableError) as ctx:
                youtube.fetch_youtube_videos("maps", api_key="TESTKEY")
        self.assertIn("quotaExceeded", str(ctx.exception))

    def test_non_json_body_raises_unparsable(self):
        with _no_key_env(), mock.patch(
            "urllib.request.urlopen",
            lambda r, timeout=None: FakeResp(b"<html>nope</html>"),
        ):
            with self.assertRaises(SourceUnparsableError):
                youtube.fetch_youtube_videos("maps", api_key="TESTKEY")


if __name__ == "__main__":
    unittest.main()
