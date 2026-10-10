"""Reddit source tests: fixture-based, no network. urlopen is stubbed."""
import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.models import SourceUnparsableError, SourceUnreachableError
from trends.radar.sources import reddit
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


def _urlopen_fixture(body: bytes):
    def _fake(request, timeout=None):
        return FakeResp(body)

    return _fake


class TestRedditSource(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_parses_listing_into_trend_items(self):
        with mock.patch(
            "urllib.request.urlopen", _urlopen_fixture(_fixture("radar_reddit_sample.json"))
        ):
            items = reddit.fetch_reddit(subreddits=("MapPorn",), limit=2)
        # third child has an empty title and is skipped, not fabricated
        self.assertEqual(len(items), 2)

        first = items[0]
        self.assertEqual(first["source"], "reddit")
        self.assertEqual(first["id"], "reddit:abc123")
        self.assertEqual(first["title"], "Wind particles flowing over North America [OC]")
        self.assertTrue(first["url"].startswith("https://www.reddit.com/r/MapPorn/"))
        self.assertEqual(first["author"], "mapmaker42")
        self.assertEqual(first["published_at"], "2025-10-01T00:00:00Z")
        self.assertEqual(first["metrics"]["score"], 12480)
        self.assertEqual(first["metrics"]["comments"], 312)
        self.assertEqual(
            first["thumbnail_url"], "https://b.thumbs.redditmedia.com/abc.jpg"
        )
        self.assertTrue(first["fetched_at"].endswith("Z"))
        self.assertIn("particle-flow", first["tags"])

        second = items[1]
        # "self" thumbnails are not URLs -> None, never downloaded
        self.assertIsNone(second["thumbnail_url"])
        self.assertIn("globe-spin", second["tags"])

    def test_trend_item_schema_keys(self):
        with mock.patch(
            "urllib.request.urlopen", _urlopen_fixture(_fixture("radar_reddit_sample.json"))
        ):
            items = reddit.fetch_reddit(subreddits=("MapPorn",))
        allowed_metrics = {"score", "upvotes", "views", "likes", "comments"}
        for item in items:
            self.assertEqual(
                set(item.keys()),
                {"id", "source", "title", "url", "author", "published_at",
                 "metrics", "tags", "thumbnail_url", "fetched_at"},
            )
            self.assertLessEqual(set(item["metrics"].keys()), allowed_metrics)
            self.assertIsInstance(item["tags"], list)

    def test_429_raises_unreachable_not_fake_data(self):
        def _boom(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {}, None
            )

        with mock.patch("urllib.request.urlopen", _boom):
            with self.assertRaises(SourceUnreachableError) as ctx:
                reddit.fetch_reddit(subreddits=("MapPorn",))
        self.assertIn("429", str(ctx.exception))

    def test_403_raises_unreachable(self):
        def _boom(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

        with mock.patch("urllib.request.urlopen", _boom):
            with self.assertRaises(SourceUnreachableError):
                reddit.fetch_reddit(subreddits=("MapPorn",))

    def test_non_json_body_raises_unparsable(self):
        with mock.patch(
            "urllib.request.urlopen", _urlopen_fixture(b"<html>blocked</html>")
        ):
            with self.assertRaises(SourceUnparsableError):
                reddit.fetch_reddit(subreddits=("MapPorn",))

    def test_sleeps_two_seconds_between_subreddits(self):
        with mock.patch(
            "urllib.request.urlopen", _urlopen_fixture(_fixture("radar_reddit_sample.json"))
        ), mock.patch("trends.radar.sources.reddit.time_module.sleep") as sleep:
            reddit.fetch_reddit(subreddits=("MapPorn", "dataisbeautiful"))
        sleep.assert_called_once_with(2)
        # ... but not before the first request
        with mock.patch(
            "urllib.request.urlopen", _urlopen_fixture(_fixture("radar_reddit_sample.json"))
        ), mock.patch("trends.radar.sources.reddit.time_module.sleep") as sleep:
            reddit.fetch_reddit(subreddits=("MapPorn",))
        sleep.assert_not_called()

    def test_per_run_cache_avoids_repoll(self):
        calls = []

        def _counting(request, timeout=None):
            calls.append(request.full_url)
            return FakeResp(_fixture("radar_reddit_sample.json"))

        with mock.patch("urllib.request.urlopen", _counting):
            reddit.fetch_reddit(subreddits=("MapPorn",))
            reddit.fetch_reddit(subreddits=("MapPorn",))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
