"""TikTok video-source tests: fixture-based, no network.

Covers the expected-today path: Creative Center answers the health probe
but no endpoint yields video entities, so fetch_tiktok_videos raises
SourceUnreachableError instead of fabricating videos.
"""
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.models import SourceUnreachableError
from trends.radar.sources import tiktok
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


def _dispatcher(mapping):
    """urlopen stub: URL substring -> fixture bytes or exception."""

    def _fake(request, timeout=None):
        url = request.full_url
        for needle, body in mapping.items():
            if needle in url:
                if isinstance(body, Exception):
                    raise body
                return FakeResp(body)
        raise AssertionError(f"unexpected URL in test: {url}")

    return _fake


class TestTiktokVideoSource(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _probe_mapping(self):
        return {
            "GetTopContentsOverview": _fixture("radar_tiktok_health.json"),
            "CreativeCenterGetTopContentsList": _fixture(
                "radar_tiktok_topcontents_empty.json"
            ),
            "popular_trend/list": _fixture("radar_tiktok_populartrend_denied.json"),
        }

    def test_zero_entities_raises_unreachable(self):
        with mock.patch(
            "urllib.request.urlopen", _dispatcher(self._probe_mapping())
        ):
            with self.assertRaises(SourceUnreachableError) as ctx:
                tiktok.fetch_tiktok_videos()
        message = str(ctx.exception)
        # names what was tried and what happened — no invented videos
        self.assertIn("zero video entities", message)
        self.assertIn("no permission", message)

    def test_health_probe_failure_raises_unreachable(self):
        mapping = dict(self._probe_mapping())
        mapping["GetTopContentsOverview"] = urllib.error.HTTPError(
            tiktok.HEALTH_URL, 403, "Forbidden", {}, None
        )
        with mock.patch("urllib.request.urlopen", _dispatcher(mapping)):
            with self.assertRaises(SourceUnreachableError) as ctx:
                tiktok.fetch_tiktok_videos()
        self.assertIn("health probe", str(ctx.exception))

    def test_never_returns_fabricated_items(self):
        # The contract: an unreachable video source raises; it never
        # returns [] masquerading as "no trends" or invented rows.
        with mock.patch(
            "urllib.request.urlopen", _dispatcher(self._probe_mapping())
        ):
            try:
                result = tiktok.fetch_tiktok_videos()
            except SourceUnreachableError:
                return
        self.fail(f"expected SourceUnreachableError, got {result!r}")

    def test_working_endpoint_would_parse_entities(self):
        payload = (
            b'{"BaseResp":{"StatusCode":0},"data":{"list":['
            b'{"title":"Wind map timelapse","video_id":"v1",'
            b'"share_url":"https://www.tiktok.com/t/abc",'
            b'"play_count":1200000,"like_count":45000}]}}'
        )
        mapping = {
            "GetTopContentsOverview": _fixture("radar_tiktok_health.json"),
            "CreativeCenterGetTopContentsList": payload,
        }
        with mock.patch("urllib.request.urlopen", _dispatcher(mapping)):
            items = tiktok.fetch_tiktok_videos()
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["source"], "tiktok")
        self.assertEqual(item["title"], "Wind map timelapse")
        self.assertEqual(item["url"], "https://www.tiktok.com/t/abc")
        self.assertEqual(item["metrics"]["views"], 1200000)
        self.assertEqual(item["metrics"]["likes"], 45000)
        self.assertIn("timelapse", item["tags"])


if __name__ == "__main__":
    unittest.main()
