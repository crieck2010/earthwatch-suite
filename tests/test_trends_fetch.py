"""Parser and fetch-flow tests. No network: sessions are faked."""
import json
import os
import sys
import unittest

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.fetch import (
    MUSIC_ENDPOINTS,
    OVERVIEW_URL,
    _parse_music_payload,
    fetch_trending_songs,
)
from trends.models import (
    SongRank,
    SourceUnparsableError,
    SourceUnreachableError,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "legacy_music_response.json")
FETCHED_AT = "2026-09-29T03:00:00Z"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    """Minimal stand-in for requests.Session with scripted GET responses."""

    def __init__(self, routes):
        # routes: list of (url_substring, FakeResponse | Exception)
        self.routes = routes
        self.closed = False

    def get(self, url, timeout=None, **kwargs):
        for needle, outcome in self.routes:
            if needle in url:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise AssertionError(f"unexpected URL in test: {url}")

    def close(self):
        self.closed = True


def overview_ok():
    return FakeResponse(200, {"BaseResp": {"StatusCode": 0, "StatusMessage": ""}})


class TestParseMusicPayload(unittest.TestCase):
    def test_parses_fixture(self):
        with open(FIXTURE) as fh:
            payload = json.load(fh)
        songs = _parse_music_payload(payload, "tiktok-creative-center", FETCHED_AT)
        self.assertEqual(len(songs), 4)  # the title-less row is skipped
        self.assertEqual(songs[0].title, "Midnight Piano")
        self.assertEqual(songs[0].artist, "Elena Voss")
        self.assertEqual(songs[0].uses, 2500000)
        self.assertEqual(songs[0].trend, "up")
        self.assertEqual(songs[0].rank, 1)
        self.assertEqual(songs[0].fetched_at, FETCHED_AT)
        # alternate key names are honored
        self.assertEqual(songs[1].uses, 1800000)
        self.assertEqual(songs[1].trend, "same")
        # missing artist becomes "" not None
        self.assertEqual(songs[3].artist, "")

    def test_non_object_payload_raises(self):
        with self.assertRaises(SourceUnparsableError):
            _parse_music_payload([1, 2, 3], "x", FETCHED_AT)

    def test_missing_data_list_raises(self):
        with self.assertRaises(SourceUnparsableError):
            _parse_music_payload({"code": 0, "data": {}}, "x", FETCHED_AT)

    def test_empty_list_raises(self):
        with self.assertRaises(SourceUnparsableError):
            _parse_music_payload({"code": 0, "data": {"list": []}}, "x", FETCHED_AT)

    def test_never_invents_missing_titles(self):
        with self.assertRaises(SourceUnparsableError):
            _parse_music_payload(
                {"code": 0, "data": {"list": [{"artist": "Nobody"}]}}, "x", FETCHED_AT
            )


class TestFetchFlow(unittest.TestCase):
    def test_happy_path(self):
        with open(FIXTURE) as fh:
            payload = json.load(fh)
        session = FakeSession(
            [
                (OVERVIEW_URL, overview_ok()),
                ("popular_trend/music/list", FakeResponse(200, payload)),
            ]
        )
        songs = fetch_trending_songs(session=session)
        self.assertEqual(len(songs), 4)
        self.assertTrue(all(isinstance(s, SongRank) for s in songs))
        self.assertEqual([s.rank for s in songs], [1, 2, 3, 4])

    def test_overview_down_means_unreachable(self):
        session = FakeSession(
            [(OVERVIEW_URL, requests.ConnectionError("dns failure"))]
        )
        with self.assertRaises(SourceUnreachableError) as ctx:
            fetch_trending_songs(session=session)
        self.assertIn("unreachable", str(ctx.exception).lower())

    def test_all_music_endpoints_404_means_unreachable(self):
        session = FakeSession(
            [
                (OVERVIEW_URL, overview_ok()),
                ("popular_trend/music/list", FakeResponse(404, text="404 page not found")),
            ]
        )
        with self.assertRaises(SourceUnreachableError) as ctx:
            fetch_trending_songs(session=session)
        self.assertIn("404", str(ctx.exception))

    def test_non_json_body_counts_as_attempted(self):
        session = FakeSession(
            [
                (OVERVIEW_URL, overview_ok()),
                ("popular_trend/music/list", FakeResponse(200, payload=None, text="<html>")),
            ]
        )
        with self.assertRaises(SourceUnreachableError):
            fetch_trending_songs(session=session)


if __name__ == "__main__":
    unittest.main()
