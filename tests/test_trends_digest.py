"""Digest rendering tests: formatting, fit heuristics, failure honesty."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.digest import (
    build_digest,
    build_unavailable_digest,
    fit_note,
)
from trends.models import SongRank

FETCHED_AT = "2026-09-29T03:00:00Z"


def song(rank, title, artist="", uses=None, trend=None):
    return SongRank(
        rank=rank,
        title=title,
        artist=artist,
        uses=uses,
        trend=trend,
        source="tiktok-creative-center",
        fetched_at=FETCHED_AT,
    )


class TestFitNote(unittest.TestCase):
    def test_instrumental_flagged_good_fit(self):
        self.assertIn("likely fit", fit_note(song(1, "Midnight Piano", "Elena Voss")))

    def test_ambient_artist_flagged_good_fit(self):
        self.assertIn("likely fit", fit_note(song(1, "Waves", "Ambient Collective")))

    def test_dance_track_flagged_caution(self):
        self.assertIn("check first", fit_note(song(1, "Dancefloor Static", "DJ Reverb")))

    def test_unknown_is_unclassified_not_good(self):
        note = fit_note(song(1, "Xyz Abc", "Qwerty"))
        self.assertIn("unclassified", note)
        self.assertNotIn("likely fit", note)


class TestBuildDigest(unittest.TestCase):
    def setUp(self):
        self.songs = [
            song(1, "Midnight Piano", "Elena Voss", 2500000, "up"),
            song(2, "Neon Skyline", "Glass Cities", 1800000, "same"),
            song(3, "Dancefloor Static", "DJ Reverb", 950000, "down"),
        ]

    def test_contains_verified_data_only(self):
        digest = build_digest(self.songs, FETCHED_AT)
        self.assertIn("Midnight Piano", digest)
        self.assertIn("Elena Voss", digest)
        self.assertIn("2.5M", digest)
        self.assertIn("▲", digest)
        self.assertIn(FETCHED_AT, digest)  # fresh stamp, never stale

    def test_week_range_rendered(self):
        digest = build_digest(self.songs, FETCHED_AT)
        self.assertIn("Sep 23", digest)
        self.assertIn("Sep 29, 2026", digest)

    def test_fit_notes_section_present(self):
        digest = build_digest(self.songs, FETCHED_AT)
        self.assertIn("Fit notes for data-viz reels", digest)
        self.assertIn("Midnight Piano", digest)  # good-fit pick listed

    def test_instagram_guidance_present(self):
        digest = build_digest(self.songs, FETCHED_AT)
        self.assertIn("Instagram: find trending audio in-app", digest)
        self.assertIn("↗", digest)

    def test_missing_counts_render_as_dash(self):
        digest = build_digest([song(1, "Silent Track")], FETCHED_AT)
        self.assertIn("—", digest)

    def test_caps_at_twenty(self):
        many = [song(i, f"Track {i}") for i in range(1, 26)]
        digest = build_digest(many, FETCHED_AT)
        self.assertIn("Track 20", digest)
        self.assertNotIn("Track 21", digest)


class TestUnavailableDigest(unittest.TestCase):
    def test_notice_is_explicit_and_stamped(self):
        digest = build_unavailable_digest("connection refused", "2026-09-29T03:00:00Z")
        self.assertIn("Trending data unavailable this week", digest)
        self.assertIn("source unreachable", digest)
        self.assertIn("connection refused", digest)
        self.assertIn("2026-09-29T03:00:00Z", digest)

    def test_lists_no_songs(self):
        digest = build_unavailable_digest("404", "2026-09-29T03:00:00Z")
        self.assertNotIn("Midnight Piano", digest)
        self.assertIn("never estimates or invents", digest)

    def test_still_carries_instagram_guidance(self):
        digest = build_unavailable_digest("404", "2026-09-29T03:00:00Z")
        self.assertIn("Instagram: find trending audio in-app", digest)


if __name__ == "__main__":
    unittest.main()
