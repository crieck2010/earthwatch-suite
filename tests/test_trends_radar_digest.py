"""Radar digest rendering tests: no network, fixture-built signals."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.radar.digest import INSPIRATION_NOTE, render_radar_digest
from trends.radar.signals import build_signals


def item(title, tags, metrics, source="reddit"):
    return {
        "id": f"test:{title}",
        "source": source,
        "title": title,
        "url": f"https://example.com/{title.replace(' ', '_')}",
        "author": None,
        "published_at": None,
        "metrics": metrics,
        "tags": tags,
        "thumbnail_url": None,
        "fetched_at": "2026-10-01T00:00:00Z",
    }


COVERAGE = [
    {"source": "reddit", "status": "ok", "n_items": 41,
     "detail": "r/MapPorn, r/dataisbeautiful via public JSON API"},
    {"source": "tiktok", "status": "unavailable", "n_items": 0,
     "detail": "no Creative Center endpoint returned video entities"},
    {"source": "youtube", "status": "unavailable", "n_items": 0,
     "detail": "YouTube Data API v3 requires an API key"},
]


class TestRenderRadarDigest(unittest.TestCase):
    def test_full_digest(self):
        items = [
            item("Wind particles A", ["particle-flow"], {"score": 10000}),
            item("Wind particles B", ["particle-flow"], {"score": 5000}),
            item("Globe spin A", ["globe-spin"], {"views": 900000}),
        ]
        signals = build_signals(items)
        digest = render_radar_digest(
            signals, COVERAGE, fetched_at="2026-10-01T21:00:00Z"
        )
        self.assertIn("# 🛰️ Map-Viz Trend Radar", digest)
        self.assertIn("2026-10-01T21:00:00Z", digest)
        self.assertIn("### 1. globe-spin", digest)
        self.assertIn("Takeaway:", digest)
        # exemplar links point back to originals
        self.assertIn("[Globe spin A](https://example.com/Globe_spin_A)", digest)
        # coverage table names what responded and what did not, with reasons
        self.assertIn("| reddit | ✅ ok | 41 |", digest)
        self.assertIn("❌ unavailable", digest)
        self.assertIn("no Creative Center endpoint returned video entities", digest)
        self.assertIn("YouTube Data API v3 requires an API key", digest)
        # ethics note is always present
        self.assertIn("Inspiration, not replication", digest)
        self.assertIn(INSPIRATION_NOTE.strip().splitlines()[0], digest)
        self.assertIn("frame-for-frame", digest)

    def test_no_signals_honest_section(self):
        digest = render_radar_digest([], COVERAGE)
        self.assertIn("No verified trend items this pull", digest)
        self.assertIn("estimated or invented", digest.lower())

    def test_timestamp_defaults_to_now(self):
        digest = render_radar_digest([], [])
        self.assertIn("Fetched 20", digest)  # ISO UTC stamp, not empty


if __name__ == "__main__":
    unittest.main()
