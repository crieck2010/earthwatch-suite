"""Priors schema tests: exact contract for the survey-autopilot wiring."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.radar.priors import priors_from_signals
from trends.radar.signals import build_signals


def item(title, tags, metrics):
    return {
        "id": f"test:{title}",
        "source": "reddit",
        "title": title,
        "url": f"https://example.com/{title}",
        "author": None,
        "published_at": None,
        "metrics": metrics,
        "tags": tags,
        "thumbnail_url": None,
        "fetched_at": "2026-10-01T00:00:00Z",
    }


def four_format_signals():
    items = [
        item("Globe A", ["globe-spin"], {"views": 900000}),
        item("Globe B", ["globe-spin"], {"views": 100000}),
        item("Wind A", ["particle-flow"], {"score": 10000}),
        item("Wind B", ["particle-flow"], {"score": 5000}),
        item("Heat A", ["heatmap"], {"score": 400}),
        item("Heat B", ["heatmap"], {"score": 300}),
        item("Dot A", ["dot-density"], {"score": 50}),
        item("Dot B", ["dot-density"], {"score": 40}),
        item("Lone", ["satellite"], {"score": 999999}),  # 1 item -> excluded
    ]
    return build_signals(items, window_days=7)


class TestPriors(unittest.TestCase):
    def test_schema_exact(self):
        payload = priors_from_signals(four_format_signals())
        self.assertEqual(
            set(payload.keys()),
            {"version", "generated_at", "window_days", "priors", "notes"},
        )
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["window_days"], 7)
        self.assertTrue(payload["generated_at"].endswith("Z"))
        self.assertIsInstance(payload["priors"], dict)
        self.assertIsInstance(payload["notes"], list)

    def test_multiplier_ladder(self):
        payload = priors_from_signals(four_format_signals())
        priors = payload["priors"]
        # rank order: globe-spin (1M) > particle-flow (15k) > heatmap (700) > dot-density (90)
        self.assertEqual(priors["globe-spin"], 1.3)
        self.assertEqual(priors["particle-flow"], 1.15)
        self.assertEqual(priors["heatmap"], 1.05)
        self.assertEqual(priors["dot-density"], 1.0)

    def test_single_item_formats_excluded(self):
        payload = priors_from_signals(four_format_signals())
        self.assertNotIn("satellite", payload["priors"])
        self.assertNotIn("satellite", " ".join(payload["notes"]))

    def test_notes_are_rank_ordered_takeaways(self):
        signals = four_format_signals()
        payload = priors_from_signals(signals)
        eligible = [s for s in signals if s["n_items"] >= 2]
        self.assertEqual(payload["notes"], [s["takeaway"] for s in eligible])

    def test_multipliers_clamped(self):
        for value in priors_from_signals(four_format_signals())["priors"].values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.25)
            self.assertLessEqual(value, 4.0)

    def test_window_days_passthrough(self):
        signals = build_signals(
            [item("A", ["heatmap"], {"score": 10}),
             item("B", ["heatmap"], {"score": 20})],
            window_days=14,
        )
        payload = priors_from_signals(signals)
        self.assertEqual(payload["window_days"], 14)

    def test_empty_signals(self):
        payload = priors_from_signals([])
        self.assertEqual(
            payload,
            {
                "version": 1,
                "generated_at": payload["generated_at"],
                "window_days": 7,
                "priors": {},
                "notes": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
