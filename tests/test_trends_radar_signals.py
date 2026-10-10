"""Signal-aggregation tests: deterministic, no network."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.radar.signals import build_signals


def item(title, tags, metrics, source="reddit"):
    return {
        "id": f"test:{title}",
        "source": source,
        "title": title,
        "url": f"https://example.com/{title}",
        "author": None,
        "published_at": None,
        "metrics": metrics,
        "tags": tags,
        "thumbnail_url": None,
        "fetched_at": "2026-10-01T00:00:00Z",
    }


def sample_items():
    return [
        item("Wind particles A", ["particle-flow"], {"score": 10000, "comments": 200}),
        item("Wind particles B", ["particle-flow"], {"score": 5000, "comments": 100}),
        item("Wind particles C", ["particle-flow"], {"score": 2500}),
        item("Globe spin A", ["globe-spin"], {"views": 900000, "likes": 10000}),
        item("Globe spin B", ["globe-spin"], {"views": 100000}),
        item("Heatmap A", ["heatmap"], {"score": 400}),
        item("Dot map A", ["dot-density"], {"score": None, "comments": 10}),
    ]


class TestBuildSignals(unittest.TestCase):
    def test_ranked_by_engagement_desc(self):
        signals = build_signals(sample_items())
        tags = [s["tag"] for s in signals]
        self.assertEqual(tags[0], "globe-spin")  # 1,010,000 engagement
        self.assertEqual(tags[1], "particle-flow")  # 17,800
        # engagement is deterministic; ties break alphabetically
        engagements = [s["engagement"] for s in signals]
        self.assertEqual(engagements, sorted(engagements, reverse=True))

    def test_stats_shape(self):
        signals = build_signals(sample_items())
        by_tag = {s["tag"]: s for s in signals}
        pf = by_tag["particle-flow"]
        self.assertEqual(pf["n_items"], 3)
        self.assertEqual(pf["engagement"], 10000 + 200 + 5000 + 100 + 2500)
        self.assertAlmostEqual(pf["share"], 3 / 7)
        self.assertIsNone(pf["velocity"])  # no baseline -> None
        self.assertIn("particle-flow", pf["takeaway"])
        self.assertEqual(pf["window_days"], 7)

    def test_engagement_ignores_none_metrics(self):
        signals = build_signals(sample_items())
        dot = {s["tag"]: s for s in signals}["dot-density"]
        self.assertEqual(dot["engagement"], 10)

    def test_exemplars_are_links_only(self):
        signals = build_signals(sample_items())
        pf = {s["tag"]: s for s in signals}["particle-flow"]
        self.assertLessEqual(len(pf["exemplars"]), 3)
        for ex in pf["exemplars"]:
            self.assertEqual(set(ex.keys()), {"title", "url", "source"})
        # highest engagement first
        self.assertEqual(pf["exemplars"][0]["title"], "Wind particles A")

    def test_velocity_with_baseline(self):
        signals = build_signals(
            sample_items(), baseline={"particle-flow": 8900.0, "globe-spin": 2020000.0}
        )
        by_tag = {s["tag"]: s for s in signals}
        self.assertAlmostEqual(
            by_tag["particle-flow"]["velocity"], (17800 - 8900.0) / 8900.0
        )
        self.assertLess(by_tag["globe-spin"]["velocity"], 0)
        self.assertIn("accelerating", by_tag["particle-flow"]["takeaway"])
        self.assertIn("cooling", by_tag["globe-spin"]["takeaway"])
        # unknown baseline tags do not invent a velocity
        self.assertIsNone(by_tag["heatmap"]["velocity"])

    def test_tie_break_is_alphabetical(self):
        items = [
            item("A", ["choropleth"], {"score": 100}),
            item("B", ["bathymetry-terrain"], {"score": 100}),
        ]
        signals = build_signals(items)
        self.assertEqual(
            [s["tag"] for s in signals], ["bathymetry-terrain", "choropleth"]
        )

    def test_deterministic(self):
        first = build_signals(sample_items())
        second = build_signals(list(reversed(sample_items())))
        self.assertEqual(
            [(s["tag"], s["engagement"]) for s in first],
            [(s["tag"], s["engagement"]) for s in second],
        )

    def test_empty_items(self):
        self.assertEqual(build_signals([]), [])


if __name__ == "__main__":
    unittest.main()
