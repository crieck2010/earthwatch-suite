"""Format-tagging tests: pure, deterministic, no network."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trends.radar import formats
from trends.radar.formats import TAG_RULES, TAG_VOCABULARY, tag_item


def item(title, tags=()):
    return {"title": title, "tags": list(tags)}


class TestTagItem(unittest.TestCase):
    def test_particle_flow(self):
        self.assertIn(
            "particle-flow",
            tag_item(item("Wind particles flowing over North America [OC]")),
        )

    def test_flow_field(self):
        self.assertIn(
            "flow-field", tag_item(item("GFS wind flow field animation"))
        )

    def test_globe_spin(self):
        self.assertIn(
            "globe-spin", tag_item(item("A rotating globe of shipping routes"))
        )

    def test_choropleth(self):
        self.assertIn(
            "choropleth", tag_item(item("US election map by county, 2024"))
        )

    def test_3d_extrusion(self):
        self.assertIn(
            "3d-extrusion", tag_item(item("3D extruded population map of Europe"))
        )

    def test_timelapse(self):
        self.assertIn(
            "timelapse", tag_item(item("Timelapse: 30 years of Arctic sea ice"))
        )

    def test_before_after(self):
        self.assertIn(
            "before-after", tag_item(item("Before and after the flood: slider compare"))
        )

    def test_heatmap(self):
        self.assertIn("heatmap", tag_item(item("Global heatmap of shipping density")))

    def test_dot_density(self):
        self.assertIn(
            "dot-density", tag_item(item("One dot per 1000 residents"))
        )

    def test_satellite(self):
        self.assertIn(
            "satellite", tag_item(item("Sentinel-2 imagery of the delta"))
        )

    def test_bathymetry_terrain(self):
        self.assertIn(
            "bathymetry-terrain",
            tag_item(item("Bathymetry of the Mariana Trench in 3D")),
        )

    def test_fallback_other(self):
        self.assertEqual(["other"], tag_item(item("Cute cat does taxes")))

    def test_multi_tag_allowed(self):
        tags = tag_item(item("3D rotating globe timelapse of ocean currents"))
        self.assertIn("globe-spin", tags)
        self.assertIn("3d-extrusion", tags)
        self.assertIn("timelapse", tags)

    def test_tags_field_is_also_scanned(self):
        self.assertIn("satellite", tag_item(item("Delta", tags=["satellite"])))

    def test_deterministic(self):
        first = tag_item(item("Wind particles and heatmap over terrain"))
        second = tag_item(item("Wind particles and heatmap over terrain"))
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first, key=TAG_VOCABULARY.index))

    def test_does_not_mutate_input(self):
        original = {"title": "Wind particles", "tags": []}
        snapshot = dict(original)
        tag_item(original)
        self.assertEqual(original, snapshot)

    def test_every_vocab_tag_has_rules(self):
        for tag in TAG_VOCABULARY:
            if tag == "other":
                continue
            rules = dict(TAG_RULES)
            self.assertIn(tag, rules)
            self.assertTrue(rules[tag], f"{tag} has no keywords")

    def test_empty_title_falls_back(self):
        self.assertEqual(["other"], tag_item({"title": "", "tags": []}))


if __name__ == "__main__":
    unittest.main()
