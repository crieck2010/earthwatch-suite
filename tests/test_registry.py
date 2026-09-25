"""Registry smoke tests: the MODULES table is well-formed and self-consistent."""

import unittest

from earthwatch import MODULES


class TestRegistry(unittest.TestCase):
    def test_every_module_has_required_keys(self):
        required = {"package", "repo", "version", "description", "status"}
        for m in MODULES:
            self.assertTrue(required <= set(m), f"missing keys in {m.get('package')}")

    def test_repo_owner_and_name(self):
        for m in MODULES:
            owner, name = m["repo"].split("/")
            self.assertEqual(owner, "crieck2010")
            self.assertEqual(name, m["package"])

    def test_status_values(self):
        for m in MODULES:
            self.assertIn(m["status"], {"shipped", "planned"})

    def test_shipped_versions_are_semver_like(self):
        import re

        for m in MODULES:
            if m["status"] == "shipped":
                self.assertRegex(m["version"], r"^\d+\.\d+\.\d+$", m["package"])

    def test_at_least_one_shipped(self):
        self.assertGreaterEqual(sum(1 for m in MODULES if m["status"] == "shipped"), 1)

    def test_no_duplicate_packages(self):
        names = [m["package"] for m in MODULES]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
