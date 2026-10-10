"""Demo-story tests: run each site's demo.sh end-to-end.

These need the earthwatch-suite engine CLIs installed (survey-3d,
survey-monitor, survey-alerts, survey-vegetation, survey-thermal,
survey-coast, survey-flood, survey-burn). They run fully offline on
synthetic engine data. Skipped when the engines are absent.
"""

import os
import shutil
import subprocess

import pytest

PACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sites"))

ENGINE_CLIS = [
    "survey-3d", "survey-monitor", "survey-alerts", "survey-vegetation",
    "survey-thermal", "survey-coast", "survey-flood", "survey-burn",
    "survey-sites",
]

engines_present = all(shutil.which(cli) for cli in ENGINE_CLIS)
needs_engines = pytest.mark.skipif(
    not engines_present, reason="engine CLIs not installed"
)


@needs_engines
@pytest.mark.parametrize("site", ["quarry", "farm", "coast", "forest"])
def test_demo_story_passes(site):
    demo = os.path.join(PACK_ROOT, site, "demo.sh")
    result = subprocess.run(
        ["bash", demo], capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"demo.sh for {site} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert f"DEMO PASS: {site}" in result.stdout
