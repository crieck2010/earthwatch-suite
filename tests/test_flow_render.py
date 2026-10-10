"""Tests for flow.render (matplotlib) and aspect/scalebar helpers."""

import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from flow.advect import AdvectionConfig, ParticleSet
from flow.fields import TimeVaryingField, VectorField
from flow.render import SIZES, _nice_scalebar_km, _pretty_timestamp, fit_aspect, render_frame

from conftest import make_fake_current_field


def test_fit_aspect_crops_wide_data():
    # Wide data, tall target -> crop longitude.
    view = fit_aspect((-92.5, 41.5, -84.5, 46.5), (1080, 1920))
    minx, miny, maxx, maxy = view
    assert miny == 41.5 and maxy == 46.5
    assert minx > -92.5 and maxx < -84.5
    # Tall data, wide target -> crop latitude.
    view2 = fit_aspect((-88.0, 40.0, -84.0, 50.0), (1920, 1080))
    assert view2[0] == -88.0 and view2[2] == -84.0
    assert view2[1] > 40.0 and view2[3] < 50.0


def test_nice_scalebar_km():
    assert _nice_scalebar_km(600.0) == 100.0
    assert _nice_scalebar_km(300.0) == 50.0
    assert _nice_scalebar_km(30.0) == 5.0
    assert _nice_scalebar_km(0.0) == 1.0


def test_pretty_timestamp():
    assert _pretty_timestamp("2026-09-16T01:20:00+00:00") == "16 Sep 2026 01:20 UTC"
    assert _pretty_timestamp("not-a-date") == "not-a-date"


def test_render_frame_creates_png(tmp_path, fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    vf = tvf.at(1.0)
    ps = ParticleSet(30, tvf.at(0.0), AdvectionConfig(seed=1))
    ps.step(vf)
    out = str(tmp_path / "frame.png")
    result = render_frame(vf, ps, "reel-dark", out, size=(270, 480), dpi=50)
    assert result == out
    assert os.path.getsize(out) > 10_000
    # PNG magic bytes.
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_render_frame_static_snapshot(tmp_path, fake_cf):
    # particles=None -> static scalar map (the survey-qgis hook shape).
    tvf = TimeVaryingField.from_current_field(fake_cf)
    out = str(tmp_path / "static.png")
    render_frame(tvf.at(0.0), None, "print", out, size="square", dpi=50)
    assert os.path.getsize(out) > 5_000


def test_render_frame_styles_and_sizes(tmp_path, fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    for style in ("reel-dark", "reel-light", "print"):
        out = str(tmp_path / f"{style}.png")
        render_frame(tvf.at(0.0), None, style, out, size=(200, 200), dpi=50)
        assert os.path.exists(out)
    assert set(SIZES) == {"reel", "wide", "square"}


def test_render_frame_rejects_bad_inputs(tmp_path, fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    with pytest.raises(ValueError):
        render_frame(tvf.at(0.0), None, "reel-dark",
                     str(tmp_path / "x.png"), size="portrait")
    no_scalar = VectorField(u=np.zeros((3, 3)), v=np.zeros((3, 3)),
                            lats=np.arange(40.0, 43.0),
                            lons=np.arange(-90.0, -87.0))
    with pytest.raises(ValueError):
        render_frame(no_scalar, None, "reel-dark",
                     str(tmp_path / "x.png"), size=(100, 100), dpi=50)


def test_render_frame_with_title(tmp_path, fake_cf):
    tvf = TimeVaryingField.from_current_field(fake_cf)
    out = str(tmp_path / "titled.png")
    render_frame(tvf.at(0.0), None, "reel-dark", out, size=(200, 200),
                 dpi=50, title="Lake Michigan demo")
    assert os.path.exists(out)


def test_sizes_values():
    assert SIZES["reel"] == (1080, 1920)
    assert SIZES["wide"] == (1920, 1080)
    assert SIZES["square"] == (1080, 1080)
