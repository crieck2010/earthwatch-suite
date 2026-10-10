"""Master pytest fixtures for earthwatch-suite."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

# --- survey-aesthetics fixtures ---
@pytest.fixture()
def uv_scalar():
    """Small synthetic (u, v, scalar) grids with a known NaN hole."""
    rng = np.random.default_rng(42)
    ny, nx = 48, 64
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    u = np.sin(x / 8.0) * 2.0
    v = np.cos(y / 10.0) * 1.5
    scalar = 10.0 + 0.1 * x + 0.05 * y
    u[5:10, 5:10] = np.nan
    v[5:10, 5:10] = np.nan
    scalar[5:10, 5:10] = np.nan
    return u, v, scalar


@pytest.fixture()
def glow_points():
    rng = np.random.default_rng(7)
    n = 60
    xs = rng.uniform(20, 100, n)
    ys = rng.uniform(20, 160, n)
    vals = rng.uniform(1, 10, n)
    return xs, ys, vals

# --- survey-animate fixtures ---
@pytest.fixture()
def tmp_png(tmp_path):
    from animate import _png
    def _make(name="f.png", shape=(32, 48, 3), seed=0):
        rng = np.random.default_rng(seed)
        img = (rng.random(shape) * 255).astype(np.uint8)
        p = str(tmp_path / name)
        _png.write_png(p, img)
        return p, img
    return _make


@pytest.fixture()
def manifest_dict(tmp_png):
    p0, _ = tmp_png("a.png", seed=1)
    p1, _ = tmp_png("b.png", seed=2)
    return {
        "schema": "survey-flow.frame-manifest/1.0",
        "generator": "survey-flow 0.1.0",
        "created": "2026-09-25T00:00:00+00:00",
        "fps": 12.0,
        "size_px": [48, 32],
        "field_source": "synthetic/test",
        "n_frames": 2,
        "frames": [
            {"frame_index": 0, "timestamp_iso": "2026-09-16T00:00:00+00:00",
             "png_path": os.path.basename(p0), "scalar": "temperature",
             "scalar_range": [10.0, 20.0], "colormap": "cividis",
             "style": "reel-dark", "render_params": {"seed": 7}},
            {"frame_index": 1, "timestamp_iso": "2026-09-16T01:00:00+00:00",
             "png_path": os.path.basename(p1), "scalar": "temperature",
             "scalar_range": [10.0, 20.0], "colormap": "cividis",
             "style": "reel-dark", "render_params": {"seed": 7}},
        ],
    }


@pytest.fixture()
def manifest_file(tmp_path, manifest_dict):
    p = str(tmp_path / "manifest.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(manifest_dict, fh)
    return p

# --- survey-flow fixtures ---
def make_fake_current_field(nt=3, ny=12, nx=16, seed=7, with_land=True,
                            with_temperature=True):
    rng = np.random.default_rng(seed)
    lats = np.linspace(41.5, 46.5, ny)
    lons = np.linspace(-92.5, -84.5, nx)
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx),
                         indexing="ij")
    land = (xx ** 2 + yy ** 2) > 0.86 if with_land else np.zeros((ny, nx), bool)
    u = np.empty((nt, ny, nx))
    v = np.empty((nt, ny, nx))
    t = np.empty((nt, ny, nx))
    t0 = dt.datetime(2026, 9, 16, tzinfo=dt.timezone.utc)
    times = []
    for k in range(nt):
        uk = -yy * 0.35 + rng.normal(0, 0.02, (ny, nx))
        vk = xx * 0.35 + rng.normal(0, 0.02, (ny, nx))
        tk = 18.0 - 4.0 * (xx ** 2 + yy ** 2) + 0.2 * k
        u[k] = np.where(land, np.nan, uk)
        v[k] = np.where(land, np.nan, vk)
        t[k] = np.where(land, np.nan, tk)
        times.append((t0 + dt.timedelta(hours=k)).isoformat())
    ns = types.SimpleNamespace()
    ns.u, ns.v = u, v
    ns.temperature = t if with_temperature else None
    ns.times, ns.lats, ns.lons = times, lats, lons
    ns.crs, ns.source = "EPSG:4326", "test/fake"
    return ns


@pytest.fixture
def fake_cf():
    return make_fake_current_field()

# --- survey-viz fixtures ---
@pytest.fixture(autouse=True)
def _offline_underlay(monkeypatch, request):
    try:
        from viz import underlay as _ul
        if request.node.get_closest_marker("real_underlay") is None:
            monkeypatch.setattr(
                _ul,
                "fetch_underlay",
                lambda *a, **k: _ul._unavailable(
                    "test: underlay disabled by tests/conftest.py"),
            )
    except ImportError:
        pass
