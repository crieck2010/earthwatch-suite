import os

import pytest

from reels.recipe import ReelRecipe, recipe_hash, example_recipe, RECIPE_SCHEMA


def _base(**kw):
    data = {
        "name": "test-reel",
        "source": "synthetic",
        "bbox": [-87.9, 41.6, -86.7, 42.9],
    }
    data.update(kw)
    return data


def test_round_trip():
    r = ReelRecipe.from_dict(_base(title="T", n_frames=12))
    assert ReelRecipe.from_dict(r.to_dict()) == r
    assert r.to_dict()["schema"] == RECIPE_SCHEMA


def test_hash_stable_and_sensitive():
    a = ReelRecipe.from_dict(_base())
    b = ReelRecipe.from_dict(_base())
    assert recipe_hash(a) == recipe_hash(b) and len(recipe_hash(a)) == 64
    c = ReelRecipe.from_dict(_base(n_frames=24))
    assert recipe_hash(c) != recipe_hash(a)


def test_name_required():
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(name="  "))


def test_bad_source():
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(source="satellite"))


@pytest.mark.parametrize("bbox", [
    [1, 2, 3], [1, 2, 3, 4, 5], [-87.9, 42.9, -86.7, 41.6],
    [-200, 41.6, -86.7, 42.9],
])
def test_bad_bbox(bbox):
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(bbox=bbox))


def test_bad_styles():
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(flow_style="neon"))
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(animate_preset="imax"))


def test_bad_counts():
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(n_frames=0))
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(n_particles=0))
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(fps=0))


def test_noaa_needs_hours():
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(source="noaa-ofs", hours=[]))
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(source="noaa-ofs", hours=[-1]))


def test_cmems_needs_window():
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base(source="cmems"))
    r = ReelRecipe.from_dict(_base(source="cmems",
                                    start="2026-09-16T00:00:00",
                                    end="2026-09-17T00:00:00"))
    assert r.source == "cmems"


def test_schema_major_mismatch_rejected():
    with pytest.raises(ValueError):
        ReelRecipe.from_dict(_base() | {"schema": "survey-reels.recipe/9.9"})
    # missing schema is fine (forward-compatible read)
    r = ReelRecipe.from_dict(_base())
    assert r.name == "test-reel"


def test_unknown_keys_ignored():
    r = ReelRecipe.from_dict(_base(future_field="kept-out"))
    assert not hasattr(r, "future_field")


def test_yaml_round_trip(tmp_path):
    pytest.importorskip("yaml")
    p = str(tmp_path / "r.yaml")
    r = example_recipe()
    r.to_yaml(p)
    assert ReelRecipe.from_yaml(p) == r


def test_example_recipe_valid():
    r = example_recipe()
    assert r.source == "noaa-ofs" and r.ofs_code == "LMHOFS"
    assert r.animate_preset == "reel"
