"""Tests for survey-style (stdlib-only core; no network, no viz import)."""

import json
import os

import pytest

import style
from style import (
    Style,
    StyleError,
    apply_style,
    get_style,
    list_styles,
    load_styles_from_dir,
    register_style,
    render_kwargs_for,
    render_template,
    suggest_style,
    validate_style,
    variable_label,
    region_label,
)
from style.presets import style_from_dict, BUILTIN_STYLES


def _spec(variable="sst", **kw):
    d = {
        "title": "Lake Michigan warmth",
        "region_key": "lake-michigan",
        "variable": variable,
        "start": "2026-07-01",
        "end": "2026-07-31",
        "style": "reel-dark",
        "caption": None,
        "underlay": False,
    }
    d.update(kw)
    return d


# --- registry ---------------------------------------------------------------

def test_list_styles_has_six_builtins():
    names = list_styles()
    assert names == ["creator", "field-notes", "midnight-ocean",
                     "reel-dark", "reel-light", "storm-chaser"]


def test_get_style_returns_style():
    s = get_style("midnight-ocean")
    assert isinstance(s, Style)
    assert s.base_style == "reel-dark"


def test_get_style_unknown_with_hint():
    with pytest.raises(StyleError, match="did you mean.*midnight-ocean"):
        get_style("midnight-oceen")


def test_get_style_unknown_no_hint_lists_available():
    with pytest.raises(StyleError, match="available:"):
        get_style("zzz-nope")


def test_builtin_styles_all_valid():
    for s in BUILTIN_STYLES:
        assert validate_style(s) == [], s.name


def test_style_constructor_rejects_bad_base_style():
    with pytest.raises(StyleError, match="base_style"):
        Style(name="bad", base_style="neon")


def test_style_constructor_rejects_bad_slug():
    with pytest.raises(StyleError, match="name"):
        Style(name="Not A Slug!")


def test_style_constructor_rejects_bad_template_field():
    with pytest.raises(StyleError, match="title_template.*unknown field"):
        Style(name="bad", title_template="{bogus}")


def test_validate_style_non_raising():
    s = Style.__new__(Style)  # bypass constructor validation
    object.__setattr__(s, "name", "ok-name")
    object.__setattr__(s, "version", "1.0")
    object.__setattr__(s, "label", "")
    object.__setattr__(s, "description", "")
    object.__setattr__(s, "base_style", "nope")
    object.__setattr__(s, "colormaps", {"sst": ""})
    object.__setattr__(s, "default_cmap", "turbo")
    object.__setattr__(s, "title_template", "{title}")
    object.__setattr__(s, "footer_template", "{nope}")
    object.__setattr__(s, "underlay", True)
    object.__setattr__(s, "tags", ())
    problems = validate_style(s)
    assert any("base_style" in p for p in problems)
    assert any("colormaps" in p for p in problems)
    assert any("footer_template" in p for p in problems)


def test_register_style_collision():
    with pytest.raises(StyleError, match="already registered"):
        register_style(Style(name="reel-dark", label="x"))
    # replace=True works and restores afterwards
    orig = get_style("reel-dark")
    register_style(Style(name="reel-dark", label="temp"), replace=True)
    assert get_style("reel-dark").label == "temp"
    register_style(orig, replace=True)


def test_register_style_rejects_non_style():
    with pytest.raises(StyleError, match="expected a Style"):
        register_style({"name": "nope"})


# --- templates --------------------------------------------------------------

def test_render_template_all_fields():
    out = render_template(
        "{title} | {variable_label} in {region_label} {start}..{end} {year}",
        {"title": "T", "variable": "sst", "variable_label": "Sea Surface Temperature",
         "region": "lake-michigan", "region_label": "Lake Michigan",
         "start": "2026-07-01", "end": "2026-07-31", "date": "2026-07-31",
         "year": "2026"})
    assert out == "T | Sea Surface Temperature in Lake Michigan 2026-07-01..2026-07-31 2026"


def test_render_template_unknown_field():
    with pytest.raises(StyleError, match="unknown field"):
        render_template("{bogus}", {})


def test_render_template_missing_channel_hint():
    with pytest.raises(StyleError, match="channel.*pass channel"):
        render_template("© {channel}", {"title": "T"})


def test_render_template_escaped_braces():
    assert render_template("{{literal}} {title}", {"title": "T"}) == "{literal} T"


# --- labels -----------------------------------------------------------------

def test_variable_label_known():
    assert variable_label("sst") == "Sea Surface Temperature"
    assert variable_label("ocean-color") == "Chlorophyll-a"


def test_variable_label_unknown_falls_back():
    assert variable_label("my-var") == "My Var"


def test_region_label():
    assert region_label("lake-michigan") == "Lake Michigan"
    assert region_label("") == ""


# --- apply ------------------------------------------------------------------

def test_apply_style_sets_spec_fields():
    out, kw = apply_style(_spec(), "midnight-ocean")
    assert out["style"] == "reel-dark"
    assert out["title"] == "Lake Michigan warmth"
    assert out["caption"] == "Sea Surface Temperature · Lake Michigan"
    assert out["underlay"] is True
    assert kw == {"cmap": "inferno"}


def test_apply_style_default_cmap_fallback():
    out, kw = apply_style(_spec(variable="brand-new-var"), "reel-dark")
    assert kw == {"cmap": "turbo"}


def test_apply_style_empty_footer_is_none():
    out, _ = apply_style(_spec(), "reel-dark")
    assert out["caption"] is None


def test_apply_style_does_not_mutate_input():
    spec = _spec()
    before = dict(spec)
    apply_style(spec, "storm-chaser")
    assert spec == before


def test_apply_style_accepts_to_dict_object():
    class Fake:
        def to_dict(self):
            return _spec(variable="wind")
    out, kw = apply_style(Fake(), "storm-chaser")
    assert kw == {"cmap": "turbo"}
    assert "Wind Speed" in out["caption"]


def test_apply_style_rejects_bad_spec():
    with pytest.raises(StyleError, match="to_dict"):
        apply_style(42, "reel-dark")
    with pytest.raises(StyleError, match="no 'variable' key"):
        apply_style({"title": "x"}, "reel-dark")


def test_apply_style_overrides():
    out, kw = apply_style(
        _spec(), "reel-dark",
        overrides={"footer_template": "hi {title}", "underlay": False,
                   "colormaps": {"sst": "plasma"}})
    assert out["caption"] == "hi Lake Michigan warmth"
    assert out["underlay"] is False
    assert kw == {"cmap": "plasma"}
    # registry untouched
    assert get_style("reel-dark").footer_template == ""


def test_apply_style_unknown_override_field():
    with pytest.raises(StyleError, match="unknown override"):
        apply_style(_spec(), "reel-dark", overrides={"bogus": 1})


def test_apply_style_creator_needs_channel():
    with pytest.raises(StyleError, match="channel"):
        apply_style(_spec(), "creator")
    out, _ = apply_style(_spec(), "creator", channel="Mapped Earth")
    assert out["caption"] == "© Mapped Earth"


def test_render_kwargs_for():
    s = get_style("field-notes")
    assert render_kwargs_for(s, "topography") == {"cmap": "gist_earth"}
    assert render_kwargs_for(s, "unknown-var") == {"cmap": "terrain"}


def test_suggest_style():
    assert suggest_style("sst") == "midnight-ocean"
    assert suggest_style("wind") == "storm-chaser"
    assert suggest_style("topography") == "field-notes"
    assert suggest_style("something-new") == "reel-dark"


# --- user styles (JSON) ------------------------------------------------------

def test_load_styles_from_dir(tmp_path):
    d = tmp_path / "styles"
    d.mkdir()
    (d / "mine.json").write_text(json.dumps({
        "name": "mine",
        "label": "Mine",
        "base_style": "light",
        "colormaps": {"sst": "coolwarm"},
        "default_cmap": "viridis",
        "title_template": "{title}!",
        "footer_template": "by {channel}",
        "underlay": False,
        "tags": ["custom"],
    }), encoding="utf-8")
    (d / "notes.txt").write_text("ignored", encoding="utf-8")
    loaded = load_styles_from_dir(str(d))
    assert [s.name for s in loaded] == ["mine"]
    assert "mine" in list_styles()
    out, kw = apply_style(_spec(), "mine", channel="me")
    assert out["title"] == "Lake Michigan warmth!"
    assert out["caption"] == "by me"
    assert out["style"] == "light"
    assert kw == {"cmap": "coolwarm"}


def test_load_styles_from_dir_bad_json(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(StyleError, match="bad.json.*cannot parse JSON"):
        load_styles_from_dir(str(d))


def test_load_styles_from_dir_invalid_style(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "bad.json").write_text(json.dumps(
        {"name": "bad", "base_style": "neon"}), encoding="utf-8")
    with pytest.raises(StyleError, match=r"bad\.json[\s\S]*base_style"):
        load_styles_from_dir(str(d))


def test_load_styles_from_dir_missing(tmp_path):
    with pytest.raises(StyleError, match="not found"):
        load_styles_from_dir(str(tmp_path / "nope"))


def test_style_from_dict_unknown_key():
    with pytest.raises(StyleError, match="unknown style field"):
        style_from_dict({"name": "x", "bogus": 1}, source="f.json")


def test_style_from_dict_not_a_dict():
    with pytest.raises(StyleError, match="must be a JSON object"):
        style_from_dict([1, 2], source="f.json")


# --- cross-peer consistency ---------------------------------------------------

def test_base_styles_match_installed_viz():
    """KNOWN_BASE_STYLES mirrors viz.spec.KNOWN_STYLES (duck-typed read)."""
    import ast
    import viz.spec  # noqa: F401  (test-only; the package itself never imports viz)
    src = open(os.path.join(os.path.dirname(viz.spec.__file__),
                            "spec.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    known = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "KNOWN_STYLES":
            known = ast.literal_eval(node.value)
    from style import KNOWN_BASE_STYLES
    assert known is not None
    assert tuple(known) == tuple(KNOWN_BASE_STYLES)


def test_builtin_cmaps_in_viz_curated_set():
    """Every built-in cmap is one viz recommends (CURATED_CMAPS)."""
    import viz.render as vr
    curated = set(vr.CURATED_CMAPS)
    for s in BUILTIN_STYLES:
        for var, cmap in s.colormaps.items():
            assert cmap in curated, f"{s.name}: {var} -> {cmap!r}"
        assert s.default_cmap in curated, s.name
