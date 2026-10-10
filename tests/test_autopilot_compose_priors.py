"""Tests for trend-prior wiring in autopilot.compose (v0.2.0)."""

import numpy as np
import pytest

from autopilot import PRESET_FORMAT_TAGS, search_composition


def _thumb_for(preset):
    """Stub render: thumbnail quality keyed by preset name.

    - "crisp": full-frame gradient -> high coverage AND contrast (best raw).
    - "dark_flow": flat mid-gray -> full coverage, zero contrast (weaker raw).
    """
    if preset == "dark_flow":
        return np.full((20, 30, 3), 0.5)
    ramp = np.linspace(0.06, 0.95, 30)[None, :, None]
    return np.repeat(np.repeat(ramp, 20, axis=0), 3, axis=2)


def _render(preset, rotation, basemap):
    return _thumb_for(preset)


def _cands():
    return [
        {"preset": "dark_flow", "rotation": 0, "basemap": "dark"},
        {"preset": "crisp", "rotation": 0, "basemap": "dark"},  # unknown preset
    ]


def _priors(prior_map):
    return {
        "version": 1,
        "generated_at": "2026-10-01T20:00:00-04:00",
        "window_days": 30,
        "priors": prior_map,
        "notes": ["test fixture"],
    }


def test_preset_format_tags_documented_as_heuristic():
    # The mapping the task asked for, verbatim.
    assert PRESET_FORMAT_TAGS == {
        "dark_strands": ["particle-flow", "flow-field"],
        "dark_flow": ["flow-field", "particle-flow"],
    }


def test_priors_flip_winner_and_keep_both_totals():
    # Unadjusted: "crisp" wins (gradient beats flat gray).
    plain = search_composition(_render, _cands())
    assert plain["winner"]["preset"] == "crisp"

    # A strong prior on "flow-field" boosts dark_flow past crisp.
    result = search_composition(_render, _cands(), trend_priors=_priors({"flow-field": 4.0}))
    assert result["trend_priors_applied"] is True
    assert result["trend_priors"] == _priors({"flow-field": 4.0})
    assert "trend_priors_reason" not in result

    by_preset = {s["candidate"]["preset"]: s for s in result["scores"]}
    flow = by_preset["dark_flow"]
    crisp = by_preset["crisp"]

    # dark_flow tags: ["flow-field", "particle-flow"] -> mean(4.0, 1.0) = 2.5
    assert flow["trend_multiplier"] == pytest.approx(2.5)
    assert flow["total_unadjusted"] == pytest.approx(0.525)
    assert flow["total"] == pytest.approx(0.525 * 2.5)
    assert flow["total"] > flow["total_unadjusted"]

    # Unknown preset "crisp": neutral multiplier, total untouched.
    assert crisp["trend_multiplier"] == pytest.approx(1.0)
    assert crisp["total"] == pytest.approx(crisp["total_unadjusted"])

    # Winner selection uses the ADJUSTED total.
    assert result["winner"]["preset"] == "dark_flow"
    totals = [s["total"] for s in result["scores"]]
    assert totals == sorted(totals, reverse=True)


def test_missing_prior_tags_count_as_neutral():
    # Only "particle-flow" has a prior; "flow-field" counts as 1.0.
    result = search_composition(_render, _cands(), trend_priors=_priors({"particle-flow": 3.0}))
    flow = next(s for s in result["scores"] if s["candidate"]["preset"] == "dark_flow")
    assert flow["trend_multiplier"] == pytest.approx(2.0)  # mean(1.0, 3.0)
    assert flow["total"] == pytest.approx(flow["total_unadjusted"] * 2.0)


def test_boundary_multipliers_are_valid():
    result = search_composition(_render, _cands(), trend_priors=_priors({"flow-field": 0.25, "particle-flow": 4.0}))
    assert result["trend_priors_applied"] is True
    flow = next(s for s in result["scores"] if s["candidate"]["preset"] == "dark_flow")
    assert flow["trend_multiplier"] == pytest.approx((0.25 + 4.0) / 2)


@pytest.mark.parametrize(
    "bad",
    [
        _priors({"flow-field": 2.0}) | {"version": 2},          # bad version
        _priors({"flow-field": 2.0}) | {"version": "1"},        # version wrong type
        _priors({"flow-field": 2.0}) | {"version": True},       # bool != int
        _priors({"flow-field": 5.0}),                            # multiplier too high
        _priors({"flow-field": 0.1}),                            # multiplier too low
        _priors({"flow-field": float("nan")}),                   # NaN multiplier
        _priors({"flow-field": "high"}),                         # non-numeric
        _priors({"flow-field": True}),                           # bool multiplier
        _priors({3: 2.0}),                                       # non-str tag
        {"version": 1, "priors": [("flow-field", 2.0)]},          # priors not a dict
        {"version": 1},                                           # priors missing
        "not a dict",                                            # not a dict
        ["a", "list"],                                           # not a dict
        42,                                                      # not a dict
    ],
    ids=[
        "bad-version", "version-str", "version-bool", "mult-high", "mult-low",
        "mult-nan", "mult-str", "mult-bool", "tag-not-str", "priors-not-dict",
        "priors-missing", "str", "list", "int",
    ],
)
def test_invalid_priors_ignored_with_recorded_reason(bad):
    plain = search_composition(_render, _cands())
    result = search_composition(_render, _cands(), trend_priors=bad)

    assert result["trend_priors_applied"] is False
    assert result["trend_priors"] is None
    assert isinstance(result["trend_priors_reason"], str)
    assert result["trend_priors_reason"].startswith("invalid schema:")
    # Identical scoring to no priors: winner and raw score entries match.
    assert result["winner"] == plain["winner"]
    assert result["scores"] == plain["scores"]
    assert all("total_unadjusted" not in s for s in result["scores"])


def test_none_is_byte_identical_to_v010_behavior():
    default = search_composition(_render, _cands())
    explicit_none = search_composition(_render, _cands(), trend_priors=None)

    assert explicit_none["winner"] == default["winner"]
    assert explicit_none["scores"] == default["scores"]
    assert explicit_none["trend_priors_applied"] is False
    assert explicit_none["trend_priors"] is None
    assert explicit_none["trend_priors_reason"] == "not provided"
    # No adjusted-total fields leak into the v0.1.0 score shape.
    for s in explicit_none["scores"]:
        assert set(s) == {"candidate", "coverage", "contrast", "label_legibility", "total"}


def test_unknown_preset_neutral_even_with_priors():
    # A preset absent from PRESET_FORMAT_TAGS is never penalized.
    cands = [{"preset": "mystery_preset", "rotation": 0, "basemap": "dark"}]
    result = search_composition(_render, cands, trend_priors=_priors({"flow-field": 4.0}))
    assert result["trend_priors_applied"] is True
    (s,) = result["scores"]
    assert s["trend_multiplier"] == pytest.approx(1.0)
    assert s["total"] == pytest.approx(s["total_unadjusted"])
    assert result["winner"]["preset"] == "mystery_preset"


def test_manifest_fields_present_when_applied():
    result = search_composition(_render, _cands(), trend_priors=_priors({"flow-field": 2.0}))
    assert result["trend_priors_applied"] is True
    assert result["trend_priors"]["version"] == 1
    assert result["trend_priors"]["priors"] == {"flow-field": 2.0}
    assert "trend_priors_reason" not in result


def test_empty_candidates_with_priors_is_honest():
    result = search_composition(_render, [], trend_priors=_priors({"flow-field": 2.0}))
    assert result["winner"] is None
    assert result["scores"] == []
    assert result["trend_priors_applied"] is True
