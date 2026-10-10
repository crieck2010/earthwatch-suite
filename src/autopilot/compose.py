"""Automatic composition search ("render 3, ship 1").

Why this exists: a full-resolution aesthetic render is expensive; a thumbnail
is cheap. Instead of a human eyeballing every candidate, render N small
candidate thumbnails (different preset / rotation / basemap), score each with
three cheap, auditable heuristics, and ship the best. The scoring uses no
randomness, so identical thumbnails always rank identically.

Honest limitations:
- Coverage treats "data" as "pixels brighter than near-black". A dark-on-dark
  colormap or a basemap that is legitimately dark everywhere will score low
  coverage even when the render is fine. This is a heuristic, not a judgment.
- Contrast is the standard deviation of luminance clipped at 0.25. It rewards
  tonal spread; a deliberately flat aesthetic (fog, night) will score lower.
- Label legibility assumes labels are light text drawn over dark areas. Dark
  text on light basemaps inverts the assumption — the caller owns that choice.
- score_thumbnail never sees the data itself, only the render. Garbage data
  that renders with good coverage and contrast will score well. Pair this
  with autopilot.qa upstream of the render.
"""

from __future__ import annotations

import numpy as np

#: Default scoring weights. Coverage dominates because a frame with no visible
#: data is unrecoverable in refinement; contrast matters next; label
#: legibility is the smallest weight because labels can be re-placed later.
DEFAULT_WEIGHTS = {"coverage": 0.4, "contrast": 0.35, "label_legibility": 0.25}

#: Heuristic mapping from render preset name to survey-trends format tags.
#:
#: The sibling survey-trends radar scores content *formats* (e.g.
#: "flow-field") and reports per-tag priors; autopilot never imports
#: survey-trends — it just receives the priors dict from the caller and needs
#: to know which tags a given preset counts as. This mapping is a
#: hand-maintained heuristic, not ground truth: it is intentionally small and
#: auditable. A preset with no entry here gets no priors and scores
#: neutrally (multiplier 1.0) — unknown does NOT mean penalized.
PRESET_FORMAT_TAGS = {
    "dark_strands": ["particle-flow", "flow-field"],
    "dark_flow": ["flow-field", "particle-flow"],
}

#: Pixels whose brightest channel exceeds this are counted as "data".
_COVERAGE_BRIGHTNESS_FLOOR = 0.05

#: Luminance std is divided by this and clipped to [0, 1] for contrast.
_CONTRAST_NORMALIZER = 0.25


def _as_rgb_float01(thumbnail) -> np.ndarray:
    """Normalize a HxWx3/HxWx4 thumbnail to float64 RGB in [0, 1].

    Accepts uint8 0..255 or float 0..1. Float input with values > 1 is treated
    as 0..255 (a common sloppy-thumbnail mistake) rather than rejected.
    """
    arr = np.asarray(thumbnail)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(
            f"thumbnail must be HxWx3 or HxWx4, got shape {arr.shape}"
        )
    rgb = arr[..., :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float64) / 255.0
    rgb = rgb.astype(np.float64)
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(rgb) if rgb.size else 0.0
    if peak > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def score_thumbnail(thumbnail, label_boxes=None, weights=None) -> dict:
    """Score a single render thumbnail on three auditable heuristics.

    Args:
        thumbnail: HxWx3 or HxWx4 numpy array, float 0..1 or uint8 0..255.
        label_boxes: optional list of (x0, y0, x1, y1) pixel boxes where labels
            will sit. Score rewards dark backgrounds under labels.
        weights: optional override of DEFAULT_WEIGHTS (partial dicts OK).

    Returns:
        dict with "coverage", "contrast", "label_legibility", "total", and the
        "weights" actually used.

    Component definitions (all in [0, 1], higher is better):
    - coverage: fraction of pixels with max(R,G,B) > 0.05. Heuristic: data
      pixels are brighter than near-black backgrounds.
    - contrast: std of luminance (0.299R + 0.587G + 0.114B), divided by 0.25
      and clipped to [0, 1].
    - label_legibility: 1 - mean luminance inside label_boxes (labels should
      sit on dark areas); 0.5 neutral when no boxes are given.
    - total: weighted sum of the three.
    """
    rgb = _as_rgb_float01(thumbnail)
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    brightest = np.nanmax(rgb, axis=-1)
    coverage = float(np.mean(brightest > _COVERAGE_BRIGHTNESS_FLOOR))

    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    contrast = float(np.clip(np.nanstd(luminance) / _CONTRAST_NORMALIZER, 0.0, 1.0))

    if label_boxes:
        h, img_w = rgb.shape[:2]
        means = []
        for x0, y0, x1, y1 in label_boxes:
            xa, xb = max(0, int(x0)), min(img_w, int(x1))
            ya, yb = max(0, int(y0)), min(h, int(y1))
            if xb > xa and yb > ya:
                means.append(float(np.nanmean(luminance[ya:yb, xa:xb])))
        label_legibility = 1.0 - float(np.mean(means)) if means else 0.5
    else:
        label_legibility = 0.5

    total = (
        w["coverage"] * coverage
        + w["contrast"] * contrast
        + w["label_legibility"] * label_legibility
    )
    return {
        "coverage": coverage,
        "contrast": contrast,
        "label_legibility": label_legibility,
        "total": float(total),
        "weights": w,
    }


#: Trend-prior multipliers must sit in this range. 0.25 can quarter a score
#: (strong negative signal) and 4.0 can quadruple it; anything outside is
#: rejected as a likely data error rather than clamped, so bad radar output
#: never silently skews the search.
_MIN_PRIOR_MULTIPLIER = 0.25
_MAX_PRIOR_MULTIPLIER = 4.0


def _validate_trend_priors(trend_priors):
    """Validate a survey-trends radar priors dict (strict).

    Expected schema:
        {"version": 1, "generated_at": <ISO str>, "window_days": <int>,
         "priors": {<format-tag str>: <multiplier>}, "notes": [<str>]}

    Returns (validated_dict_or_None, reason_or_None). Strict and honest:
    any deviation from the schema means the priors are ignored entirely and
    the reason says why — autopilot never half-applies a malformed prior.
    """
    if trend_priors is None:
        return None, "not provided"
    if not isinstance(trend_priors, dict):
        return None, (
            f"invalid schema: expected a dict, got {type(trend_priors).__name__}"
        )
    version = trend_priors.get("version")
    if type(version) is not int or version != 1:
        return None, f"invalid schema: version must be 1, got {version!r}"
    priors = trend_priors.get("priors")
    if not isinstance(priors, dict):
        return None, "invalid schema: 'priors' must be a dict of tag -> multiplier"
    for tag, mult in priors.items():
        if not isinstance(tag, str):
            return None, (
                f"invalid schema: priors keys must be strings, got {tag!r}"
            )
        if isinstance(mult, bool) or not isinstance(mult, (int, float)):
            return None, (
                f"invalid schema: multiplier for tag {tag!r} must be a "
                f"number, got {mult!r}"
            )
        if not (_MIN_PRIOR_MULTIPLIER <= mult <= _MAX_PRIOR_MULTIPLIER):
            return None, (
                f"invalid schema: multiplier for tag {tag!r} out of range "
                f"[0.25, 4.0], got {mult!r}"
            )
    return trend_priors, None


def _prior_multiplier(preset, priors):
    """Mean prior multiplier for a candidate preset.

    Looks up the preset's format tags in PRESET_FORMAT_TAGS, averages the
    priors for those tags, and treats missing tags as 1.0 (neutral). A preset
    with no tags at all is neutral too — unknown presets are never penalized.
    """
    tags = PRESET_FORMAT_TAGS.get(preset, [])
    if not tags:
        return 1.0
    return float(sum(priors.get(tag, 1.0) for tag in tags) / len(tags))


def search_composition(render_thumbnail, candidates, weights=None, label_boxes=None, trend_priors=None) -> dict:
    """Render every candidate thumbnail, score each, return the winner.

    Args:
        render_thumbnail: caller-supplied callback
            ``render_thumbnail(preset, rotation, basemap) -> thumbnail array``.
            The caller owns rendering entirely, including errors: exceptions
            raised by the callback propagate unchanged so failures are never
            silently swallowed or re-ranked.
        candidates: list of dicts, each with keys "preset", "rotation",
            "basemap" (values are opaque and passed straight through to the
            callback). Extra keys are preserved untouched.
        weights: optional scoring-weight overrides, as in score_thumbnail.
        label_boxes: optional label boxes, as in score_thumbnail.
        trend_priors: optional survey-trends radar priors dict (strictly
            validated; see _validate_trend_priors). When valid, each
            candidate's total is multiplied by the mean prior of its preset's
            format tags (missing tags count as 1.0). Autopilot never imports
            survey-trends — the caller supplies the dict.

    Returns:
        {"winner": <candidate dict>, "scores": [...], "weights": {...},
         "n_candidates": N, "trend_priors_applied": bool,
         "trend_priors": <validated dict or None>} plus
        "trend_priors_reason" when priors were not applied.
        With an empty candidate list, "winner" is None (honest, not an error).

    Honest limitations of trend priors:
    - Priors amplify the heuristic score; they do not fix it. A strong
      multiplier can crown a genuinely weak render, so the per-score
      "total_unadjusted" field is kept for audit.
    - PRESET_FORMAT_TAGS is a hand-maintained heuristic mapping. A preset
      missing from it scores neutrally (1.0) rather than being penalized.
    - Invalid priors are ignored entirely with the reason recorded; they
      never partially apply.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    priors, priors_reason = _validate_trend_priors(trend_priors)
    priors_applied = priors is not None
    prior_map = priors["priors"] if priors_applied else {}

    scored = []
    for cand in candidates:
        thumb = render_thumbnail(cand["preset"], cand["rotation"], cand["basemap"])
        s = score_thumbnail(thumb, label_boxes=label_boxes, weights=w)
        entry = {
            "candidate": dict(cand),
            "coverage": s["coverage"],
            "contrast": s["contrast"],
            "label_legibility": s["label_legibility"],
            "total": s["total"],
        }
        if priors_applied:
            multiplier = _prior_multiplier(cand.get("preset"), prior_map)
            entry["total_unadjusted"] = s["total"]
            entry["trend_multiplier"] = multiplier
            entry["total"] = s["total"] * multiplier
        scored.append(entry)
    scored.sort(key=lambda d: d["total"], reverse=True)
    result = {
        "winner": scored[0]["candidate"] if scored else None,
        "scores": scored,
        "weights": w,
        "n_candidates": len(scored),
        "trend_priors_applied": priors_applied,
        "trend_priors": priors,
    }
    if not priors_applied:
        result["trend_priors_reason"] = priors_reason
    return result
