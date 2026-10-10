"""Tests for burned-in timestamps and title cards."""

import numpy as np
import pytest

from animate.overlays import (TextOverlayUnavailable, burn_timestamps,
                              make_title_card, pretty_timestamp)

mpl = pytest.importorskip("matplotlib")


def _frame(h=240, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_pretty_timestamp():
    assert pretty_timestamp("2026-09-16T00:00:00+00:00") == "16 Sep 2026 00:00 UTC"
    assert pretty_timestamp("") == ""
    assert pretty_timestamp("not-a-date") == "not-a-date"


def test_burn_timestamps_changes_pixels():
    out = burn_timestamps([_frame()], ["2026-09-16T00:00:00+00:00"])
    assert out[0].shape == (240, 160, 3)
    assert np.count_nonzero(out[0]) > 0


def test_burn_timestamps_positions():
    for pos in ("lower-left", "lower-right", "upper-left", "upper-right"):
        out = burn_timestamps([_frame()], ["2026-09-16T00:00:00+00:00"],
                              position=pos)
        assert np.count_nonzero(out[0]) > 0


def test_burn_timestamps_blank_left_alone():
    f = _frame()
    out = burn_timestamps([f], [""])
    assert np.array_equal(out[0], f)


def test_burn_timestamps_all_blank_no_mpl_needed(monkeypatch):
    # no labels -> never touches matplotlib
    import animate.overlays as ov
    monkeypatch.setattr(ov, "_require_mpl",
                        lambda: (_ for _ in ()).throw(AssertionError("no mpl")))
    f = _frame()
    out = ov.burn_timestamps([f], [""])
    assert np.array_equal(out[0], f)


def test_make_title_card():
    cards = make_title_card("Lake Michigan", "one week of currents",
                            320, 480, n_frames=4)
    assert len(cards) == 4
    assert cards[0].shape == (480, 320, 3)
    assert cards[0].dtype == np.uint8
    assert np.count_nonzero(cards[0]) > 0  # text drawn


def test_make_title_card_no_subtitle():
    cards = make_title_card("Title only", width=320, height=480, n_frames=1)
    assert cards[0].shape == (480, 320, 3)


def test_text_overlay_unavailable_without_mpl(monkeypatch):
    import animate.overlays as ov
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("blocked")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(TextOverlayUnavailable, match="matplotlib"):
        ov.make_title_card("t")
