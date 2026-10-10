import matplotlib

matplotlib.use("Agg")

import numpy as np

from aesthetics.backgrounds import close, fig_to_rgba, new_canvas
from aesthetics.watermark import (
    draw_copyright,
    draw_data_source,
    draw_watermark,
    frame_furniture,
)


def test_watermark_text():
    fig, ax = new_canvas(400, 600, background="black")
    t = draw_watermark(ax, 0.94, 0.05, "my-ocean-page")
    assert t.get_text() == "my-ocean-page"
    assert t.get_ha() == "right"
    close(fig)


def test_copyright_format():
    fig, ax = new_canvas(400, 600, background="black")
    t = draw_copyright(ax, 0.94, 0.03, "Charles Rieck", 2026)
    assert t.get_text() == "© 2026 Charles Rieck"
    close(fig)


def test_data_source_multiline():
    fig, ax = new_canvas(400, 600, background="black")
    arts = draw_data_source(ax, 0.06, 0.02,
                            "NOAA GLOFS MODEL (LOOFS)\n16-23 SEPTEMBER 2026")
    assert len(arts) == 2
    assert arts[0].get_text() == "16-23 SEPTEMBER 2026"  # top line first
    close(fig)


def test_frame_furniture_composes_and_draws():
    fig, ax = new_canvas(400, 600, background="black")
    before = fig_to_rgba(fig).copy()
    frame_furniture(
        ax, brand="my-ocean-page", holder="Charles Rieck", year=2026,
        data_source="NOAA GLOFS MODEL (LOOFS)\n16-23 SEPTEMBER 2026",
        encoding="BRIGHTNESS = SPEED",
    )
    after = fig_to_rgba(fig)
    assert not np.array_equal(before, after)
    texts = [t.get_text() for t in ax.texts]
    assert "my-ocean-page" in texts
    assert "© 2026 Charles Rieck" in texts
    assert "BRIGHTNESS = SPEED" in texts
    close(fig)


def test_frame_furniture_without_encoding():
    fig, ax = new_canvas(400, 600, background="black")
    frame_furniture(ax, brand="b", holder="h", year=2026,
                    data_source="src")
    texts = [t.get_text() for t in ax.texts]
    assert "b" in texts
    close(fig)
