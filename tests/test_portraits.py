"""Portrait loading, animation timing, and compositing edge cases.

A portrait is drawn relative to wherever your hands are, so it routinely lands
partly outside the frame. Every clipping case here is a crash or a silent
no-draw if the bounds maths is wrong.

    python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jjk.portraits import (
    MAX_FRAMES,
    MAX_LOOP_MS,
    TARGET_HEIGHT,
    Portrait,
    draw_portrait,
    load_portraits,
)

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720


@pytest.fixture
def frame():
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def card(value=255, alpha=255, size=(300, 200)):
    image = np.zeros((*size, 4), dtype=np.uint8)
    image[:, :, :3] = value
    image[:, :, 3] = alpha
    return image


@pytest.fixture
def portrait():
    return Portrait([card()], [100])


def hand_at(x, y):
    return {"Left": np.column_stack([
        np.full(21, x, dtype=np.float32),
        np.full(21, y, dtype=np.float32),
        np.zeros(21, dtype=np.float32),
    ])}


# --- compositing ---------------------------------------------------------

def test_draws_something(frame, portrait):
    draw_portrait(frame, portrait, hand_at(0.4, 0.5))
    assert frame.sum() > 0


@pytest.mark.parametrize(
    "x, y",
    [
        (0.02, 0.5),   # hands at the left edge
        (0.98, 0.5),   # hands at the right edge -- portrait must flip sides
        (0.5, 0.02),   # top
        (0.5, 0.98),   # bottom
        (0.99, 0.99),  # corner
    ],
)
def test_stays_inside_the_frame_wherever_the_hands_are(frame, portrait, x, y):
    draw_portrait(frame, portrait, hand_at(x, y))
    assert frame.sum() > 0, "portrait vanished instead of being clamped"
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)


def test_no_hands_falls_back_to_a_fixed_position(frame, portrait):
    draw_portrait(frame, portrait, {})
    assert frame.sum() > 0


def test_missing_portrait_is_a_no_op(frame):
    draw_portrait(frame, None, hand_at(0.5, 0.5))
    assert frame.sum() == 0


def test_zero_opacity_draws_nothing(frame, portrait):
    draw_portrait(frame, portrait, hand_at(0.5, 0.5), opacity=0.0)
    assert frame.sum() == 0


def test_opacity_scales_the_blend(frame, portrait):
    draw_portrait(frame, portrait, hand_at(0.4, 0.5), opacity=0.5)
    half = frame.max()
    frame[:] = 0
    draw_portrait(frame, portrait, hand_at(0.4, 0.5), opacity=1.0)
    assert half < frame.max()


def test_transparent_pixels_leave_the_frame_untouched(frame):
    invisible = Portrait([card(alpha=0)], [100])
    frame[:] = 40
    draw_portrait(frame, invisible, hand_at(0.4, 0.5))
    assert (frame == 40).all()


# --- animation timing ----------------------------------------------------

@pytest.fixture
def animated():
    """Three frames, 100ms each, distinguishable by brightness."""
    return Portrait([card(60), card(160), card(250)], [100, 100, 100])


def test_static_portrait_ignores_time(portrait):
    assert np.array_equal(portrait.frame_at(0.0), portrait.frame_at(99.0))
    assert not portrait.animated


@pytest.mark.parametrize(
    "elapsed, expected",
    [(0.00, 0), (0.05, 0), (0.15, 1), (0.25, 2), (0.35, 0), (0.45, 1)],
)
def test_animation_advances_and_loops(animated, elapsed, expected):
    """Frame three at 250ms, then back to frame one at 350ms."""
    assert np.array_equal(animated.frame_at(elapsed), animated.frames[expected])


def test_uneven_frame_durations_are_respected():
    slow_then_fast = Portrait([card(60), card(250)], [300, 50])
    assert np.array_equal(slow_then_fast.frame_at(0.20), slow_then_fast.frames[0])
    assert np.array_equal(slow_then_fast.frame_at(0.32), slow_then_fast.frames[1])


def test_zero_duration_frames_do_not_hang():
    """Some GIFs declare 0ms; a naive loop would spin or divide by zero."""
    portrait = Portrait([card(60), card(250)], [0, 0])
    assert portrait.total_ms > 0
    assert portrait.frame_at(1.234) is not None


# --- loading from disk ---------------------------------------------------

def test_loads_animated_gif(tmp_path, monkeypatch):
    frames = [
        Image.new("RGBA", (100, 140), (255, 0, 0, 255)),
        Image.new("RGBA", (100, 140), (0, 255, 0, 255)),
        Image.new("RGBA", (100, 140), (0, 0, 255, 255)),
    ]
    path = tmp_path / "malevolent_shrine.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)

    monkeypatch.setattr("jjk.portraits.ASSETS", tmp_path)
    loaded = load_portraits()

    assert "malevolent_shrine" in loaded
    portrait = loaded["malevolent_shrine"]
    assert portrait.animated
    assert len(portrait.frames) == 3
    assert portrait.shape[0] == TARGET_HEIGHT, "frames should be scaled to target height"


def test_long_gif_is_subsampled_and_retimed(tmp_path, monkeypatch):
    """A long clip must be condensed, not truncated.

    A portrait is only on screen while you hold the sign, so a 20-second GIF is
    mostly animation nobody sees -- paid for in memory and startup time.
    """
    frames = [
        Image.new("RGBA", (80, 100), (index % 256, 0, 0, 255))
        for index in range(200)
    ]
    path = tmp_path / "interlocked.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    monkeypatch.setattr("jjk.portraits.ASSETS", tmp_path)
    portrait = load_portraits()["interlocked"]

    assert len(portrait.frames) <= MAX_FRAMES, "frame count not capped"
    assert portrait.total_ms <= MAX_LOOP_MS + 100, "loop not retimed to fit"
    assert portrait.animated


def test_short_gif_keeps_its_own_timing(tmp_path, monkeypatch):
    """Clips already inside the budget must not be sped up."""
    frames = [Image.new("RGBA", (80, 100), (index * 40, 0, 0, 255)) for index in range(6)]
    path = tmp_path / "palm_thrust.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    monkeypatch.setattr("jjk.portraits.ASSETS", tmp_path)
    portrait = load_portraits()["palm_thrust"]

    assert len(portrait.frames) == 6, "short clip should keep every frame"
    assert portrait.total_ms == pytest.approx(600, abs=60)


def test_filename_matching_is_forgiving(tmp_path, monkeypatch):
    """Art arrives named however it arrives."""
    for filename in ("Crossed Hands.gif", "palm-and-fist.png"):
        Image.new("RGBA", (80, 100), (255, 255, 255, 255)).save(tmp_path / filename)

    monkeypatch.setattr("jjk.portraits.ASSETS", tmp_path)
    loaded = load_portraits()

    assert "crossed_hands" in loaded
    assert "palm_and_fist" in loaded


def test_loads_static_png_and_skips_junk(tmp_path, monkeypatch):
    Image.new("RGBA", (100, 140), (255, 255, 255, 255)).save(tmp_path / "interlocked.png")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "broken.png").write_bytes(b"this is not a png")

    monkeypatch.setattr("jjk.portraits.ASSETS", tmp_path)
    loaded = load_portraits()

    assert set(loaded) == {"interlocked"}, "junk files must not become portraits"
    assert not loaded["interlocked"].animated


def test_missing_assets_directory_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("jjk.portraits.ASSETS", tmp_path / "nope")
    assert load_portraits() == {}
