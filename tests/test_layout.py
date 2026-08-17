"""Split-screen composition.

    python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jjk.layout import BACKGROUND, compose_split, fit_into

CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720


@pytest.fixture
def camera():
    """A distinctly-coloured camera frame, so we can tell which half is which."""
    return np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), (200, 100, 50), dtype=np.uint8)


@pytest.fixture
def portrait():
    image = np.zeros((260, 460, 4), dtype=np.uint8)
    image[:, :, :3] = (50, 220, 90)
    image[:, :, 3] = 255
    return image


def test_output_is_twice_as_wide_as_the_camera(camera):
    canvas = compose_split(camera, max_width=10_000)
    assert canvas.shape[1] == CAMERA_WIDTH * 2
    assert canvas.shape[0] == CAMERA_HEIGHT


def test_camera_half_is_pixel_exact(camera):
    """The webcam panel must not be resampled -- it's already the right size."""
    canvas = compose_split(camera, max_width=10_000)
    assert np.array_equal(canvas[:, :CAMERA_WIDTH], camera)


def test_scales_down_to_fit_the_screen(camera):
    canvas = compose_split(camera, max_width=1600)
    assert canvas.shape[1] == 1600
    # Aspect ratio preserved.
    assert canvas.shape[0] == pytest.approx(1600 * CAMERA_HEIGHT / (CAMERA_WIDTH * 2), abs=2)


def test_no_portrait_leaves_the_right_panel_empty(camera):
    canvas = compose_split(camera, None, max_width=10_000)
    right = canvas[:, CAMERA_WIDTH + 4 :]
    assert (right == np.array(BACKGROUND, dtype=np.uint8)).all()


def test_portrait_fills_the_right_panel(camera, portrait):
    canvas = compose_split(camera, portrait, max_width=10_000)
    right = canvas[:, CAMERA_WIDTH + 4 :]
    # The portrait's green should dominate somewhere in the right half.
    assert (right[:, :, 1] > 150).any()
    # ...and must not have leaked into the camera half.
    assert np.array_equal(canvas[:, :CAMERA_WIDTH], camera)


def test_opacity_fades_the_panel(camera, portrait):
    faint = compose_split(camera, portrait, opacity=0.2, max_width=10_000)
    solid = compose_split(camera, portrait, opacity=1.0, max_width=10_000)
    assert faint[:, CAMERA_WIDTH:].max() < solid[:, CAMERA_WIDTH:].max()


def test_zero_opacity_shows_nothing(camera, portrait):
    canvas = compose_split(camera, portrait, opacity=0.0, max_width=10_000)
    right = canvas[:, CAMERA_WIDTH + 4 :]
    assert (right == np.array(BACKGROUND, dtype=np.uint8)).all()


@pytest.mark.parametrize(
    "shape",
    [(100, 400, 4), (400, 100, 4), (260, 460, 3), (50, 50, 4)],
)
def test_fit_handles_any_portrait_shape(shape):
    """Tall, wide, square, with or without alpha -- none may overflow the panel."""
    panel = np.zeros((300, 500, 3), dtype=np.uint8)
    image = np.full(shape, 200, dtype=np.uint8)
    fit_into(panel, image)
    assert panel.shape == (300, 500, 3)


def test_fit_preserves_aspect_ratio():
    """A wide image must not be stretched to fill a square panel."""
    panel = np.zeros((400, 400, 3), dtype=np.uint8)
    image = np.full((100, 400, 4), 255, dtype=np.uint8)
    fit_into(panel, image)
    # Rows that differ from the background are the ones the image landed on --
    # the panel itself is filled with BACKGROUND, not zeros.
    background = np.array(BACKGROUND, dtype=np.uint8)
    filled_rows = np.where((panel != background).any(axis=(1, 2)))[0]
    # 400x100 scaled into a 400-wide panel is 100 tall, centred.
    assert len(filled_rows) == pytest.approx(100, abs=2)


def test_fit_composites_alpha_against_the_background():
    panel = np.zeros((300, 300, 3), dtype=np.uint8)
    transparent = np.zeros((100, 100, 4), dtype=np.uint8)
    transparent[:, :, :3] = 255
    transparent[:, :, 3] = 0
    fit_into(panel, transparent)
    assert (panel == np.array(BACKGROUND, dtype=np.uint8)).all()
