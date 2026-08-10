"""The feature vector has to survive you moving around.

If these fail, no amount of training data will save the classifier -- it would be
learning where you sat rather than what your hands did.

    python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jjk.features import FEATURE_DIM, build_feature_vector, hands_from_result


def hand(seed):
    return np.random.default_rng(seed).normal(0.5, 0.08, size=(21, 3)).astype(np.float32)


@pytest.fixture
def both_hands():
    return {"Left": hand(1), "Right": hand(2)}


@pytest.mark.parametrize(
    "hands, flags",
    [
        ({"Left": hand(1), "Right": hand(2)}, [1.0, 1.0]),
        ({"Right": hand(2)}, [0.0, 1.0]),
        ({"Left": hand(1)}, [1.0, 0.0]),
        ({}, [0.0, 0.0]),
    ],
)
def test_shape_and_presence_flags(hands, flags):
    """One- and two-handed signs must share a feature space."""
    vector = build_feature_vector(hands)
    assert vector.shape == (FEATURE_DIM,)
    assert np.isfinite(vector).all()
    assert vector[-2:].tolist() == flags


def test_invariant_to_position_and_distance(both_hands):
    """The same sign, further away and off to one side, is the same sign."""
    moved = {
        side: (landmarks + np.array([0.2, -0.1, 0.0], dtype=np.float32)) * 1.4
        for side, landmarks in both_hands.items()
    }
    assert build_feature_vector(both_hands) == pytest.approx(
        build_feature_vector(moved), abs=1e-3
    )


def test_inter_hand_distance_is_preserved(both_hands):
    """Sliding the hands apart must change the features.

    This is the one that matters for JJK signs specifically: several of them
    differ only in how the hands sit relative to each other, so normalising each
    hand in its own frame would make them indistinguishable.
    """
    apart = dict(both_hands)
    apart["Right"] = apart["Right"] + np.array([0.25, 0.0, 0.0], dtype=np.float32)
    difference = np.abs(build_feature_vector(both_hands) - build_feature_vector(apart)).max()
    assert difference > 0.1, "inter-hand offset was normalised away"


def test_degenerate_hand_does_not_divide_by_zero():
    """A fist viewed end-on can collapse the scale reference to nothing."""
    flat = {"Right": np.full((21, 3), 0.5, dtype=np.float32)}
    assert np.isfinite(build_feature_vector(flat)).all()


class FakeCategory:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class FakeResult:
    def __init__(self, detections):
        self.hand_landmarks = [
            [type("P", (), {"x": i, "y": i, "z": i})() for i in range(21)] for _ in detections
        ]
        self.handedness = [[FakeCategory(name, score)] for name, score in detections]


def test_duplicate_handedness_keeps_the_confident_detection():
    """Overlapping hands make MediaPipe report the same side twice."""
    parsed = hands_from_result(FakeResult([("Left", 0.40), ("Left", 0.95)]))
    assert set(parsed) == {"Left"}


def test_empty_result():
    assert hands_from_result(FakeResult([])) == {}
