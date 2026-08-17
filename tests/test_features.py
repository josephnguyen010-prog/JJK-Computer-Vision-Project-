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

from jjk.features import (
    FEATURE_DIM,
    NUM_LANDMARKS,
    build_feature_vector,
    hands_from_result,
    mirror_feature_vector,
)


def hand(seed):
    return np.random.default_rng(seed).normal(0.5, 0.08, size=(21, 3)).astype(np.float32)


@pytest.fixture
def both_hands():
    return {"Left": hand(1), "Right": hand(2)}


@pytest.mark.parametrize(
    "hands, presence",
    [
        ({"Left": hand(1), "Right": hand(2)}, [1.0, 1.0]),
        # A lone hand always fills the first slot, whichever hand it is -- slots
        # are screen order, not identity.
        ({"Right": hand(2)}, [1.0, 0.0]),
        ({"Left": hand(1)}, [1.0, 0.0]),
        ({}, [0.0, 0.0]),
    ],
)
def test_shape_and_presence_flags(hands, presence):
    """One- and two-handed signs must share a feature space."""
    vector = build_feature_vector(hands)
    assert vector.shape == (FEATURE_DIM,)
    assert np.isfinite(vector).all()
    assert vector[-4:-2].tolist() == presence


def test_handedness_label_swap_does_not_scramble_the_geometry():
    """The reason slots are ordered by screen position.

    MediaPipe flips its Left/Right labels between frames on symmetric poses --
    two fists being the obvious case. If slots were keyed on that label, a flip
    would swap both halves of the vector and read as an entirely different sign.
    Ordering by screen position means the geometry is identical either way, and
    only the soft handedness features change.
    """
    left, right = hand(1), hand(2)
    original = build_feature_vector({"Left": left, "Right": right})
    mislabelled = build_feature_vector({"Left": right, "Right": left})

    geometry = slice(0, FEATURE_DIM - 4)
    assert original[geometry] == pytest.approx(mislabelled[geometry], abs=1e-6)
    assert original[-2:].tolist() == [-value for value in mislabelled[-2:].tolist()]


def test_slots_follow_screen_order():
    """Whichever hand is further left on screen fills the first slot."""
    left_hand = hand(1) - np.array([0.3, 0, 0], dtype=np.float32)
    right_hand = hand(2) + np.array([0.3, 0, 0], dtype=np.float32)
    vector = build_feature_vector({"Left": left_hand, "Right": right_hand})
    first_slot_x = vector[0]
    second_slot_x = vector[NUM_LANDMARKS * 3]
    assert first_slot_x < second_slot_x


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


def mirrored_landmarks(hands):
    """Reflect raw landmarks about the vertical axis, swapping handedness."""
    flip = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    swap = {"Left": "Right", "Right": "Left"}
    return {swap[side]: landmarks * flip for side, landmarks in hands.items()}


def test_mirroring_features_matches_mirroring_the_hands(both_hands):
    """The definitive property: reflecting the vector == reflecting reality.

    If these disagree, mirror augmentation is training the model on vectors that
    no real hand could produce.
    """
    from_vector = mirror_feature_vector(build_feature_vector(both_hands))
    from_landmarks = build_feature_vector(mirrored_landmarks(both_hands))
    assert from_vector == pytest.approx(from_landmarks, abs=1e-5)


def test_mirroring_one_hand_matches_mirroring_the_hand():
    one = {"Right": hand(3)}
    from_vector = mirror_feature_vector(build_feature_vector(one))
    from_landmarks = build_feature_vector(mirrored_landmarks(one))
    assert from_vector == pytest.approx(from_landmarks, abs=1e-5)


@pytest.mark.parametrize(
    "hands",
    [
        {"Left": hand(1), "Right": hand(2)},
        {"Right": hand(2)},
        {},
    ],
)
def test_mirroring_twice_is_the_identity(hands):
    original = build_feature_vector(hands)
    assert mirror_feature_vector(mirror_feature_vector(original)) == pytest.approx(
        original, abs=1e-6
    )


def test_mirroring_preserves_presence_flags(both_hands):
    original = build_feature_vector(both_hands)
    assert mirror_feature_vector(original)[-4:-2].tolist() == original[-4:-2].tolist()


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
