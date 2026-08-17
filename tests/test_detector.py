"""The charge/lockout state machine.

This is the logic that decides whether the demo is usable, and it is completely
independent of how good the classifier is -- so it gets tested against a scripted
model that returns whatever we tell it.

Without this state machine the classifier emits a prediction every frame, and
holding a sign for two seconds means sixty domain expansions.

    python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detect
from detect import CONFIDENCE, SignDetector
from jjk.signs import BY_NAME

DT = 1 / 30
CLASSES = ["idle", "sukuna_domain", "gojo_domain"]
SOME_HAND = {"Left": np.zeros((21, 3), dtype=np.float32)}


class ScriptedModel:
    """Returns a near-certain probability for whichever label is currently set."""

    def __init__(self):
        self.label = "idle"

    def predict_proba(self, _features):
        probabilities = np.full(len(CLASSES), 0.01)
        probabilities[CLASSES.index(self.label)] = 0.98
        return probabilities.reshape(1, -1)


@pytest.fixture
def detector():
    """A SignDetector wired to the scripted model instead of a trained one.

    These class names aren't in signs.py, so the hand-count gate treats them as
    one-handed and stays out of the way -- leaving the charge/lockout logic
    tested in isolation, which is the point of this fixture.
    """
    instance = SignDetector.__new__(SignDetector)
    instance.model = ScriptedModel()
    instance.classes = list(CLASSES)
    instance.probabilities = np.zeros(len(CLASSES), dtype=np.float32)
    instance.charge = 0.0
    instance.locked = False
    instance.hand_gate = True
    instance.needs_two_hands = np.array(
        [bool(BY_NAME[name].two_handed) if name in BY_NAME else False for name in CLASSES],
        dtype=bool,
    )
    return instance


def hold(detector, label, seconds, hands=SOME_HAND):
    """Hold `label` for `seconds` of simulated time; returns activation count."""
    detector.model.label = label
    fires = 0
    for _ in range(int(seconds / DT)):
        *_, fired = detector.update(hands, DT)
        fires += bool(fired)
    return fires


def test_holding_a_sign_fires_exactly_once(detector):
    """The whole point. Five seconds of holding is 150 frames and one activation."""
    assert hold(detector, "sukuna_domain", 5.0) == 1


def test_continuing_to_hold_does_not_refire(detector):
    hold(detector, "sukuna_domain", 5.0)
    assert hold(detector, "sukuna_domain", 5.0) == 0


def test_returning_to_idle_rearms(detector):
    hold(detector, "sukuna_domain", 5.0)
    hold(detector, "idle", 0.5)
    assert not detector.locked
    assert hold(detector, "sukuna_domain", 3.0) == 1


def test_brief_flash_does_not_fire(detector):
    """Passing through a sign on the way to another one must not trigger it."""
    assert hold(detector, "sukuna_domain", 0.3) == 0


def test_charge_decays_on_release(detector):
    """Two half-holds must not add up to one activation."""
    hold(detector, "sukuna_domain", 0.6)
    charged = detector.charge
    assert charged > 0
    hold(detector, "idle", 0.6)
    assert detector.charge < charged * 0.1


def test_idle_never_fires(detector):
    assert hold(detector, "idle", 5.0) == 0


def test_no_hands_never_fires(detector):
    """Even if the model is confident, an empty frame can't be a sign."""
    assert hold(detector, "sukuna_domain", 5.0, hands={}) == 0


def test_fires_close_to_the_configured_charge_time(detector):
    detector.model.label = "gojo_domain"
    elapsed = 0.0
    for _ in range(int(6.0 / DT)):
        *_, fired = detector.update(SOME_HAND, DT)
        elapsed += DT
        if fired:
            break
    assert detector.locked, "never fired"
    # Slightly over CHARGE_SECONDS: the smoothed probability needs a few frames
    # to climb past the confidence threshold before charging even starts.
    assert detect.CHARGE_SECONDS <= elapsed <= detect.CHARGE_SECONDS + 0.6


"""The hand-count gate.

Malevolent Shrine is a sign MediaPipe merges into one detection; Gojo genuinely
needs two hands. Without a gate, a single-detection frame could be confidently
called Gojo -- half the evidence missing and no way for the classifier to know.
"""

REAL_CLASSES = ["idle", "gojo", "malevolent_shrine", "megumi", "sukuna", "yuta"]
ONE_HAND = {"Left": np.zeros((21, 3), dtype=np.float32)}
TWO_HANDS = {
    "Left": np.zeros((21, 3), dtype=np.float32),
    "Right": np.zeros((21, 3), dtype=np.float32),
}


class RealNameModel:
    def __init__(self, label):
        self.label = label

    def predict_proba(self, _features):
        probabilities = np.full(len(REAL_CLASSES), 0.01)
        probabilities[REAL_CLASSES.index(self.label)] = 0.95
        return probabilities.reshape(1, -1)


def gated_detector(label, hand_gate=True):
    instance = SignDetector.__new__(SignDetector)
    instance.model = RealNameModel(label)
    instance.classes = list(REAL_CLASSES)
    instance.probabilities = np.zeros(len(REAL_CLASSES), dtype=np.float32)
    instance.charge = 0.0
    instance.locked = False
    instance.hand_gate = hand_gate
    instance.needs_two_hands = np.array(
        [bool(BY_NAME[name].two_handed) if name in BY_NAME else False for name in REAL_CLASSES],
        dtype=bool,
    )
    return instance


def test_gojo_is_two_handed_and_malevolent_shrine_is_not():
    """Guards the assumption the gate rests on."""
    assert BY_NAME["gojo"].two_handed is True
    assert BY_NAME["malevolent_shrine"].two_handed is False


def test_two_handed_sign_cannot_win_on_a_single_detection():
    """The reported bug: one hand seen, and it confidently answers Gojo."""
    detector = gated_detector("gojo")
    for _ in range(30):
        label, *_ = detector.update(ONE_HAND, DT)
    assert label != "gojo"


def test_two_handed_sign_wins_normally_with_both_hands():
    detector = gated_detector("gojo")
    for _ in range(30):
        label, *_ = detector.update(TWO_HANDS, DT)
    assert label == "gojo"


def test_merged_sign_still_wins_on_a_single_detection():
    """Malevolent Shrine must not be caught by the gate -- one hand is its normal state."""
    detector = gated_detector("malevolent_shrine")
    for _ in range(30):
        label, confidence, *_ = detector.update(ONE_HAND, DT)
    assert label == "malevolent_shrine"
    assert confidence >= CONFIDENCE, "gate must not depress confidence below the firing threshold"


def test_gate_renormalises_so_confidence_still_means_something():
    """Zeroing classes without renormalising would quietly starve the survivors."""
    detector = gated_detector("malevolent_shrine")
    for _ in range(30):
        detector.update(ONE_HAND, DT)
    assert detector.probabilities.sum() == pytest.approx(1.0, abs=0.05)


def test_gate_can_be_disabled():
    detector = gated_detector("gojo", hand_gate=False)
    for _ in range(30):
        label, *_ = detector.update(ONE_HAND, DT)
    assert label == "gojo"


def test_single_bad_frame_does_not_cancel_the_charge(detector):
    """Smoothing exists so one dropped frame doesn't reset your build-up."""
    hold(detector, "sukuna_domain", 0.7)
    before = detector.charge
    detector.model.label = "idle"
    detector.update(SOME_HAND, DT)
    assert detector.charge > before * 0.8
