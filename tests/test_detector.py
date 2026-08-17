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
    instance.since_two_hands = float("inf")
    instance.hand_gate =True
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


# --- the hand-count gate -------------------------------------------------
#
# Gojo's fingers interlace, so MediaPipe merges both hands into a single
# detection; Megumi and the rest genuinely resolve two. Without a gate, a
# single-detection frame can be confidently called a two-handed sign -- half the
# evidence missing and no way for the classifier to know it.
#
# Which signs are which is measured from the recordings rather than declared,
# because getting it backwards vetoes a good sign on every frame and looks
# exactly like a recognition failure.

REAL_CLASSES = ["idle", "gojo", "malevolent_shrine", "megumi", "sukuna", "yuta"]

# Which signs genuinely need two detections, as measured from the recordings.
# Gojo's fingers interlace and MediaPipe merges both hands into one detection on
# 100% of its recorded frames, so gating it on two hands vetoes it permanently --
# which is exactly the bug that made the gate measure this rather than declare it.
MEASURED_TWO_HANDED = {
    "gojo": False,
    "malevolent_shrine": True,
    "megumi": True,
    "sukuna": True,
    "yuta": True,
    "idle": False,
}
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
    instance.since_two_hands = float("inf")
    instance.hand_gate =hand_gate
    instance.needs_two_hands = np.array(
        [MEASURED_TWO_HANDED[name] for name in REAL_CLASSES], dtype=bool
    )
    return instance


def test_signs_py_agrees_with_what_was_measured():
    """signs.py is documentation, but documentation that lies is worse than none.

    The gate reads the measured values out of the trained model; these flags only
    describe them. When they drifted apart, Gojo was vetoed on every frame and
    looked like a recognition failure.
    """
    for name, two_handed in MEASURED_TWO_HANDED.items():
        if name in BY_NAME:
            assert BY_NAME[name].two_handed is two_handed, f"{name} is out of step"


def test_idle_is_never_gated():
    """Gating idle would mean a single visible hand could not be idle, forcing
    some sign to win instead and firing the demo at random."""
    assert MEASURED_TWO_HANDED["idle"] is False


def test_two_handed_sign_cannot_win_on_a_single_detection():
    """The original bug: one hand seen, and it confidently answers a two-hand sign."""
    detector = gated_detector("megumi")
    for _ in range(30):
        label, *_ = detector.update(ONE_HAND, DT)
    assert label != "megumi"


def test_two_handed_sign_wins_normally_with_both_hands():
    detector = gated_detector("megumi")
    for _ in range(30):
        label, *_ = detector.update(TWO_HANDS, DT)
    assert label == "megumi"


def test_merged_sign_still_wins_on_a_single_detection():
    """Gojo must not be caught by the gate -- one detection is its normal state."""
    detector = gated_detector("gojo")
    for _ in range(30):
        label, confidence, *_ = detector.update(ONE_HAND, DT)
    assert label == "gojo"
    assert confidence >= CONFIDENCE, "gate must not depress confidence below the firing threshold"


def test_gate_renormalises_so_confidence_still_means_something():
    """Zeroing classes without renormalising would quietly starve the survivors."""
    detector = gated_detector("gojo")
    for _ in range(30):
        detector.update(ONE_HAND, DT)
    assert detector.probabilities.sum() == pytest.approx(1.0, abs=0.05)


def test_gate_can_be_disabled():
    detector = gated_detector("megumi", hand_gate=False)
    for _ in range(30):
        label, *_ = detector.update(ONE_HAND, DT)
    assert label == "megumi"


def test_merging_mid_gesture_does_not_veto_the_sign():
    """The gate asks about the gesture, not the frame.

    A two-handed sign can still lose one hand to occlusion for stretches at a
    time. Two hands seen at the start has to keep it eligible, or it becomes
    impossible to throw rather than merely harder.
    """
    detector = gated_detector("megumi")
    for _ in range(5):
        detector.update(TWO_HANDS, DT)
    for _ in range(30):
        label, *_ = detector.update(ONE_HAND, DT)
    assert label == "megumi"


def test_two_hand_evidence_expires():
    """One glimpse of two hands must not license a two-handed answer forever."""
    detector = gated_detector("megumi")
    for _ in range(5):
        detector.update(TWO_HANDS, DT)
    for _ in range(int(3.0 / DT)):
        label, *_ = detector.update(ONE_HAND, DT)
    assert label != "megumi"


def test_hands_leaving_frame_clears_the_evidence():
    """An empty frame ends the gesture, so nothing carries into the next one."""
    detector = gated_detector("megumi")
    for _ in range(5):
        detector.update(TWO_HANDS, DT)
    detector.update({}, DT)
    for _ in range(30):
        label, *_ = detector.update(ONE_HAND, DT)
    assert label != "megumi"


def test_single_bad_frame_does_not_cancel_the_charge(detector):
    """Smoothing exists so one dropped frame doesn't reset your build-up."""
    hold(detector, "sukuna_domain", 0.7)
    before = detector.charge
    detector.model.label = "idle"
    detector.update(SOME_HAND, DT)
    assert detector.charge > before * 0.8
