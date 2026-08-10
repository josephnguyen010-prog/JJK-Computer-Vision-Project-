"""Turning raw MediaPipe landmarks into something a classifier can learn from.

The whole project rests on this file. Raw landmark coordinates are useless as
features: they move when you move across the frame, and they shrink when you
lean back. Two recordings of the same sign look completely different to a
classifier unless we normalise first.

The subtlety for JJK signs specifically is that most of them are TWO-handed, and
what separates them is often the relationship between the hands (interlocked vs
apart vs stacked) rather than the shape of either hand alone. So we deliberately
do NOT normalise each hand in its own little frame -- that would throw away the
exact information we need. Both hands get placed in one shared coordinate frame.
"""

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9  # knuckle of the middle finger

# 21 landmarks x 3 coords x 2 hands, plus a present/absent flag per hand.
FEATURE_DIM = NUM_LANDMARKS * 3 * 2 + 2

HAND_SLOTS = ("Left", "Right")


def _hand_scale(landmarks):
    """A size reference that survives the hand rotating.

    Wrist-to-middle-knuckle is the standard choice: it barely changes as the
    fingers move, so it tracks distance-from-camera and not pose.
    """
    span = np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST])
    # A closed fist viewed end-on can collapse this to nearly zero.
    return max(float(span), 1e-6)


def build_feature_vector(hands):
    """Flatten a frame's hands into a fixed-length vector.

    `hands` maps "Left"/"Right" to a (21, 3) array of landmarks. Either or both
    may be missing -- a missing hand contributes zeros plus a cleared flag, so
    one-handed and two-handed signs share the same feature space.
    """
    present = {side: hands.get(side) is not None for side in HAND_SLOTS}
    available = [hands[side] for side in HAND_SLOTS if present[side]]

    if not available:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # One shared origin and scale for both hands. With two hands we centre on
    # the midpoint between the wrists, which keeps the inter-hand offset intact
    # and encoded in the features rather than normalised away.
    origin = np.mean([hand[WRIST] for hand in available], axis=0)
    scale = float(np.mean([_hand_scale(hand) for hand in available]))

    parts = []
    for side in HAND_SLOTS:
        if present[side]:
            parts.append(((hands[side] - origin) / scale).ravel())
        else:
            parts.append(np.zeros(NUM_LANDMARKS * 3))

    parts.append([1.0 if present[side] else 0.0 for side in HAND_SLOTS])
    return np.concatenate(parts).astype(np.float32)


def hands_from_result(result):
    """Pull a {"Left": (21,3), "Right": (21,3)} dict out of a MediaPipe result.

    Handedness is reported per detection and can duplicate when the hands
    overlap -- which happens constantly with interlocked signs. We keep the
    more confident detection for each side rather than letting a spurious
    second "Left" overwrite a good one.
    """
    hands = {}
    scores = {}

    landmark_sets = getattr(result, "hand_landmarks", None) or []
    handedness_sets = getattr(result, "handedness", None) or []

    for landmarks, handedness in zip(landmark_sets, handedness_sets):
        if not handedness:
            continue
        label = handedness[0].category_name
        score = handedness[0].score
        if label not in HAND_SLOTS:
            continue
        if label in scores and scores[label] >= score:
            continue
        hands[label] = np.array(
            [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
        )
        scores[label] = score

    return hands
