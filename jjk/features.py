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

# 21 landmarks x 3 coords x 2 slots, a present/absent flag per slot, and the
# handedness MediaPipe reported for each slot.
FEATURE_DIM = NUM_LANDMARKS * 3 * 2 + 2 + 2

HAND_SLOTS = ("Left", "Right")

# Hands go into slots by screen position -- leftmost first -- rather than by
# MediaPipe's Left/Right label. On symmetric poses (two fists especially) the
# handedness label flips between frames, and with label-keyed slots that flip
# swaps both halves of the feature vector and looks like a totally different
# sign. Screen order is decided by the pixels, so it can't disagree with itself.
#
# The label isn't thrown away: it rides along as a soft feature, so the model can
# lean on it while it's reliable without being wrecked when it isn't.
HANDEDNESS_VALUE = {"Left": -1.0, "Right": 1.0}


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

    `hands` maps "Left"/"Right" to a (21, 3) array of landmarks; either or both
    may be missing. Hands are then re-slotted by screen position, so a missing
    hand always leaves the *second* slot empty and one- and two-handed signs
    share the same feature space.
    """
    detected = [(side, hands[side]) for side in HAND_SLOTS if hands.get(side) is not None]

    if not detected:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # Leftmost hand on screen goes first. See HANDEDNESS_VALUE above for why we
    # order by pixels rather than by MediaPipe's Left/Right label.
    detected.sort(key=lambda item: float(item[1][WRIST][0]))

    # One shared origin and scale for both hands. With two hands we centre on
    # the midpoint between the wrists, which keeps the inter-hand offset intact
    # and encoded in the features rather than normalised away.
    landmark_sets = [landmarks for _, landmarks in detected]
    origin = np.mean([hand[WRIST] for hand in landmark_sets], axis=0)
    scale = float(np.mean([_hand_scale(hand) for hand in landmark_sets]))

    slots = []
    presence = []
    handedness = []
    for index in range(2):
        if index < len(detected):
            side, landmarks = detected[index]
            slots.append(((landmarks - origin) / scale).ravel())
            presence.append(1.0)
            handedness.append(HANDEDNESS_VALUE.get(side, 0.0))
        else:
            slots.append(np.zeros(NUM_LANDMARKS * 3))
            presence.append(0.0)
            handedness.append(0.0)

    return np.concatenate(slots + [presence, handedness]).astype(np.float32)


def mirror_feature_vector(vector):
    """Reflect a feature vector left-to-right.

    Doing a sign with your hands swapped produces a mirrored feature vector, not
    a similar one, so a model trained on only one orientation won't recognise the
    other. Rather than record every sign twice, we reflect the training data:
    every recording teaches both orientations.

    Because features are already normalised relative to the midpoint between the
    wrists, reflecting is just negating the x components -- the origin negates
    along with the landmarks. The slots then swap, since whichever hand was
    leftmost on screen is now rightmost, and handedness flips with them.

    Only valid for signs that mean the same thing either way round. Every sign in
    the current vocabulary qualifies; a sign where left-vs-right actually
    distinguished it would need this turned off.
    """
    vector = np.asarray(vector, dtype=np.float32)
    slot_size = NUM_LANDMARKS * 3
    slots = [vector[:slot_size].copy(), vector[slot_size : slot_size * 2].copy()]
    presence = vector[-4:-2].copy()
    handedness = vector[-2:].copy()

    for slot in slots:
        slot[0::3] *= -1.0

    if presence[1] > 0:
        # Two hands: the leftmost is now the rightmost, so the slots exchange.
        slots.reverse()
        handedness = np.array([-handedness[1], -handedness[0]], dtype=np.float32)
    else:
        # One hand stays in the first slot; only its handedness flips.
        handedness = np.array([-handedness[0], 0.0], dtype=np.float32)

    return np.concatenate(slots + [presence, handedness]).astype(np.float32)


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
