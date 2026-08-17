"""The sign vocabulary.

Edit this list freely -- it is the only place signs are defined. The recorder
reads it to know what to prompt you for, and the classifier learns exactly the
labels listed here.

`two_handed` is documentation for you while recording, not a constraint the code
enforces. `occlusion` is your day-one triage note: signs where the hands
interlock will give MediaPipe trouble, so start with the clean ones, get the
whole pipeline working end to end, and only then fight the hard ones.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sign:
    key: str          # keyboard key used to record this sign
    name: str         # label the classifier learns
    display: str      # what gets drawn on screen
    two_handed: bool
    occlusion: str    # "low" | "medium" | "high" -- expected tracking difficulty


# Record keys are unchanged from the original session -- each key still means the
# same physical sign, so muscle memory from recording still applies. Only the
# names changed, from descriptions of the hand shape to who the sign belongs to.
SIGNS = [
    # Two closed fists held apart. The easiest sign of the set: compact, stable,
    # and with no hand-on-hand occlusion for MediaPipe to lose track of.
    Sign("1", "megumi", "Megumi", True, "low"),

    # Palm thrust toward the camera, other hand a raised fist. The thrust hand is
    # foreshortened and reads much larger than the fist -- that size difference
    # survives normalisation and is most of what separates this from yuta.
    Sign("2", "sukuna", "Sukuna", True, "low"),

    # Flat palm forward with fingers together, fist held lower at the chest.
    # Watch this one against sukuna in the confusion matrix: same two hand
    # shapes, differing mainly in depth and how high the fist sits.
    Sign("3", "yuta", "Yuta", True, "low"),

    # Fingers interlace, and MediaPipe merges both hands into a single detection
    # essentially always -- measured at 100% of recorded frames. A consistently
    # merged detection is still a distinctive, stable signature, and the [1, 0]
    # presence flags become part of what identifies it.
    Sign("4", "gojo", "Gojo", False, "medium"),

    # Hands stay separate enough that both are found reliably.
    Sign("5", "malevolent_shrine", "Malevolent Shrine", True, "medium"),
]

# `two_handed` above is documentation. What the detector's hand gate actually
# uses is measured from the recordings during training -- see measure_hand_counts
# in train.py. These two disagreeing is precisely the bug that motivated the
# measurement, so the values here are kept in step but are not the authority.

# Deferred, not dropped:
#   - The face-referenced sign (hand over the eye, finger at the cheek) needs
#     face landmarks before it can work at all. Hand landmarks alone cannot tell
#     a splayed palm over your eye from a splayed palm anywhere else, because
#     absolute position is normalised away on purpose.
#   - The two heavily-interlocked signs, where the hands press together and
#     MediaPipe tends to merge them into one detection. Retry these with
#     view_signs.py once the pipeline is proven end to end.

# Recorded whenever your hands are doing nothing in particular. Without a
# negative class the classifier has to assign *some* sign to every frame, so it
# will fire constantly on your resting hands. This is the single most important
# class in the set -- record more of it than anything else, and record it while
# actually moving around.
IDLE = Sign("0", "idle", "--", False, "low")

ALL_SIGNS = [IDLE] + SIGNS

BY_KEY = {sign.key: sign for sign in ALL_SIGNS}
BY_NAME = {sign.name: sign for sign in ALL_SIGNS}


def display_name(name):
    sign = BY_NAME.get(name)
    return sign.display if sign else name
