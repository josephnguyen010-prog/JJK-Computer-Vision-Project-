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


SIGNS = [
    Sign("1", "gojo_blue", "Lapse: Blue", False, "low"),
    Sign("2", "gojo_red", "Reversal: Red", False, "low"),
    Sign("3", "gojo_purple", "Hollow Purple", True, "medium"),
    Sign("4", "nobara_resonance", "Resonance", True, "medium"),
    Sign("5", "megumi_shikigami", "Divine Dogs", True, "medium"),
    Sign("6", "megumi_domain", "Chimera Shadow Garden", True, "high"),
    Sign("7", "sukuna_domain", "Malevolent Shrine", True, "high"),
    Sign("8", "gojo_domain", "Unlimited Void", True, "high"),
]

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
