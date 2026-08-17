"""Live sign detection with charge-up and domain expansion.

    python detect.py

Hold a sign. Energy gathers at your fingertips, a ring closes, and when it
completes the domain fires.

That charge-up is the debounce wearing a costume. The classifier emits a
prediction every single frame, so a sign held for two seconds is sixty
activations unless something stops it. Three things do:

  SMOOTHING   -- predictions are averaged over recent frames, so one bad frame
                 can't fire anything.
  CHARGE TIME -- the sign has to stay predicted for ~1s of real time. This is
                 the debounce, and it's also the drama.
  LOCKOUT     -- after firing, nothing else can fire until you return to idle.
                 Without it, keeping your hands up re-triggers immediately.

Keys: q quit  |  d toggle the debug readout
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from jjk.effects import (
    DomainExpansion,
    Particles,
    apply_glow,
    draw_charge_ring,
    fingertip_points,
    palm_center,
)
from jjk.effects import GLOW_SCALE
from jjk.features import build_feature_vector, hands_from_result
from jjk.layout import compose_split
from jjk.portraits import draw_portrait, load_portraits
from jjk.signs import BY_NAME, display_name
from jjk.tracker import FrameSource, draw_hands, draw_text

MODEL = Path(__file__).resolve().parent / "models" / "classifier.joblib"

CHARGE_SECONDS = 1.0     # how long a sign must be held before it fires
CONFIDENCE = 0.80        # smoothed probability needed to charge at all
SMOOTHING = 0.35         # EMA weight on the newest frame
DISCHARGE_RATE = 2.5     # charge drains this many times faster than it builds

ENERGY_COLOR = np.array([255.0, 150.0, 60.0])   # BGR: cursed blue-white
CHARGED_COLOR = np.array([120.0, 90.0, 255.0])  # shifts red as it completes


class SignDetector:
    """Smoothed prediction plus the charge/lockout state machine."""

    def __init__(self, model_path=MODEL, hand_gate=True):
        if not model_path.exists():
            raise SystemExit(f"No model at {model_path}. Run: python record.py, then python train.py")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.probabilities = np.zeros(len(self.classes), dtype=np.float32)
        self.charge = 0.0
        self.locked = False

        # A two-handed sign cannot be identified from a single detection -- half
        # the evidence simply isn't there. Without this the classifier is free to
        # answer anyway, and it will: signs that MediaPipe merges into one hand
        # get confidently mistaken for two-handed ones whose second hand happened
        # to drop out. Which sign needs how many hands is already recorded in
        # signs.py, so the gate is metadata, not a special case.
        self.needs_two_hands = np.array(
            [bool(BY_NAME[name].two_handed) if name in BY_NAME else False
             for name in self.classes],
            dtype=bool,
        )
        self.hand_gate = hand_gate

    def update(self, hands, dt):
        """Returns (label, confidence, charge, fired)."""
        features = build_feature_vector(hands).reshape(1, -1)
        frame_probabilities = self.model.predict_proba(features)[0]

        if self.hand_gate and len(hands) < 2:
            # Rule out the impossible, then put the probability mass back so the
            # remaining classes are still judged against the usual confidence
            # threshold rather than being quietly penalised for existing.
            frame_probabilities = frame_probabilities * ~self.needs_two_hands
            total = frame_probabilities.sum()
            if total > 1e-6:
                frame_probabilities = frame_probabilities / total

        self.probabilities += SMOOTHING * (frame_probabilities - self.probabilities)

        index = int(np.argmax(self.probabilities))
        label = self.classes[index]
        confidence = float(self.probabilities[index])

        charging = label != "idle" and confidence >= CONFIDENCE and hands

        if charging and not self.locked:
            self.charge = min(1.0, self.charge + dt / CHARGE_SECONDS)
        else:
            self.charge = max(0.0, self.charge - dt / CHARGE_SECONDS * DISCHARGE_RATE)

        fired = False
        if self.charge >= 1.0 and not self.locked:
            fired = True
            self.locked = True
            self.charge = 0.0

        # Releasing the sign is what re-arms it -- not a timer, so you can't
        # double-fire by holding.
        if self.locked and (label == "idle" or not hands):
            self.locked = False

        return label, confidence, self.charge, fired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="start with the readout visible")
    parser.add_argument(
        "--layout",
        choices=("split", "overlay"),
        default="split",
        help="split: camera left, character right. overlay: character beside your hands.",
    )
    parser.add_argument(
        "--effects",
        choices=("minimal", "full"),
        default="minimal",
        help="minimal: just the charge ring. full: particles, glow and the domain expansion.",
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="hide the hand landmark overlay (toggle live with h)",
    )
    parser.add_argument(
        "--no-hand-gate",
        action="store_true",
        help="allow two-handed signs to be predicted from a single detection",
    )
    args = parser.parse_args()

    full_effects = args.effects == "full"
    show_skeleton = not args.no_skeleton

    detector = SignDetector(hand_gate=not args.no_hand_gate)
    particles = Particles() if full_effects else None
    domain = DomainExpansion() if full_effects else None
    portraits = load_portraits()
    print(f"Loaded {len(portraits)} character portraits: {', '.join(sorted(portraits)) or '(none)'}")
    show_debug = args.debug
    last_time = time.monotonic()

    # The portrait fades in with recognition rather than popping on the instant a
    # probability crosses the line -- at 30fps a hard cut flickers every time the
    # classifier wavers, and a fade reads as the system growing more certain.
    portrait_sign = None
    portrait_opacity = 0.0

    canvas = None                 # reused between frames to avoid reallocating
    frame_times = deque(maxlen=45)

    with FrameSource(camera=args.camera) as source:
        for frame, result in source:
            now = time.monotonic()
            dt = min(now - last_time, 0.1)  # clamp so a stutter can't jump the charge
            frame_times.append(dt)
            last_time = now

            height, width = frame.shape[:2]
            hands = hands_from_result(result)
            label, confidence, charge, fired = detector.update(hands, dt)

            center = palm_center(hands, width, height)
            color = ENERGY_COLOR * (1 - charge) + CHARGED_COLOR * charge

            if full_effects:
                if fired:
                    domain.trigger(display_name(label).upper())

                # Energy gathers harder the closer the sign is to firing.
                tips = fingertip_points(hands, width, height)
                if len(tips):
                    ambient = 2 if charge == 0 else int(2 + 26 * charge)
                    particles.emit(tips, ambient, color, speed=1.5 + 3.0 * charge)
                particles.update(attractor=center, pull=0.9 * charge)

                glow = np.zeros((height // GLOW_SCALE, width // GLOW_SCALE, 3), dtype=np.float32)
                particles.draw(glow)
                frame = apply_glow(frame, glow, strength=0.7 + 0.8 * charge)

            # Drawn on the camera frame before compositing, so it scales down
            # with it rather than sitting at full size over a shrunken feed.
            if show_skeleton:
                draw_hands(frame, hands, thickness=2)

            # Show the character as soon as the sign is recognised, not only when
            # it fires -- the point is feedback while you're still holding it.
            recognised = label != "idle" and confidence >= 0.5 and hands
            if recognised and label in portraits:
                portrait_sign = label
                portrait_opacity = min(1.0, portrait_opacity + dt * 4.0)
            else:
                portrait_opacity = max(0.0, portrait_opacity - dt * 3.0)
                if portrait_opacity <= 0.0:
                    portrait_sign = None

            if charge > 0.02:
                draw_charge_ring(frame, center, charge, tuple(color.tolist()))

            if full_effects:
                frame = domain.render(frame, dt)

            # Everything above draws on the camera image. Only now does the
            # character get placed, either beside the hands or on its own panel.
            portrait = portraits.get(portrait_sign) if portrait_sign else None
            if args.layout == "split":
                image = portrait.frame_at(now) if portrait is not None else None
                canvas = compose_split(frame, image, portrait_opacity, canvas=canvas)
                frame = canvas
            elif portrait is not None:
                draw_portrait(frame, portrait, hands, portrait_opacity, elapsed=now)

            height, width = frame.shape[:2]

            if show_debug:
                ranked = np.argsort(detector.probabilities)[::-1][:3]
                fps = len(frame_times) / sum(frame_times) if sum(frame_times) else 0.0
                draw_text(
                    frame,
                    [f"{detector.classes[i]:<20} {detector.probabilities[i]:.2f}" for i in ranked]
                    + [
                        f"charge {charge:.2f}   {'LOCKED' if detector.locked else 'ready'}",
                        f"hands  {len(hands)}    {fps:.0f} fps",
                    ],
                    scale=0.55,
                    gap=24,
                )
            else:
                draw_text(
                    frame,
                    [display_name(label) if label != "idle" and confidence > 0.5 else ""],
                    origin=(20, 44),
                    scale=0.8,
                )

            draw_text(
                frame,
                ["q quit   d debug   h hands"],
                origin=(20, height - 18),
                scale=0.5,
                color=(180, 180, 180),
            )

            cv2.imshow("Jujutsu Kaisen - hand signs", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
            if key == ord("h"):
                show_skeleton = not show_skeleton


if __name__ == "__main__":
    main()
