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
from jjk.signs import display_name
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

    def __init__(self, model_path=MODEL):
        if not model_path.exists():
            raise SystemExit(f"No model at {model_path}. Run: python record.py, then python train.py")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.probabilities = np.zeros(len(self.classes), dtype=np.float32)
        self.charge = 0.0
        self.locked = False

    def update(self, hands, dt):
        """Returns (label, confidence, charge, fired)."""
        features = build_feature_vector(hands).reshape(1, -1)
        frame_probabilities = self.model.predict_proba(features)[0]
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
    args = parser.parse_args()

    detector = SignDetector()
    particles = Particles()
    domain = DomainExpansion()
    show_debug = args.debug
    last_time = time.monotonic()

    with FrameSource(camera=args.camera) as source:
        for frame, result in source:
            now = time.monotonic()
            dt = min(now - last_time, 0.1)  # clamp so a stutter can't jump the charge
            last_time = now

            height, width = frame.shape[:2]
            hands = hands_from_result(result)
            label, confidence, charge, fired = detector.update(hands, dt)

            if fired:
                domain.trigger(display_name(label).upper())

            # Energy gathers harder the closer the sign is to firing.
            tips = fingertip_points(hands, width, height)
            center = palm_center(hands, width, height)
            color = ENERGY_COLOR * (1 - charge) + CHARGED_COLOR * charge

            if len(tips):
                ambient = 2 if charge == 0 else int(2 + 26 * charge)
                particles.emit(tips, ambient, color, speed=1.5 + 3.0 * charge)
            particles.update(attractor=center, pull=0.9 * charge)

            glow = np.zeros((height // GLOW_SCALE, width // GLOW_SCALE, 3), dtype=np.float32)
            particles.draw(glow)
            frame = apply_glow(frame, glow, strength=0.7 + 0.8 * charge)

            if show_debug:
                draw_hands(frame, hands, thickness=1)

            if charge > 0.02:
                draw_charge_ring(frame, center, charge, tuple(color.tolist()))

            frame = domain.render(frame, dt)

            if show_debug:
                ranked = np.argsort(detector.probabilities)[::-1][:3]
                draw_text(
                    frame,
                    [f"{detector.classes[i]:<20} {detector.probabilities[i]:.2f}" for i in ranked]
                    + [
                        f"charge {charge:.2f}   {'LOCKED' if detector.locked else 'ready'}",
                        f"hands  {len(hands)}",
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
                ["q quit   d debug"],
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


if __name__ == "__main__":
    main()
