"""Day-one triage tool: is this sign actually trackable?

Hold a JJK sign in front of the camera and this tells you whether MediaPipe can
see it well enough to be worth building on. Two things kill a sign:

  DETECTION -- with the hands interlocked, MediaPipe intermittently loses one of
    them. If a hand vanishes for a third of your frames, no classifier will save
    you.

  STABILITY -- even when both hands are found, occluded fingers make the
    landmarks jitter. The model guesses where a hidden fingertip is, and its
    guess changes every frame. Training on jittery data gives you a classifier
    that flickers between labels.

Both are measured over a rolling window while you hold still, so the verdict
reflects a held pose rather than a lucky frame.

    python view_signs.py

Press q to quit, r to reset the meters between signs.
"""

import argparse
from collections import deque

import cv2
import numpy as np

from jjk.features import build_feature_vector
from jjk.tracker import FrameSource, draw_hands, draw_text
from jjk.features import hands_from_result

WINDOW = 30  # frames of history, about a second

# Jitter thresholds, in normalised hand-widths of landmark movement per frame.
# Calibrated so a cleanly-tracked open palm sits comfortably in "good".
JITTER_GOOD = 0.020
JITTER_OK = 0.045


class Meters:
    """Rolling detection-rate and jitter measurements."""

    def __init__(self, window=WINDOW):
        self.features = deque(maxlen=window)
        self.hand_counts = deque(maxlen=window)

    def reset(self):
        self.features.clear()
        self.hand_counts.clear()

    def update(self, hands):
        self.hand_counts.append(len(hands))
        self.features.append(build_feature_vector(hands) if hands else None)

    @property
    def samples(self):
        return len(self.hand_counts)

    def detection_rate(self, expected_hands):
        """Fraction of recent frames where all expected hands were found."""
        if not self.hand_counts:
            return 0.0
        hits = sum(1 for count in self.hand_counts if count >= expected_hands)
        return hits / len(self.hand_counts)

    def jitter(self):
        """Mean per-frame landmark movement while holding still.

        Consecutive frames only -- averaging against the window mean would
        conflate slow drift (you shifting position, which is harmless) with
        frame-to-frame flicker (the model guessing, which is not).
        """
        usable = [f for f in self.features if f is not None]
        if len(usable) < 3:
            return None
        stacked = np.stack(usable)
        deltas = np.abs(np.diff(stacked, axis=0))
        return float(deltas.mean())


def verdict(detection_rate, jitter):
    """Turn the two meters into one call you can act on."""
    if jitter is None or detection_rate < 0.5:
        return "POOR - hands not reliably found", (60, 60, 255)
    if detection_rate < 0.85:
        return "MARGINAL - hands drop out", (60, 200, 255)
    if jitter > JITTER_OK:
        return "POOR - landmarks too jittery", (60, 60, 255)
    if jitter > JITTER_GOOD:
        return "MARGINAL - some jitter, needs more data", (60, 200, 255)
    return "GOOD - build on this one", (120, 255, 120)


def bar(frame, origin, value, width=260, height=16, color=(120, 255, 120)):
    """A simple 0..1 filled meter."""
    x, y = origin
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 70, 70), -1)
    filled = int(width * max(0.0, min(1.0, value)))
    cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (220, 220, 220), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--hands",
        type=int,
        default=2,
        choices=(1, 2),
        help="how many hands the sign you're testing uses",
    )
    args = parser.parse_args()

    meters = Meters()

    with FrameSource(camera=args.camera) as source:
        for frame, result in source:
            hands = hands_from_result(result)
            meters.update(hands)
            draw_hands(frame, hands)

            detection_rate = meters.detection_rate(args.hands)
            jitter = meters.jitter()
            message, color = verdict(detection_rate, jitter)

            height = frame.shape[0]
            draw_text(
                frame,
                [
                    f"hands found: {len(hands)}/{args.hands}",
                    f"detection rate: {detection_rate * 100:5.1f}%",
                    f"jitter: {jitter:.4f}" if jitter is not None else "jitter: --",
                ],
            )
            bar(frame, (20, height - 90), detection_rate)
            if jitter is not None:
                # Draw jitter inverted so that, like detection rate, fuller is better.
                quality = 1.0 - min(1.0, jitter / (JITTER_OK * 2))
                bar(frame, (20, height - 62), quality, color=color)

            draw_text(frame, [message], origin=(20, height - 25), scale=0.8, color=color)
            draw_text(
                frame,
                ["hold a sign steady for ~1s   |   r: reset   q: quit"],
                origin=(20, height - 115),
                scale=0.5,
                color=(200, 200, 200),
            )

            cv2.imshow("JJK sign viability", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                meters.reset()


if __name__ == "__main__":
    main()
