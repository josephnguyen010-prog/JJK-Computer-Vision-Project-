"""Photograph each sign next to the name the classifier gives it.

    python identify.py

Throw each of your signs in turn. When one is recognised and held steady, this
saves a snapshot to identify/<class_name>.jpg. After you've been through all of
them you have one labelled photo per class, showing exactly which physical sign
the model calls what.

This exists because the class names were guessed from screenshots before any
data was recorded, so the mapping between "the sign you throw" and "the name on
screen" was never verified. The model doesn't care what the classes are called --
only the display names and the portrait filenames need correcting, and both are
just labels.

Press q to quit, r to clear what you've captured and start over.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from detect import CONFIDENCE, SignDetector
from jjk.features import hands_from_result
from jjk.tracker import FrameSource, draw_hands, draw_text

OUTPUT = Path(__file__).resolve().parent / "identify"
HOLD_FRAMES = 12  # steady frames before we take the shot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    detector = SignDetector()
    expected = [name for name in detector.classes if name != "idle"]

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    captured = {}
    steady_label = None
    steady_count = 0
    last_time = None

    print(f"\nThrow each sign. Looking for: {', '.join(expected)}\n")

    with FrameSource(camera=args.camera) as source:
        import time

        for frame, result in source:
            now = time.monotonic()
            dt = 0.033 if last_time is None else min(now - last_time, 0.1)
            last_time = now

            hands = hands_from_result(result)
            label, confidence, _, _ = detector.update(hands, dt)
            clean = frame.copy()  # snapshot without the overlay drawn on it

            if label != "idle" and confidence >= CONFIDENCE and hands:
                steady_count = steady_count + 1 if label == steady_label else 1
                steady_label = label
            else:
                steady_label, steady_count = None, 0

            if steady_label and steady_count == HOLD_FRAMES and steady_label not in captured:
                path = OUTPUT / f"{steady_label}.jpg"
                labelled = clean.copy()
                draw_text(labelled, [steady_label], origin=(24, 56), scale=1.1)
                cv2.imwrite(str(path), labelled)
                captured[steady_label] = path
                print(f"  captured {steady_label}")

            draw_hands(frame, hands, thickness=1)

            lines = []
            for name in expected:
                mark = "[x]" if name in captured else "[ ]"
                lines.append(f"{mark} {name}")
            draw_text(frame, lines, origin=(20, 40), scale=0.6, gap=26)

            if steady_label:
                progress = min(1.0, steady_count / HOLD_FRAMES)
                status = (
                    f"already captured: {steady_label}"
                    if steady_label in captured
                    else f"hold... {steady_label}"
                )
                draw_text(
                    frame, [status],
                    origin=(20, frame.shape[0] - 70), scale=0.8,
                    color=(120, 255, 120) if progress >= 1 else (60, 200, 255),
                )

            remaining = [name for name in expected if name not in captured]
            draw_text(
                frame,
                [f"{len(captured)}/{len(expected)} captured   |   q quit   r reset"],
                origin=(20, frame.shape[0] - 24), scale=0.55, color=(200, 200, 200),
            )

            cv2.imshow("identify signs", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not remaining:
                break
            if key == ord("r"):
                captured.clear()
                shutil.rmtree(OUTPUT)
                OUTPUT.mkdir(parents=True)
                print("  reset")

    print(f"\nSaved {len(captured)} photos to {OUTPUT}")
    for name in sorted(captured):
        print(f"  {name}.jpg")
    if len(captured) < len(expected):
        missing = [n for n in expected if n not in captured]
        print(f"\nStill missing: {', '.join(missing)}  -- run again to add them.")


if __name__ == "__main__":
    main()
