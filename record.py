"""Collect training samples for each sign.

Press a sign's number key, get a short countdown to set your hands, then it
records a burst of frames. Repeat until every class has a few hundred samples.

    python record.py

The quality of this dataset decides the quality of the whole project, and there
are only three rules:

  MOVE WHILE RECORDING. Drift closer and further, rotate your wrists, shift left
    and right, tilt. A burst captured perfectly still teaches the classifier one
    exact pose, and it will fail the moment you sit differently.

  RECORD IDLE THE MOST. Key 0. Hands relaxed, reaching for the keyboard,
    scratching your face, half-way between two signs. Every frame that isn't a
    sign has to look like idle or the live demo will fire at random.

  DO SEVERAL SHORT BURSTS, not one long one. Get up, sit back down, change the
    lighting, record again. Variation between bursts is what makes it robust.

Keys: 0-8 record that sign  |  u undo last burst  |  q quit (saves on exit)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from jjk.features import FEATURE_DIM, build_feature_vector, hands_from_result
from jjk.signs import ALL_SIGNS, BY_KEY
from jjk.tracker import FrameSource, draw_hands, draw_text

DATASET = Path(__file__).resolve().parent / "data" / "samples.npz"

COUNTDOWN_SECONDS = 3
BURST_FRAMES = 150


def load_dataset():
    """Returns features, labels, and the burst each sample came from.

    The burst id is not bookkeeping -- it is what makes honest evaluation
    possible. Frames captured back-to-back are near-identical, so a random
    train/test split puts near-duplicates on both sides and reports an accuracy
    that has nothing to do with reality. train.py splits on burst instead.
    """
    if not DATASET.exists():
        return (
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int32),
        )
    stored = np.load(DATASET, allow_pickle=True)
    if "groups" in stored:
        return stored["X"], stored["y"], stored["groups"]
    # Datasets recorded before bursts were tracked: treat as one big burst.
    return stored["X"], stored["y"], np.zeros(len(stored["y"]), dtype=np.int32)


def save_dataset(X, y, groups):
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET, X=X, y=y, groups=groups)


def counts(y):
    if len(y) == 0:
        return {}
    labels, totals = np.unique(y, return_counts=True)
    return dict(zip(labels, totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=BURST_FRAMES, help="frames per burst")
    args = parser.parse_args()

    X, y, groups = load_dataset()
    next_burst = int(groups.max()) + 1 if len(groups) else 0
    X = list(X)
    y = list(y)
    groups = list(groups)
    print(f"Loaded {len(y)} existing samples from {DATASET}")

    # Recording state: either idle, counting down, or capturing.
    pending = None       # sign selected, waiting out the countdown
    countdown_until = 0.0
    capturing = None     # sign currently being captured
    burst = []
    burst_sizes = []     # lets `u` undo the last burst

    with FrameSource(camera=args.camera) as source:
        for frame, result in source:
            hands = hands_from_result(result)
            draw_hands(frame, hands)
            height = frame.shape[0]
            now = time.monotonic()

            if pending is not None:
                remaining = countdown_until - now
                if remaining <= 0:
                    capturing, pending, burst = pending, None, []
                else:
                    draw_text(
                        frame,
                        [f"{pending.display}   {int(remaining) + 1}"],
                        origin=(20, height // 2),
                        scale=2.0,
                        color=(60, 200, 255),
                    )

            if capturing is not None:
                # Frames with no hands at all are noise for every class except
                # idle, where an empty frame is a perfectly valid example.
                if hands or capturing.name == "idle":
                    burst.append(build_feature_vector(hands))
                progress = len(burst) / args.frames
                draw_text(
                    frame,
                    [f"RECORDING {capturing.display}  {len(burst)}/{args.frames}"],
                    origin=(20, height // 2),
                    scale=1.2,
                    color=(60, 60, 255),
                )
                cv2.rectangle(
                    frame,
                    (0, height - 8),
                    (int(frame.shape[1] * progress), height),
                    (60, 60, 255),
                    -1,
                )
                if len(burst) >= args.frames:
                    X.extend(burst)
                    y.extend([capturing.name] * len(burst))
                    groups.extend([next_burst] * len(burst))
                    burst_sizes.append(len(burst))
                    print(f"  +{len(burst)} {capturing.name}  (total {len(y)})")
                    next_burst += 1
                    capturing, burst = None, []

            tally = counts(np.array(y, dtype=object))
            lines = [
                f"[{sign.key}] {sign.display:<24} {tally.get(sign.name, 0):>5}"
                for sign in ALL_SIGNS
            ]
            draw_text(frame, lines, origin=(20, 36), scale=0.55, gap=24)
            draw_text(
                frame,
                ["0-8 record   u undo last burst   q save & quit"],
                origin=(20, height - 20),
                scale=0.55,
                color=(200, 200, 200),
            )

            cv2.imshow("JJK sign recorder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("u") and burst_sizes and capturing is None:
                size = burst_sizes.pop()
                del X[-size:]
                del y[-size:]
                del groups[-size:]
                next_burst -= 1
                print(f"  undid last burst (-{size}, total {len(y)})")
            elif key != 255 and capturing is None:
                sign = BY_KEY.get(chr(key))
                if sign is not None:
                    pending = sign
                    countdown_until = now + COUNTDOWN_SECONDS

    save_dataset(
        np.array(X, dtype=np.float32),
        np.array(y, dtype=object),
        np.array(groups, dtype=np.int32),
    )
    print(f"\nSaved {len(y)} samples in {next_burst} bursts to {DATASET}")
    for name, total in sorted(counts(np.array(y, dtype=object)).items()):
        print(f"  {name:<24} {total:>5}")


if __name__ == "__main__":
    main()
