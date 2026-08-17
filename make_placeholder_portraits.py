"""Generate stand-in character portraits so the display works before you have art.

    python make_placeholder_portraits.py

Writes one transparent PNG per sign into assets/characters/, named after the
sign. Replace any of them with real artwork at the same filename and it gets
picked up automatically -- nothing in the code refers to these by name.

Real portraits should be transparent PNGs, roughly portrait-shaped, and will be
scaled to 260px tall. Anything much wider than it is tall will crowd the frame.
"""

from pathlib import Path

import cv2
import numpy as np

from jjk.signs import SIGNS

ASSETS = Path(__file__).resolve().parent / "assets" / "characters"

WIDTH, HEIGHT = 300, 420
RADIUS = 28

# One accent per sign, so they're instantly distinguishable on screen (BGR).
ACCENTS = [
    (90, 70, 210),
    (200, 140, 60),
    (90, 190, 120),
    (200, 90, 190),
    (70, 180, 220),
]


def rounded_mask(width, height, radius):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    for cx, cy in [
        (radius, radius),
        (width - radius, radius),
        (radius, height - radius),
        (width - radius, height - radius),
    ]:
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def wrap(text, limit=12):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def make_card(display, accent):
    card = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    card[:, :, :3] = (28, 26, 32)

    cv2.rectangle(card, (0, 0), (WIDTH, 90), accent, -1)
    cv2.rectangle(card, (12, 12), (WIDTH - 12, HEIGHT - 12), accent, 3)

    # A silhouette standing in for wherever the character will be.
    cv2.circle(card, (WIDTH // 2, 210), 54, accent, -1)
    cv2.ellipse(card, (WIDTH // 2, 360), (96, 84), 0, 180, 360, accent, -1)

    for i, line in enumerate(wrap(display)):
        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.putText(
            card,
            line,
            ((WIDTH - size[0]) // 2, 44 + i * 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        card, "PLACEHOLDER", (WIDTH // 2 - 78, HEIGHT - 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
    )

    card[:, :, 3] = rounded_mask(WIDTH, HEIGHT, RADIUS)
    return card


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for index, sign in enumerate(SIGNS):
        path = ASSETS / f"{sign.name}.png"
        if path.exists():
            print(f"  skip  {path.name} (already exists -- delete it to regenerate)")
            continue
        cv2.imwrite(str(path), make_card(sign.display, ACCENTS[index % len(ACCENTS)]))
        print(f"  wrote {path.name}")
    print(f"\nDrop real artwork into {ASSETS} using the same filenames.")


if __name__ == "__main__":
    main()
