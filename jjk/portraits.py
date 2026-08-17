"""Character portraits shown next to the hands when a sign is recognised.

Drop an image into assets/characters/ named after the sign -- so
`malevolent_shrine.gif` for the sign whose name is "malevolent_shrine" -- and it
appears automatically. Nothing else needs editing.

Animated GIFs and WebP play on loop at their own frame timing; PNG, JPG and
static WebP work too. Transparency is preserved in all of them. Signs with no
image simply don't show one, so a half-filled folder is fine while you're still
collecting art.

GIFs go through Pillow rather than OpenCV: OpenCV has no GIF decoder, and even
where it can read one it returns a single frame. Pillow also handles the parts of
the format that are genuinely fiddly -- palette transparency, and frames that
only store the pixels that changed since the last one.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "characters"

STATIC_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
ANIMATED_SUFFIXES = (".gif", ".webp")

TARGET_HEIGHT = 260   # portraits are scaled to this tall
MARGIN = 24
DEFAULT_FRAME_MS = 100  # some GIFs declare 0; browsers treat that as ~100ms

# A portrait is only on screen while you hold the sign -- a couple of seconds.
# Anything longer than that is animation nobody ever sees, paid for in memory and
# startup time: a 12-second GIF costs 160MB and shows you its first sixth.
# Long clips get evenly subsampled and retimed to fit, so the whole animation
# plays in the window you actually have.
MAX_LOOP_MS = 2500
MAX_FRAMES = 60


class Portrait:
    """One or more BGRA frames, with the timing to play them back."""

    def __init__(self, frames, durations_ms):
        self.frames = frames
        self.durations = [max(10, int(d)) for d in durations_ms]
        self.total_ms = sum(self.durations)

    @property
    def animated(self):
        return len(self.frames) > 1

    @property
    def shape(self):
        return self.frames[0].shape

    def frame_at(self, elapsed_seconds):
        """The frame that should be showing after `elapsed_seconds`, looping."""
        if not self.animated:
            return self.frames[0]
        position = (elapsed_seconds * 1000.0) % self.total_ms
        for frame, duration in zip(self.frames, self.durations):
            position -= duration
            if position < 0:
                return frame
        return self.frames[-1]


def _to_bgra(image):
    """Normalise any decoded image to 4-channel BGRA."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def _scaled(image):
    scale = TARGET_HEIGHT / image.shape[0]
    return cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _load_animated(path):
    frames, durations = [], []

    with Image.open(path) as image:
        count = getattr(image, "n_frames", 1)
        keep = min(count, MAX_FRAMES)
        # Evenly spaced picks across the whole clip, so a subsampled animation
        # still covers the same action rather than just its opening.
        wanted = set(np.linspace(0, count - 1, keep).astype(int).tolist())

        source_ms = 0
        for index, frame in enumerate(ImageSequence.Iterator(image)):
            source_ms += frame.info.get("duration") or DEFAULT_FRAME_MS
            if index not in wanted:
                # Still iterated -- GIF frames can be deltas against the previous
                # one, so skipping the decode entirely would corrupt what follows.
                # We just don't pay to convert and resize this one.
                continue
            # Converting in sequence lets Pillow apply the GIF's disposal rules,
            # so partial frames composite onto what came before instead of
            # appearing as fragments on transparency.
            rgba = np.array(frame.convert("RGBA"))
            frames.append(_scaled(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)))

    if not frames:
        return frames, durations

    # Retime the kept frames to fill whichever is shorter: the clip's own length
    # or the window a portrait is actually visible for.
    target_ms = min(source_ms, MAX_LOOP_MS) if source_ms else MAX_LOOP_MS
    per_frame = max(10, int(round(target_ms / len(frames))))
    return frames, [per_frame] * len(frames)


def _key_for(path):
    """Match a filename to a sign name forgivingly.

    "Crossed Hands.gif", "crossed-hands.gif" and "crossed_hands.gif" all resolve
    to the same sign. Filenames come from wherever the art came from, and a
    silently unmatched portrait looks identical to a broken one.
    """
    return path.stem.strip().lower().replace(" ", "_").replace("-", "_")


def load_portraits():
    """Read every portrait once at startup. Returns {sign_name: Portrait}."""
    portraits = {}
    if not ASSETS.exists():
        return portraits

    for path in sorted(ASSETS.iterdir()):
        suffix = path.suffix.lower()
        try:
            if suffix in ANIMATED_SUFFIXES:
                frames, durations = _load_animated(path)
            elif suffix in STATIC_SUFFIXES:
                # IMREAD_UNCHANGED keeps the alpha channel; without it PNG
                # transparency flattens onto black and every portrait gets a box.
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("decoder returned nothing")
                frames, durations = [_scaled(_to_bgra(image))], [DEFAULT_FRAME_MS]
            else:
                continue
        except Exception as error:
            print(f"  ! skipped {path.name}: {error}")
            continue

        if not frames:
            print(f"  ! skipped {path.name}: no frames")
            continue

        portrait = Portrait(frames, durations)
        portraits[_key_for(path)] = portrait
        detail = f"{len(frames)} frames, {portrait.total_ms}ms loop" if portrait.animated else "static"
        print(f"  loaded {path.name} ({detail})")

    # A portrait that loads fine but matches no sign is invisible at runtime and
    # looks exactly like one that failed to load, so say so out loud.
    from jjk.signs import ALL_SIGNS

    known = {sign.name for sign in ALL_SIGNS}
    for name in sorted(set(portraits) - known):
        print(f"  ! {name}: no sign by that name -- this portrait will never show")

    return portraits


def _paste(frame, image, x, y, opacity=1.0):
    """Alpha-composite a BGRA image onto the frame at (x, y), clipped to bounds."""
    height, width = image.shape[:2]

    # Clip against every edge -- a portrait drawn beside hands near the frame
    # border would otherwise index out of range or silently vanish.
    left, top = max(0, x), max(0, y)
    right, bottom = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if right <= left or bottom <= top:
        return

    patch = image[top - y : bottom - y, left - x : right - x]
    region = frame[top:bottom, left:right]

    alpha = (patch[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    blended = patch[:, :, :3].astype(np.float32) * alpha + region.astype(np.float32) * (1 - alpha)
    frame[top:bottom, left:right] = blended.astype(frame.dtype)


def draw_portrait(frame, portrait, hands, opacity=1.0, elapsed=None):
    """Draw the portrait beside the hands, flipping sides near the frame edge."""
    if portrait is None or opacity <= 0.01:
        return

    image = portrait.frame_at(time.monotonic() if elapsed is None else elapsed)

    frame_height, frame_width = frame.shape[:2]
    height, width = image.shape[:2]

    if hands:
        points = np.vstack([landmarks[:, :2] for landmarks in hands.values()])
        left_edge = float(points[:, 0].min()) * frame_width
        right_edge = float(points[:, 0].max()) * frame_width
        centre_y = float(points[:, 1].mean()) * frame_height

        # Prefer the right of the hands; flip to the left if there's no room.
        x = int(right_edge + MARGIN)
        if x + width > frame_width:
            x = int(left_edge - MARGIN - width)
        y = int(centre_y - height / 2)
    else:
        x = frame_width - width - MARGIN
        y = (frame_height - height) // 2

    x = max(MARGIN, min(x, frame_width - width - MARGIN))
    y = max(MARGIN, min(y, frame_height - height - MARGIN))

    _paste(frame, image, x, y, opacity)
