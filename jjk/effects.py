"""Cursed energy: particles, glow, and the domain expansion hit.

All of this rides on landmark coordinates you already have every frame, so it
costs almost nothing to run and does most of the work of selling the demo.

The glow is done at quarter resolution -- blurring a full 720p frame every tick
is what would actually cost you framerate, and nobody can tell the difference
once it's blurred and screen-blended.
"""

import numpy as np
import cv2

FINGERTIPS = [4, 8, 12, 16, 20]
PALM = 9

GLOW_SCALE = 4  # render the glow layer at 1/4 size


class Particles:
    """A small additive particle system that emits from the fingertips.

    While a sign is charging, particles are pulled toward the palm instead of
    drifting away -- energy gathering rather than dissipating. That inward pull
    is what reads as "powering up" and it's a two-line difference.
    """

    def __init__(self, capacity=700):
        self.capacity = capacity
        self.position = np.zeros((capacity, 2), dtype=np.float32)
        self.velocity = np.zeros((capacity, 2), dtype=np.float32)
        self.life = np.zeros(capacity, dtype=np.float32)
        self.color = np.zeros((capacity, 3), dtype=np.float32)
        self.cursor = 0

    def emit(self, points, count, color, speed=2.0, spread=1.0):
        """Spawn `count` particles spread across the given screen points."""
        if len(points) == 0 or count <= 0:
            return
        picks = np.random.randint(0, len(points), size=count)
        slots = (self.cursor + np.arange(count)) % self.capacity
        self.cursor = int((self.cursor + count) % self.capacity)

        jitter = np.random.normal(0, 4.0 * spread, size=(count, 2))
        angles = np.random.uniform(0, 2 * np.pi, size=count)
        magnitudes = np.random.uniform(0.2, 1.0, size=count) * speed

        self.position[slots] = points[picks] + jitter
        self.velocity[slots] = np.column_stack(
            [np.cos(angles) * magnitudes, np.sin(angles) * magnitudes]
        )
        self.life[slots] = np.random.uniform(0.6, 1.0, size=count)
        self.color[slots] = color

    def update(self, attractor=None, pull=0.0, drag=0.94, fade=0.03):
        alive = self.life > 0
        if not alive.any():
            return
        if attractor is not None and pull > 0:
            offset = attractor - self.position[alive]
            distance = np.linalg.norm(offset, axis=1, keepdims=True) + 1e-6
            self.velocity[alive] += (offset / distance) * pull
        self.velocity[alive] *= drag
        self.position[alive] += self.velocity[alive]
        self.life[alive] -= fade

    def draw(self, glow):
        """Render into a (downscaled) glow layer."""
        alive = np.where(self.life > 0)[0]
        if len(alive) == 0:
            return
        height, width = glow.shape[:2]
        points = (self.position[alive] / GLOW_SCALE).astype(int)
        inside = (
            (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )
        points, indices = points[inside], alive[inside]
        if len(points) == 0:
            return
        intensity = self.life[indices][:, None]
        # Accumulate rather than assign, so overlapping particles build brightness.
        np.add.at(glow, (points[:, 1], points[:, 0]), self.color[indices] * intensity)


def apply_glow(frame, glow, strength=1.0):
    """Blur the glow layer and screen-blend it over the frame."""
    blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=3.0)
    full = cv2.resize(blurred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    full = np.clip(full * strength, 0, 255)
    # Screen blend keeps highlights from clipping to flat white the way adding does.
    base = frame.astype(np.float32)
    blended = 255.0 - (255.0 - base) * (255.0 - full) / 255.0
    return blended.astype(np.uint8)


def fingertip_points(hands, width, height):
    """Screen-space fingertip positions for every detected hand."""
    points = []
    for landmarks in hands.values():
        for index in FINGERTIPS:
            points.append([landmarks[index][0] * width, landmarks[index][1] * height])
    return np.array(points, dtype=np.float32) if points else np.empty((0, 2), dtype=np.float32)


def palm_center(hands, width, height):
    if not hands:
        return None
    centers = [[lm[PALM][0] * width, lm[PALM][1] * height] for lm in hands.values()]
    return np.array(centers, dtype=np.float32).mean(axis=0)


def draw_charge_ring(frame, center, charge, color):
    """A ring that closes as the sign charges -- the visible half of the debounce."""
    if center is None:
        return
    center = tuple(center.astype(int))
    radius = int(90 - 40 * charge)
    cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
    cv2.ellipse(
        frame, center, (radius + 10, radius + 10), -90, 0, int(360 * charge), color, 4, cv2.LINE_AA
    )


class DomainExpansion:
    """The activation hit: white flash, expanding shockwave, the domain's name."""

    def __init__(self, duration=1.6):
        self.duration = duration
        self.elapsed = duration + 1.0
        self.title = ""

    @property
    def active(self):
        return self.elapsed < self.duration

    def trigger(self, title):
        self.title = title
        self.elapsed = 0.0

    def render(self, frame, dt):
        if not self.active:
            return frame
        self.elapsed += dt
        progress = min(1.0, self.elapsed / self.duration)
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)

        # Flash, hottest at the instant of activation.
        flash = max(0.0, 1.0 - progress * 4.0)
        if flash > 0:
            frame = cv2.addWeighted(
                frame, 1.0 - flash, np.full_like(frame, 255), flash, 0
            )

        # Shockwave, thinning as it expands.
        wave_radius = int(progress * max(width, height) * 0.9)
        wave_alpha = max(0.0, 1.0 - progress)
        if wave_alpha > 0:
            overlay = frame.copy()
            cv2.circle(overlay, center, wave_radius, (255, 240, 220), max(1, int(28 * wave_alpha)))
            frame = cv2.addWeighted(overlay, wave_alpha * 0.8, frame, 1 - wave_alpha * 0.8, 0)

        # Darken toward the edges so the title reads.
        vignette = min(1.0, progress * 2.0) * 0.55
        if vignette > 0:
            frame = cv2.addWeighted(frame, 1 - vignette, np.zeros_like(frame), vignette, 0)

        title_alpha = min(1.0, progress * 3.0) * max(0.0, 1.0 - (progress - 0.6) / 0.4)
        if title_alpha > 0:
            scale = 1.6 + 0.4 * progress
            size, _ = cv2.getTextSize(self.title, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
            origin = (center[0] - size[0] // 2, center[1] + size[1] // 2)
            layer = frame.copy()
            cv2.putText(layer, self.title, origin, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(layer, self.title, origin, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
            frame = cv2.addWeighted(layer, title_alpha, frame, 1 - title_alpha, 0)

        return frame
