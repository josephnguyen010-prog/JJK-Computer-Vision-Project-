"""Split-screen composition: camera on the left, character on the right.

The window is two equal square panels side by side. The camera is letterboxed
into its half rather than cropped -- cropping a 16:9 feed to a square would cut
the sides off, which is exactly where your hands go when you throw a two-handed
sign.
"""

import cv2
import numpy as np

BACKGROUND = (20, 18, 24)
DIVIDER = (60, 56, 68)
DIVIDER_WIDTH = 2

# Each panel takes the camera's own shape, so the webcam half fills completely
# instead of sitting letterboxed inside a square. Both panels together are then
# twice as wide as one camera frame, which is more than fits comfortably on a
# normal screen -- so the finished canvas is scaled down to this width.
MAX_OUTPUT_WIDTH = 1600


def _composite(region, resized, background):
    """Draw a BGRA image over `region`, as cheaply as the alpha allows.

    The obvious implementation -- convert to float32, multiply by alpha, add,
    convert back -- costs about 50ms for a panel-sized image, which alone is
    slower than the hand tracking. It also does that work for nothing most of
    the time: GIF only supports 1-bit transparency, so every pixel of an
    animated portrait is either fully opaque or fully clear. Soft alpha is only
    possible from a PNG, and even then usually only around the edges.

    So there are three paths, cheapest first, and we take the best one that is
    exactly correct for the image at hand -- no approximation.
    """
    colour = resized[:, :, :3]
    if resized.shape[2] == 3:
        region[:] = colour
        return

    alpha = resized[:, :, 3]
    lowest = int(alpha.min())

    if lowest == 255:                       # fully opaque -- just copy
        region[:] = colour
        return

    if not ((alpha == 0) | (alpha == 255)).all():
        # Genuine soft alpha. Pay for the float blend, but only here.
        weight = alpha[:, :, None].astype(np.float32) / 255.0
        region[:] = (
            colour.astype(np.float32) * weight
            + np.asarray(background, dtype=np.float32) * (1.0 - weight)
        ).astype(region.dtype)
        return

    # Binary alpha: fill the background, then stamp the opaque pixels over it.
    region[:] = background
    np.copyto(region, colour, where=(alpha == 255)[:, :, None])


def fit_into(target, image, background=BACKGROUND):
    """Scale `image` to fit inside `target` preserving aspect, centred.

    Writes into `target` in place. `image` may be BGR or BGRA; alpha is
    composited against the background rather than dropped.
    """
    panel_height, panel_width = target.shape[:2]
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        target[:] = background
        return

    scale = min(panel_width / width, panel_height / height)
    new_width = max(1, min(panel_width, int(width * scale)))
    new_height = max(1, min(panel_height, int(height * scale)))

    # INTER_AREA is for shrinking; it is both slower and no better when growing.
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    x = (panel_width - new_width) // 2
    y = (panel_height - new_height) // 2

    # Clear only the letterbox bars. Blanking the whole panel first would repaint
    # every pixel the image is about to cover anyway.
    if y > 0:
        target[:y] = background
        target[y + new_height :] = background
    if x > 0:
        target[y : y + new_height, :x] = background
        target[y : y + new_height, x + new_width :] = background

    _composite(target[y : y + new_height, x : x + new_width], resized, background)


def panel_size(camera_shape, max_width=MAX_OUTPUT_WIDTH):
    """Size of one panel in the finished canvas."""
    height, width = camera_shape[:2]
    panel_width = min(width, max_width // 2)
    panel_height = max(1, round(height * panel_width / width))
    return panel_width, panel_height


def compose_split(camera, portrait_image=None, opacity=1.0, max_width=MAX_OUTPUT_WIDTH,
                  canvas=None):
    """Build the two-panel canvas: camera left, character right.

    Composed directly at output size. An earlier version built the canvas at
    twice the camera's width and downscaled the whole thing at the end, which
    meant allocating and resampling a 2560x720 image every frame -- most of a
    millisecond of pure waste per frame, for pixels nobody ever saw.

    Pass `canvas` to reuse a buffer across frames and skip the allocation too.
    """
    panel_width, panel_height = panel_size(camera.shape, max_width)
    total_width = panel_width * 2

    if canvas is None or canvas.shape[:2] != (panel_height, total_width):
        canvas = np.empty((panel_height, total_width, 3), dtype=np.uint8)

    if (camera.shape[0], camera.shape[1]) == (panel_height, panel_width):
        canvas[:, :panel_width] = camera[:, :, :3]
    else:
        # INTER_LINEAR rather than INTER_AREA: three times faster on a live
        # webcam downscale, and the difference is invisible at this size.
        canvas[:, :panel_width] = cv2.resize(
            camera[:, :, :3], (panel_width, panel_height), interpolation=cv2.INTER_LINEAR
        )

    right = canvas[:, panel_width:]
    if portrait_image is not None and opacity > 0.01:
        if opacity >= 0.999:
            fit_into(right, portrait_image)
        else:
            # Fade the panel up from the background rather than fading the art
            # against black, which would read as a dark flash on the way in.
            faded = np.empty_like(right)
            fit_into(faded, portrait_image)
            cv2.addWeighted(
                faded, opacity, np.full_like(faded, BACKGROUND), 1 - opacity, 0, dst=right
            )
    else:
        right[:] = BACKGROUND

    # Written as a slice, not cv2.line: a line of thickness 2 centres itself on
    # the coordinate and bleeds a column back into the camera panel. The slice
    # lands entirely on the character side.
    canvas[:, panel_width : panel_width + DIVIDER_WIDTH] = DIVIDER
    return canvas
