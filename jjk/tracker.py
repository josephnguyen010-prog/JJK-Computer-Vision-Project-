"""Camera + MediaPipe hand landmarker, wrapped so the rest of the code stays clean.

Everything that touches the MediaPipe API lives here. If MediaPipe changes its
surface, this is the only file that needs to move.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

# Which landmark pairs to join when drawing the skeleton: the palm outline
# followed by the five fingers.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

HAND_COLORS = {"Left": (255, 170, 60), "Right": (120, 120, 255)}


def ensure_model():
    """Fetch the landmarker weights on first run."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")
    return MODEL_PATH


def create_landmarker(num_hands=2, min_confidence=0.5):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera(index=0, width=1280, height=720):
    # CAP_DSHOW dodges the ~2s MSMF startup stall on Windows.
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Close anything else using the webcam, "
            f"or pass a different --camera index."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


class FrameSource:
    """Iterates webcam frames and runs the landmarker over each one.

    Yields `(frame, result)` where frame is a mirrored BGR image ready to draw
    on. Mirroring matters: it makes the preview behave like a mirror, and it
    means MediaPipe's "Left"/"Right" labels line up with your actual hands. As
    long as recording and inference both go through here, the convention stays
    consistent -- which is all the classifier cares about.
    """

    def __init__(self, camera=0, num_hands=2, min_confidence=0.5):
        self.capture = open_camera(camera)
        self.landmarker = create_landmarker(num_hands, min_confidence)
        self._start = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        self.capture.release()
        self.landmarker.close()
        cv2.destroyAllWindows()

    def __iter__(self):
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode wants a monotonically increasing timestamp; it uses the
            # gaps between them to decide when to re-detect vs. keep tracking.
            timestamp_ms = int((time.monotonic() - self._start) * 1000)
            yield frame, self.landmarker.detect_for_video(image, timestamp_ms)


def draw_hands(frame, hands, thickness=2):
    """Draw the skeleton for each detected hand, in place."""
    height, width = frame.shape[:2]
    for side, landmarks in hands.items():
        color = HAND_COLORS.get(side, (255, 255, 255))
        points = np.column_stack(
            [landmarks[:, 0] * width, landmarks[:, 1] * height]
        ).astype(int)
        for start, end in CONNECTIONS:
            cv2.line(frame, tuple(points[start]), tuple(points[end]), color, thickness)
        for point in points:
            cv2.circle(frame, tuple(point), thickness + 2, (255, 255, 255), -1)
            cv2.circle(frame, tuple(point), thickness + 1, color, -1)


def draw_text(frame, lines, origin=(20, 40), scale=0.7, color=(255, 255, 255), gap=32):
    """Draw a stack of text with a dark outline so it stays readable on any background."""
    x, y = origin
    for i, line in enumerate(lines):
        position = (x, y + i * gap)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
