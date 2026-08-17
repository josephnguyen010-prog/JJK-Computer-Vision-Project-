/**
 * Webcam plus the MediaPipe hand landmarker, in the browser.
 *
 * Everything touching MediaPipe lives here, mirroring jjk/tracker.py on the
 * desktop side. The WASM bundle and the landmarker model are the heavy part of
 * this page -- several megabytes -- so nothing here runs until start() is
 * called. Load it behind an explicit user action rather than on page load.
 */

const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
  "hand_landmarker/float16/1/hand_landmarker.task";

/** Landmark pairs to join when drawing the skeleton: palm, then five fingers. */
export const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export class HandTracker {
  constructor({ numHands = 2, minConfidence = 0.5 } = {}) {
    this.numHands = numHands;
    this.minConfidence = minConfidence;
    this.video = null;
    this.landmarker = null;
    this.stream = null;
    this.lastTimestamp = -1;
  }

  /**
   * Ask for the camera and load MediaPipe. Both can fail in ways worth
   * distinguishing, so the errors carry a `reason` the UI can act on.
   */
  async start(video) {
    this.video = video;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
    } catch (error) {
      const denied = error?.name === "NotAllowedError";
      const failure = new Error(
        denied
          ? "Camera permission was denied."
          : `Could not open the camera: ${error?.message ?? error}`
      );
      failure.reason = denied ? "permission" : "camera";
      throw failure;
    }

    video.srcObject = this.stream;
    await video.play();
    // Metadata can still be pending right after play(), and a zero-sized frame
    // makes MediaPipe throw rather than return nothing.
    if (!video.videoWidth) {
      await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
      });
    }

    const { FilesetResolver, HandLandmarker } = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22"
    );
    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
    this.landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: this.numHands,
      minHandDetectionConfidence: this.minConfidence,
      minHandPresenceConfidence: this.minConfidence,
      minTrackingConfidence: this.minConfidence,
    });

    return { width: video.videoWidth, height: video.videoHeight };
  }

  /**
   * Run the landmarker over the current video frame.
   * @returns {object|null} null when the video has not advanced yet
   */
  detect(timestampMs) {
    if (!this.landmarker || !this.video?.videoWidth) return null;
    // VIDEO mode rejects a timestamp that has not moved forward, which happens
    // whenever the render loop outruns the camera.
    if (timestampMs <= this.lastTimestamp) return null;
    this.lastTimestamp = timestampMs;
    return this.landmarker.detectForVideo(this.video, timestampMs);
  }

  stop() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.landmarker?.close();
    this.stream = null;
    this.landmarker = null;
    if (this.video) this.video.srcObject = null;
  }
}
