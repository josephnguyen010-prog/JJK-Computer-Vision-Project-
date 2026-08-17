/**
 * Wires the camera, classifier and detector into a running demo.
 *
 * Exported as a class rather than as top-level code so the same logic backs both
 * the standalone page and the React component -- the only difference between
 * them is who owns the DOM nodes.
 */

import { Classifier } from "./classifier.js";
import { SignDetector } from "./detector.js";
import { buildFeatureVector, handsFromResult } from "./features.js";
import {
  drawChargeRing,
  drawHands,
  drawLabel,
  drawVideo,
  palmCenter,
  resizeCanvas,
} from "./renderer.js";
import { HandTracker } from "./tracker.js";

export class SignDemo {
  /**
   * @param {object} options
   * @param {HTMLVideoElement} options.video    hidden, holds the camera stream
   * @param {HTMLCanvasElement} options.canvas  the camera panel
   * @param {(state: object) => void} [options.onState]  called every frame
   */
  constructor({ video, canvas, onState = () => {}, modelUrl = "./model.json", handGate = true }) {
    this.video = video;
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onState = onState;
    this.modelUrl = modelUrl;
    this.handGate = handGate;

    this.tracker = new HandTracker();
    this.classifier = null;
    this.detector = null;
    this.signs = {};
    this.running = false;
    this.frameHandle = null;
    this.lastFrameTime = null;
    this.showSkeleton = true;
    this.peakHands = 0;
  }

  /**
   * Show the camera as soon as it opens, then load the models behind it.
   *
   * Both downloads start before the camera is awaited, so the network time
   * overlaps the permission prompt instead of following it. The render loop
   * begins the moment there is video, and detection switches on when the
   * landmarker is ready -- so the user sees themselves immediately rather than
   * staring at a black panel with the camera light on.
   */
  async start() {
    const modelRequest = fetch(this.modelUrl);
    const mediapipeReady = this.tracker.loadModel();
    // Nothing awaits this until later; without a handler a slow failure would
    // surface as an unhandled rejection before we get a chance to report it.
    mediapipeReady.catch(() => {});

    await this.tracker.openCamera(this.video);

    this.running = true;
    this.lastFrameTime = null;
    this.loop();

    const response = await modelRequest;
    if (!response.ok) {
      throw new Error(`Could not load the model (${response.status}).`);
    }
    const model = await response.json();

    this.classifier = new Classifier(model);
    this.signs = model.signs ?? {};
    const twoHanded = Object.fromEntries(
      Object.entries(this.signs).map(([name, sign]) => [name, sign.twoHanded])
    );
    this.detector = new SignDetector(this.classifier, { twoHanded, handGate: this.handGate });

    await mediapipeReady;
  }

  stop() {
    this.running = false;
    if (this.frameHandle) cancelAnimationFrame(this.frameHandle);
    this.frameHandle = null;
    this.tracker.stop();
  }

  displayName(label) {
    return this.signs[label]?.display ?? label;
  }

  loop = () => {
    if (!this.running) return;
    this.frameHandle = requestAnimationFrame(this.loop);

    const now = performance.now();
    // Clamp the step so a background tab or a stutter cannot jump the charge
    // straight to full on the first frame back.
    const dt = this.lastFrameTime === null ? 1 / 60 : Math.min((now - this.lastFrameTime) / 1000, 0.1);
    this.lastFrameTime = now;

    resizeCanvas(this.canvas);
    drawVideo(this.context, this.video);

    // The video is on screen well before the models finish downloading. Until
    // both are ready there is simply nothing to detect with, so draw the frame
    // and report that we are still loading.
    if (!this.detector || !this.tracker.ready) {
      this.onState({ loading: true, label: "idle", display: "", confidence: 0,
                     charge: 0, fired: false, handCount: 0, recognised: false });
      return;
    }

    const result = this.tracker.detect(now);

    // detect() returns null when the video has not advanced. Reuse the previous
    // hands rather than treating it as "hands gone", which would collapse the
    // charge every time the render loop outran the camera.
    if (result) this.hands = handsFromResult(result);
    const hands = this.hands ?? {};
    const handCount = Object.keys(hands).length;

    const features = buildFeatureVector(hands);
    const { label, confidence, charge, fired } = this.detector.update(features, handCount, dt);

    if (this.showSkeleton) drawHands(this.context, hands);
    drawChargeRing(this.context, palmCenter(this.context, hands), charge);

    const recognised = label !== "idle" && confidence >= 0.5 && handCount > 0;
    if (recognised) drawLabel(this.context, this.displayName(label));

    // Highest hand count since your hands last left the frame. If a two-handed
    // sign never reaches 2, the gate is what is blocking it -- worth showing
    // rather than leaving it to be guessed at, since a live count flickers too
    // fast to read.
    if (handCount === 0) this.peakHands = 0;
    else if (handCount > this.peakHands) this.peakHands = handCount;

    this.onState({
      loading: false,
      label,
      display: this.displayName(label),
      confidence,
      charge,
      fired,
      handCount,
      peakHands: this.peakHands,
      recognised,
      probabilities: this.detector.probabilities,
      classes: this.classifier.classes,
    });
  };
}
