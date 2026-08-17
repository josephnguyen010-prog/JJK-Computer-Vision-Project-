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
  constructor({ video, canvas, onState = () => {}, modelUrl = "./model.json" }) {
    this.video = video;
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onState = onState;
    this.modelUrl = modelUrl;

    this.tracker = new HandTracker();
    this.classifier = null;
    this.detector = null;
    this.signs = {};
    this.running = false;
    this.frameHandle = null;
    this.lastFrameTime = null;
    this.showSkeleton = true;
  }

  async start() {
    const response = await fetch(this.modelUrl);
    if (!response.ok) {
      throw new Error(`Could not load the model (${response.status}).`);
    }
    const model = await response.json();

    this.classifier = new Classifier(model);
    this.signs = model.signs ?? {};
    const twoHanded = Object.fromEntries(
      Object.entries(this.signs).map(([name, sign]) => [name, sign.twoHanded])
    );
    this.detector = new SignDetector(this.classifier, { twoHanded });

    await this.tracker.start(this.video);

    this.running = true;
    this.lastFrameTime = null;
    this.loop();
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

    const result = this.tracker.detect(now);
    resizeCanvas(this.canvas);
    drawVideo(this.context, this.video);

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

    this.onState({
      label,
      display: this.displayName(label),
      confidence,
      charge,
      fired,
      handCount,
      recognised,
      probabilities: this.detector.probabilities,
      classes: this.classifier.classes,
    });
  };
}
