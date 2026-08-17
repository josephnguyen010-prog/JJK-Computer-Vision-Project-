/**
 * Smoothed prediction plus the charge/lockout state machine.
 *
 * Ported from detect.py. The classifier emits a prediction every frame, so a
 * sign held for two seconds is sixty activations unless something stops it.
 * Three things do, and all three are visible on screen:
 *
 *   SMOOTHING   probabilities are averaged over recent frames, so one bad frame
 *               cannot fire anything.
 *   CHARGE TIME the sign has to stay predicted for about a second of real time.
 *               This is the debounce, and it is also the drama.
 *   LOCKOUT     after firing, nothing else fires until you return to idle.
 *
 * On top of that, a two-handed sign cannot be identified from a single
 * detection -- half the evidence is not there. Without that gate the classifier
 * answers anyway, and confidently mistakes a sign that merges into one hand for
 * a two-handed sign whose second hand happened to drop out.
 */

export const CHARGE_SECONDS = 1.0;
export const CONFIDENCE = 0.8;
export const SMOOTHING = 0.35;
export const DISCHARGE_RATE = 2.5;

export class SignDetector {
  /**
   * @param {import("./classifier.js").Classifier} classifier
   * @param {object} options
   * @param {Record<string, boolean>} options.twoHanded  sign name -> needs two hands
   * @param {boolean} [options.handGate=true]
   */
  constructor(classifier, { twoHanded = {}, handGate = true } = {}) {
    this.classifier = classifier;
    this.handGate = handGate;
    this.needsTwoHands = classifier.classes.map((name) => Boolean(twoHanded[name]));
    this.probabilities = new Float64Array(classifier.classes.length);
    this.charge = 0;
    this.locked = false;
  }

  reset() {
    this.probabilities.fill(0);
    this.charge = 0;
    this.locked = false;
  }

  /**
   * @param {Float32Array} features
   * @param {number} handCount
   * @param {number} dt seconds since the previous frame
   * @returns {{label: string, confidence: number, charge: number, fired: boolean}}
   */
  update(features, handCount, dt) {
    const frame = Float64Array.from(this.classifier.predictProba(features));

    if (this.handGate && handCount < 2) {
      // Rule out the impossible, then put the probability mass back so the
      // survivors are still judged against the usual confidence threshold
      // rather than being quietly penalised for existing.
      let total = 0;
      for (let i = 0; i < frame.length; i += 1) {
        if (this.needsTwoHands[i]) frame[i] = 0;
        total += frame[i];
      }
      if (total > 1e-6) {
        for (let i = 0; i < frame.length; i += 1) frame[i] /= total;
      }
    }

    for (let i = 0; i < frame.length; i += 1) {
      this.probabilities[i] += SMOOTHING * (frame[i] - this.probabilities[i]);
    }

    let best = 0;
    for (let i = 1; i < this.probabilities.length; i += 1) {
      if (this.probabilities[i] > this.probabilities[best]) best = i;
    }
    const label = this.classifier.classes[best];
    const confidence = this.probabilities[best];

    const charging = label !== "idle" && confidence >= CONFIDENCE && handCount > 0;

    if (charging && !this.locked) {
      this.charge = Math.min(1, this.charge + dt / CHARGE_SECONDS);
    } else {
      this.charge = Math.max(0, this.charge - (dt / CHARGE_SECONDS) * DISCHARGE_RATE);
    }

    let fired = false;
    if (this.charge >= 1 && !this.locked) {
      fired = true;
      this.locked = true;
      this.charge = 0;
    }

    // Releasing the sign is what re-arms it -- not a timer, so holding on
    // cannot double-fire.
    if (this.locked && (label === "idle" || handCount === 0)) {
      this.locked = false;
    }

    return { label, confidence, charge: this.charge, fired };
  }
}
