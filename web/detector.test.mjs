/**
 * The charge/lockout state machine, tested the same way the Python is.
 *
 *     node web/detector.test.mjs
 *
 * These mirror tests/test_detector.py deliberately. The browser build and the
 * desktop build have to behave identically, and the only way to know that is to
 * ask them the same questions.
 */

import { SignDetector, CHARGE_SECONDS, CONFIDENCE, TWO_HAND_GRACE_SECONDS } from "./detector.js";

const DT = 1 / 30;
const CLASSES = ["idle", "gojo", "malevolent_shrine", "megumi"];
const TWO_HANDED = { gojo: true, megumi: true, malevolent_shrine: false };

let failures = 0;
const ok = (message) => console.log(`  PASS  ${message}`);
const fail = (message) => {
  console.log(`  FAIL  ${message}`);
  failures += 1;
};
const check = (condition, message) => (condition ? ok(message) : fail(message));

/** Stands in for the real classifier, returning whatever label we set. */
class ScriptedClassifier {
  constructor() {
    this.classes = CLASSES;
    this.label = "idle";
  }

  predictProba() {
    const probabilities = new Float64Array(CLASSES.length).fill(0.01);
    probabilities[CLASSES.indexOf(this.label)] = 0.97;
    return probabilities;
  }
}

function makeDetector(handGate = true) {
  const classifier = new ScriptedClassifier();
  const detector = new SignDetector(classifier, { twoHanded: TWO_HANDED, handGate });
  detector.scripted = classifier;
  return detector;
}

const FEATURES = new Float32Array(130);

/** Hold `label` for `seconds`; returns how many times it fired. */
function hold(detector, label, seconds, handCount = 2) {
  detector.scripted.label = label;
  let fires = 0;
  const frames = Math.round(seconds / DT);
  for (let i = 0; i < frames; i += 1) {
    if (detector.update(FEATURES, handCount, DT).fired) fires += 1;
  }
  return fires;
}

console.log("\n[1] charge and lockout");

let detector = makeDetector();
check(hold(detector, "gojo", 5) === 1, "holding a sign for 5s fires exactly once (not 150)");
check(hold(detector, "gojo", 5) === 0, "holding for another 5s does not re-fire");

hold(detector, "idle", 0.5);
check(!detector.locked, "returning to idle re-arms");
check(hold(detector, "gojo", 3) === 1, "fires again after re-arming");

detector = makeDetector();
check(hold(detector, "gojo", 0.3) === 0, "a 0.3s flash of a sign does not fire");

detector = makeDetector();
hold(detector, "gojo", 0.6);
const charged = detector.charge;
hold(detector, "idle", 0.6);
check(
  charged > 0 && detector.charge < charged * 0.1,
  `charge decays on release (${charged.toFixed(2)} -> ${detector.charge.toFixed(2)})`
);

detector = makeDetector();
check(hold(detector, "idle", 5) === 0, "idle never fires");

detector = makeDetector();
check(hold(detector, "gojo", 5, 0) === 0, "no hands in frame never fires");

detector = makeDetector();
detector.scripted.label = "gojo";
let elapsed = 0;
for (let i = 0; i < Math.round(6 / DT); i += 1) {
  elapsed += DT;
  if (detector.update(FEATURES, 2, DT).fired) break;
}
check(
  elapsed >= CHARGE_SECONDS && elapsed <= CHARGE_SECONDS + 0.6,
  `fires ${elapsed.toFixed(2)}s after the sign appears (charge ${CHARGE_SECONDS}s + EMA warm-up)`
);

detector = makeDetector();
hold(detector, "gojo", 0.7);
const before = detector.charge;
detector.scripted.label = "idle";
detector.update(FEATURES, 2, DT);
check(detector.charge > before * 0.8, "a single misclassified frame does not cancel the charge");

console.log("\n[2] hand-count gate");

detector = makeDetector();
let result = { label: null };
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 1, DT);
check(result.label !== "gojo", "a two-handed sign cannot win on a single detection");

detector = makeDetector();
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 2, DT);
detector.scripted.label = "gojo";
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 2, DT);
check(result.label === "gojo", "a two-handed sign wins normally with both hands");

detector = makeDetector();
detector.scripted.label = "malevolent_shrine";
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 1, DT);
check(result.label === "malevolent_shrine", "a merged sign still wins on a single detection");
check(
  result.confidence >= CONFIDENCE,
  "the gate does not depress confidence below the firing threshold"
);

let total = 0;
for (const value of detector.probabilities) total += value;
check(Math.abs(total - 1) < 0.05, "probabilities still sum to 1 after the gate renormalises");

detector = makeDetector(false);
detector.scripted.label = "gojo";
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 1, DT);
check(result.label === "gojo", "the gate can be disabled");

console.log("\n[3] the gate asks about the gesture, not the frame");

// A sign whose fingers interlace makes MediaPipe merge both hands for stretches
// at a time. Two hands seen at the start of the gesture has to keep it eligible,
// or the sign becomes impossible to throw rather than merely harder.
detector = makeDetector();
detector.scripted.label = "gojo";
for (let i = 0; i < 5; i += 1) detector.update(FEATURES, 2, DT);
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 1, DT);
check(result.label === "gojo", "merging to one hand mid-gesture does not veto the sign");

// ...but the evidence has to expire, or one glimpse of two hands would license
// a two-handed answer indefinitely.
detector = makeDetector();
detector.scripted.label = "gojo";
for (let i = 0; i < 5; i += 1) detector.update(FEATURES, 2, DT);
for (let i = 0; i < Math.round(3 / DT); i += 1) result = detector.update(FEATURES, 1, DT);
check(result.label !== "gojo", "the two-hand evidence expires after the grace period");

// Hands leaving the frame ends the gesture, so evidence must not carry into the
// next one.
detector = makeDetector();
detector.scripted.label = "gojo";
for (let i = 0; i < 5; i += 1) detector.update(FEATURES, 2, DT);
detector.update(FEATURES, 0, DT);
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 1, DT);
check(result.label !== "gojo", "hands leaving the frame clears the two-hand evidence");

// The original bug still has to stay fixed: a single detection with no two-hand
// evidence at all cannot be called a two-handed sign.
detector = makeDetector();
detector.scripted.label = "gojo";
for (let i = 0; i < 30; i += 1) result = detector.update(FEATURES, 1, DT);
check(result.label !== "gojo", "one hand and no evidence still cannot be a two-handed sign");

console.log(failures === 0 ? "\nALL PASSED\n" : `\n${failures} FAILED\n`);
process.exit(failures === 0 ? 0 : 1);
