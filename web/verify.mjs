/**
 * Check the JavaScript classifier against scikit-learn's own output.
 *
 *     node web/verify.mjs
 *
 * export_model.py bakes real feature vectors and the exact probabilities Python
 * produced for them into model.json. If this passes, the browser build and the
 * desktop build agree; if it drifts later, this fails loudly instead of the demo
 * quietly misreading signs.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { Classifier } from "./classifier.js";
import { buildFeatureVector, FEATURE_DIM } from "./features.js";

const here = dirname(fileURLToPath(import.meta.url));
const model = JSON.parse(readFileSync(join(here, "model.json"), "utf8"));
const classifier = new Classifier(model);

let failures = 0;
const ok = (message) => console.log(`  PASS  ${message}`);
const fail = (message) => {
  console.log(`  FAIL  ${message}`);
  failures += 1;
};

console.log("\n[1] shapes");
const shape = [model.scaler.mean.length, ...model.layers.map((l) => l.b.length)].join(" -> ");
console.log(`  network ${shape}, classes: ${model.classes.join(", ")}`);
if (model.scaler.mean.length === FEATURE_DIM) ok(`input matches FEATURE_DIM (${FEATURE_DIM})`);
else fail(`model wants ${model.scaler.mean.length} features, features.js builds ${FEATURE_DIM}`);

console.log("\n[2] probabilities match scikit-learn");
if (!model.tests?.length) {
  fail("no test vectors in model.json -- re-run export_model.py with a dataset present");
} else {
  let worst = 0;
  let worstLabel = null;
  let agreed = 0;

  for (const test of model.tests) {
    const got = classifier.predictProba(Float64Array.from(test.features));
    for (let i = 0; i < got.length; i += 1) {
      const delta = Math.abs(got[i] - test.expected[i]);
      if (delta > worst) {
        worst = delta;
        worstLabel = test.label;
      }
    }
    const bestIndex = got.indexOf(Math.max(...got));
    const expectedIndex = test.expected.indexOf(Math.max(...test.expected));
    if (bestIndex === expectedIndex) agreed += 1;
  }

  console.log(`  ${model.tests.length} vectors, largest probability difference ${worst.toExponential(2)}`);
  if (agreed === model.tests.length) ok(`predicted class agrees on all ${agreed}`);
  else fail(`predicted class differs on ${model.tests.length - agreed} of ${model.tests.length}`);

  // 1e-6 is far tighter than anything that could change a decision, while
  // leaving room for the last bits of floating-point ordering.
  if (worst < 1e-6) ok(`probabilities agree to ${worst.toExponential(2)}`);
  else fail(`probabilities drift by ${worst.toExponential(2)} (worst on ${worstLabel})`);
}

console.log("\n[3] feature builder");
const zeros = buildFeatureVector({});
if (zeros.length === FEATURE_DIM && zeros.every((v) => v === 0)) ok("no hands -> all zeros");
else fail("no hands should give an all-zero vector");

const fakeHand = (seed) =>
  Array.from({ length: 21 }, (_, i) => [
    0.4 + Math.sin(seed + i) * 0.05,
    0.5 + Math.cos(seed + i) * 0.05,
    Math.sin(seed * i) * 0.01,
  ]);

const both = { Left: fakeHand(1), Right: fakeHand(2) };
const vector = buildFeatureVector(both);
if (vector.length === FEATURE_DIM && vector.every(Number.isFinite)) ok("two hands -> finite vector");
else fail("two hands produced a bad vector");

const presence = [vector[126], vector[127]];
if (presence[0] === 1 && presence[1] === 1) ok("presence flags set for both slots");
else fail(`presence flags wrong: ${presence}`);

const oneHand = buildFeatureVector({ Right: fakeHand(2) });
if (oneHand[126] === 1 && oneHand[127] === 0) ok("one hand fills the first slot only");
else fail(`one hand presence flags wrong: ${[oneHand[126], oneHand[127]]}`);

// Slots are ordered by screen position, so relabelling the hands must leave the
// geometry untouched and flip only the handedness features.
const swapped = buildFeatureVector({ Left: both.Right, Right: both.Left });
let geometryDrift = 0;
for (let i = 0; i < FEATURE_DIM - 4; i += 1) {
  geometryDrift = Math.max(geometryDrift, Math.abs(vector[i] - swapped[i]));
}
if (geometryDrift < 1e-6) ok("swapping Left/Right labels leaves the geometry identical");
else fail(`handedness swap changed the geometry by ${geometryDrift}`);

console.log(failures === 0 ? "\nALL PASSED\n" : `\n${failures} FAILED\n`);
process.exit(failures === 0 ? 0 : 1);
