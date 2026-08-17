/**
 * Feature extraction, ported from jjk/features.py.
 *
 * This has to agree with the Python exactly. The model was trained on vectors
 * built by that file, so any difference here -- a different slot order, a
 * different origin, a missing flag -- is not a small inaccuracy. It feeds the
 * network something from a distribution it has never seen, and the predictions
 * become noise that still looks confident.
 *
 * tests/verify.mjs checks this against vectors built by the Python.
 */

export const NUM_LANDMARKS = 21;
export const WRIST = 0;
export const MIDDLE_MCP = 9;
export const FEATURE_DIM = NUM_LANDMARKS * 3 * 2 + 2 + 2;

const HANDEDNESS_VALUE = { Left: -1, Right: 1 };

/**
 * Size reference that survives the hand rotating: wrist to middle knuckle
 * barely changes as fingers move, so it tracks distance from the camera
 * rather than pose.
 */
function handScale(landmarks) {
  const wrist = landmarks[WRIST];
  const knuckle = landmarks[MIDDLE_MCP];
  const dx = knuckle[0] - wrist[0];
  const dy = knuckle[1] - wrist[1];
  const dz = knuckle[2] - wrist[2];
  // A fist viewed end-on can collapse this to nearly zero.
  return Math.max(Math.hypot(dx, dy, dz), 1e-6);
}

/**
 * Build the fixed-length feature vector for one frame.
 *
 * @param {{Left?: number[][], Right?: number[][]}} hands
 *   Landmarks as [x, y, z] triples, 21 per hand. Either or both may be absent.
 * @returns {Float32Array} length FEATURE_DIM
 */
export function buildFeatureVector(hands) {
  const features = new Float32Array(FEATURE_DIM);

  const detected = [];
  for (const side of ["Left", "Right"]) {
    if (hands[side]) detected.push({ side, landmarks: hands[side] });
  }
  if (detected.length === 0) return features;

  // Leftmost hand on screen goes first. Ordering by pixels rather than by
  // MediaPipe's Left/Right label means a flipped label -- which happens
  // constantly on symmetric poses -- cannot swap both halves of the vector.
  detected.sort((a, b) => a.landmarks[WRIST][0] - b.landmarks[WRIST][0]);

  // One shared origin and scale, so the offset between the hands survives
  // normalisation instead of being divided away per hand.
  const origin = [0, 0, 0];
  let scale = 0;
  for (const { landmarks } of detected) {
    origin[0] += landmarks[WRIST][0];
    origin[1] += landmarks[WRIST][1];
    origin[2] += landmarks[WRIST][2];
    scale += handScale(landmarks);
  }
  origin[0] /= detected.length;
  origin[1] /= detected.length;
  origin[2] /= detected.length;
  scale /= detected.length;

  const slotSize = NUM_LANDMARKS * 3;
  for (let slot = 0; slot < 2; slot += 1) {
    if (slot < detected.length) {
      const { side, landmarks } = detected[slot];
      const base = slot * slotSize;
      for (let i = 0; i < NUM_LANDMARKS; i += 1) {
        features[base + i * 3] = (landmarks[i][0] - origin[0]) / scale;
        features[base + i * 3 + 1] = (landmarks[i][1] - origin[1]) / scale;
        features[base + i * 3 + 2] = (landmarks[i][2] - origin[2]) / scale;
      }
      features[slotSize * 2 + slot] = 1;
      features[slotSize * 2 + 2 + slot] = HANDEDNESS_VALUE[side] ?? 0;
    }
  }

  return features;
}

/**
 * Pull a {Left, Right} map out of a MediaPipe Tasks result.
 *
 * Handedness can be reported twice for the same side when the hands overlap,
 * which happens constantly with interlocked signs -- keep the more confident
 * detection rather than letting a spurious duplicate overwrite a good one.
 */
export function handsFromResult(result) {
  const hands = {};
  const scores = {};
  const landmarkSets = result?.landmarks ?? [];
  const handednessSets = result?.handedness ?? [];

  for (let i = 0; i < landmarkSets.length; i += 1) {
    const category = handednessSets[i]?.[0];
    if (!category) continue;
    const label = category.categoryName;
    if (label !== "Left" && label !== "Right") continue;
    if (scores[label] !== undefined && scores[label] >= category.score) continue;
    hands[label] = landmarkSets[i].map((point) => [point.x, point.y, point.z]);
    scores[label] = category.score;
  }

  return hands;
}
