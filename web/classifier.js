/**
 * The trained classifier, running in plain JavaScript.
 *
 * The network is 130 -> 128 -> 64 -> 6: three matrix multiplies and two
 * activations, about 25k multiply-adds per frame. That runs in well under a
 * millisecond, so the browser build needs no ML runtime at all -- no
 * TensorFlow.js, no ONNX, nothing to download but model.json.
 *
 * Weights come from export_model.py and are laid out exactly as scikit-learn
 * stores them, so the multiplication happens in the same order and the
 * floating-point rounding matches.
 */

function relu(values) {
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] < 0) values[i] = 0;
  }
  return values;
}

function softmax(values) {
  // Subtract the max before exponentiating. Without it a confident logit
  // overflows to Infinity and the whole vector comes back NaN.
  let max = -Infinity;
  for (const value of values) if (value > max) max = value;

  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    values[i] = Math.exp(values[i] - max);
    total += values[i];
  }
  for (let i = 0; i < values.length; i += 1) values[i] /= total;
  return values;
}

export class Classifier {
  /** @param {object} model parsed model.json */
  constructor(model) {
    this.classes = model.classes;
    this.mean = Float64Array.from(model.scaler.mean);
    this.scale = Float64Array.from(model.scaler.scale);
    this.layers = model.layers.map((layer) => ({
      weights: layer.w.map((row) => Float64Array.from(row)),
      bias: Float64Array.from(layer.b),
    }));

    // Float64 throughout, matching numpy's default. Float32 accumulates
    // visibly different rounding over 25k multiply-adds, which is enough to
    // flip the ranking when two signs are close.
    this.inputSize = this.mean.length;
  }

  /**
   * @param {ArrayLike<number>} features length must match the trained input
   * @returns {Float64Array} probability per class, in `this.classes` order
   */
  predictProba(features) {
    if (features.length !== this.inputSize) {
      throw new Error(
        `Expected ${this.inputSize} features, got ${features.length}. ` +
          `The feature builder and the model have drifted apart.`
      );
    }

    let activations = new Float64Array(this.inputSize);
    for (let i = 0; i < this.inputSize; i += 1) {
      activations[i] = (features[i] - this.mean[i]) / this.scale[i];
    }

    for (let index = 0; index < this.layers.length; index += 1) {
      const { weights, bias } = this.layers[index];
      const output = new Float64Array(bias.length);
      output.set(bias);

      // weights[i][j] is input i -> unit j, so we walk inputs on the outside.
      // Skipping zero activations is worth it: after a relu, most are zero.
      for (let i = 0; i < activations.length; i += 1) {
        const activation = activations[i];
        if (activation === 0) continue;
        const row = weights[i];
        for (let j = 0; j < output.length; j += 1) {
          output[j] += activation * row[j];
        }
      }

      activations = index === this.layers.length - 1 ? softmax(output) : relu(output);
    }

    return activations;
  }

  /** @returns {{label: string, confidence: number, probabilities: Float64Array}} */
  predict(features) {
    const probabilities = this.predictProba(features);
    let best = 0;
    for (let i = 1; i < probabilities.length; i += 1) {
      if (probabilities[i] > probabilities[best]) best = i;
    }
    return {
      label: this.classes[best],
      confidence: probabilities[best],
      probabilities,
    };
  }
}

export async function loadClassifier(url = "./model.json") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load model from ${url}: ${response.status}`);
  return new Classifier(await response.json());
}
