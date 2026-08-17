"""Export the trained classifier to JSON for the browser build.

    python export_model.py

The model is a StandardScaler feeding a two-hidden-layer MLP -- three matrix
multiplies and two activations. That is small enough to run in JavaScript
directly, so the browser version needs no ML runtime at all: no TensorFlow.js,
no ONNX, nothing to download but this file.

Also writes a set of test vectors: real feature vectors paired with the exact
probabilities scikit-learn produces for them. The JavaScript port is checked
against those, so a porting mistake shows up as a failed comparison rather than
as a demo that quietly misreads every third sign.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from jjk.signs import ALL_SIGNS

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "classifier.joblib"
DATASET = ROOT / "data" / "samples.npz"
OUTPUT = ROOT / "web" / "model.json"

TEST_VECTORS = 40


def main():
    if not MODEL.exists():
        raise SystemExit(f"No model at {MODEL}. Run: python train.py")

    bundle = joblib.load(MODEL)
    pipeline = bundle["model"]
    measured = bundle.get("two_handed") or {}
    scaler = pipeline.named_steps["standardscaler"]
    net = pipeline.named_steps["mlpclassifier"]

    if net.activation != "relu" or net.out_activation_ != "softmax":
        raise SystemExit(
            f"The JS port implements relu + softmax; this model uses "
            f"{net.activation} + {net.out_activation_}."
        )

    export = {
        "classes": [str(c) for c in net.classes_],
        # Display names and hand counts travel with the model so the browser
        # build reads them rather than keeping a second copy that can drift
        # out of step with signs.py.
        # twoHanded comes from what training measured, not from the declaration
        # in signs.py -- see measure_hand_counts in train.py for why.
        "signs": {
            sign.name: {
                "display": sign.display,
                "twoHanded": bool(measured.get(sign.name, sign.two_handed)),
            }
            for sign in ALL_SIGNS
        },
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        # Weights are stored column-major-free: layer["w"][i][j] is the weight
        # from input i to unit j, matching scikit-learn's coefs_ layout exactly
        # so the JS can multiply in the same order and get the same rounding.
        "layers": [
            {"w": weight.tolist(), "b": bias.tolist()}
            for weight, bias in zip(net.coefs_, net.intercepts_)
        ],
        "activation": net.activation,
        "output": net.out_activation_,
    }

    # Test vectors drawn from real recordings rather than random noise: random
    # inputs land far outside the training distribution, where a subtly wrong
    # port can still agree by accident because everything saturates.
    if DATASET.exists():
        stored = np.load(DATASET, allow_pickle=True)
        X, y = stored["X"], stored["y"]
        rng = np.random.default_rng(0)
        picks = rng.choice(len(X), size=min(TEST_VECTORS, len(X)), replace=False)
        export["tests"] = [
            {
                "features": X[index].tolist(),
                "label": str(y[index]),
                "expected": pipeline.predict_proba(X[index].reshape(1, -1))[0].tolist(),
            }
            for index in picks
        ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(export), encoding="utf-8")

    shapes = " -> ".join(
        [str(len(export["scaler"]["mean"]))] + [str(len(layer["b"])) for layer in export["layers"]]
    )
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT}")
    print(f"  network      {shapes}")
    print(f"  classes      {', '.join(export['classes'])}")
    print(f"  test vectors {len(export.get('tests', []))}")
    print(f"  size         {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
