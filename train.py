"""Train the sign classifier.

    python train.py

Small MLP over normalised landmark vectors. It trains in seconds on CPU because
the input is 128 numbers per frame rather than an image -- that is the entire
reason this project is a weekend rather than a month.

The interesting part here is the evaluation. Frames recorded back-to-back are
near-duplicates of each other, so a normal random train/test split scatters
copies of the same instant across both sides and reports something close to
100% no matter how bad the model is. We split on burst instead: every frame from
a given recording lands wholly in train or wholly in test. That number is the
one that predicts how the live demo will feel.

Both numbers get printed, because the gap between them is worth seeing.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jjk.features import mirror_feature_vector
from jjk.signs import ALL_SIGNS, display_name

DATASET = Path(__file__).resolve().parent / "data" / "samples.npz"
MODEL = Path(__file__).resolve().parent / "models" / "classifier.joblib"

MIN_SAMPLES_PER_CLASS = 100
MIN_BURSTS_PER_CLASS = 3


def build_model():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=0,
        ),
    )


def measure_hand_counts(X, y, classes):
    """Decide which signs need two hands by looking at what was recorded.

    The presence flags are already in the feature vector, so how many hands
    MediaPipe actually found for a sign is a measurement, not a guess. Deciding
    it by hand is how a sign ends up flagged two-handed while producing a single
    merged detection every time -- which makes the detector's hand gate veto it
    on every frame, and no amount of tuning elsewhere will bring it back.

    Signs whose fingers interlace routinely merge into one detection. That is a
    fact about the sign as MediaPipe sees it, and the gate has to agree with it.
    """
    print("\nHands seen per sign, measured from the recordings:\n")
    hand_count = X[:, -4] + X[:, -3]
    measured = {}

    for name in classes:
        counts = hand_count[y == name]
        if len(counts) == 0:
            continue
        two_ratio = float((counts == 2).mean())

        if str(name) == "idle":
            # Idle is never gated, whatever the ratio says. It is the answer for
            # every frame that isn't a sign, including the one-handed and empty
            # ones -- gate it and a single visible hand could not be idle, so
            # something else would have to win and the demo would fire at random.
            measured[str(name)] = False
            print(f"  {display_name(name):<24} {two_ratio:6.0%} of frames show two   -> never gated")
            continue

        # Comfortably above half, so a sign that merges half the time is treated
        # as the single-detection sign it effectively is.
        measured[str(name)] = bool(two_ratio >= 0.6)
        shape = "two hands" if measured[str(name)] else "one detection"
        print(f"  {display_name(name):<24} {two_ratio:6.0%} of frames show two   -> {shape}")

    declared = {sign.name: sign.two_handed for sign in ALL_SIGNS}
    for name, actual in measured.items():
        if name == "idle":
            continue
        if name in declared and declared[name] != actual:
            print(
                f"\n  ! signs.py calls {name} "
                f"{'two-handed' if declared[name] else 'one-handed'}, but the "
                f"recordings say otherwise. The measurement wins."
            )

    return measured


def audit(y, groups):
    """Warn about the dataset problems that produce a good score and a bad demo."""
    problems = []
    labels, totals = np.unique(y, return_counts=True)

    if len(labels) < 2:
        problems.append("Need at least two classes -- record more signs.")

    for label, total in zip(labels, totals):
        bursts = len(np.unique(groups[y == label]))
        if total < MIN_SAMPLES_PER_CLASS:
            problems.append(f"{label}: only {total} samples (want {MIN_SAMPLES_PER_CLASS}+)")
        if bursts < MIN_BURSTS_PER_CLASS:
            problems.append(
                f"{label}: only {bursts} burst(s) (want {MIN_BURSTS_PER_CLASS}+ recorded "
                f"at different times, so it learns the sign and not the sitting position)"
            )

    if "idle" not in labels:
        problems.append(
            "No 'idle' class. Without it every frame gets assigned some sign and "
            "the live demo will fire constantly. Record key 0."
        )
    elif totals[labels == "idle"][0] < max(totals) * 0.5:
        problems.append(
            "Fewer idle samples than the biggest sign class. Idle should be the "
            "largest class -- it has to cover everything your hands do normally."
        )

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--force", action="store_true", help="train despite dataset warnings")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="skip left/right mirror augmentation (use if a sign's meaning depends on which hand does what)",
    )
    args = parser.parse_args()

    if not DATASET.exists():
        raise SystemExit(f"No dataset at {DATASET}. Run: python record.py")

    stored = np.load(DATASET, allow_pickle=True)
    X, y = stored["X"], stored["y"]
    groups = stored["groups"] if "groups" in stored else np.zeros(len(y), dtype=np.int32)

    print(f"{len(y)} samples, {len(np.unique(y))} classes, {len(np.unique(groups))} bursts\n")
    for label in np.unique(y):
        bursts = len(np.unique(groups[y == label]))
        print(f"  {display_name(label):<26} {(y == label).sum():>5} samples  {bursts:>3} bursts")

    problems = audit(y, groups)
    if problems:
        print("\nDataset warnings:")
        for problem in problems:
            print(f"  ! {problem}")
        if not args.force:
            raise SystemExit("\nFix these and re-record, or pass --force to train anyway.")
        print("\n--force given, training anyway.\n")

    if not args.no_mirror:
        # Reflect every sample so each sign is learned in both orientations --
        # you get the hands-swapped version of every recording without doing it.
        #
        # The mirror keeps its original's burst id. That matters: burst id is
        # what the split holds out, so a frame and its reflection always land on
        # the same side. Give them different ids and the test set fills up with
        # mirrored copies of the training data, which is leakage wearing a hat.
        mirrored = np.stack([mirror_feature_vector(sample) for sample in X])
        X = np.vstack([X, mirrored])
        y = np.concatenate([y, y])
        groups = np.concatenate([groups, groups])
        print(f"\nMirror augmentation: {len(y)} samples after reflecting ({len(y) // 2} recorded)")

    # The honest split: whole bursts held out, but stratified so every class is
    # actually represented in the test set. A plain group split is free to shove
    # every burst of some sign onto one side, and then you get a clean-looking
    # report that silently never tested that sign at all.
    bursts_per_class = min(
        len(np.unique(groups[y == label])) for label in np.unique(y)
    )
    n_splits = max(2, min(round(1 / args.test_size), bursts_per_class))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    missing = set(np.unique(y)) - set(np.unique(y[test_idx]))
    if missing:
        print(
            f"\nNote: {', '.join(sorted(missing))} did not land in the test split. "
            f"Record more separate bursts for those to get a real number on them."
        )

    model = build_model()
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    grouped_accuracy = (predictions == y[test_idx]).mean()

    # The flattering split, purely to show what it would have told you.
    leaky_train, leaky_test, leaky_y_train, leaky_y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=0, stratify=y
    )
    leaky_model = build_model()
    leaky_model.fit(leaky_train, leaky_y_train)
    leaky_accuracy = (leaky_model.predict(leaky_test) == leaky_y_test).mean()

    print(f"\n  random split (leaks near-duplicate frames): {leaky_accuracy:6.1%}  <- ignore this")
    print(f"  held-out bursts (what the demo will feel like): {grouped_accuracy:6.1%}  <- trust this")

    print("\nPer class, on held-out bursts:\n")
    print(classification_report(y[test_idx], predictions, zero_division=0))

    # Columns are numbered rather than truncated -- sign names share prefixes
    # ("megumi_domain" / "megumi_shikigami") and a clipped header turns the most
    # useful diagnostic you have into a guessing game.
    labels = sorted(np.unique(np.concatenate([y[test_idx], predictions])))
    matrix = confusion_matrix(y[test_idx], predictions, labels=labels)
    width = max(len(label) for label in labels) + 2
    print("Confusion (rows = actual, cols = predicted):\n")
    print(" " * width + "".join(f"{i:>7}" for i in range(len(labels))))
    for i, (label, row) in enumerate(zip(labels, matrix)):
        cells = "".join(f"{value:>7}" for value in row)
        print(f"{f'{i} {label}':<{width}}{cells}")
    print("\nOff-diagonal pairs are the signs worth re-recording together.")

    # Retrain on everything for the model that actually ships -- the split above
    # was for measurement, and there's no reason to throw away a quarter of the
    # data once the number is known.
    final = build_model()
    final.fit(X, y)

    two_handed = measure_hand_counts(X, y, final.classes_)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final, "classes": list(final.classes_), "two_handed": two_handed},
        MODEL,
    )
    print(f"\nSaved model to {MODEL}")

    if grouped_accuracy < 0.85:
        print(
            "\nUnder 85% on held-out bursts. Before touching the model, check the "
            "confusion matrix above -- if two signs trade off against each other "
            "they probably look too similar to MediaPipe, and more data for those "
            "two specifically will help more than any hyperparameter."
        )


if __name__ == "__main__":
    main()
