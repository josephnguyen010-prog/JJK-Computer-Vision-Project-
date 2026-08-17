"""Relabel sign classes throughout the dataset, then retrain.

    python rename_signs.py --swap gojo malevolent_shrine
    python rename_signs.py --rename old_name new_name
    python rename_signs.py --swap a b --dry-run

Use this whenever a class turns out to be attached to the wrong sign. It rewrites
the labels in data/samples.npz and retrains, which takes seconds -- there is
never a reason to re-record over a naming mistake.

Every change is applied as one simultaneous mapping, never in sequence. A swap
done one rename at a time would have the first overwrite the class the second
still needed to find, silently merging two signs into one.

Recording is the irreplaceable part of this project, so the dataset is backed up
before anything is written.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "samples.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swap", nargs=2, action="append", metavar=("A", "B"), default=[],
        help="exchange two class names; repeatable",
    )
    parser.add_argument(
        "--rename", nargs=2, action="append", metavar=("OLD", "NEW"), default=[],
        help="rename one class; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would change")
    parser.add_argument("--skip-train", action="store_true", help="relabel only, don't retrain")
    args = parser.parse_args()

    if not args.swap and not args.rename:
        raise SystemExit("Nothing to do. Pass --swap A B or --rename OLD NEW.")

    if not DATASET.exists():
        raise SystemExit(f"No dataset at {DATASET}")

    stored = np.load(DATASET, allow_pickle=True)
    X, y, groups = stored["X"], stored["y"], stored["groups"]
    present = set(np.unique(y).tolist())

    mapping = {label: label for label in present}
    for first, second in args.swap:
        for name in (first, second):
            if name not in present:
                raise SystemExit(f"No class called {name!r}. Present: {', '.join(sorted(present))}")
        mapping[first], mapping[second] = second, first
    for old, new in args.rename:
        if old not in present:
            raise SystemExit(f"No class called {old!r}. Present: {', '.join(sorted(present))}")
        mapping[old] = new

    # Build the new labels in one pass from the original array. Applying the
    # changes one at a time would let an earlier rename clobber rows a later one
    # still needs to find -- which for a swap silently merges two classes.
    renamed = np.array([mapping[label] for label in y], dtype=object)

    print(f"{len(y)} samples\n")
    for old in sorted(present):
        new = mapping[old]
        count = int((y == old).sum())
        arrow = "unchanged" if old == new else f"-> {new}"
        print(f"  {old:<20} {count:>5} samples   {arrow}")

    collisions = [
        name for name in set(mapping.values())
        if sum(1 for value in mapping.values() if value == name) > 1
    ]
    if collisions:
        raise SystemExit(f"\nTwo classes would end up sharing a name: {collisions}")

    if args.dry_run:
        print("\nDry run -- nothing written.")
        return

    backup = DATASET.with_suffix(".npz.bak")
    shutil.copy2(DATASET, backup)
    print(f"\nBacked up dataset to {backup.name}")

    np.savez_compressed(DATASET, X=X, y=renamed, groups=groups)
    print(f"Rewrote labels in {DATASET.name}")

    if args.skip_train:
        print("\n--skip-train given. Run `python train.py` when ready.")
        return

    print("\nRetraining...\n")
    subprocess.run([sys.executable, str(ROOT / "train.py")], check=True)


if __name__ == "__main__":
    main()
