"""Assemble the browser build into a single deployable folder.

    python build_web.py                     # -> dist/
    python build_web.py --target "C:\\...\\react-portfolio-template\\public\\jjk"

In the repo the page lives in web/ and the media in assets/, so the page refers
to "../assets/...". Deployed, everything sits under one directory and those
paths have to become "assets/...". Rather than keep a second hand-edited copy
that drifts, this copies and rewrites in one step -- so redeploying after
changing a GIF or retraining is one command.

Run export_model.py first if the model has changed; this copies model.json and
signs.json as they are.
"""

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ASSETS = ROOT / "assets"

# Everything the page needs at runtime. verify.mjs and detector.test.mjs are
# deliberately absent -- they are development tools and the test file bundles
# nothing useful to a visitor.
WEB_FILES = [
    "index.html",
    "app.js",
    "audio.js",
    "classifier.js",
    "detector.js",
    "features.js",
    "renderer.js",
    "tracker.js",
    "model.json",
    "signs.json",
]

# README files document the folders for whoever is editing them; a visitor has
# no use for them and they would be served publicly.
EXCLUDE = {"README.md"}


def copy_assets(target):
    destination = target / "assets"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        ASSETS,
        destination,
        ignore=lambda _directory, names: [n for n in names if n in EXCLUDE],
    )
    return sum(1 for path in destination.rglob("*") if path.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=ROOT / "dist",
        help="where to write the build (default: dist/)",
    )
    args = parser.parse_args()
    target = args.target.resolve()

    missing = [name for name in WEB_FILES if not (WEB / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing from web/: {', '.join(missing)}\n"
            f"Run: python export_model.py"
        )

    target.mkdir(parents=True, exist_ok=True)

    for name in WEB_FILES:
        source = WEB / name
        text = source.read_text(encoding="utf-8")
        if name.endswith((".html", ".js")):
            # The only path change the flattening needs. Anchored on the quote so
            # it cannot touch an unrelated "../" elsewhere in the file.
            text, count = re.subn(r'(["\'`])\.\./assets/', r"\1assets/", text)
            if count:
                print(f"  {name}: rewrote {count} asset path(s)")
        (target / name).write_text(text, encoding="utf-8")

    asset_count = copy_assets(target)

    total = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    print(f"\nBuilt {target}")
    print(f"  {len(WEB_FILES)} page files, {asset_count} assets, {total / 1024 / 1024:.1f} MB total")

    biggest = sorted(
        (p for p in target.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )[:4]
    print("\n  largest files:")
    for path in biggest:
        print(f"    {path.stat().st_size / 1024 / 1024:5.1f} MB  {path.relative_to(target)}")


if __name__ == "__main__":
    main()
