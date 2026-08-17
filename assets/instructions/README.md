# Reference images

One image per sign, showing a visitor what to do with their hands. They appear
on the page before the camera starts, and behind the **How to use** button after.

| Filename | Sign |
|---|---|
| `gojo.png` | Gojo |
| `sukuna.png` | Sukuna |
| `megumi.png` | Megumi |
| `yuta.png` | Yuta |
| `malevolent_shrine.png` | Malevolent Shrine |

`.png`, `.jpg`, `.jpeg`, `.webp` and `.gif` all work — the page tries each in
turn and uses the first that loads, so a mix of formats is fine. A sign with no
image yet shows a "no reference image" placeholder rather than a broken icon,
so you can add them one at a time.

## What works well here

- **Square-ish images.** The card frames are square and the image is contained
  inside, so a very wide or tall picture will letterbox.
- **A photo of your own hands doing the sign** reads better than artwork from the
  show. It shows the exact pose the model was trained on, from roughly the angle
  the camera sees — and it sidesteps the copyright question entirely.
- **Plain background, hands filling the frame.** These render around 140–200px
  wide, so detail is lost quickly.

The names come from `web/signs.json`, which `export_model.py` writes. Rename a
sign in `jjk/signs.py`, re-export, and the filenames expected here change with it.
