# Audio

Drop files in here and they get picked up automatically. Nothing needs editing,
and anything missing is simply skipped — a folder with only a theme in it works
fine, as does an empty one.

| Filename | When it plays |
|---|---|
| `theme.mp3` | Loops in the background from the moment you press Start |
| `gojo.mp3` | Once, when Gojo fires |
| `sukuna.mp3` | Once, when Sukuna fires |
| `megumi.mp3` | Once, when Megumi fires |
| `yuta.mp3` | Once, when Yuta fires |
| `malevolent_shrine.mp3` | Once, when Malevolent Shrine fires |

`.mp3`, `.ogg`, `.m4a` and `.wav` all work — the loader tries each extension and
uses the first that loads.

## Notes

- Background music plays at 35% volume, activation sounds at 80%. Both are
  adjustable in [`web/audio.js`](../../web/audio.js).
- Playback can only begin from a user gesture, which is why it starts on the
  Start button rather than on page load. That's a browser rule, not a choice.
- Keep activation sounds short — a second or two. They're one-shots, and they
  retrigger from the start if the same sign fires again quickly.
- Long background tracks are fine here; unlike the portrait GIFs, audio streams
  rather than being decoded into memory up front.

## A word on sourcing

If this ends up public on your portfolio, the soundtrack is the part most likely
to draw an automated copyright complaint — audio fingerprinting is far more
aggressive than image matching. Options, roughly in order of safety:

- Royalty-free or Creative Commons tracks (check the licence allows web use)
- Something you made
- The official release, if you accept the risk on a non-commercial fan project

Same reasoning applies to the character GIFs, but audio is the one that gets
caught.
