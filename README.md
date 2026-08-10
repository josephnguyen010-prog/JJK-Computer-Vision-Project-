# Jujutsu Kaisen hand sign recognition

Throw a JJK hand sign at your webcam. Cursed energy gathers at your fingertips,
a ring closes, and the domain expands.

MediaPipe finds the hand landmarks; a small MLP trained on your own recordings
classifies the sign; a charge-up state machine decides when to actually fire.

## Setup

Already done — dependencies are installed in `.venv` and the MediaPipe landmarker
model has been downloaded. To run anything:

```
.venv\Scripts\python.exe view_signs.py
```

(or activate the venv first with `.venv\Scripts\Activate.ps1`.)

## The workflow, in order

### 1. Find out which signs are viable — `view_signs.py`

**Do this before anything else.** It answers the only question that can sink the
project: can MediaPipe actually see these signs?

```
python view_signs.py            # two-handed signs
python view_signs.py --hands 1  # one-handed
```

Hold each sign steady for about a second and read the verdict. Two things are
being measured:

- **detection rate** — how often both hands are found at all
- **jitter** — how much the landmarks twitch while you hold still

Interlocked fingers make MediaPipe guess at hidden fingertips, and its guess
changes every frame. That jitter is what a sign failing looks like.

Sort your signs into GOOD / MARGINAL / POOR and **start with the GOOD ones
only.** Get the whole pipeline working end to end on four or five easy signs
before going anywhere near Malevolent Shrine. Press `r` to reset between signs.

### 2. Record training data — `record.py`

```
python record.py
```

Press a sign's number key, get a 3-second countdown to set your hands, then it
captures 150 frames. Press `u` to undo a bad burst, `q` to save and quit.
Recording appends, so you can stop and come back.

Three rules, and they matter more than anything in the model:

- **Move while recording.** Drift nearer and further, rotate your wrists, tilt,
  shift side to side. A burst held perfectly still teaches one exact pose.
- **Record idle (key `0`) more than anything else.** Hands relaxed, reaching for
  the keyboard, scratching your face, halfway between two signs. Every frame
  that isn't a sign has to look like idle, or the demo fires at random.
- **Several short bursts, not one long one.** Get up, sit back down, change the
  lighting, record again. Variation *between* bursts is what makes it robust —
  and it's what the evaluation in step 3 measures.

Aim for 300+ samples and 3+ separate bursts per sign; roughly double that for idle.

### 3. Train — `train.py`

```
python train.py
```

Trains in seconds, and prints two accuracy numbers:

```
random split (leaks near-duplicate frames):  99.8%  <- ignore this
held-out bursts (what the demo will feel like): 91.2%  <- trust this
```

Frames recorded back to back are near-identical. A normal random train/test
split scatters copies of the same instant across both sides and reports a number
close to 100% no matter how bad the model is. The second number holds out whole
recording sessions, so it measures what you actually care about: does this work
when you sit down tomorrow. Both are printed because the gap is the point.

`train.py` refuses to run on a dataset with obvious problems (too few samples,
too few bursts, missing or undersized idle class). Pass `--force` to override.

If accuracy is low, read the confusion matrix before touching the model. Two
signs trading off against each other means they look alike to MediaPipe, and
more data for that specific pair beats any hyperparameter.

### 4. Run it — `detect.py`

```
python detect.py
python detect.py --debug    # live probabilities, charge, hand count
```

Hold a sign; energy gathers and pulls toward your palm as the charge builds; at
full charge the domain fires. Press `d` to toggle the debug readout, `q` to quit.

## The charge-up is the debounce

The classifier emits a prediction every frame, so a sign held for two seconds is
sixty activations. Three things prevent that, and all three are visible on screen:

| Mechanism | What it does | Where |
|---|---|---|
| Smoothing | Averages probabilities over recent frames so one bad frame can't fire | `SMOOTHING` |
| Charge time | The sign must hold for ~1s of real time | `CHARGE_SECONDS` |
| Lockout | Nothing else fires until you return to idle | `SignDetector.locked` |

The charge time is a debounce that happens to look like powering up. Tuning
knobs are at the top of `detect.py`.

## Layout

```
view_signs.py     sign viability triage — run this first
record.py         dataset collection
train.py          training + honest evaluation
detect.py         live detection, charge-up, domain expansion
jjk/features.py   landmark normalisation (the heart of it)
jjk/tracker.py    camera + MediaPipe, the only file touching the MP API
jjk/effects.py    particles, glow, domain expansion hit
jjk/signs.py      the sign vocabulary — edit this to add or rename signs
tests/            feature invariance + the charge state machine
```

Run the tests with `python -m pytest tests/` — 18 of them, no webcam needed.

## Adding or changing signs

Everything reads from `SIGNS` in `jjk/signs.py`. Add an entry, re-record, retrain.
The `occlusion` field is your own triage note from step 1; nothing enforces it.

## Notes

- `jjk/features.py` places **both hands in one shared coordinate frame** rather
  than normalising each hand separately. Several domain expansion signs differ
  mainly in how the hands sit relative to each other, and per-hand normalisation
  would erase exactly that.
- The preview is mirrored, which also makes MediaPipe's Left/Right labels match
  your actual hands. Recording and detection both go through `FrameSource`, so
  the convention stays consistent — which is all the classifier needs.
- If the webcam won't open, close anything else using it, or try
  `--camera 1`.
