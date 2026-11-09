# dartbot: Throwy Darts helper for Legends of Idleon

A screen-reading bot for the Throwy Darts minigame. It watches the screen, works out the
moment that puts the dart in the bullseye, and clicks. The goal is the "9 Bullseye in a row" trophy.

> Automating input may break the game's terms of service. Use it at your own risk.

## How it works

- **No game hooks.** The bot finds the game by matching the "THROWY DARTS" banner and the
  EXIT button anywhere on screen. Window size, Retina scaling, Steam, browser or CrossOver
  make no difference.
- **Reference coordinates.** Everything is converted to the 3020x1710 frame of
  `board.png`, so `model.json` works on any machine.
- **Vision**:
  - platform position by template matching
  - dart angle from the shape of the white shaft
  - wind speed from the digits on the panel, wind direction from the arrow's orientation
  - score by glyph matching
- **Swing.** The angle follows `center + amp * sin(w t + phi)`. The speed `w` grows during
  a game, so the bot re-fits it from about 1.6 s of observation before every throw.
- **Flight.** A time-free ballistic model, `y(x)` with gravity plus wind acceleration
  `beta * mph * direction`. It is fitted on whole flight paths. On the bundled recording,
  the path fit error is ~1.4 px and the predicted segment matches the actual one for
  16 of 16 throws.
- **Per-machine latency.** The delay from click to launch is learned automatically. After
  each throw, the real launch angle is measured from the flight path and compared with the
  angle the bot aimed for.

## Install

```
python -m venv .venv
.venv/bin/pip install -e .        # Windows: .venv\Scripts\pip install -e .
```

macOS needs **Screen Recording** and **Accessibility** permission for your terminal.

## Use

1. Open Throwy Darts and keep the whole game window visible.
2. `dartbot check`: prints what the bot sees (canvas, platform, dart angle, wind, score,
   sampling rate) and the angle it would aim at.
3. `dartbot play --dry-run`: tracks and plans without clicking. Every throw you make
   yourself is logged to `data/live_throws.jsonl` as training data.
4. `dartbot play`: plays until 9 bullseyes in a row (`--streak N`) or until no dart
   shows up for 10 s. Don't move the mouse while it plays. The learned latency is saved
   back to `model.json`.

## Improving the model

```
dartbot analyze recording.mov        # recording -> data/video_throws.jsonl
dartbot fit                          # all data/*.jsonl -> model.json
```

The model is fitted on throws from every wind direction it has seen. Sharing
`data/live_throws.jsonl` helps everyone, especially throws in strong or unusual wind.

## Files

| path | role |
| --- | --- |
| `dartbot/config.py` | reference geometry: board segments, panels, anchors |
| `dartbot/screen.py` | canvas locator; live, video and replay sources |
| `dartbot/vision.py` | detectors |
| `dartbot/swing.py` | swing fit and click timing |
| `dartbot/physics.py` | flight model, solver, fitting |
| `dartbot/bot.py` | live loop |
| `dartbot/analyze.py` | recording -> throw records |
| `model.json` | fitted parameters |

## Known limits

- The bundled data covers only 16 throws from one recording: no wind, 3 mph and 8 mph.
  Stronger winds are extrapolated until real throws are logged.
- Glyph `9` is synthesised from `6`, because no 9 appeared in the recording. If the wind
  or score is misread, add a correct crop to `dartbot/assets/glyphs/`.
