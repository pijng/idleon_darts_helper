# Idleon Darts Helper

**Get the "9 Bullseye in a row" trophy without the grind.**

Idleon Darts Helper watches the Throwy Darts minigame, works out where every dart will fly (including the wind), and clicks at the
right moment. It ships with a model trained on almost 2,000 real throws, so there is
nothing to train: install, open the minigame, run one command.

- **Up to ~90% bullseyes** early in a game, and streaks of **18, 17 and 16 bullseyes in a
  row** in real games (the trophy needs 9).
- It only looks at the screen and clicks the mouse: no game files, no memory reading, no injection.
- Any window size and any display scaling (a bigger game window is more accurate, see below).
- It keeps playing until the game is over and tells you when the trophy streak is reached.
- Tested on Windows (Steam) and macOS (CrossOver).

## Install (Windows)

1. Install **Python 3.12** from [python.org](https://www.python.org/downloads/windows/).
   In the installer, tick **"Add python.exe to PATH"**.
2. Download this project (**Code → Download ZIP**, then unzip it) or `git clone` it.
3. Open **PowerShell** in the project folder (Shift + right-click the folder → _Open in
   Terminal_) and run:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```

   If PowerShell refuses to run `activate`, run this once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Quick start

1. Start Legends of Idleon and open **Throwy Darts**. Keep the whole game window visible,
   not minimised and not covered by other windows, including the PowerShell window.
   **Make the game window as large as you can**: the bot reads the dart angle from pixels,
   and a bigger window means fewer misses.
2. In PowerShell, in the project folder (run `.venv\Scripts\activate` first if it's a new
   window):

   ```powershell
   dartbot play
   ```

3. Don't touch the mouse while it plays. It stops by itself when the game is over;
   **Ctrl+C** stops it earlier.

Every throw prints a line like `#22 wind 9mph@16 aim 13.59 ... -> 30 pts, bullseye streak 9`.
`9 bullseyes in a row: trophy streak reached` means the trophy is yours.

Not sure the bot sees the game? `dartbot check` prints what it detects (window, player,
dart angle, wind, score) and the angle it would aim at, without clicking.

## Features

| command                                  | what it does                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------- |
| `dartbot play`                           | plays until the game ends (no dart in hand for 30 s)                                        |
| `dartbot play --streak 9`                | stops after 9 bullseyes in a row instead of playing on                                      |
| `dartbot play --throws 20`               | stops after 20 throws                                                                       |
| `dartbot play --dry-run`                 | tracks and plans but never clicks; your own throws are logged as training data              |
| `dartbot play --debug`                   | step-by-step log, plus `logs/<date-time>/` with a log file and screenshots of odd moments   |
| `dartbot play --direction up\|down\|any` | which way the dart must be swinging when clicked (default `up`, which is far more accurate) |
| `dartbot play --monitor N`               | capture only monitor N (default: all monitors)                                              |
| `dartbot check`                          | one-shot detection report                                                                   |
| `dartbot fit`                            | retrains `model.json` from all logged throws (~2 min)                                       |
| `dartbot analyze video.mp4`              | extracts throws from a gameplay recording into training data                                |

What the bot handles on its own:

- **Wind** of any strength and direction, read from the mph number and the arrow.
- **The swing speeds up during a game.** The bot re-measures the swing before every throw.
- **Game hitches.** Frozen frames are ignored; if the game hitches right before a click,
  the bot waits for the next swing.
- **Ignored clicks.** The game occasionally drops a click (from ~1 in 25 to ~1 in 10 in
  our sessions). The bot notices (same platform and score afterwards), doesn't count it
  as a miss, and throws again.
- **Moved or resized game window.** The bot finds it again.
- **Bullseye streak bonuses** (+10, +30) count as bullseyes in the streak counter.

### Expected accuracy

Two things decide how often the bot hits the bullseye: the size of the game window and
the speed of the swing.

**Game window size.** The dart is only a few pixels thick, so a larger window gives a more
precise angle reading. **Make the game as large as your screen allows.** Bullseye rate by
swing speed:

| swing speed                      | when       | game ~1,100 px wide | game ~1,500 px wide |
| -------------------------------- | ---------- | ------------------- | ------------------- |
| up to ~200°/s                    | early game | ~70–80%             | **~80–90%**         |
| ~200–330°/s                      | mid game   | ~40–50%             | ~60%                |
| over ~330°/s                     | late game  | ~33%                | ~42%                |
| launch-angle scatter, early game |            | ~2.1–2.3°           | ~0.9–1.3°           |

The ~1,500 px column comes from two games, 493 throws in total. The best one: 419 throws,
83% bullseyes over the first 100, a streak of 18, final score 2008. Widths are as
captured: real pixels on Windows, points on macOS (~1,500 was the game filling a
1512-point laptop screen). The bullseye tolerates about ±2.3° of launch angle. Smaller
windows work, just less accurately.

The angle reading also has a small bias that depends on the window size, about 1°
between these two sizes. The model corrects for it automatically; for window sizes it
hasn't seen, it uses the nearest size it has.

**Swing speed.** The swing gets faster as a game goes on, down to a ~0.7 s period. The
bot's click timing scatters by about ±10 ms, most likely the game's own frame rate, and
that costs more the faster the dart moves. That's why the rates drop later in a game.

Long streaks therefore happen early in a game: in the large window, the first 100 throws
are the best place for the trophy. If they don't get it, restart the minigame; a new game
starts with a slow swing again.

### Troubleshooting

- **"Game canvas not found"**: the whole minigame (including the "THROWY DARTS" banner and
  the EXIT button) must be visible on screen.
- **Nothing happens when it clicks**: if the game runs as administrator, run PowerShell as
  administrator too. Windows blocks clicks from normal programs into elevated windows.
- **It stops or waits oddly**: run `dartbot play --debug` and look at the `waiting: ...`
  lines. Every couple of seconds they say what the bot is waiting for. The
  `logs/<date-time>/` folder is the thing to share in a bug report.

## Training data and the model

`model.json` contains the fitted parameters: gravity, wind response, launch-angle offset
(and how it changes with window size) and click latency. All coordinates are in a fixed
reference frame, so it works on any
screen. Every live throw is appended to `data/live_throws.jsonl`. After a few hundred new
throws, or if you play on a very different machine, retrain:

```powershell
dartbot fit
```

Sharing `data/live_throws.jsonl` helps everyone, especially throws in strong winds.

## How it works

- **Finding the game.** The bot looks for the "THROWY DARTS" banner and the EXIT button
  anywhere on screen. Their positions give the window's location and scale. Everything is
  converted into the 3020x1710 frame of `board.png`.
- **Vision.**
  - The platform is found by template matching.
  - The dart angle comes from the shape of its shaft near the tip.
  - Wind speed is read from the digits; wind direction from the arrow's orientation.
  - The score is read by glyph matching.
- **Swing.** The dart angle follows `center + amp * sin(w t + phi)`. All four parameters
  drift during a game: the range grows from about −21..53° to −30..65°, and the period
  shrinks from ~4 s to under 1 s. Before every throw the bot fits them to the last 3 s,
  dropping frozen frames and outliers.
- **Flight.** A ballistic model `y(x)` with gravity plus a wind acceleration proportional
  to the mph vector. It is fitted on whole tracked flight paths; the median path error is
  ~1 px.
- **Aiming.** The bot solves for the launch angle that lands in the middle of the red
  segment. The launch angle equals the dart's on-screen angle at `click + latency`, plus a
  small offset that depends on the window size; `fit` learns all of it. The bot then
  predicts when the swing will next pass that
  angle on the upswing and clicks that much earlier.
- **Feedback.** After each throw, the bot measures the real launch angle from the flight
  path and the result from the score, then logs everything for the next `fit`.

### Files

| path                 | role                                                |
| -------------------- | --------------------------------------------------- |
| `dartbot/config.py`  | reference geometry: board segments, panels, anchors |
| `dartbot/screen.py`  | canvas locator; live, video and replay sources      |
| `dartbot/vision.py`  | detectors                                           |
| `dartbot/swing.py`   | swing fit and click timing                          |
| `dartbot/physics.py` | flight model, solver, fitting                       |
| `dartbot/bot.py`     | live loop                                           |
| `dartbot/analyze.py` | recording → throw records                           |
| `model.json`         | fitted parameters                                   |

### Notes

- Throws logged before the dart-angle fix (no `angle_v` in their `extra`) read ~2° low
  and were wrong between 32° and 56°. `fit` shifts their samples by +2.1°, drops readings
  above 30°, and calibrates timing only on current throws.
- Glyph `9` is synthesised from `6`, because no 9 appeared in the recordings. If the wind
  or score is misread, add a correct crop to `dartbot/assets/glyphs/`.
- macOS (for example the game under CrossOver) also works: use `.venv/bin/activate`, and
  give your terminal **Screen Recording** and **Accessibility** permission.

## License

MIT, see [LICENSE](LICENSE).
