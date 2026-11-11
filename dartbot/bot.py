"""Live play: see, plan, click, learn."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import records
from . import vision as V
from .physics import Model, latency_from_throw, measure_launch
from .records import Throw
from .screen import ReplaySource, ScreenSource
from .swing import SwingFit, fit_swing, recent

log = logging.getLogger("dartbot")

HELD_CONFIRM = 3          # consecutive held reads before we trust a new dart
MIN_SWING_S = 2.0         # observe the swing this long before planning
REFIT_BEFORE_S = 0.12     # last refit this long before the click
MIN_LEAD_S = 0.15         # never plan a click sooner than this
RELEASE_S = 0.25          # dart unseen this long = it left the hand
FLIGHT_MAX_S = 2.0
IDLE_TIMEOUT_S = 30.0     # no dart in hand this long = game over / minigame closed
RELOCATE_S = 3.0          # platform unseen this long = look for the game window again
HEARTBEAT_S = 2.0
FIT_OK = dict(rms=2.5, inliers=0.7)        # a fit we trust
FIT_FALLBACK = dict(rms=4.0, inliers=0.5)  # accepted after PLAN_PATIENCE_S of trying
PLAN_PATIENCE_S = 4.0
CLICK_HOLD_S = 0.03       # some clicks are still ignored (17 of 252 with the hold); finish_throw retries them
HITCH_S = 0.06           # identical readings this long = the game is hitching
KEEP_SWING_S = 6.0        # swing history stored per throw
CURSOR_PARK = (1400, 1600)  # bottom of the field: out of the dart's way
TROPHY_STREAK = 9


# ---------------------------------------------------------------- setup

def setup_logging(debug: bool, log_dir: Path) -> Path | None:
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S"))
    log.addHandler(console)
    if not debug:
        return None
    session = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(session / "dartbot.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    log.info(f"debug log and snapshots: {session}")
    return session


class Snapshots:
    """Saves the game canvas when something unexpected happens (debug mode only)."""

    MIN_GAP_S = 1.0

    def __init__(self, src, folder: Path | None):
        self.src, self.folder, self.last = src, folder, 0.0

    def __call__(self, reason: str):
        if self.folder is None or time.perf_counter() - self.last < self.MIN_GAP_S:
            return
        self.last = time.perf_counter()
        img, _ = self.src.grab((0, 0, C.REF_W, C.REF_H))
        name = f"{datetime.now():%H%M%S_%f}"[:-3] + f"_{reason}.png"
        C.write_png(self.folder / name, cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        log.debug(f"snapshot {name}")


def _connect(args):
    if getattr(args, "replay", None):
        return ReplaySource(args.replay)
    src = ScreenSource(args.monitor)
    if src.locate() is None:
        raise SystemExit("Game canvas not found: open the Throwy Darts minigame and keep it fully visible.")
    return src


def _load_model(path):
    if not path.exists():
        raise SystemExit(f"{path} not found: run `dartbot fit` first.")
    return Model.load(path)


# ---------------------------------------------------------------- reading

@dataclass
class Panel:
    plat: tuple
    plat_score: float
    wind: V.Wind | None
    score: int | None


def read_panel(src) -> Panel:
    field, _ = src.grab(C.FIELD)
    plat, pv = V.find_platform(field)
    wind = V.read_wind(src.grab(C.WIND_TEXT)[0], src.grab(C.WIND_ARROW)[0])
    score = V.read_score(src.grab(C.SCORE_TEXT)[0])
    return Panel(plat, pv, wind, score)


def read_dart(src, plat):
    img, t = src.grab(V.swing_rect(plat))
    return V.dart_angle(img, plat), t


def _fmt_dart(r):
    return "nothing" if r is None else f"angle {r.angle:.1f} tip {r.tip_dist:.0f} held={r.held}"


def fit_ok(fit: SwingFit | None, limits) -> bool:
    return fit is not None and fit.rms <= limits["rms"] and fit.inliers >= limits["inliers"]


def hitching(samples, min_s=HITCH_S):
    """True if the latest readings are frozen (the game or capture is stalling)."""
    t_last, a_last = samples[-1]
    i = len(samples) - 1
    while i > 0 and samples[i - 1][1] == a_last:
        i -= 1
    return t_last - samples[i][0] >= min_s


def observe_swing(src, plat, seconds):
    samples = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        r, t = read_dart(src, plat)
        if r is not None and r.held:
            samples.append((t, r.angle))
    return samples


def check(args):
    setup_logging(False, Path("logs"))
    src = _connect(args)
    print(f"canvas: {src.canvas}  dpr={getattr(src, 'dpr', 1.0)}")
    p = read_panel(src)
    print(f"platform: {p.plat} (match {p.plat_score:.2f}, need > 0.7)")
    r, _ = read_dart(src, p.plat)
    print(f"dart: {_fmt_dart(r)}\nwind: {p.wind}\nscore: {p.score}")
    t0 = time.perf_counter()
    samples = observe_swing(src, p.plat, 3.0)
    print(f"swing sampling rate: {len(samples) / (time.perf_counter() - t0):.0f} Hz (held samples)")
    fit = fit_swing(*np.asarray(samples).T) if len(samples) >= 20 else None
    print(f"swing: {fit.describe() if fit else 'no fit (is a dart in hand?)'}")
    if fit and args.model.exists() and p.wind is not None:
        m = Model.load(args.model).at_scale(src.canvas.scale)
        sol = m.solve(p.plat, p.wind.mph, p.wind.direction, fit.range)
        print(f"plan: {sol and f'launch {sol[0]:.2f} deg (sprite {sol[0] - m.angle_offset:.2f}), {sol[1]:.1f} px/deg'}")


# ---------------------------------------------------------------- game loop

class Bot:
    def __init__(self, args, src, model, snap):
        self.args, self.src, self.model, self.snap = args, src, model, snap
        self.direction = args.direction
        self.mouse = None
        self.streak = 0
        self.n = 0
        self.next_dart = None   # dart already found while reading the previous result

    # -------------------------------------------------- waiting for a dart

    def wait_for_dart(self, timeout):
        """Block until a dart is held. Returns (panel, first swing samples) or None."""
        start = time.perf_counter()
        last_beat, plat_lost_since = start, None
        while time.perf_counter() - start < timeout:
            p = read_panel(self.src)
            now = time.perf_counter()
            if p.plat_score < 0.7:
                plat_lost_since = plat_lost_since or now
                if now - last_beat > HEARTBEAT_S:
                    last_beat = now
                    log.info(f"waiting: player platform not visible (match {p.plat_score:.2f})")
                    self.snap("no_platform")
                if now - plat_lost_since > RELOCATE_S and isinstance(self.src, ScreenSource):
                    log.info("looking for the game window again")
                    if self.src.locate() is None:
                        log.info("game window not found")
                    else:
                        log.info(f"game window at {self.src.canvas}")
                    plat_lost_since = now
                continue
            plat_lost_since = None
            samples, last = [], None
            for _ in range(HELD_CONFIRM * 2):
                r, t = read_dart(self.src, p.plat)
                last = r
                if r is not None and r.held:
                    samples.append((t, r.angle))
                elif samples:
                    break
            if len(samples) >= HELD_CONFIRM:
                log.debug(f"dart in hand: platform {p.plat}, wind {p.wind}, score {p.score}")
                return p, samples
            if now - last_beat > HEARTBEAT_S:
                last_beat = now
                log.info(f"waiting: platform {p.plat}, no dart in hand ({_fmt_dart(last)})")
                self.snap("no_dart")
        return None

    def sample_until(self, plat, samples, t_end):
        """Collect held-dart angles until t_end. False if the dart left the hand."""
        while time.perf_counter() < t_end:
            r, t = read_dart(self.src, plat)
            if r is not None and r.held:
                samples.append((t, r.angle))
            elif samples and t - samples[-1][0] > RELEASE_S:
                log.debug(f"dart gone for {t - samples[-1][0]:.2f} s ({_fmt_dart(r)})")
                return False
        return True

    # -------------------------------------------------- planning

    def aim_model(self) -> Model:
        """The model with the angle offset for the current game window size."""
        return self.model.at_scale(self.src.canvas.scale)

    def plan_click(self, panel, wind, samples):
        """Refit swing + aim until shortly before the click.

        Returns (fit, theta, slope, click time) or None if the dart left the hand."""
        plat = panel.plat
        t_cross = None
        started = time.perf_counter()
        last_beat = started
        while True:
            fit = fit_swing(*recent(samples).T)
            now = time.perf_counter()
            patient = now - started < PLAN_PATIENCE_S
            if not fit_ok(fit, FIT_OK if patient else FIT_FALLBACK):
                if now - last_beat > HEARTBEAT_S:
                    last_beat = now
                    log.info(f"swing fit not trusted yet ({fit.describe() if fit else 'too few samples'})")
                    self.snap("bad_fit")
                if not self.sample_until(plat, samples, now + 0.1):
                    return None
                continue
            if not patient and not fit_ok(fit, FIT_OK):
                log.info(f"using a weaker swing fit after {PLAN_PATIENCE_S:.0f} s: {fit.describe()}")

            m = self.aim_model()
            theta, slope = aim(m, plat, wind, fit.range)
            target = theta - m.angle_offset
            if t_cross is None:
                t_min = now + MIN_LEAD_S + self.model.latency
            else:  # refining: stay on the crossing we already committed to
                t_min = max(now + self.model.latency, t_cross - 0.1)
            t_cross = fit.next_crossing(target, t_min, self.direction)
            t_click = t_cross - self.model.latency
            if t_click - now <= REFIT_BEFORE_S + 0.02:
                if hitching(samples):
                    # Clicks during a hitch get ignored or land late: take the next crossing.
                    log.info("game hitch right before the click; waiting for the next swing")
                    t_cross = None
                    if not self.sample_until(plat, samples, t_click + 0.15):
                        return None
                    continue
                return fit, theta, slope, t_click
            log.debug(f"plan: launch {theta:.2f} (sprite {target:.2f}, {self.direction}), "
                      f"click in {(t_click - now) * 1000:.0f} ms, swing {fit.describe()}")
            if not self.sample_until(plat, samples, t_click - REFIT_BEFORE_S):
                return None

    # -------------------------------------------------- after the throw

    def track_flight(self, plat, t_start):
        tracker = V.FlightTracker(plat)
        while time.perf_counter() - t_start < FLIGHT_MAX_S:
            field, t = self.src.grab(C.FIELD)
            p, pv = V.find_platform(field)
            if pv > 0.7 and abs(p[0] - plat[0]) + abs(p[1] - plat[1]) > 20:
                break  # the platform jumps once the dart has landed
            tracker.add(field, t)
        return tracker.path()

    def finish_throw(self, panel, wind, samples, click_t, target) -> Throw | None:
        """Track the flight, read the result, log it, and learn the click latency.

        None if the game ignored the click (dart back in hand on the same platform, same score)."""
        release_t = click_t if click_t is not None else samples[-1][0]
        path = self.track_flight(panel.plat, release_t)
        log.debug(f"flight: {len(path)} points")
        # The score updates when the dart lands; the next dart appears a bit later.
        got = self.wait_for_dart(3.0)
        self.next_dart = got
        score_after = got[0].score if got else read_panel(self.src).score
        # The platform moves after every real throw, hit or miss.
        if (click_t is not None and got is not None
                and got[0].score == panel.score
                and abs(got[0].plat[0] - panel.plat[0]) + abs(got[0].plat[1] - panel.plat[1]) <= 20):
            log.info("the game ignored the click (same platform and score afterwards); trying again")
            self.snap("click_ignored")
            return None
        keep = [tuple(s) for s in samples if s[0] >= samples[-1][0] - KEEP_SWING_S]
        t = Throw(source="live", plat=tuple(map(float, panel.plat)), wind_mph=wind.mph,
                  wind_dir=wind.direction, swing=keep, release_t=release_t,
                  path=path, score_before=panel.score, score_after=score_after,
                  click_t=click_t, target_sprite=target,
                  extra={"angle_v": V.ANGLE_VERSION, "scale": self.src.canvas.scale})
        records.append(self.args.log, t)

        m = measure_launch(self.model, t)
        if m is not None:
            th_meas, _ = m
            y_hit = self.model.hit_y(panel.plat, th_meas, wind.mph, wind.direction)
            t.extra["y_hit"] = y_hit
            summary = f"got {th_meas:.2f} -> y {y_hit:.0f} (bull {C.BULLSEYE[0]}-{C.BULLSEYE[1]})"
            if click_t is not None:
                # Logged only: adapting latency online chased noise (55 -> 66 ms in one
                # session, ~1.5 deg low on fast swings). `fit` estimates it from all throws.
                lat = latency_from_throw(t, self.model, th_meas)
                if lat is not None:
                    summary += f", latency {lat * 1000:.0f} ms (model {self.model.latency * 1000:.0f})"
            t.extra["summary"] = summary
        elif len(path) < 10:
            self.snap("no_flight")
        return t

    @staticmethod
    def is_bullseye(t: Throw):
        """Bullseyes score 5 plus a growing streak bonus (10, 30, ...)."""
        if t.points is not None:
            return t.points >= 5
        y = t.extra.get("y_hit")
        return y is not None and C.BULLSEYE[0] <= y < C.BULLSEYE[1]

    # -------------------------------------------------- one throw

    def throw_once(self):
        got, self.next_dart = self.next_dart or self.wait_for_dart(IDLE_TIMEOUT_S), None
        if got is None:
            log.info(f"no dart in hand for {IDLE_TIMEOUT_S:.0f} s: game over or minigame closed.")
            return False
        panel, samples = got
        wind = panel.wind or V.NO_WIND
        click_t = target = None
        theta = slope = None

        planned = None
        if self.sample_until(panel.plat, samples, samples[0][0] + MIN_SWING_S):
            # The arrow animates in when the wind changes, so read it again now.
            w = V.read_wind(self.src.grab(C.WIND_TEXT)[0], self.src.grab(C.WIND_ARROW)[0])
            if w is None:
                log.info("wind panel unreadable, assuming no wind")
                self.snap("wind_unreadable")
            wind = w or V.NO_WIND
            planned = self.plan_click(panel, wind, samples)

        if planned is None:
            log.info("dart left the hand before we clicked (manual throw?)")
        else:
            fit, theta, slope, t_click = planned
            target = theta - self.aim_model().angle_offset
            log.debug(f"final: swing {fit.describe()}, rate at click {fit.rate(t_click + self.model.latency):.0f} deg/s")
            if self.args.dry_run:
                log.info(f"[dry] would click in {(t_click - time.perf_counter()) * 1000:.0f} ms "
                         f"for launch {theta:.2f} (sprite {target:.2f})")
                while self.sample_until(panel.plat, samples, time.perf_counter() + 0.5):
                    pass
            else:
                click_t = _click_at(t_click, self.mouse)
                log.debug(f"clicked {(click_t - t_click) * 1000:+.1f} ms vs plan")

        t = self.finish_throw(panel, wind, samples, click_t, target)
        if t is None:
            return True  # the game ignored the click; the same dart is thrown again
        self.n += 1
        self.streak = self.streak + 1 if self.is_bullseye(t) else 0
        plan = f"aim {theta:.2f} ({slope:.0f} px/deg) " if theta is not None else ""
        log.info(f"#{self.n} wind {wind.mph:g}mph@{wind.direction:.0f} {plan}"
                 f"{t.extra.get('summary', '')} -> {t.points} pts, bullseye streak {self.streak}")
        return True


def aim(model, plat, wind, sprite_range):
    sol = model.solve(plat, wind.mph, wind.direction, sprite_range)
    if sol is not None:
        return sol
    log.info("no launch angle reaches the bullseye from here; throwing at the closest")
    lo, hi = (a + model.angle_offset for a in sprite_range)
    ths = np.linspace(lo, hi, 400)
    err = [abs(model.hit_y(plat, th, wind.mph, wind.direction) - model.target_y) for th in ths]
    return float(ths[int(np.nanargmin(err))]), float("nan")


def _click_at(t_click, mouse):
    from pynput.mouse import Button
    while True:
        dt = t_click - time.perf_counter()
        if dt <= 0:
            break
        if dt > 0.003:
            time.sleep(dt - 0.002)
    t = time.perf_counter()
    mouse.press(Button.left)
    time.sleep(CLICK_HOLD_S)
    mouse.release(Button.left)
    return t


def play(args):
    if args.replay:
        args.dry_run = True
    session = setup_logging(args.debug, Path("logs"))
    model = _load_model(args.model)
    src = _connect(args)
    bot = Bot(args, src, model, Snapshots(src, session))
    if not args.dry_run:
        from pynput.mouse import Controller
        bot.mouse = Controller()
        bot.mouse.position = src.ref_to_point(*CURSOR_PARK)
        time.sleep(0.2)

    log.info(f"canvas {src.canvas}; model from {model.n_throws} throws, latency {model.latency * 1000:.0f} ms, "
             f"angle offset {model.offset(src.canvas.scale):.2f} deg for this window size, "
             f"clicks on {args.direction} swing")
    try:
        while bot.throw_once():
            if bot.streak == TROPHY_STREAK:
                log.info(f"{TROPHY_STREAK} bullseyes in a row: trophy streak reached")
            if args.streak and bot.streak >= args.streak:
                log.info(f"{bot.streak} bullseyes in a row. Done!")
                break
            if args.throws and bot.n >= args.throws:
                break
    except EOFError:
        log.info("end of recording")
    except KeyboardInterrupt:
        log.info("stopped")
