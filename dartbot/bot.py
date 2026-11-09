"""Live play: see, plan, click, learn."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import config as C
from . import records
from . import vision as V
from .physics import Model, latency_from_throw, measure_launch
from .records import Throw
from .screen import ReplaySource, ScreenSource
from .swing import fit_swing, recent

HELD_CONFIRM = 3          # consecutive held reads before we trust a new dart
MIN_SWING_S = 1.6         # observe the swing this long before planning (see swing.WINDOW_S)
REFIT_BEFORE_S = 0.12     # last refit this long before the click
MIN_LEAD_S = 0.15         # never plan a click sooner than this
FLIGHT_MAX_S = 2.0
IDLE_TIMEOUT_S = 10.0
CURSOR_PARK = (1400, 1600)  # bottom of the field: out of the dart's way


def _connect(args):
    if getattr(args, "replay", None):
        return ReplaySource(args.replay)
    src = ScreenSource(args.monitor)
    if src.locate() is None:
        raise SystemExit("Game canvas not found: open the Throwy Darts minigame and keep it fully visible.")
    return src


def _load_model(path):
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m dartbot fit` first.")
    return Model.load(path)


@dataclass
class Panel:
    plat: tuple
    wind: V.Wind | None
    score: int | None


def read_panel(src) -> Panel | None:
    field, _ = src.grab(C.FIELD)
    plat, pv = V.find_platform(field)
    if pv < 0.7:
        return None
    wind = V.read_wind(src.grab(C.WIND_TEXT)[0], src.grab(C.WIND_ARROW)[0])
    score = V.read_score(src.grab(C.SCORE_TEXT)[0])
    return Panel(plat, wind, score)


def read_dart(src, plat):
    img, t = src.grab(V.swing_rect(plat))
    return V.dart_angle(img, plat), t


def check(args):
    src = _connect(args)
    print(f"canvas: {src.canvas}  dpr={src.dpr}")
    p = read_panel(src)
    if p is None:
        raise SystemExit("player platform not found")
    r, _ = read_dart(src, p.plat)
    print(f"platform: {p.plat}\ndart: {r}\nwind: {p.wind}\nscore: {p.score}")
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 1.0:
        read_dart(src, p.plat)
        n += 1
    print(f"swing sampling rate: {n} Hz")
    if args.model.exists() and p.wind is not None:
        m = Model.load(args.model)
        sol = m.solve(p.plat, p.wind.mph, p.wind.direction)
        print(f"plan: {sol and f'launch {sol[0]:.2f} deg (sprite {sol[0] - m.angle_offset:.2f}), {sol[1]:.1f} px/deg'}")


def wait_for_dart(src, timeout):
    """Block until a dart is held. Returns (panel, first swing samples) or None."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        p = read_panel(src)
        if p is None:
            continue
        samples = []
        for _ in range(HELD_CONFIRM * 2):
            r, t = read_dart(src, p.plat)
            if r is not None and r.held:
                samples.append((t, r.angle))
            elif samples:
                break
        if len(samples) >= HELD_CONFIRM:
            return p, samples
    return None


def sample_until(src, plat, samples, t_end):
    while time.perf_counter() < t_end:
        r, t = read_dart(src, plat)
        if r is not None and r.held:
            samples.append((t, r.angle))
        elif r is not None and not r.held and len(samples) > 5 and t - samples[-1][0] > 0.1:
            return False  # dart left the hand (someone else clicked)
    return True


def track_flight(src, plat, t_start):
    tracker = V.FlightTracker(plat)
    while time.perf_counter() - t_start < FLIGHT_MAX_S:
        field, t = src.grab(C.FIELD)
        p, pv = V.find_platform(field)
        if pv > 0.7 and abs(p[0] - plat[0]) + abs(p[1] - plat[1]) > 20:
            break  # the platform jumps once the dart has landed
        tracker.add(field, t)
    return tracker.path()


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
    mouse.release(Button.left)
    return t


def play(args):
    if args.replay:
        args.dry_run = True
    try:
        _play(args)
    except EOFError:
        print("end of recording")


def _play(args):
    model = _load_model(args.model)
    src = _connect(args)
    mouse = None
    if not args.dry_run:
        from pynput.mouse import Controller
        mouse = Controller()
        mouse.position = src.ref_to_point(*CURSOR_PARK)
        time.sleep(0.2)

    print(f"canvas {src.canvas}; model from {model.n_throws} throws, latency {model.latency * 1000:.0f} ms")
    streak, n, w_hint = 0, 0, None
    while True:
        got = wait_for_dart(src, IDLE_TIMEOUT_S)
        if got is None:
            print("no dart in hand for a while: game over or minigame closed.")
            break
        panel, samples = got
        wind = panel.wind or V.NO_WIND
        click_t = planned = theta = slope = sprite_target = None
        if sample_until(src, panel.plat, samples, samples[0][0] + MIN_SWING_S):
            # The arrow animates in when the wind changes, so read it again now.
            w = V.read_wind(src.grab(C.WIND_TEXT)[0], src.grab(C.WIND_ARROW)[0])
            if w is None:
                print("! wind panel unreadable, assuming no wind")
            wind = w or V.NO_WIND
            theta, slope = aim(model, panel.plat, wind)
            sprite_target = theta - model.angle_offset
            planned = plan_click(src, model, panel.plat, samples, sprite_target, w_hint)
        if planned is None:
            print("dart released before we acted (manual throw?)")
        else:
            fit, t_click = planned
            w_hint = fit.w
            if args.dry_run:
                print(f"[dry] aim launch {theta:.2f} (sprite {sprite_target:.2f}), swing {fit.w:.2f} rad/s "
                      f"rms {fit.rms:.2f}, would click in {(t_click - time.perf_counter()) * 1000:.0f} ms")
                # keep observing until somebody throws; their throw is logged as data
                while sample_until(src, panel.plat, samples, time.perf_counter() + 0.5):
                    pass
            else:
                click_t = _click_at(t_click, mouse)

        t = finish_throw(src, model, args.log, panel, wind, samples, click_t, sprite_target)
        n += 1
        streak = streak + 1 if t.points == 5 else 0
        plan = f"aim {theta:.2f} ({slope:.0f} px/deg) " if theta is not None else ""
        print(f"#{n} wind {wind.mph:g}mph@{wind.direction:.0f} {plan}"
              f"{t.extra.get('summary', '')} -> {t.points} pts, bullseye streak {streak}")
        if streak >= args.streak:
            print(f"{streak} bullseyes in a row. Done!")
            break
        if args.throws and n >= args.throws:
            break
    if not args.dry_run:
        model.save(args.model)  # keeps the learned click latency


def aim(model, plat, wind):
    sol = model.solve(plat, wind.mph, wind.direction)
    if sol is not None:
        return sol
    print("! no launch angle reaches the bullseye from here; throwing at the closest")
    lo, hi = model.launch_range()
    ths = np.linspace(lo, hi, 400)
    err = [abs(model.hit_y(plat, th, wind.mph, wind.direction) - model.target_y) for th in ths]
    return float(ths[int(np.nanargmin(err))]), float("nan")


def plan_click(src, model, plat, samples, sprite_target, w_hint):
    """Refit the swing until shortly before the click.

    Returns (fit, click time), or None if the dart left the hand meanwhile."""
    t_cross = None
    while True:
        S = recent(samples)
        fit = fit_swing(S[:, 0], S[:, 1], model.swing_center, model.swing_amp, w_hint)
        if fit is None or fit.rms > 3.0:
            fit = fit_swing(S[:, 0], S[:, 1], model.swing_center, model.swing_amp)
        now = time.perf_counter()
        if fit is None or fit.rms > 3.0:
            if not sample_until(src, plat, samples, now + 0.1):
                return None
            continue
        w_hint = fit.w
        if t_cross is None:
            t_min = now + MIN_LEAD_S + model.latency
        else:  # refining: stay on the crossing we already committed to
            t_min = max(now + model.latency, t_cross - 0.1)
        t_cross = fit.next_crossing(sprite_target, t_min)
        t_click = t_cross - model.latency
        if t_click - now <= REFIT_BEFORE_S + 0.02:
            return fit, t_click
        if not sample_until(src, plat, samples, t_click - REFIT_BEFORE_S):
            return None


def finish_throw(src, model, log_path, panel, wind, samples, click_t, sprite_target) -> Throw:
    """Track the flight, read the result, log it, and learn the click latency."""
    release_t = click_t if click_t is not None else samples[-1][0]
    path = track_flight(src, panel.plat, release_t)
    got = wait_for_dart(src, 3.0)
    if got is not None:
        score_after = got[0].score
    else:
        p = read_panel(src)
        score_after = p.score if p else None
    t = Throw(source="live", plat=tuple(map(float, panel.plat)), wind_mph=wind.mph,
              wind_dir=wind.direction, swing=[tuple(s) for s in samples], release_t=release_t,
              path=path, score_before=panel.score, score_after=score_after,
              click_t=click_t, target_sprite=sprite_target)
    records.append(log_path, t)

    m = measure_launch(model, t)
    if m is not None:
        th_meas, _ = m
        y_hit = model.hit_y(panel.plat, th_meas, wind.mph, wind.direction)
        summary = f"got {th_meas:.2f} -> y {y_hit:.0f} (bull {C.BULLSEYE[0]}-{C.BULLSEYE[1]})"
        if click_t is not None:
            lat = latency_from_throw(t, model, th_meas)
            if lat is not None and 0.0 <= lat <= 0.3:
                model.latency = 0.7 * model.latency + 0.3 * lat
                summary += f", latency {lat * 1000:.0f} -> {model.latency * 1000:.0f} ms"
        t.extra["summary"] = summary
    return t
