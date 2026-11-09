"""Extract throw records from a gameplay recording."""
from __future__ import annotations

from pathlib import Path

from . import config as C
from . import vision as V
from .records import Throw
from .screen import VideoSource

RELEASE_DEBOUNCE = 3   # consecutive not-held frames that mean "released"
REGRAB_DEBOUNCE = 5    # consecutive held frames that mean "new dart in hand"
MIN_HELD = 10


def analyze_video(path: Path, log=print) -> list[Throw]:
    src = VideoSource(path)
    throws: list[Throw] = []

    held_run, free_run = 0, 0
    swing: list[tuple[float, float]] = []
    plat = wind = score = None
    flight: V.FlightTracker | None = None
    pending: Throw | None = None
    last_held_t = None

    while src.next():
        field, t = src.grab(C.FIELD)
        p, pv = V.find_platform(field)
        r = V.dart_angle(src.grab(V.swing_rect(p))[0], p) if pv > 0.7 else None
        held = bool(r and r.held)

        if held:
            held_run, free_run = held_run + 1, 0
        else:
            free_run, held_run = free_run + 1, 0

        if flight is None:
            # waiting for / tracking a dart in hand
            if held:
                if not swing:
                    plat = p
                swing.append((t, r.angle))
                last_held_t = t
                if len(swing) % 10 == 1:  # panel values are only needed occasionally
                    w = V.read_wind(src.grab(C.WIND_TEXT)[0], src.grab(C.WIND_ARROW)[0])
                    wind = w if w is not None else wind
                    s = V.read_score(src.grab(C.SCORE_TEXT)[0])
                    score = s if s is not None else score
            elif free_run >= RELEASE_DEBOUNCE and len(swing) >= MIN_HELD:
                w = V.read_wind(src.grab(C.WIND_TEXT)[0], src.grab(C.WIND_ARROW)[0])
                wind = w if w is not None else wind
                # drop the transition frames recorded while the arm was already moving
                sw = [s for s in swing if s[0] <= last_held_t][:-2]
                pending = Throw(source=f"video:{Path(path).name}", plat=tuple(map(float, plat)),
                                wind_mph=wind.mph if wind else 0.0,
                                wind_dir=wind.direction if wind else 0.0,
                                swing=sw, release_t=last_held_t, path=[], score_before=score)
                flight = V.FlightTracker(plat)
                flight.add(field, t)
            elif free_run >= RELEASE_DEBOUNCE:
                swing = []
        else:
            if abs(p[0] - plat[0]) + abs(p[1] - plat[1]) < 20:  # platform jumps after landing
                flight.add(field, t)
            if held_run >= REGRAB_DEBOUNCE:
                pending.path = flight.path()
                pending.score_after = V.read_score(src.grab(C.SCORE_TEXT)[0])
                throws.append(pending)
                log(f"throw {len(throws):2d}: t={pending.release_t:6.2f}s plat={pending.plat} "
                    f"wind={pending.wind_mph:g}mph@{pending.wind_dir:.0f} path={len(pending.path)}pts "
                    f"score {pending.score_before}->{pending.score_after}")
                flight, pending = None, None
                swing, plat = [], p
                # the frames that confirmed the new grab belong to the new swing
                swing.append((t, r.angle))
    return throws
