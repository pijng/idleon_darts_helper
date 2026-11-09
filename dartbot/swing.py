"""The held dart swings as center + amp * sin(w t + phi).

center/amp are game constants (fitted once, stored in model.json); the angular
speed w grows during a game, so w and phi are re-fitted from fresh samples
before every throw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


# Fit on this much recent history: short windows extrapolate badly
# (0.6 s of data -> ~6 deg error 0.2 s ahead; 1.5 s -> ~1 deg).
WINDOW_S = 3.0


def recent(samples):
    S = np.asarray(samples, float)
    return S[S[:, 0] >= S[-1, 0] - WINDOW_S]


@dataclass
class SwingFit:
    center: float
    amp: float
    w: float
    phi: float
    t0: float
    rms: float

    def angle(self, t):
        return self.center + self.amp * np.sin(self.w * (np.asarray(t) - self.t0) + self.phi)

    def rate(self, t):
        """Angular speed in deg/s."""
        return self.amp * self.w * np.cos(self.w * (np.asarray(t) - self.t0) + self.phi)

    def next_crossing(self, target, t_min):
        """Earliest t >= t_min with angle(t) == target (clamped to the swing range)."""
        s = np.clip((target - self.center) / self.amp, -1.0, 1.0)
        base = np.arcsin(s)
        period = 2 * np.pi / self.w
        best = None
        for u in (base, np.pi - base):
            t = self.t0 + (u - self.phi) / self.w
            k = np.ceil((t_min - t) / period)
            t += k * period
            if best is None or t < best:
                best = t
        return float(best)


def fit_swing(ts, angles, center, amp, w_hint=None, fit_shape=False) -> SwingFit | None:
    ts = np.asarray(ts, float)
    ys = np.asarray(angles, float)
    if len(ts) < 8:
        return None
    t0 = ts[0]
    tt = ts - t0
    # Seed: for each candidate w, (A sin + B cos + c) is linear -> phase in closed form.
    ws = [w_hint] if w_hint else np.linspace(1.2, 7.0, 59)
    seeds = []
    for w in ws:
        X = np.stack([np.sin(w * tt), np.cos(w * tt), np.ones_like(tt)], 1)
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        err = float(np.sum((X @ coef - ys) ** 2))
        seeds.append((err, w, float(np.arctan2(coef[1], coef[0]))))
    seeds.sort()
    best = None
    for _, w, ph in seeds[:3]:
        if fit_shape:
            p0 = [w, ph, center, amp]
            f = lambda p: p[2] + p[3] * np.sin(p[0] * tt + p[1]) - ys
        else:
            p0 = [w, ph]
            f = lambda p: center + amp * np.sin(p[0] * tt + p[1]) - ys
        r = least_squares(f, p0, loss="soft_l1", f_scale=2.0)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    c, a = (p[2], p[3]) if fit_shape else (center, amp)
    w, ph = p[0], p[1]
    if a < 0:
        a, ph = -a, ph + np.pi
    if w < 0:
        w, ph = -w, np.pi - ph
    res = c + a * np.sin(w * tt + ph) - ys
    rms = float(np.sqrt(np.median(res ** 2)))
    return SwingFit(c, a, w, ph % (2 * np.pi), t0, rms)
