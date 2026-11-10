"""The held dart swings as center + amp * sin(w t + phi).

All four parameters drift during a game (live data: the range grows from about
-21..53 deg to -30..65 deg and the period shrinks from ~4 s to ~2 s), so every
throw fits all of them from its own recent samples.

The angle readings contain two kinds of junk that a plain least-squares fit
chokes on: frozen frames (the same reading repeated while the game or the
capture hitches) and short excursions where the detector is pulled off the
dart. Frozen samples are dropped, then a robust fit is refined on inliers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# Fit on this much recent history (live data, 0.15 s ahead, median error:
# 2 s -> 0.95 deg, 3 s -> 0.84 deg with a much shorter tail).
WINDOW_S = 3.0
INLIER_DEG = 4.0
W_RANGE = (0.8, 8.0)        # rad/s; ~0.8 s .. ~8 s period

UP, DOWN, ANY = "up", "down", "any"


def recent(samples, window=WINDOW_S):
    S = np.asarray(samples, float)
    return S[S[:, 0] >= S[-1, 0] - window]


def drop_frozen(S):
    """Remove samples identical to the previous one (stale frame)."""
    if len(S) < 2:
        return S
    keep = np.ones(len(S), bool)
    keep[1:] = S[1:, 1] != S[:-1, 1]
    return S[keep]


@dataclass
class SwingFit:
    center: float
    amp: float
    w: float
    phi: float
    t0: float
    rms: float          # rms residual of inliers, deg
    inliers: float      # fraction of (non-frozen) samples within INLIER_DEG
    n: int

    def angle(self, t):
        return self.center + self.amp * np.sin(self.w * (np.asarray(t) - self.t0) + self.phi)

    def rate(self, t):
        """Angular speed in deg/s (positive = dart moving up)."""
        return self.amp * self.w * np.cos(self.w * (np.asarray(t) - self.t0) + self.phi)

    @property
    def period(self):
        return 2 * np.pi / self.w

    @property
    def range(self):
        return self.center - self.amp, self.center + self.amp

    def next_crossing(self, target, t_min, direction=ANY):
        """Earliest t >= t_min with angle(t) == target, moving in `direction`.

        Targets outside the swing range are clamped to the nearest extreme."""
        s = np.clip((target - self.center) / self.amp, -1.0, 1.0)
        base = float(np.arcsin(s))
        phases = {UP: [base], DOWN: [np.pi - base], ANY: [base, np.pi - base]}[direction]
        best = None
        for u in phases:
            t = self.t0 + (u - self.phi) / self.w
            t += np.ceil((t_min - t) / self.period) * self.period
            if best is None or t < best:
                best = t
        return float(best)

    def describe(self):
        lo, hi = self.range
        return (f"range {lo:.0f}..{hi:.0f} deg, period {self.period:.2f} s, "
                f"rms {self.rms:.2f}, inliers {self.inliers:.0%} of {self.n}")


def _seeds(t, y, k=3):
    """For a fixed w the model is linear in (A, B, c): grid w, solve, keep the best."""
    out = []
    for w in np.linspace(*W_RANGE, 73):
        X = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = X @ coef - y
        out.append((float(np.sum(np.minimum(r * r, 25.0))), w,
                    float(np.arctan2(coef[1], coef[0])), float(coef[2]), float(np.hypot(coef[0], coef[1]))))
    out.sort()
    return out[:k]


def _solve(t, y, p0, loss, f_scale):
    f = lambda p: p[2] + p[3] * np.sin(p[0] * t + p[1]) - y
    return least_squares(f, p0, loss=loss, f_scale=f_scale)


def fit_swing(ts, angles) -> SwingFit | None:
    S = drop_frozen(np.stack([np.asarray(ts, float), np.asarray(angles, float)], 1))
    if len(S) < 20:
        return None
    t0 = S[0, 0]
    t, y = S[:, 0] - t0, S[:, 1]

    best = None
    for _, w, ph, c, a in _seeds(t, y):
        r = _solve(t, y, [w, ph, c, a], "cauchy", 1.5)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    good = np.abs(p[2] + p[3] * np.sin(p[0] * t + p[1]) - y) < INLIER_DEG
    if good.sum() >= 20 and good.mean() > 0.5:
        p = _solve(t[good], y[good], p, "soft_l1", 1.0).x

    w, ph, c, a = p
    if a < 0:
        a, ph = -a, ph + np.pi
    if w < 0:
        w, ph = -w, np.pi - ph
    res = np.abs(c + a * np.sin(w * t + ph) - y)
    inl = res < INLIER_DEG
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("inf")
    return SwingFit(float(c), float(a), float(w), float(ph % (2 * np.pi)), float(t0), rms,
                    float(inl.mean()), len(S))
