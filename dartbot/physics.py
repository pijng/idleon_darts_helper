"""Dart flight model and its calibration.

Time-free ballistic model in reference pixels. The launch speed is fixed, so we
use it as the unit: path parameter tau is "pixels travelled at launch speed".

    x(tau) = xr + cos(th) tau + 1/2 ax tau^2
    y(tau) = yr - sin(th) tau + 1/2 (g + ay) tau^2      (screen y points down)
    (ax, ay) = (beta_x * wind_x, -beta_y * wind_y)       (wind in mph, y up)

Only shapes y(x) are fitted, never timings: screen recordings have variable
frame rate and live captures have jitter, but the geometry is exact.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

from . import config as C
from .records import Throw
from .swing import fit_swing, recent
from .vision import ANGLE_VERSION, LEGACY_ANGLE_MAX, LEGACY_ANGLE_SHIFT

MIN_CURRENT_THROWS = 30
REF_INV_SCALE = 2.5          # 1/scale where angle_offset applies (scale 0.4)
MIN_INV_SCALE_SPREAD = 0.3   # throws this far (in 1/scale) from the typical size ...
MIN_SCALE_THROWS = 30        # ... needed in this number before the scale slope is fitted


@dataclass
class Model:
    g: float = 3.9e-4
    beta_x: float = 0.0
    beta_y: float = 0.0
    rel_y0: float = -175.0          # release y = py + rel_y0 + rel_y1 * theta_deg
    rel_y1: float = -1.75
    # launch angle = sprite angle at (click + latency) + offset, where
    # offset = angle_offset + offset_per_inv_scale * (1/scale - REF_INV_SCALE).
    # The sprite angle is read from pixels, so its small bias depends on how large the game
    # window is (scale = window size / reference frame).
    angle_offset: float = 1.0
    offset_per_inv_scale: float = 0.0
    # Window sizes the slope was fitted on; outside them the offset is held at the edge
    # instead of extrapolated.
    scale_min: float = 0.4
    scale_max: float = 0.4
    latency: float = 0.05           # click -> game reads input, seconds (per machine)
    board_x: float = C.BOARD_X
    target_y: float = C.BULLSEYE_Y
    n_throws: int = 0

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Model":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in json.loads(Path(path).read_text()).items() if k in known})

    def offset(self, scale: float | None) -> float:
        if scale is None:
            return self.angle_offset
        s = min(max(scale, self.scale_min), self.scale_max)
        return self.angle_offset + self.offset_per_inv_scale * (1.0 / s - REF_INV_SCALE)

    def at_scale(self, scale: float | None) -> "Model":
        """This model with the angle offset for one window size baked in."""
        return replace(self, angle_offset=self.offset(scale), offset_per_inv_scale=0.0)

    # ------------------------------------------------------------ prediction

    def accel(self, wind_mph, wind_dir):
        r = np.radians(wind_dir)
        return (self.beta_x * wind_mph * np.cos(r),
                -self.beta_y * wind_mph * np.sin(r))

    def release(self, plat, theta):
        return plat[0] + C.RELEASE_DX, plat[1] + self.rel_y0 + self.rel_y1 * theta

    def y_at(self, x, theta, xr, yr, ax, ay):
        """y of the path at horizontal position(s) x."""
        th = np.radians(theta)
        c, s = np.cos(th), np.sin(th)
        D = np.asarray(x, float) - xr
        if abs(ax) < 1e-12:
            tau = D / c
        else:
            disc = c * c + 2 * ax * D
            tau = (-c + np.sqrt(np.where(disc >= 0, disc, np.nan))) / ax
        return yr - s * tau + 0.5 * (self.g + ay) * tau ** 2

    def hit_y(self, plat, theta, wind_mph, wind_dir):
        ax, ay = self.accel(wind_mph, wind_dir)
        xr, yr = self.release(plat, theta)
        return float(self.y_at(self.board_x, theta, xr, yr, ax, ay))

    def solve(self, plat, wind_mph, wind_dir, sprite_range, target_y=None):
        """Launch angle hitting target_y, preferring the least sensitive solution.

        `sprite_range` is the (lo, hi) of the current swing. Returns
        (theta, dy/dtheta in px/deg) or None if unreachable."""
        target_y = self.target_y if target_y is None else target_y
        lo, hi = (a + self.angle_offset for a in sprite_range)
        th = np.arange(lo + 0.5, hi - 0.5, 0.05)
        y = np.array([self.hit_y(plat, t, wind_mph, wind_dir) for t in th])
        f = y - target_y
        sol = []
        for i in np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
            t = th[i] - f[i] * (th[i + 1] - th[i]) / (f[i + 1] - f[i])
            slope = (y[i + 1] - y[i]) / (th[i + 1] - th[i])
            sol.append((abs(slope), t, slope))
        if not sol:
            return None
        _, t, slope = min(sol)
        return float(t), float(slope)


# ---------------------------------------------------------------- fitting

def _path_xy(t: Throw):
    P = np.asarray(t.path, float)
    if len(P) < 10:
        return None
    xr = t.plat[0] + C.RELEASE_DX
    P = P[P[:, 1] > xr + 60]
    return P if len(P) >= 10 else None


def _fit_launch(m: Model, t: Throw, P, x0=None):
    """Best (theta, release y) for one path under fixed global physics."""
    ax, ay = m.accel(t.wind_mph, t.wind_dir)
    xr = t.plat[0] + C.RELEASE_DX
    if x0 is None:
        q = np.polyfit(P[:, 1] - xr, P[:, 2], 2)
        x0 = [float(np.degrees(np.arctan(-q[1]))), float(q[2])]

    def resid(p):
        return np.nan_to_num(m.y_at(P[:, 1], p[0], xr, p[1], ax, ay) - P[:, 2], nan=1e3)

    r = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, x_scale=[1.0, 10.0], xtol=1e-12, ftol=1e-12)
    return r.x, r.fun


def fit_paths(model: Model, throws: list[Throw], fit_wind=True):
    """Fit g / wind betas (global) and launch angle + release y (per throw).

    Variable projection: the outer search only sees the 1-3 global parameters;
    every evaluation re-solves each throw's own 2-parameter fit (warm-started).
    Nelder-Mead on the robust total cost: finite-difference gradients through the
    inner fits were too noisy to make progress.
    Returns per-throw (throw, theta, yr, path error in px)."""
    data = [(t, _path_xy(t)) for t in throws]
    data = [(t, P) for t, P in data if P is not None]
    if not data:
        raise ValueError("no usable flight paths")
    warm = [None] * len(data)

    def with_globals(p):
        g, bx, by = (p[0], p[1], p[2]) if fit_wind else (p[0], model.beta_x * 1e5, model.beta_y * 1e5)
        return Model(**{**asdict(model), "g": g * 1e-4, "beta_x": bx * 1e-5, "beta_y": by * 1e-5})

    def cost(p):
        m = with_globals(p)
        total = 0.0
        for k, (t, P) in enumerate(data):
            x, fun = _fit_launch(m, t, P, warm[k])
            warm[k] = x
            total += float(np.sum(np.sqrt(1 + (fun / 3.0) ** 2) - 1))  # soft_l1, 3 px scale
        return total

    head = [model.g * 1e4, model.beta_x * 1e5, model.beta_y * 1e5] if fit_wind else [model.g * 1e4]
    simplex = [head] + [list(np.add(head, np.eye(len(head))[i] * (0.3 if i == 0 else 1.0)))
                        for i in range(len(head))]
    r = minimize(cost, head, method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=0.5, initial_simplex=simplex))
    m = with_globals(r.x)
    model.g, model.beta_x, model.beta_y = m.g, m.beta_x, m.beta_y
    out = []
    for k, (t, P) in enumerate(data):
        x, fun = _fit_launch(m, t, P, warm[k])
        # median: the first tracked point often sits on the hand
        out.append((t, float(x[0]), float(x[1]), float(np.median(np.abs(fun)))))
    return out


def swing_samples(t: Throw):
    """Swing samples in the current angle measurement (see vision.ANGLE_VERSION)."""
    S = np.asarray(t.swing, float).reshape(-1, 2)
    if t.extra.get("angle_v", 1) < 2:
        S = S[S[:, 1] < LEGACY_ANGLE_MAX]
        S[:, 1] += LEGACY_ANGLE_SHIFT
    return S


def swing_before(t: Throw, until: float | None = None):
    """Swing fit from the samples a throw recorded (optionally only those before `until`)."""
    S = swing_samples(t)
    if until is not None:
        S = S[S[:, 0] <= until]
    if len(S) < 20:
        return None
    f = fit_swing(*recent(S).T)
    if f is None or f.inliers < 0.6 or f.rms > 3.0:
        return None
    return f


def fit_model(throws: list[Throw], model: Model | None = None, log=print) -> Model:
    model = model or Model()

    # 1. flight physics
    # beta_x / beta_y are only separable with at least two distinct wind directions
    wind_dirs = {round(t.wind_dir / 15) for t in throws if t.wind_mph > 0}
    fits = fit_paths(model, throws, fit_wind=len(wind_dirs) >= 2)
    log(f"physics: g={model.g:.4e} beta_x={model.beta_x:.3e} beta_y={model.beta_y:.3e} "
        f"over {len(fits)} paths, median path error {np.median([f[3] for f in fits]):.2f}px")

    # 2. release point vs launch angle
    th = np.array([f[1] for f in fits])
    dy = np.array([f[2] - f[0].plat[1] for f in fits])
    q0 = np.polyfit(th, dy, 1)
    q = least_squares(lambda q: q[0] * th + q[1] - dy, q0, loss="cauchy", f_scale=5.0).x
    model.rel_y1, model.rel_y0 = float(q[0]), float(q[1])
    log(f"release: y = py + {model.rel_y0:.1f} + {model.rel_y1:.3f}*theta")

    # 3. launch angle = sprite(click + latency) + offset.
    #    Video throws have no click: their release time is the last held frame, latency 0.
    #    Live throws mix swing speeds and directions, which separates latency from offset.
    #    Legacy throws are only approximately converted to the current angle measurement
    #    (they left a +1.2 deg aiming bias), so they are used only while there are too
    #    few current ones.
    current = [f for f in fits if f[0].extra.get("angle_v", 1) >= ANGLE_VERSION]
    use = current if len(current) >= MIN_CURRENT_THROWS else fits
    # Window size per throw (recorded since the scale-aware offset; older throws were
    # matched to their session logs). Throws without it are dropped once enough have it.
    scaled = [f for f in use if f[0].extra.get("scale")]
    if len(scaled) >= MIN_CURRENT_THROWS:
        use = scaled
    obs = []  # (swing fit, time, measured theta, is_live, 1/scale - ref)
    for t, theta, _, _ in use:
        when = t.click_t if t.click_t is not None else t.release_t
        f = swing_before(t, when)
        if f is not None:
            s = t.extra.get("scale")
            du = 1.0 / s - REF_INV_SCALE if s else 0.0
            obs.append((f, when, theta, t.click_t is not None, du))
    if obs:
        log(f"launch calibration from {'current-detector' if use is not fits else 'all'} throws")
        live = any(o[3] for o in obs)
        # The scale slope is only identifiable with enough throws away from the typical size.
        du = np.array([o[4] for o in obs])
        by_scale = np.sum(np.abs(du - np.median(du)) > MIN_INV_SCALE_SPREAD) >= MIN_SCALE_THROWS

        def resid(p):
            off = p[0]
            k = p[1] if by_scale else 0.0
            lat = p[-1] if live else 0.0
            return np.array([th_ - (f.angle(w + (lat if is_live else 0.0)) + off + k * d)
                             for f, w, th_, is_live, d in obs])

        p0 = ([model.angle_offset] + ([model.offset_per_inv_scale] if by_scale else [])
              + ([model.latency] if live else []))
        r = least_squares(resid, p0, loss="cauchy", f_scale=2.0)
        model.angle_offset = float(r.x[0])
        model.offset_per_inv_scale = float(r.x[1]) if by_scale else 0.0
        seen = 1.0 / (du + REF_INV_SCALE)
        model.scale_min, model.scale_max = float(seen.min()), float(seen.max())
        if live:
            model.latency = float(np.clip(r.x[-1], 0.0, 0.3))
        res = np.abs(r.fun)
        scales = sorted({round(1.0 / (d + REF_INV_SCALE), 2) for d in du})
        log(f"launch: offset {model.angle_offset:.2f} deg at scale {1 / REF_INV_SCALE:.2f}"
            + (f", {model.offset_per_inv_scale:+.2f} deg per unit of 1/scale "
               f"(scale 0.36: {model.offset(0.36):.2f}, 0.50: {model.offset(0.5):.2f})"
               if by_scale else f" (one window size only, scales seen: {scales})")
            + (f", latency {model.latency * 1000:.0f} ms" if live else "")
            + f" (n={len(obs)}, median |err| {np.median(res):.2f} deg, p90 {np.percentile(res, 90):.2f})")

    model.n_throws = len(fits)
    return model


def latency_from_throw(t: Throw, model: Model, theta):
    """Delay between our click and the launch, inferred from the measured launch angle."""
    f = swing_before(t, t.click_t)
    if f is None:
        return None
    target = theta - model.offset(t.extra.get("scale"))
    # The crossing of the target nearest to click_t + current latency estimate.
    ts = np.linspace(t.click_t - 0.05, t.click_t + 0.3, 701)
    err = np.abs(f.angle(ts) - target)
    cand = ts[err < 1.5]
    if len(cand) == 0:
        return None
    guess = t.click_t + model.latency
    return float(cand[np.argmin(np.abs(cand - guess))] - t.click_t)


def measure_launch(model: Model, t: Throw):
    """Launch angle of one live throw using the fitted global physics."""
    P = _path_xy(t)
    if P is None:
        return None
    x, fun = _fit_launch(model, t, P)
    return float(x[0]), float(np.sqrt(np.mean(fun ** 2)))
