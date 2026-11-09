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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import config as C
from .records import Throw
from .swing import fit_swing, recent


@dataclass
class Model:
    g: float = 3.9e-4
    beta_x: float = 0.0
    beta_y: float = 0.0
    rel_y0: float = -175.0          # release y = py + rel_y0 + rel_y1 * theta_deg
    rel_y1: float = -1.75
    angle_offset: float = 4.3       # launch angle = sprite angle + offset
    swing_center: float = 16.9
    swing_amp: float = 41.5
    latency: float = 0.05           # click -> game reads input, seconds (per machine)
    board_x: float = C.BOARD_X
    target_y: float = C.BULLSEYE_Y
    n_throws: int = 0

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Model":
        return cls(**json.loads(Path(path).read_text()))

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

    def launch_range(self):
        lo = self.swing_center - self.swing_amp + self.angle_offset
        hi = self.swing_center + self.swing_amp + self.angle_offset
        return lo, hi

    def solve(self, plat, wind_mph, wind_dir, target_y=None):
        """Launch angle hitting target_y, preferring the least sensitive solution.

        Returns (theta, dy/dtheta in px/deg) or None if unreachable."""
        target_y = self.target_y if target_y is None else target_y
        lo, hi = self.launch_range()
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


def fit_paths(model: Model, throws: list[Throw], fit_wind=True):
    """Fit g / wind betas (global) and launch angle + release y (per throw).

    Returns per-throw (theta, yr, rms)."""
    data = [(t, _path_xy(t)) for t in throws]
    data = [(t, P) for t, P in data if P is not None]
    if not data:
        raise ValueError("no usable flight paths")

    init = []
    for t, P in data:
        xr = t.plat[0] + C.RELEASE_DX
        q = np.polyfit(P[:, 1] - xr, P[:, 2], 2)
        init += [float(np.degrees(np.arctan(-q[1]))), float(q[2])]

    def unpack(p):
        g, bx, by = (p[0], p[1], p[2]) if fit_wind else (p[0], model.beta_x, model.beta_y)
        return g * 1e-4, bx * 1e-5, by * 1e-5, p[3 if fit_wind else 1:]

    def resid(p):
        g, bx, by, per = unpack(p)
        m = Model(**{**asdict(model), "g": g, "beta_x": bx, "beta_y": by})
        out = []
        for k, (t, P) in enumerate(data):
            th, yr = per[2 * k], per[2 * k + 1]
            ax, ay = m.accel(t.wind_mph, t.wind_dir)
            yp = m.y_at(P[:, 1], th, t.plat[0] + C.RELEASE_DX, yr, ax, ay)
            out.append(np.nan_to_num(yp - P[:, 2], nan=1e3))
        return np.concatenate(out)

    head = [model.g * 1e4, model.beta_x * 1e5, model.beta_y * 1e5] if fit_wind else [model.g * 1e4]
    r = least_squares(resid, head + init, loss="soft_l1", f_scale=3.0)
    g, bx, by, per = unpack(r.x)
    model.g, model.beta_x, model.beta_y = float(g), float(bx), float(by)
    res = resid(r.x)
    out, i = [], 0
    for k, (t, P) in enumerate(data):
        n = len(P)
        out.append((t, float(per[2 * k]), float(per[2 * k + 1]),
                    float(np.sqrt(np.mean(res[i:i + n] ** 2)))))
        i += n
    return out


def sprite_at(t: Throw, model: Model, when: float):
    """Sprite angle at time `when` from the swing samples recorded before the throw."""
    if len(t.swing) < 8:
        return None
    S = recent(t.swing)
    f = fit_swing(S[:, 0], S[:, 1], model.swing_center, model.swing_amp)
    if f is None or f.rms > 3.0:
        return None
    return float(f.angle(when)), f


def fit_model(throws: list[Throw], model: Model | None = None, log=print) -> Model:
    model = model or Model()

    # 1. swing shape from long held segments
    segs = [np.asarray(t.swing, float) for t in throws if len(t.swing) >= 60]
    if segs:
        cs, as_ = [], []
        for S in segs:
            f = fit_swing(S[:, 0], S[:, 1], model.swing_center, model.swing_amp, fit_shape=True)
            if f and f.rms < 2.5:
                cs.append(f.center)
                as_.append(f.amp)
        if cs:
            model.swing_center, model.swing_amp = float(np.median(cs)), float(np.median(as_))
            log(f"swing: center={model.swing_center:.2f} amp={model.swing_amp:.2f} ({len(cs)} segments)")

    # 2. flight physics
    # beta_x / beta_y are only separable with at least two distinct wind directions
    wind_dirs = {round(t.wind_dir / 15) for t in throws if t.wind_mph > 0}
    fits = fit_paths(model, throws, fit_wind=len(wind_dirs) >= 2)
    log(f"physics: g={model.g:.4e} beta_x={model.beta_x:.3e} beta_y={model.beta_y:.3e} "
        f"over {len(fits)} paths, path rms median={np.median([f[3] for f in fits]):.2f}px")

    # 3. release point vs launch angle
    th = np.array([f[1] for f in fits])
    dy = np.array([f[2] - f[0].plat[1] for f in fits])
    model.rel_y1, model.rel_y0 = (float(v) for v in np.polyfit(th, dy, 1))
    log(f"release: y = py + {model.rel_y0:.1f} + {model.rel_y1:.3f}*theta")

    # 4. sprite -> launch offset (video: release_t is the last held frame) and
    #    latency (live: launch happens at click_t + latency)
    offs, lats = [], []
    for t, theta, _, _ in fits:
        if t.click_t is None:
            r = sprite_at(t, model, t.release_t)
            if r:
                offs.append(theta - r[0])
    if offs:
        model.angle_offset = float(np.median(offs))
        log(f"angle offset: {model.angle_offset:.2f} deg (std {np.std(offs):.2f}, n={len(offs)})")
    for t, theta, _, _ in fits:
        if t.click_t is not None:
            lat = latency_from_throw(t, model, theta)
            if lat is not None:
                lats.append(lat)
    if lats:
        model.latency = float(np.median(lats))
        log(f"latency: {model.latency * 1000:.0f} ms (n={len(lats)})")

    model.n_throws = len(fits)
    return model


def latency_from_throw(t: Throw, model: Model, theta):
    """Delay between our click and the launch, inferred from the measured launch angle."""
    r = sprite_at(t, model, t.click_t)
    if r is None:
        return None
    _, f = r
    target = theta - model.angle_offset
    # Search the crossing nearest to click_t + current latency estimate.
    ts = np.linspace(t.click_t - 0.05, t.click_t + 0.3, 701)
    err = np.abs(f.angle(ts) - target)
    guess = t.click_t + model.latency
    cand = ts[err < 1.5]
    if len(cand) == 0:
        return None
    return float(cand[np.argmin(np.abs(cand - guess))] - t.click_t)


def measure_launch(model: Model, t: Throw):
    """Launch angle of one live throw using the fitted global physics."""
    P = _path_xy(t)
    if P is None:
        return None
    ax, ay = model.accel(t.wind_mph, t.wind_dir)
    xr = t.plat[0] + C.RELEASE_DX

    def resid(p):
        return np.nan_to_num(model.y_at(P[:, 1], p[0], xr, p[1], ax, ay) - P[:, 2], nan=1e3)

    q = np.polyfit(P[:, 1] - xr, P[:, 2], 2)
    r = least_squares(resid, [float(np.degrees(np.arctan(-q[1]))), float(q[2])],
                      loss="soft_l1", f_scale=3.0)
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    return float(r.x[0]), rms
