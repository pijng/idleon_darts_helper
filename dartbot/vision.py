"""Detectors. Every function takes images already in reference scale."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C

_plat = cv2.imread(str(C.ASSETS / "platform.png"), cv2.IMREAD_GRAYSCALE)
_plat_small = cv2.resize(_plat, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- platform

def find_platform(field_img, origin=(C.FIELD[0], C.FIELD[1])):
    """Platform template top-left in reference coords and match score."""
    g = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    _, _, _, l = cv2.minMaxLoc(cv2.matchTemplate(s, _plat_small, cv2.TM_CCOEFF_NORMED))
    x, y = l[0] * 4, l[1] * 4
    x0, y0 = max(0, x - 8), max(0, y - 8)
    sub = g[y0:y + _plat.shape[0] + 8, x0:x + _plat.shape[1] + 8]
    _, v, _, l = cv2.minMaxLoc(cv2.matchTemplate(sub, _plat, cv2.TM_CCOEFF_NORMED))
    return (origin[0] + x0 + l[0], origin[1] + y0 + l[1]), float(v)


def swing_rect(plat):
    px, py = plat
    dx0, dy0, dx1, dy1 = C.SWING_ROI
    return (px + dx0, py + dy0, px + dx1, py + dy1)


# ---------------------------------------------------------------- dart in hand

@dataclass
class DartReading:
    angle: float      # degrees, counter-clockwise from +x (screen up = positive)
    tip_dist: float
    held: bool


def dart_angle(roi_img, plat) -> DartReading | None:
    """Angle of the held dart. `roi_img` is the image of swing_rect(plat)."""
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] < 70) & (hsv[..., 2] > 140)
    ys, xs = np.nonzero(m)
    pvx = C.SHOULDER[0] - C.SWING_ROI[0]
    pvy = C.SHOULDER[1] - C.SWING_ROI[1]
    dx, dy = xs - pvx, ys - pvy
    right = dx > 0
    if not right.any():
        return None
    d2 = (dx * dx + dy * dy).astype(np.int64)
    d2[~right] = -1
    k = int(np.argmax(d2))
    tx, ty = xs[k], ys[k]
    near = ((xs - tx) ** 2 + (ys - ty) ** 2 < 130 ** 2) & right
    if near.sum() < 20:
        return None
    P = np.stack([xs[near], ys[near]], 1).astype(float)
    P -= P.mean(0)
    _, v = np.linalg.eigh(P.T @ P)
    d = v[:, 1]
    if d[0] < 0:
        d = -d
    ang = float(np.degrees(np.arctan2(-d[1], d[0])))
    dist = float(np.sqrt(d2[k]))
    held = C.HELD_TIP_DIST[0] < dist < C.HELD_TIP_DIST[1] and ang > C.HELD_ANGLE_MIN
    return DartReading(ang, dist, held)


# ---------------------------------------------------------------- flying dart

def dart_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = (hsv[..., 1] < 70) & (hsv[..., 2] > 150)
    yel = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)
    return low | yel


class FlightTracker:
    """Collects frames of the field after release and extracts the dart centroid path.

    Static bright things (cursor, stuck darts, HUD) are removed by discarding pixels
    lit in most frames; the player is excluded by an x cut-off.
    """

    STEP = 2  # work at half resolution

    def __init__(self, plat):
        self.x_min = plat[0] + 380
        self.masks, self.times = [], []

    def add(self, field_img, t):
        s = self.STEP
        small = field_img[::s, ::s]
        m = dart_mask(small)
        m[:, : max(0, (self.x_min - C.FIELD[0]) // s)] = False
        self.masks.append(m)
        self.times.append(t)

    def path(self, x_max=2700.0):
        """List of (t, x, y) in reference coords, monotone in x."""
        if len(self.masks) < 3:
            return []
        M = np.array(self.masks)
        static = M.mean(0) > 0.5
        s = self.STEP
        out = []
        for m, t in zip(M, self.times):
            mm = (m & ~static).astype(np.uint8)
            n, _, st, cen = cv2.connectedComponentsWithStats(mm, connectivity=8)
            if n < 2:
                continue
            j = 1 + int(np.argmax(st[1:, 4]))
            if st[j, 4] * s * s < 80:
                continue
            x = C.FIELD[0] + cen[j][0] * s
            y = C.FIELD[1] + cen[j][1] * s
            if x > x_max:
                continue
            if out and x <= out[-1][1] + 2:
                continue
            out.append((t, float(x), float(y)))
        return out


# ---------------------------------------------------------------- text

def _load_glyphs():
    glyphs = []
    for p in sorted((C.ASSETS / "glyphs").glob("*.png")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
        glyphs.append((p.stem.split("_")[0], g))
    return glyphs


_GLYPHS = _load_glyphs()
_NORM = (22, 28)


def _norm(g):
    return cv2.resize(g.astype(np.float32), _NORM, interpolation=cv2.INTER_AREA)


_GLYPHS_N = [(ch, _norm(g), g.shape[1] / g.shape[0]) for ch, g in _GLYPHS]


def _components(img):
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    th = max(120, int(v.max() * 0.6))
    m = (v > th).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = sorted(tuple(st[j]) for j in range(1, n) if st[j, 4] > 15)
    return m, comps


def classify_glyph(g):
    gn, ar = _norm(g), g.shape[1] / g.shape[0]
    best = min(((np.abs(gn - t).mean() + 0.3 * abs(ar - a), ch) for ch, t, a in _GLYPHS_N))
    return best[1], float(best[0])


def read_digits(img, min_height=22, max_dist=0.2):
    """Read the leading run of digits in a text image; None if nothing/unsure."""
    m, comps = _components(img)
    digits = ""
    for x, y, w, h, _ in comps:
        if h < min_height:
            if digits:
                break
            continue
        ch, d = classify_glyph(m[y:y + h, x:x + w] > 0)
        if not ch.isdigit():
            break
        if d > max_dist:
            return None
        digits += ch
    return int(digits) if digits else None


def read_score(score_img):
    """`score_img` = SCORE_TEXT rect. Skips the 'Score:' label (x < 70 in the crop)."""
    m, comps = _components(score_img)
    digits = ""
    for x, y, w, h, _ in comps:
        if x < 70 or h < 22:
            continue
        ch, d = classify_glyph(m[y:y + h, x:x + w] > 0)
        if not ch.isdigit() or d > 0.2:
            return None
        digits += ch
    return int(digits) if digits else None


# ---------------------------------------------------------------- wind

@dataclass
class Wind:
    mph: float
    direction: float  # degrees, screen convention with y up (0 = right, 90 = up)

    @property
    def vector(self):
        r = np.radians(self.direction)
        return self.mph * np.cos(r), self.mph * np.sin(r)


NO_WIND = Wind(0.0, 0.0)


def arrow_direction(arrow_img):
    """Direction the wind arrow points, or None if no arrow is drawn."""
    hsv = cv2.cvtColor(arrow_img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 110) & (hsv[..., 1] > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(m)
    if len(xs) < 300:
        return None
    P = np.stack([xs, ys], 1).astype(float)
    P -= P.mean(0)
    _, v = np.linalg.eigh(P.T @ P)
    d = v[:, 1]
    proj = P @ d
    # The tail block is heavier than the head, so the centroid sits toward the tail:
    # the head is the end farther from the centroid.
    if (proj.max() + proj.min()) / 2 < 0:
        d = -d
    return float(np.degrees(np.arctan2(-d[1], d[0])))


def read_wind(text_img, arrow_img):
    """Wind, NO_WIND, or None when the panel can't be parsed."""
    direction = arrow_direction(arrow_img)
    if direction is None:
        return NO_WIND
    mph = read_digits(text_img)
    if mph is None:
        return None
    return Wind(float(mph), direction)
