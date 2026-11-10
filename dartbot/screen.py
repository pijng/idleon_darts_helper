"""Locating the game canvas and grabbing frames in reference coordinates."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C

_logo = C.read_image(C.ASSETS / "logo.png", cv2.IMREAD_GRAYSCALE)
_exit = C.read_image(C.ASSETS / "exit.png", cv2.IMREAD_GRAYSCALE)


@dataclass
class Canvas:
    """Maps reference coordinates to physical image pixels: phys = origin + scale * ref."""
    ox: float
    oy: float
    scale: float

    def phys_rect(self, rect):
        x0, y0, x1, y1 = rect
        return (self.ox + x0 * self.scale, self.oy + y0 * self.scale,
                self.ox + x1 * self.scale, self.oy + y1 * self.scale)


def _best_match(gray, tmpl, scales):
    best = (-1.0, 1.0, (0, 0))
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] < 8 or t.shape[0] > gray.shape[0] or t.shape[1] > gray.shape[1]:
            continue
        _, v, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED))
        if v > best[0]:
            best = (v, s, loc)
    return best


def _locate_anchor(gray, tmpl, work=0.25):
    small = cv2.resize(gray, None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    v, s, loc = _best_match(small, tmpl, np.geomspace(0.15, 1.6, 48) * work)
    s_full = s / work
    # Refine at full resolution around the coarse hit.
    th, tw = int(tmpl.shape[0] * s_full), int(tmpl.shape[1] * s_full)
    x, y = int(loc[0] / work), int(loc[1] / work)
    pad = max(16, int(0.15 * max(th, tw)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    sub = gray[y0:y + th + pad, x0:x + tw + pad]
    v, s2, loc2 = _best_match(sub, tmpl, np.linspace(s_full * 0.93, s_full * 1.07, 15))
    return v, s2, (x0 + loc2[0], y0 + loc2[1])


def locate_canvas(img, min_score=0.7) -> Canvas | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    v1, s1, l1 = _locate_anchor(gray, _logo)
    v2, s2, l2 = _locate_anchor(gray, _exit)
    if min(v1, v2) < min_score:
        return None
    # Scale from anchor separation is far more precise than template scale.
    scale = (l2[1] - l1[1]) / (C.ANCHOR_EXIT[1] - C.ANCHOR_LOGO[1])
    if not 0.8 < scale / ((s1 + s2) / 2) < 1.25:
        return None
    ox = l1[0] - scale * C.ANCHOR_LOGO[0]
    oy = l1[1] - scale * C.ANCHOR_LOGO[1]
    return Canvas(ox, oy, scale)


def to_ref(img, canvas: Canvas, rect):
    """Crop `rect` (reference coords) from a physical image and resample to reference scale."""
    x0, y0, x1, y1 = rect
    w, h = int(round(x1 - x0)), int(round(y1 - y0))
    M = np.float32([[canvas.scale, 0, canvas.ox + x0 * canvas.scale],
                    [0, canvas.scale, canvas.oy + y0 * canvas.scale]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class ScreenSource:
    """Live capture through mss. Coordinates handed to mss/pynput are logical points;
    images may come back in physical pixels (dpr is measured, not assumed)."""

    def __init__(self, monitor_index: int | None = None):
        import mss
        self._sct = mss.MSS()
        mons = self._sct.monitors
        self.monitor = mons[monitor_index] if monitor_index is not None else mons[0]
        self.canvas: Canvas | None = None
        self.dpr = 1.0

    def full(self):
        shot = self._sct.grab(self.monitor)
        img = np.asarray(shot)[:, :, :3]
        self.dpr = img.shape[1] / self.monitor["width"]
        return img

    def locate(self) -> Canvas | None:
        self.canvas = locate_canvas(self.full())
        return self.canvas

    def grab(self, rect):
        """Return (image in reference scale, timestamp at capture midpoint)."""
        cv_ = self.canvas
        px0, py0, px1, py1 = cv_.phys_rect(rect)
        d = self.dpr
        region = {
            "left": int(np.floor(px0 / d)) + self.monitor["left"],
            "top": int(np.floor(py0 / d)) + self.monitor["top"],
        }
        region["width"] = int(np.ceil(px1 / d)) + self.monitor["left"] - region["left"]
        region["height"] = int(np.ceil(py1 / d)) + self.monitor["top"] - region["top"]
        t0 = time.perf_counter()
        img = np.asarray(self._sct.grab(region))[:, :, :3]
        t1 = time.perf_counter()
        # The region we got starts at a whole point; build a local canvas for it.
        lx = (region["left"] - self.monitor["left"]) * d
        ly = (region["top"] - self.monitor["top"]) * d
        local = Canvas(cv_.ox - lx, cv_.oy - ly, cv_.scale)
        return to_ref(img, local, rect), (t0 + t1) / 2

    def ref_to_point(self, x, y):
        """Reference coordinate -> logical screen point for the mouse."""
        cv_ = self.canvas
        return ((cv_.ox + x * cv_.scale) / self.dpr + self.monitor["left"],
                (cv_.oy + y * cv_.scale) / self.dpr + self.monitor["top"])


class VideoSource:
    """Offline frames from a recording, with true (VFR) timestamps."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.canvas: Canvas | None = None
        self.frame = None
        self.t = 0.0
        self.index = -1

    def next(self) -> bool:
        ok, f = self.cap.read()
        if not ok:
            return False
        self.frame = f
        self.t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.index += 1
        if self.canvas is None:
            self.canvas = locate_canvas(f)
            if self.canvas is None:
                raise RuntimeError("game canvas not found in first video frame")
        return True

    def grab(self, rect):
        return to_ref(self.frame, self.canvas, rect), self.t


class ReplaySource:
    """Plays a recording in real time behind the live-capture interface (dry-run testing)."""

    def __init__(self, path, speed=1.0):
        self.video = VideoSource(path)
        self.video.next()
        self.canvas = self.video.canvas
        self.speed = speed
        self.start = None

    def locate(self):
        return self.canvas

    def grab(self, rect):
        now = time.perf_counter()
        if self.start is None:
            self.start = now - self.video.t
        target = (now - self.start) * self.speed
        while self.video.t < target:
            if not self.video.next():
                raise EOFError("end of recording")
        img, _ = self.video.grab(rect)
        return img, now
