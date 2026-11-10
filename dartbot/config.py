"""Game geometry in reference coordinates.

All detection and physics work in a fixed "reference" pixel space: the game
canvas as captured in board.png (3020x1710). Any real window is mapped into
this space by `screen.Canvas`, so model parameters are resolution independent.
"""
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).parent / "assets"


def read_image(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also works for non-ASCII paths on Windows (e.g. C:\\Users\\Марк)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_png(path, img):
    """cv2.imwrite counterpart of read_image."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


REF_W, REF_H = 3020, 1710

# Anchor templates and their top-left positions in reference space.
ANCHOR_LOGO = (2540, 15)    # "THROWY DARTS" banner, assets/logo.png
ANCHOR_EXIT = (2815, 1595)  # EXIT button, assets/exit.png

# Rects are (x0, y0, x1, y1) in reference space.
# The platform can sit as low as y ~1670 (seen live), so the field runs to the bottom.
FIELD = (0, 200, 2885, 1710)
# Darts never fly below the board; the bottom strip shows banner text between throws.
FLIGHT_MAX_Y = 1640
SCORE_TEXT = (80, 100, 330, 160)
WIND_TEXT = (1540, 95, 1760, 160)
WIND_ARROW = (1760, 30, 1905, 165)

# Board: vertical strip on the right. A dart's centroid trajectory evaluated
# at BOARD_X decides which segment it scores (validated on 15/15 video throws).
BOARD_X = 2890.0
SEGMENTS = (  # (y0, y1, points)
    (298, 515, 1),
    (515, 699, 2),
    (699, 869, 3),
    (869, 995, 5),
    (995, 1157, 3),
    (1157, 1343, 2),
    (1343, 1566, 1),
)
BULLSEYE = (869, 995)
BULLSEYE_Y = (BULLSEYE[0] + BULLSEYE[1]) / 2

# Player geometry relative to the platform template top-left (px, py).
SWING_ROI = (50, -420, 650, 60)   # region holding the player + dart
SHOULDER = (200, -150)            # rough rotation pivot used to find the dart tip
RELEASE_DX = 300.0                # trajectories are parameterised at x = px + RELEASE_DX

# Held-dart test: distance from SHOULDER to the farthest bright pixel.
HELD_TIP_DIST = (183.0, 235.0)
HELD_ANGLE_MIN = -60.0


def segment_points(y: float) -> int:
    for y0, y1, pts in SEGMENTS:
        if y0 <= y < y1:
            return pts
    return 0
