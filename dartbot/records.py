"""Throw records: the dataset the physics model is fitted on (JSON lines)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Throw:
    source: str
    plat: tuple[float, float]
    wind_mph: float
    wind_dir: float
    swing: list[tuple[float, float]]           # (t, sprite angle) while held, before release
    release_t: float                          # last held frame (video) or click time (live)
    path: list[tuple[float, float, float]]    # (t, x, y) dart centroid in flight
    score_before: int | None = None
    score_after: int | None = None
    click_t: float | None = None              # live only
    target_sprite: float | None = None        # live only: sprite angle we aimed for
    extra: dict = field(default_factory=dict)

    @property
    def points(self):
        if self.score_before is None or self.score_after is None:
            return None
        return self.score_after - self.score_before


def append(path: Path, t: Throw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(t)) + "\n")


def load(*paths: Path) -> list[Throw]:
    out = []
    for p in paths:
        if not Path(p).exists():
            continue
        for line in open(p):
            if line.strip():
                out.append(Throw(**json.loads(line)))
    return out
