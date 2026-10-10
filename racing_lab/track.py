"""Original, hand-authored circuit-inspired layouts. Coordinates are metres."""

import hashlib
import json
from pathlib import Path
import numpy as np

TRACK_DIR = Path(__file__).parent / "tracks"


def names():
    return sorted(p.stem for p in TRACK_DIR.glob("*.json"))


class Track:
    def __init__(self, name):
        path = TRACK_DIR / f"{name}.json" if name in names() else Path(name)
        raw = path.read_bytes()
        spec = json.loads(raw)
        self.name = spec["name"]
        self.version = spec["version"]
        self.digest = hashlib.sha256(raw).hexdigest()
        self.width = float(spec["width"])
        controls = np.asarray(spec["points"], dtype=float)
        if not np.isfinite(self.width) or self.width <= 2:
            raise ValueError("Track width must be finite and greater than 2 metres")
        if (
            controls.ndim != 2
            or controls.shape[1] != 2
            or len(controls) < 4
            or not np.isfinite(controls).all()
        ):
            raise ValueError("Track requires at least four finite [x,y] control points")
        if np.any(np.linalg.norm(np.roll(controls, -1, axis=0) - controls, axis=1) < 1e-6):
            raise ValueError(
                "Adjacent control points must differ; do not repeat the starting point"
            )
        # Periodic Catmull-Rom interpolation; retain tight turns without global splines.
        points = []
        for i in range(len(controls)):
            a, b, c, d = [controls[j % len(controls)] for j in (i - 1, i, i + 1, i + 2)]
            n = max(12, int(np.linalg.norm(c - b) / 3))
            t = np.arange(n)[:, None] / n
            points.extend(
                0.5
                * (
                    (2 * b)
                    + (-a + c) * t
                    + (2 * a - 5 * b + 4 * c - d) * t * t
                    + (-a + 3 * b - 3 * c + d) * t * t * t
                )
            )
        self.points = np.asarray(points)
        self.vec = np.roll(self.points, -1, axis=0) - self.points
        self.seglen = np.linalg.norm(self.vec, axis=1)
        if np.any(self.seglen < 1e-9):
            raise ValueError("Track interpolation produced a zero-length segment")
        self.s = np.r_[0, np.cumsum(self.seglen)]
        self.length = self.s[-1]
        self.tangents = self.vec / self.seglen[:, None]

    def sample(self, distance):
        distance = np.asarray(distance) % self.length
        idx = np.searchsorted(self.s, distance, side="right") - 1
        return (
            self.points[idx]
            + self.vec[idx] * ((distance - self.s[idx]) / self.seglen[idx])[..., None]
        )

    def project(self, pos):
        t = np.clip(np.sum((pos - self.points) * self.vec, axis=1) / self.seglen**2, 0, 1)
        nearest = self.points + t[:, None] * self.vec
        idx = np.argmin(np.sum((nearest - pos) ** 2, axis=1))
        tangent = self.tangents[idx]
        error = pos - nearest[idx]
        lateral = tangent[0] * error[1] - tangent[1] * error[0]
        return (
            float(self.s[idx] + t[idx] * self.seglen[idx]),
            float(lateral),
            float(np.arctan2(tangent[1], tangent[0])),
        )
