"""Original, hand-authored circuit-inspired layouts. Coordinates are metres."""

import hashlib
import json
from pathlib import Path
import numpy as np

TRACK_DIR = Path(__file__).parent / "tracks"


def names():
    return sorted(
        (p.stem for p in TRACK_DIR.glob("*.json")), key=lambda n: (n.endswith("_long"), n)
    )


class Track:
    def __init__(self, name):
        path = TRACK_DIR / f"{name}.json" if name in names() else Path(name)
        raw = path.read_bytes()
        spec = json.loads(raw)
        self.name = spec["name"]
        self.version = spec["version"]
        self.digest = hashlib.sha256(raw).hexdigest()
        self.width = float(spec["width"])
        self.shoulder = float(spec.get("shoulder", 2.0))
        if not np.isfinite(self.shoulder) or not 0 <= self.shoulder <= 10:
            raise ValueError("Shoulder must be between zero and ten metres")
        self.barrier_offset = self.width / 2 + self.shoulder
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
        profile = np.asarray(spec.get("profile", [[0, 0, 0], [0.5, 0, 0]]), dtype=float)
        if (
            profile.ndim != 2
            or profile.shape[1] != 3
            or len(profile) < 2
            or not np.isfinite(profile).all()
            or profile[0, 0] != 0
            or np.any(np.diff(profile[:, 0]) <= 0)
            or profile[-1, 0] >= 1
            or np.max(np.abs(profile[:, 2])) > 15
        ):
            raise ValueError("Profile requires ordered [lap_fraction, height_m, bank_degrees] rows")
        self.profile_s = profile[:, 0] * self.length
        self.profile_values = profile[:, 1:].copy()
        self.profile_values[:, 1] = np.radians(self.profile_values[:, 1])
        prev = np.roll(self.profile_s, 1)
        prev[0] -= self.length
        nxt = np.roll(self.profile_s, -1)
        nxt[-1] += self.length
        self.profile_slopes = (
            np.roll(self.profile_values, -1, axis=0) - np.roll(self.profile_values, 1, axis=0)
        ) / (nxt - prev)[:, None]

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

    def project_many(self, positions, hint=None):
        """Batch projection; a hint restricts collision queries to the local road."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        indices = np.arange(len(self.points))
        if hint is not None:
            delta = (self.s[:-1] - hint + self.length / 2) % self.length - self.length / 2
            indices = indices[np.abs(delta) < 40]
        pts, vec, length = self.points[indices], self.vec[indices], self.seglen[indices]
        all_s, all_lateral, all_angle = [], [], []
        for start in range(0, len(positions), 256):
            p = positions[start : start + 256]
            t = np.clip(np.sum((p[:, None] - pts) * vec, axis=2) / length**2, 0, 1)
            error = p[:, None] - (pts + t[:, :, None] * vec)
            closest = np.argmin(np.sum(error**2, axis=2), axis=1)
            rows = np.arange(len(p))
            idx = indices[closest]
            tangent = self.tangents[idx]
            e = error[rows, closest]
            all_s.append(self.s[idx] + t[rows, closest] * self.seglen[idx])
            all_lateral.append(tangent[:, 0] * e[:, 1] - tangent[:, 1] * e[:, 0])
            all_angle.append(np.arctan2(tangent[:, 1], tangent[:, 0]))
        return tuple(np.concatenate(values) for values in (all_s, all_lateral, all_angle))

    def profile(self, distance):
        """Periodic cubic Hermite height, grade (dz/ds), and bank angle."""
        distance = np.asarray(distance) % self.length
        i = np.searchsorted(self.profile_s, distance, side="right") - 1
        j = (i + 1) % len(self.profile_s)
        span = (self.profile_s[j] - self.profile_s[i]) % self.length
        t = (distance - self.profile_s[i]) / span
        a, b = self.profile_values[i], self.profile_values[j]
        ma, mb = self.profile_slopes[i] * span[..., None], self.profile_slopes[j] * span[..., None]
        values = (
            (2 * t**3 - 3 * t**2 + 1)[..., None] * a
            + (t**3 - 2 * t**2 + t)[..., None] * ma
            + (-2 * t**3 + 3 * t**2)[..., None] * b
            + (t**3 - t**2)[..., None] * mb
        )
        derivative = (
            (6 * t * t - 6 * t)[..., None] * a
            + (3 * t * t - 4 * t + 1)[..., None] * ma
            + (-6 * t * t + 6 * t)[..., None] * b
            + (3 * t * t - 2 * t)[..., None] * mb
        ) / span[..., None]
        return values[..., 0], derivative[..., 0], values[..., 1]

    def surface(self, positions):
        if not np.any(self.profile_values):
            return np.zeros(len(positions))
        s, lateral, _ = self.project_many(positions)
        height, _, bank = self.profile(s)
        # Banking blends into the surrounding terrain beyond the shoulder.
        return height + np.clip(
            lateral, -self.barrier_offset - 10, self.barrier_offset + 10
        ) * np.tan(bank)
