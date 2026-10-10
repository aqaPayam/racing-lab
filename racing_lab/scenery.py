"""Trackside scale cues and braking boards for the procedural circuits."""

import numpy as np
from .scene3d import Scene3D


def details(track):
    faces = []
    rng = np.random.default_rng(412)
    # Original procedural broadleaf woodland. Nearby parallel track sections
    # are checked so scenery cannot grow through another portion of the road.
    for distance in np.arange(0, track.length, 22):
        if distance < 100 or distance > track.length - 50:
            continue
        p = track.sample(distance)
        _, _, angle = track.project(p)
        normal = np.array([-np.sin(angle), np.cos(angle)])
        for side in (-1, 1):
            center = p + normal * side * (track.width / 2 + rng.uniform(14, 45))
            if abs(track.project(center)[1]) < track.width / 2 + 10:
                continue
            height = rng.uniform(6, 11)
            faces += Scene3D.box([*center, height * 0.3], [0.35, 0.35, height * 0.6], (91, 76, 55))
            for crown in range(2):
                radius = rng.uniform(2, 3.2)
                origin = np.array([*center, height * 0.68]) + [
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                    crown * 1.1,
                ]
                rings = []
                for lat in np.linspace(-np.pi / 2, np.pi / 2, 6):
                    angles = np.arange(10) * 2 * np.pi / 10
                    ring = np.column_stack(
                        [
                            radius * np.cos(lat) * np.cos(angles),
                            radius * np.cos(lat) * np.sin(angles),
                            np.full(10, radius * 1.2 * np.sin(lat)),
                        ]
                    )
                    rings.append(ring + origin)
                for a, b in zip(rings, rings[1:]):
                    for i in range(10):
                        j = (i + 1) % 10
                        color = (
                            int(rng.integers(42, 58)),
                            int(rng.integers(89, 116)),
                            int(rng.integers(35, 51)),
                        )
                        faces.append((np.array([a[i], a[j], b[j], b[i]]), color))
    # Seated spectators give the pit straight recognizable human scale.
    p = track.sample(30)
    _, _, angle = track.project(p)
    f = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-f[1], f[0]])
    colors = [(133, 45, 43), (66, 89, 115), (207, 193, 161), (63, 78, 69)]
    for row in range(5):
        for x in np.arange(-15, 25, 1.1):
            if rng.random() < 0.15:
                continue
            loc = p + f * x + n * (-track.width / 2 - 11 - row * 1.6)
            z = 1 + row * 0.65
            faces += Scene3D.box(
                [*loc, z + 0.42], [0.4, 0.38, 0.62], colors[int(rng.integers(4))], angle
            )
            faces += Scene3D.box([*loc, z + 0.87], [0.23, 0.23, 0.25], (172, 135, 108), angle)
    # Detect separated bends, then place 150/100/50 m approach boards.
    s = np.arange(0, track.length, 10.0)
    p = track.sample(s)
    d = np.roll(p, -1, axis=0) - p
    h = np.arctan2(d[:, 1], d[:, 0])
    curvature = np.abs((np.roll(h, -1) - h + np.pi) % (2 * np.pi) - np.pi)
    bends = []
    for i in np.argsort(-curvature):
        if curvature[i] < 0.10:
            break
        if all(
            abs((s[i] - b + track.length / 2) % track.length - track.length / 2) > 180
            for b in bends
        ):
            bends.append(s[i])
    digits = {"0": "abcdef", "1": "bc", "5": "afgcd"}
    segments = {
        "a": (0, 0.4, 0.24, 0.055),
        "b": (0.12, 0.2, 0.05, 0.22),
        "c": (0.12, -0.2, 0.05, 0.22),
        "d": (0, -0.4, 0.24, 0.055),
        "e": (-0.12, -0.2, 0.05, 0.22),
        "f": (-0.12, 0.2, 0.05, 0.22),
        "g": (0, 0, 0.24, 0.055),
    }
    for bend in bends:
        for distance in (150, 100, 50):
            center = track.sample(bend - distance)
            _, _, angle = track.project(center)
            f = np.array([np.cos(angle), np.sin(angle)])
            n = np.array([-f[1], f[0]])
            center += n * (track.width / 2 + 3.7)
            # Board plane faces the approaching driver.
            faces += Scene3D.box([*center, 1.45], [0.10, 1.45, 1.05], (238, 237, 218), angle)
            faces += Scene3D.box([*center, 0.6], [0.1, 0.1, 1.2], (110, 115, 110), angle)
            text = str(distance)
            for j, char in enumerate(text):
                for segment in digits[char]:
                    y, z, width, height = segments[segment]
                    loc = center - f * 0.061 + n * (y + (j - (len(text) - 1) / 2) * 0.39)
                    faces += Scene3D.box(
                        [*loc, 1.45 + z], [0.008, width, height], (25, 29, 31), angle
                    )
    # Regular small reflector posts provide a reliable sense of distance/speed.
    for distance in np.arange(0, track.length, 25):
        center = track.sample(distance)
        _, _, angle = track.project(center)
        n = np.array([-np.sin(angle), np.cos(angle)])
        for side in (-1, 1):
            loc = center + n * side * (track.width / 2 + 4.3)
            faces += Scene3D.box([*loc, 0.48], [0.1, 0.15, 0.96], (220, 219, 204), angle)
            faces += Scene3D.box([*loc, 0.78], [0.12, 0.17, 0.14], (212, 57, 42), angle)
    return faces
