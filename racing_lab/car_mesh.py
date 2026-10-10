"""Original procedural open-wheel car, in metres. No external assets required."""

import numpy as np
from .scene3d import Scene3D

RED = (203, 23, 35)
CARBON = (24, 28, 33)


def tube(a, b, radius, color, segments=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    axis /= np.linalg.norm(axis)
    reference = [0, 0, 1] if abs(axis[2]) < 0.9 else [1, 0, 0]
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = np.arange(segments) * 2 * np.pi / segments
    ring = radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    return [
        (
            np.array(
                [
                    a + ring[i],
                    a + ring[(i + 1) % segments],
                    b + ring[(i + 1) % segments],
                    b + ring[i],
                ]
            ),
            color,
        )
        for i in range(segments)
    ]


def loft(sections, color):
    """Sections: longitudinal position, half width, bottom, top; bevel shoulders."""
    rings = []
    for x, w, bottom, top in sections:
        rings.append(
            np.array(
                [
                    [x, -w * 0.7, top],
                    [x, w * 0.7, top],
                    [x, w, top * 0.75 + bottom * 0.25],
                    [x, w, bottom],
                    [x, -w, bottom],
                    [x, -w, top * 0.75 + bottom * 0.25],
                ]
            )
        )
    faces = [(rings[0][::-1], color), (rings[-1], color)]
    for a, b in zip(rings, rings[1:]):
        for i in range(6):
            j = (i + 1) % 6
            faces.append((np.array([a[i], b[i], b[j], a[j]]), color))
    return faces


def wheel():
    faces = []
    # Rounded shoulder and recessed sidewall; axle is the local Y direction.
    profile = [
        (-0.20, 0.245),
        (-0.20, 0.31),
        (-0.17, 0.355),
        (-0.13, 0.36),
        (0.13, 0.36),
        (0.17, 0.355),
        (0.20, 0.31),
        (0.20, 0.245),
    ]
    t = np.arange(40) * 2 * np.pi / 40

    def ring(y, r):
        return np.column_stack([r * np.cos(t), np.full(40, y), r * np.sin(t)])

    for (a, ra), (b, rb) in zip(profile, profile[1:]):
        u, v = ring(a, ra), ring(b, rb)
        for i in range(40):
            j = (i + 1) % 40
            faces.append((np.array([u[i], u[j], v[j], v[i]]), (22, 23, 25)))
    for side in (-1, 1):
        y = side * 0.202
        faces.append((ring(y, 0.245), (36, 39, 43)))
        # Thin tyre compound marking, with distinct depths to avoid z-fighting.
        u, v = ring(y, 0.294), ring(y, 0.282)
        for i in range(40):
            j = (i + 1) % 40
            faces.append((np.array([u[i], u[j], v[j], v[i]]), (218, 178, 53)))
        faces.append((ring(side * 0.204, 0.20), (72, 76, 81)))
        faces.append((ring(side * 0.21, 0.065), (177, 181, 182)))
        for theta in np.arange(10) * 2 * np.pi / 10:
            vertices = [
                [r * np.cos(theta + a), side * 0.215, r * np.sin(theta + a)]
                for r, a in [(0.055, -0.15), (0.193, -0.06), (0.193, 0.06), (0.055, 0.15)]
            ]
            faces.append((np.array(vertices), (158, 164, 172)))
    return Scene3D.pack(faces)


class CarMesh:
    def __init__(self):
        faces = []
        box = Scene3D.box
        faces += loft(
            [
                (-2.25, 0.19, 0.18, 0.40),
                (-1.5, 0.43, 0.16, 0.64),
                (-0.3, 0.44, 0.16, 0.68),
                (0.7, 0.27, 0.18, 0.59),
                (2.55, 0.12, 0.22, 0.32),
            ],
            RED,
        )
        faces += loft(
            [(-1.6, 0.10, 0.4, 0.65), (-0.95, 0.20, 0.4, 1.02), (-0.60, 0.22, 0.4, 0.96)], RED
        )
        faces += box([-0.45, 0, 0.16], [3.45, 1.55, 0.07], CARBON)
        for side in (-1, 1):
            pod = loft(
                [
                    (-1.8, 0.15, 0.23, 0.39),
                    (-1.0, 0.25, 0.22, 0.54),
                    (0.05, 0.22, 0.24, 0.53),
                    (0.3, 0.15, 0.3, 0.48),
                ],
                RED,
            )
            faces += [(v + [0, side * 0.46, 0], c) for v, c in pod]
            faces += box([0.23, side * 0.47, 0.44], [0.035, 0.29, 0.14], CARBON)
            for x in (-1.6, 1.8):
                for z in (0.25, 0.46):
                    for dx in (-0.4, 0.3):
                        faces += tube(
                            [x + dx, side * 0.25, z], [x, side * 0.74, 0.36], 0.025, CARBON
                        )
            faces += tube([0.3, side * 0.24, 0.65], [0.35, side * 0.57, 0.76], 0.022, CARBON)
            faces += box([0.35, side * 0.59, 0.77], [0.22, 0.13, 0.08], RED)
            faces += box([2.47, side * 0.86, 0.28], [0.48, 0.05, 0.27], RED)
            faces += box([-2.27, side * 0.88, 0.87], [0.58, 0.045, 0.5], RED)
            faces += tube([-0.55, side * 0.28, 0.69], [0.35, side * 0.28, 0.91], 0.035, CARBON)
        faces += tube([0.35, -0.28, 0.91], [0.35, 0.28, 0.91], 0.035, CARBON)
        faces += tube([0.35, 0, 0.91], [0.58, 0, 0.57], 0.032, CARBON)
        for x, z, width in [
            (2.5, 0.14, 1.8),
            (2.3, 0.20, 1.65),
            (2.12, 0.25, 1.5),
            (-2.3, 0.97, 1.8),
            (-2.15, 1.06, 1.72),
        ]:
            faces += box([x, 0, z], [0.23, width, 0.055], CARBON)
        faces += box([-0.15, 0, 0.685], [0.72, 0.45, 0.05], CARBON)
        # Helmet ellipsoid and dark visor.
        for j in range(10):
            a, b = -np.pi / 2 + j * np.pi / 10, -np.pi / 2 + (j + 1) * np.pi / 10
            for i in range(20):
                ts = [i * np.pi / 10, (i + 1) * np.pi / 10]
                v = [
                    [
                        -0.2 + 0.17 * np.cos(lat) * np.cos(t),
                        0.16 * np.cos(lat) * np.sin(t),
                        0.87 + 0.20 * np.sin(lat),
                    ]
                    for lat, t in [(a, ts[0]), (a, ts[1]), (b, ts[1]), (b, ts[0])]
                ]
                color = (32, 50, 64) if j in (5, 6) and (i < 5 or i > 14) else (238, 192, 44)
                faces.append((np.array(v), color))
        self.body, self.body_colors = Scene3D.pack(faces)
        self.wheel, self.wheel_colors = wheel()

    def mesh(self, env):
        phase = env.distance / 0.36
        c, s = np.cos(phase), np.sin(phase)
        spin = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        meshes, colors = [self.body], [self.body_colors]
        for x in (-1.6, 1.8):
            steer = env.steering if x > 0 else 0
            c, s = np.cos(steer), np.sin(steer)
            turn = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            for side in (-1, 1):
                meshes.append(
                    (self.wheel.reshape(-1, 3) @ (spin @ turn)).reshape(self.wheel.shape)
                    + [x, side * 0.75, 0.37]
                )
                colors.append(self.wheel_colors)
        c, s = np.cos(env.heading), np.sin(env.heading)
        rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        mesh = np.concatenate(meshes)
        return (mesh.reshape(-1, 3) @ rotation).reshape(mesh.shape) + [*env.pos, 0], np.concatenate(
            colors
        )
