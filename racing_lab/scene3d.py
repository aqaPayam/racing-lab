"""Portable perspective renderer. All scenery is visual; simulation stays in metres.

Meshes are shaded in world space, clipped at the near plane, and painter-sorted.
No graphics driver, assets, or network downloads are required.
"""

import numpy as np
import pygame


def clip_near(vertices, near=0.35):
    """Clip a camera-space polygon before perspective division."""
    output = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        inside_a, inside_b = a[2] >= near, b[2] >= near
        if inside_a:
            output.append(a)
        if inside_a != inside_b:
            output.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
    return np.asarray(output)


class Scene3D:
    def __init__(self):
        self.camera = 0
        self.track_hash = None
        self.faces = []

    @staticmethod
    def box(center, size, color, angle=0):
        x, y, z = np.asarray(size) / 2
        corners = np.array(
            [
                [-x, -y, -z],
                [x, -y, -z],
                [x, y, -z],
                [-x, y, -z],
                [-x, -y, z],
                [x, -y, z],
                [x, y, z],
                [-x, y, z],
            ]
        )
        c, s = np.cos(angle), np.sin(angle)
        corners[:, :2] = corners[:, :2] @ np.array([[c, s], [-s, c]])
        corners += center
        return [
            (corners[list(ids)], tuple(int(v * shade) for v in color))
            for ids, shade in [
                ((0, 1, 5, 4), 0.7),
                ((1, 2, 6, 5), 0.82),
                ((2, 3, 7, 6), 0.6),
                ((3, 0, 4, 7), 0.9),
                ((4, 5, 6, 7), 1.0),
            ]
        ]

    def build_track(self, track):
        self.faces = []
        samples = track.sample(np.arange(0, track.length, 6.0))
        tangent = np.roll(samples, -1, axis=0) - np.roll(samples, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        width = track.width / 2

        def strip(a, b, color, z=0.02):
            inner = samples + normal * a
            outer = samples + normal * b
            for i in range(len(samples)):
                j = (i + 1) % len(samples)
                quad = np.column_stack(
                    [np.array([inner[i], inner[j], outer[j], outer[i]]), np.full(4, z)]
                )
                self.faces.append((quad, color(i)))

        strip(-width, width, lambda i: (62, 67, 72) if i % 2 else (64, 69, 74))
        for side in (-1, 1):
            strip(side * (width - 0.12), side * width, lambda i: (224, 224, 210), 0.035)
            strip(
                side * width,
                side * (width + 0.8),
                lambda i: (201, 54, 45) if i % 2 else (226, 226, 214),
                0.015,
            )
            strip(side * (width + 0.8), side * (width + 3.0), lambda i: (141, 150, 111), 0.005)
        # Trackside guardrails and evenly spaced posts; kept outside the runoff.
        for i in range(0, len(samples), 4):
            j = (i + 4) % len(samples)
            for side in (-1, 1):
                a = samples[i] + normal[i] * side * (width + 5)
                b = samples[j] + normal[j] * side * (width + 5)
                q = np.array([[*a, 0.3], [*b, 0.3], [*b, 1.1], [*a, 1.1]])
                self.faces.append((q, (157, 170, 178)))
                self.faces.append((q[::-1], (157, 170, 178)))
                self.faces.extend(self.box([*a, 0.55], [0.16, 0.16, 1.1], (111, 123, 131)))
        # Trees are deterministic scenery, never obstacles or random simulator state.
        rng = np.random.default_rng(921)
        for i in range(0, len(samples), 10):
            side = 1 if i % 20 else -1
            p = samples[i] + normal[i] * side * (width + 15 + rng.uniform(0, 18))
            _, offset, _ = track.project(p)
            if abs(offset) < width + 9:
                continue
            height = rng.uniform(5, 9)
            self.faces.extend(self.box([*p, 1.4], [0.45, 0.45, 2.8], (100, 78, 54)))
            for z, r in [(2.2, 2.4), (3.8, 1.9), (5.2, 1.3)]:
                base = np.array(
                    [
                        [p[0] + r * np.cos(t), p[1] + r * np.sin(t), z]
                        for t in np.arange(6) * np.pi / 3
                    ]
                )
                tip = np.array([*p, z + height * 0.4])
                for k in range(6):
                    self.faces.append(
                        (
                            np.array([base[k], base[(k + 1) % 6], tip]),
                            (36 + k * 3, 87 + k * 4, 54 + k * 2),
                        )
                    )
        # Pit building, grandstand and start gantry aligned to the opening straight.
        p = track.sample(30)
        _, _, angle = track.project(p)
        f = np.array([np.cos(angle), np.sin(angle)])
        n = np.array([-f[1], f[0]])

        def local(x, y, z):
            return [*(p + x * f + y * n), z]

        self.faces.extend(self.box(local(0, width + 13, 6.3), [50, 11, 0.6], (44, 63, 73), angle))
        for x in range(-21, 24, 6):
            for z, height, color in [
                (1.5, 3, (48, 68, 81)),
                (3.6, 1.2, (205, 201, 179)),
                (4.85, 1.3, (117, 171, 188)),
                (5.75, 0.5, (205, 201, 179)),
            ]:
                self.faces.extend(
                    self.box(local(x, width + 13, z), [5.5, 10, height], color, angle)
                )
            self.faces.extend(
                self.box(local(x + 2.75, width + 13, 3), [0.5, 10, 6], (205, 201, 179), angle)
            )
        for row in range(5):
            self.faces.extend(
                self.box(
                    local(5, -width - 11 - row * 1.6, 0.5 + row * 0.65),
                    [44, 1.6, 1 + row * 1.3],
                    (104 + row * 13, 113 + row * 9, 126 + row * 5),
                    angle,
                )
            )
        self.faces.extend(
            self.box(local(5, -width - 15, 6.2), [48, 12, 0.4], (224, 225, 214), angle)
        )
        for y in (-width - 1, width + 1):
            self.faces.extend(self.box(local(-25, y, 3.7), [0.5, 0.5, 7.4], (70, 83, 92), angle))
        self.faces.extend(
            self.box(local(-25, 0, 7.1), [1.1, track.width + 3, 1.4], (26, 44, 57), angle)
        )
        # Checkerboard start stripe, using actual start geometry.
        start = track.points[0]
        f = track.tangents[0]
        n = np.array([-f[1], f[0]])
        for row in range(2):
            for col in range(int(track.width)):
                v = [
                    start + f * (row * 0.6 + a) + n * (-width + col + b)
                    for a, b in [(0, 0), (0.6, 0), (0.6, 1), (0, 1)]
                ]
                self.faces.append(
                    (
                        np.column_stack([v, np.full(4, 0.06)]),
                        (239, 238, 222) if (row + col) % 2 else (28, 32, 37),
                    )
                )
        self.centers = np.array([v.mean(axis=0) for v, _ in self.faces])
        self.mesh, self.colors = self.pack(self.faces)
        self.mesh_centers = self.mesh.mean(axis=1)
        self.track_hash = track.digest

    @staticmethod
    def pack(faces):
        triangles = []
        colors = []
        for vertices, color in faces:
            if (
                len(vertices) == 4
                and np.max(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)) > 30
            ):
                # Subdivide long building faces so depth sorting does not hide
                # nearby windows behind the far half of a single huge triangle.
                count = max(2, int(np.linalg.norm(vertices[1] - vertices[0]) / 4))
                for t in range(count):
                    a = t / count
                    b = (t + 1) / count
                    quad = np.array(
                        [
                            vertices[0] * (1 - a) + vertices[1] * a,
                            vertices[0] * (1 - b) + vertices[1] * b,
                            vertices[3] * (1 - b) + vertices[2] * b,
                            vertices[3] * (1 - a) + vertices[2] * a,
                        ]
                    )
                    triangles.extend([quad[[0, 1, 2]], quad[[0, 2, 3]]])
                    colors.extend([color, color])
                continue
            for i in range(1, len(vertices) - 1):
                triangles.append(vertices[[0, i, i + 1]])
                colors.append(color)
        return np.asarray(triangles), np.asarray(colors)

    def car_faces(self, env):
        faces = []
        angle = env.heading
        f = np.array([np.cos(angle), np.sin(angle)])
        n = np.array([-f[1], f[0]])

        def part(x, y, z, size, color, steer=0):
            faces.extend(self.box([*(env.pos + f * x + n * y), z], size, color, angle + steer))

        # Dimensions follow the 5.5 x 1.9 m simulator envelope.
        part(-0.35, 0, 0.42, [3.5, 1.02, 0.44], (204, 40, 41))
        part(1.65, 0, 0.4, [2.2, 0.34, 0.3], (239, 61, 46))
        part(-0.7, 0, 0.72, [1.4, 0.6, 0.35], (228, 53, 44))
        part(-0.05, 0, 0.76, [0.7, 0.52, 0.24], (23, 31, 39))
        part(-0.12, 0, 0.98, [0.35, 0.34, 0.3], (243, 199, 61))
        part(-2.38, 0, 0.91, [0.42, 1.86, 0.12], (31, 39, 49))
        part(-2.3, -0.7, 0.64, [0.1, 0.09, 0.5], (52, 57, 64))
        part(-2.3, 0.7, 0.64, [0.1, 0.09, 0.5], (52, 57, 64))
        part(2.5, 0, 0.21, [0.5, 1.85, 0.1], (37, 43, 51))
        part(0.45, 0, 0.94, [0.07, 0.63, 0.07], (26, 30, 36))
        for side in (-1, 1):
            part(-0.02, side * 0.3, 0.96, [0.94, 0.07, 0.07], (26, 30, 36))
        # Cylindrical tyres with rim hubs and steering angle on the front axle.
        for x in (-1.7, 1.7):
            for y in (-0.75, 0.75):
                wheel_angle = angle + (env.steering if x > 0 else 0)
                wf = np.array([np.cos(wheel_angle), np.sin(wheel_angle)])
                wn = np.array([-wf[1], wf[0]])
                origin = env.pos + f * x + n * y
                rings = []
                for side in (-0.2, 0.2):
                    rings.append(
                        np.array(
                            [
                                [
                                    *(origin + wf * 0.36 * np.cos(t) + wn * side),
                                    0.37 + 0.36 * np.sin(t),
                                ]
                                for t in np.arange(12) * np.pi / 6
                            ]
                        )
                    )
                for k in range(12):
                    j = (k + 1) % 12
                    faces.append(
                        (
                            np.array([rings[0][k], rings[0][j], rings[1][j], rings[1][k]]),
                            (23 + k % 3 * 3, 25 + k % 3 * 3, 28 + k % 3 * 3),
                        )
                    )
                for ring in rings:
                    faces.append((ring, (39, 43, 48)))
                    center = ring.mean(axis=0)
                    faces.append((center + (ring - center) * 0.5, (132, 141, 147)))
        return faces

    def draw(self, viewer):
        env = viewer.env
        screen = viewer.screen
        w, h = screen.get_size()
        if self.track_hash != env.track.digest:
            self.build_track(env.track)
        f = np.array([np.cos(env.heading), np.sin(env.heading), 0.0])
        pos = np.array([*env.pos, 0.0])
        if self.camera == 0:
            eye = pos - f * (10 + env.speed * 0.025) + [0, 0, 4.2]
            target = pos + f * 15 + [0, 0, 0.3]
        elif self.camera == 1:
            eye = pos + f * 0.3 + [0, 0, 1.25]
            target = pos + f * 35 + [0, 0, 1.05]
        else:
            eye = pos - f * 22 + [0, 0, 48]
            target = pos + f * 10
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross([0, 0, 1.0], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        basis = np.column_stack([right, up, forward])
        focal = w * 0.83
        # Sky gradient, with the ground horizon derived from camera pitch.
        for y in range(0, h, 4):
            t = y / h
            pygame.draw.rect(
                screen, (int(100 + 70 * t), int(155 + 47 * t), int(190 + 28 * t)), (0, y, w, 4)
            )
        horizon = int(
            np.clip(h * 0.45 + focal * forward[2] / max(0.01, np.linalg.norm(forward[:2])), 0, h)
        )
        pygame.draw.rect(screen, (79, 109, 69), (0, horizon, w, h - horizon))
        mask = np.linalg.norm(self.mesh_centers[:, :2] - env.pos, axis=1) < 230
        shadow = np.array(
            [
                [*(env.pos + f[:2] * x + right[:2] * y), 0.07]
                for x, y in [(-2.8, -1), (2.8, -1), (2.8, 1), (-2.8, 1)]
            ]
        )
        dynamic = [(shadow, (34, 42, 40))]
        if self.camera != 1:
            dynamic += self.car_faces(env)
        car_mesh, car_colors = self.pack(dynamic)
        mesh = np.concatenate([self.mesh[mask], car_mesh])
        colors = np.concatenate([self.colors[mask], car_colors])
        camera_vertices = (mesh - eye) @ basis
        z = camera_vertices[:, :, 2]
        normals = np.cross(
            camera_vertices[:, 1] - camera_vertices[:, 0],
            camera_vertices[:, 2] - camera_vertices[:, 0],
        )
        front = (normals * camera_vertices[:, 0]).sum(axis=1) < 0
        keep = (z.max(axis=1) >= 0.35) & (front | (mesh[:, :, 2].max(axis=1) < 0.1))
        camera_vertices = camera_vertices[keep]
        colors = colors[keep]
        ground = mesh[keep, :, 2].max(axis=1) < 0.1
        z = camera_vertices[:, :, 2]
        depth = z.mean(axis=1)
        fog = np.clip((depth - 90) / 240, 0, 0.72)[:, None]
        colors = (colors * (1 - fog) + np.array([166, 189, 181]) * fog).astype(int)
        projected = np.empty((len(z), 3, 2))
        safez = np.maximum(z, 0.35)
        projected[:, :, 0] = w / 2 + focal * camera_vertices[:, :, 0] / safez
        projected[:, :, 1] = h * 0.45 - focal * camera_vertices[:, :, 1] / safez
        visible = (
            (projected[:, :, 0].max(axis=1) >= 0)
            & (projected[:, :, 0].min(axis=1) <= w)
            & (projected[:, :, 1].max(axis=1) >= 0)
            & (projected[:, :, 1].min(axis=1) <= h)
        )
        # Flat road decals render before above-ground objects; large grass polygons
        # must never paint over the road just because their centroids are nearer.
        minimum_z = z.min(axis=1)
        order = np.argsort(-depth + (~ground) * 10000)
        for i in order:
            if not visible[i]:
                continue
            polygon = projected[i]
            if minimum_z[i] < 0.35:
                clipped = clip_near(camera_vertices[i])
                if len(clipped) < 3:
                    continue
                polygon = np.column_stack(
                    [
                        w / 2 + focal * clipped[:, 0] / clipped[:, 2],
                        h * 0.45 - focal * clipped[:, 1] / clipped[:, 2],
                    ]
                )
            pygame.draw.polygon(screen, colors[i], polygon)
        self.hud(viewer)

    def hud(self, viewer):
        env, screen = viewer.env, viewer.screen
        info = env.info()

        def panel(rect):
            surf = pygame.Surface(rect[2:], pygame.SRCALPHA)
            surf.fill((12, 22, 30, 215))
            screen.blit(surf, rect[:2])

        panel((22, 20, 1156, 77))
        screen.blit(viewer.title.render("RACING LAB", True, (238, 242, 241)), (40, 30))
        screen.blit(viewer.font.render(env.track.name, True, (137, 219, 194)), (295, 35))
        camera = ("CHASE", "NOSE", "AERIAL")[self.camera]
        screen.blit(
            viewer.font.render(f"3D / {camera}    C camera    V 2D view", True, (225, 233, 231)),
            (40, 70),
        )
        screen.blit(
            viewer.font.render(
                f"TIME {info['elapsed']:6.2f}   LAP {max(0, info['progress']) * 100:5.1f}%",
                True,
                (238, 242, 241),
            ),
            (755, 39),
        )
        panel((22, 725, 280, 120))
        screen.blit(
            viewer.title.render(f"{env.speed * 3.6:3.0f} km/h", True, (247, 249, 242)), (40, 738)
        )
        for i, (name, value, color) in enumerate(
            zip(("THR", "BRK"), env.action[1:], ((69, 206, 155), (237, 89, 73)))
        ):
            y = 782 + i * 26
            screen.blit(viewer.font.render(name, True, (211, 222, 224)), (40, y))
            pygame.draw.rect(screen, (45, 58, 66), (90, y + 3, 180, 12))
            pygame.draw.rect(screen, color, (90, y + 3, int(180 * value), 12))
        # Track map is useful for approaching corners and keeps all four tracks recognizable.
        panel((970, 620, 207, 225))
        points = env.track.points
        lo, hi = points.min(axis=0), points.max(axis=0)
        scale = min(170 / (hi - lo)[0], 185 / (hi - lo)[1])

        def map_point(p):
            return (p - (lo + hi) / 2) * scale + [1073, 733]

        pygame.draw.lines(
            screen, (158, 176, 181), True, (points - (lo + hi) / 2) * scale + [1073, 733], 3
        )
        pygame.draw.circle(screen, (255, 196, 73), map_point(env.pos), 5)
        if info["offtrack"]:
            screen.blit(viewer.title.render("OFF TRACK", True, (255, 171, 88)), (490, 115))
        panel((22, 855, 1156, 32))
        screen.blit(viewer.font.render(viewer.note[:105], True, (247, 220, 148)), (32, 861))
