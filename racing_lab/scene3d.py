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
        self.wheel_phase = 0.0
        self.last_step = 0

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
        samples = track.points
        tangent = track.tangents
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
        for i in range(len(samples)):
            j = (i + 1) % len(samples)
            for side in (-1, 1):
                a = samples[i] + normal[i] * side * track.barrier_offset
                b = samples[j] + normal[j] * side * track.barrier_offset
                q = np.array([[*a, 0.3], [*b, 0.3], [*b, 1.1], [*a, 1.1]])
                self.faces.append((q, (157, 170, 178)))
                self.faces.append((q[::-1], (157, 170, 178)))
                if i % 8 == 0:
                    self.faces.extend(self.box([*a, 0.55], [0.16, 0.16, 1.1], (111, 123, 131)))
        # Trees are deterministic scenery, never obstacles or random simulator state.
        tree_start = len(self.faces)
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
        self.tree_face_range = (tree_start, len(self.faces))
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
                    local(5, -width - 11 - row * 1.6, 0.5 + row * 0.325),
                    [44, 1.6, 1 + row * 0.65],
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
        self.mesh_ground = self.mesh[:, :, 2].max(axis=1) < 0.1
        self.mesh_layer = np.where(self.mesh_ground, 1.0, 2.0)
        if not getattr(self, "defer_terrain", False):
            from .terrain import warp_mesh, ground_mesh

            self.mesh = warp_mesh(track, self.mesh)
            ground = ground_mesh(track)
            self.mesh = np.concatenate([self.mesh, ground])
            self.colors = np.concatenate([self.colors, np.tile([81, 112, 61], (len(ground), 1))])
            self.mesh_ground = np.r_[self.mesh_ground, np.ones(len(ground), dtype=bool)]
            self.mesh_layer = np.r_[self.mesh_layer, np.zeros(len(ground))]
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
        angle = 0.0
        local_pos = np.zeros(2)
        f = np.array([np.cos(angle), np.sin(angle)])
        n = np.array([-f[1], f[0]])

        def part(x, y, z, size, color, steer=0):
            # Physical reduced-order pitch and roll.
            pitch = env.vehicle.pitch
            roll = env.vehicle.roll
            faces.extend(
                self.box(
                    [*(local_pos + f * x + n * y), z + x * pitch + y * roll],
                    size,
                    color,
                    angle + steer,
                )
            )

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
        for axle, x in enumerate((1.8, -1.6)):
            for side_index, y in enumerate((-0.75, 0.75)):
                wheel_angle = angle + (env.steering if x > 0 else 0)
                wf = np.array([np.cos(wheel_angle), np.sin(wheel_angle)])
                wn = np.array([-wf[1], wf[0]])
                origin = local_pos + f * x + n * y
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
                    # Alternating bright spokes provide visible wheel rotation.
                    for phase in (
                        np.arange(5) * 2 * np.pi / 5 - env.vehicle.angle[axle * 2 + side_index]
                    ):
                        spoke = []
                        for radius, offset in [
                            (0.035, -0.12),
                            (0.17, -0.12),
                            (0.17, 0.12),
                            (0.035, 0.12),
                        ]:
                            spoke.append(
                                center
                                + np.r_[
                                    wf * radius * np.cos(phase + offset),
                                    radius * np.sin(phase + offset),
                                ]
                            )
                        faces.append((np.asarray(spoke), (207, 215, 219)))
        from .terrain import pose_frame

        origin, frame = pose_frame(env)
        return [(vertices @ frame + origin, color) for vertices, color in faces]

    def draw(self, viewer):
        env = viewer.env
        screen = viewer.screen
        w, h = screen.get_size()
        if self.track_hash != env.track.digest:
            self.build_track(env.track)
        self.wheel_phase = -env.distance / 0.36
        self.last_step = env.steps
        from .terrain import pose_frame

        pos, frame = pose_frame(env)
        f = frame[0]
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
        shadow[:, 2] += env.track.surface(shadow[:, :2])
        dynamic = [(shadow, (34, 42, 40))]
        if self.camera != 1:
            dynamic += self.car_faces(env)
        if viewer.show_ghost and viewer.ghost_reference:
            pose = viewer.ghost_pose()
            if pose is not None:
                from .practice import ghost_environment

                dynamic += [
                    (v, (60, 180, 190))
                    for v, _ in self.car_faces(
                        ghost_environment(env.track, pose, env.steps * env.dt)
                    )
                ]
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
        ground_flags = np.r_[self.mesh_ground[mask], np.zeros(len(car_mesh), dtype=bool)]
        dynamic_flags = np.r_[np.zeros(mask.sum(), dtype=bool), np.ones(len(car_mesh), dtype=bool)]
        keep = (z.max(axis=1) >= 0.35) & (front | ground_flags | dynamic_flags)
        camera_vertices = camera_vertices[keep]
        colors = colors[keep]
        layers = np.r_[self.mesh_layer[mask], [1.5, 1.5], np.full(len(car_mesh) - 2, 2.0)][keep]
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
        order = np.lexsort((-depth, layers))
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
        from .hud import draw

        draw(
            viewer, f"3D / {('CHASE', 'COCKPIT', 'AERIAL')[self.camera]} / {viewer.renderer_status}"
        )
