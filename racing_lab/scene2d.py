"""Readable top-down racing view with a smoothed forward-looking camera."""

import numpy as np
import pygame
import pygame.gfxdraw
from .hud import draw as draw_hud


class Scene2D:
    def __init__(self):
        self.track_hash = None
        self.center = None
        self.heading = 0.0
        self.last_step = -1
        self.scale = 12.0

    def build(self, track):
        points = track.points
        normals = np.column_stack([-track.tangents[:, 1], track.tangents[:, 0]])
        self.strips = []
        half = track.width / 2
        for a, b, kind in [
            (-half - track.shoulder, half + track.shoulder, "shoulder"),
            (-half, half, "road"),
            (-half - 0.6, -half, "curb"),
            (half, half + 0.6, "curb"),
            (-half, -half + 0.12, "line"),
            (half - 0.12, half, "line"),
        ]:
            left, right = points + normals * a, points + normals * b
            self.strips.append(
                (
                    np.stack(
                        [left, np.roll(left, -1, axis=0), np.roll(right, -1, axis=0), right], axis=1
                    ),
                    kind,
                )
            )
        self.barriers = [points + normals * side * track.barrier_offset for side in (-1, 1)]
        rng = np.random.default_rng(623)
        self.trees = []
        for i in range(0, len(points), 12):
            for side in (-1, 1):
                p = points[i] + normals[i] * side * (track.barrier_offset + rng.uniform(5, 18))
                if abs(track.project(p)[1]) > track.barrier_offset + 3:
                    self.trees.append((p, rng.uniform(1.3, 2.4)))
        self.track_hash = track.digest

    def draw(self, viewer, rgb=False):
        env, screen = viewer.env, viewer.screen
        if self.track_hash != env.track.digest:
            self.build(env.track)
            self.center = None
        if self.center is None or env.steps < self.last_step:
            self.center = env.pos.copy()
            self.heading = env.heading
        elapsed = max(0, env.steps - self.last_step) * env.dt
        blend = 1 - np.exp(-elapsed * 8)
        target = env.pos + np.array([np.cos(env.heading), np.sin(env.heading)]) * (
            180 / max(1, self.scale)
        )
        self.center += (target - self.center) * blend
        self.heading += ((env.heading - self.heading + np.pi) % (2 * np.pi) - np.pi) * blend
        self.scale += (max(7.0, 14.0 - abs(env.speed) * 0.06) - self.scale) * blend
        if viewer.zoom:
            center = self.center
            scale = self.scale
            basis = np.array(
                [
                    [-np.sin(self.heading), -np.cos(self.heading)],
                    [np.cos(self.heading), -np.sin(self.heading)],
                ]
            )
        else:
            lo, hi = env.track.points.min(axis=0), env.track.points.max(axis=0)
            center = (lo + hi) / 2
            scale = min(930 / (hi - lo)[0], 600 / (hi - lo)[1])
            basis = np.eye(2)

        def xy(points):
            return (np.asarray(points) - center) @ basis * scale + [550, 415]

        screen.fill((49, 76, 49))
        screen.set_clip((0, 95, 1200, 641))
        # Mown grass bands are tied to world coordinates, so they move naturally.
        radius = 850 / scale
        for x in np.arange(np.floor((center[0] - radius) / 30) * 30, center[0] + radius, 30):
            poly = xy(
                [
                    [x, center[1] - radius],
                    [x + 15, center[1] - radius],
                    [x + 15, center[1] + radius],
                    [x, center[1] + radius],
                ]
            )
            pygame.draw.polygon(screen, (53, 81, 50), poly)
        mask = np.linalg.norm(env.track.points - center, axis=1) < radius + 20
        colors = {"road": (53, 60, 65), "shoulder": (154, 157, 130), "line": (222, 224, 204)}
        for quads, kind in self.strips:
            projected = xy(quads[mask])
            indices = np.flatnonzero(mask)
            for q, i in zip(projected, indices):
                color = (
                    ((208, 58, 61) if i % 4 < 2 else (232, 227, 206))
                    if kind == "curb"
                    else colors[kind]
                )
                pygame.draw.polygon(screen, color, q)
        for barrier in self.barriers:
            for i in np.flatnonzero(mask):
                a, b = xy(barrier[[i, (i + 1) % len(barrier)]])
                pygame.draw.line(
                    screen, (24, 33, 37), a + [2, 3], b + [2, 3], max(2, round(scale * 0.45))
                )
                pygame.draw.line(screen, (169, 186, 188), a, b, max(1, round(scale * 0.25)))
        for p, r in self.trees:
            if np.linalg.norm(p - center) > radius:
                continue
            q = xy(p)
            rr = max(2, int(r * scale))
            pygame.draw.circle(screen, (33, 52, 37), q + [rr * 0.35, rr * 0.35], rr)
            pygame.draw.circle(screen, (37, 88, 51), q, rr)
            pygame.draw.circle(screen, (54, 106, 59), q + [-rr * 0.2, -rr * 0.25], int(rr * 0.7))
        p = env.track.points[0]
        f = env.track.tangents[0]
        n = np.array([-f[1], f[0]])
        for row in range(2):
            for col in range(int(env.track.width)):
                poly = [
                    p + f * (row * 0.5 + a) + n * (-env.track.width / 2 + col + b)
                    for a, b in ((0, 0), (0.5, 0), (0.5, 1), (0, 1))
                ]
                pygame.draw.polygon(
                    screen, (235, 234, 217) if (row + col) % 2 else (29, 35, 39), xy(poly)
                )
        if env.steps != self.last_step:
            viewer.trail.append(env.pos.copy())
            viewer.trail = viewer.trail[-24000:]
        if len(viewer.trail) > 1 and not viewer.zoom:
            pygame.draw.lines(screen, (59, 188, 181), False, xy(viewer.trail), 1)
        if viewer.agent_ghost and viewer.show_lines:
            pygame.draw.lines(
                screen, (65, 220, 208), False, xy(viewer.agent_ghost.states[:, :2]), 2
            )
            if viewer.human_line and len(viewer.human_line) > 1:
                pygame.draw.lines(
                    screen, (255, 174, 93), False, xy(np.asarray(viewer.human_line)[:, :2]), 2
                )
        practice = viewer.ghost_reference
        if practice and viewer.show_ghost:
            ghost = viewer.ghost_pose()
            if ghost is not None:
                self.car(screen, xy, ghost[:2], ghost[2], ghost[6], scale, ghost=True)
        self.car(screen, xy, env.pos, env.heading, env.steering, scale)
        screen.set_clip(None)
        draw_hud(viewer, "2D / FOLLOW" if viewer.zoom else "2D / TRACK OVERVIEW")
        self.last_step = env.steps
        if not viewer.offscreen:
            pygame.display.flip()
        if rgb:
            return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))

    @staticmethod
    def car(screen, xy, pos, heading, steer, scale, ghost=False):
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, s], [-s, c]])

        def poly(vertices, color):
            points = np.rint(xy(np.asarray(vertices) @ rot + pos)).astype(int).tolist()
            if ghost:
                pygame.draw.polygon(screen, (87, 224, 229), points, max(1, int(scale * 0.12)))
            else:
                pygame.gfxdraw.filled_polygon(screen, points, color)
                pygame.gfxdraw.aapolygon(screen, points, color)

        if not ghost:
            poly(
                [
                    [-2.6, -0.8],
                    [-1.3, -0.72],
                    [0.8, -0.48],
                    [2.8, -0.8],
                    [2.8, 0.96],
                    [0.8, 0.62],
                    [-1.3, 0.9],
                    [-2.6, 0.96],
                ],
                (39, 46, 45),
            )
        for x in (-1.6, 1.8):
            poly(
                [[x - 0.22, -0.77], [x + 0.15, -0.77], [x + 0.15, 0.77], [x - 0.22, 0.77]],
                (99, 108, 113),
            )
            for y in (-0.75, 0.75):
                angle = steer if x > 0 else 0
                turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
                vertices = np.array(
                    [[-0.36, -0.2], [0.36, -0.2], [0.36, 0.2], [-0.36, 0.2]]
                ) @ turn + [x, y]
                poly(vertices, (14, 19, 25))
        poly(
            [
                [-2.3, -0.24],
                [-1.3, -0.65],
                [0.2, -0.60],
                [0.8, -0.23],
                [2.65, -0.11],
                [2.65, 0.11],
                [0.8, 0.23],
                [0.2, 0.60],
                [-1.3, 0.65],
                [-2.3, 0.24],
            ],
            (226, 49, 62),
        )
        poly([[-1.5, -0.08], [2.35, -0.04], [2.35, 0.04], [-1.5, 0.08]], (255, 120, 122))
        for x in (-2.3, 2.4):
            poly(
                [[x - 0.17, -0.91], [x + 0.17, -0.91], [x + 0.17, 0.91], [x - 0.17, 0.91]],
                (27, 33, 41),
            )
        poly([[-0.58, -0.24], [0.20, -0.24], [0.30, 0.24], [-0.58, 0.24]], (20, 28, 36))
        if not ghost:
            pygame.draw.circle(
                screen,
                (251, 205, 89),
                xy(pos + np.array([-0.2, 0]) @ rot),
                max(2, int(scale * 0.17)),
            )
