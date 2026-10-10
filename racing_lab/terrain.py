"""Presentation geometry for the same height/bank profile used by physics."""

import numpy as np


def warp_mesh(track, mesh):
    vertices = mesh.reshape(-1, 3).copy()
    unique, inverse = np.unique(vertices[:, :2], axis=0, return_inverse=True)
    vertices[:, 2] += track.surface(unique)[inverse]
    return vertices.reshape(mesh.shape)


def ground_mesh(track):
    low = track.points.min(axis=0) - 140
    high = track.points.max(axis=0) + 140
    x = np.arange(low[0], high[0] + 24, 24)
    y = np.arange(low[1], high[1] + 24, 24)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    zz = track.surface(xy) - 0.12
    grid = np.column_stack([xy, zz]).reshape(len(y), len(x), 3)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]
    return np.concatenate(
        [
            np.stack([a, b, c], axis=2).reshape(-1, 3, 3),
            np.stack([a, c, d], axis=2).reshape(-1, 3, 3),
        ]
    )


def pose_frame(env):
    s, lateral, tangent = env.track.project(env.pos)
    height, grade, bank = env.track.profile(s)
    error = env.heading - tangent
    along = float(grade * np.cos(error) + np.tan(bank) * np.sin(error))
    across = float(np.tan(bank) * np.cos(error) - grade * np.sin(error))
    f = np.array([np.cos(env.heading), np.sin(env.heading), along])
    f /= np.linalg.norm(f)
    r = np.array([-np.sin(env.heading), np.cos(env.heading), across])
    r -= f * np.dot(r, f)
    r /= np.linalg.norm(r)
    up = np.cross(f, r)
    origin = np.array([*env.pos, float(height + lateral * np.tan(bank))])
    return origin, np.stack([f, r, up])
