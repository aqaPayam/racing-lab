import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from racing_lab import RacingEnv
from racing_lab.scene3d import clip_near


def test_near_plane_clips_crossing_polygon():
    vertices = np.array([[-1, 0, -2], [1, 0, 2], [0, 1, 2]], dtype=float)
    clipped = clip_near(vertices)
    assert len(clipped) == 4
    assert np.isfinite(clipped).all()
    assert (clipped[:, 2] >= 0.35 - 1e-9).all()
    assert len(clip_near(vertices - [0, 0, 10])) == 0


def test_three_cameras_do_not_change_simulation_and_rebuild_track():
    env = RacingEnv(render_mode="rgb_array", view="3d")
    control = RacingEnv()
    env.reset(seed=12)
    control.reset(seed=12)
    images = []
    for _ in range(3):
        before = env.state().copy()
        pixels = env.render()
        assert pixels.shape == (900, 1200, 3)
        assert pixels.std() > 20
        np.testing.assert_array_equal(before, env.state())
        images.append(pixels)
        env.viewer.cycle_camera()
    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[0], images[2])
    for _ in range(10):
        a = env.step([0.1, 0.7, 0])
        b = control.step([0.1, 0.7, 0])
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    from racing_lab.track import Track

    env.track = Track("spa")
    env.reset()
    env.render()
    assert env.viewer.scene3d.track_hash == env.track.digest
    env.viewer.mode = "2d"
    assert env.render().shape == (900, 1200, 3)
    env.close()
