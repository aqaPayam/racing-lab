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
    env = RacingEnv(render_mode="rgb_array", view="3d", renderer="software")
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


def test_gpu_renderer_and_wheel_geometry():
    import pytest
    from racing_lab.gpu_scene import GPUScene

    # Only context availability may skip the test, never shader/mesh failures.
    import moderngl

    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception:
        try:
            context = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception as exc:
            pytest.skip(f"No OpenGL context on this host: {exc}")
    context.release()
    env = RacingEnv(render_mode="rgb_array", view="3d", renderer="gpu")
    env.reset()
    try:
        before = env.snapshot()
        image = env.render()
        assert isinstance(env.viewer.scene3d, GPUScene)
        assert image.shape == (900, 1200, 3) and image.std() > 20
        np.testing.assert_array_equal(
            env.state(),
            list(before["state"]["pos"])
            + [
                before["state"][k]
                for k in ("heading", "speed", "lateral_speed", "yaw_rate", "steering")
            ],
        )
        from types import SimpleNamespace

        pose = env.state().copy()
        pose[:2] = env.track.sample(20)
        env.viewer.practice = SimpleNamespace(
            ghost=lambda _: pose.copy(), best_time=100.0, delta=lambda *_: 0.0
        )
        ghost_image = env.render()
        assert not np.array_equal(image, ghost_image)
        np.testing.assert_array_equal(env.vehicle.vector(), before["state"]["vehicle"].vector())
        car = env.viewer.scene3d.car
        still, _ = car.mesh(env)
        env.vehicle.angle[:] = 1 / 0.36
        rolling, _ = car.mesh(env)
        assert not np.array_equal(still, rolling)
        env.vehicle.angle[:] = 2 * np.pi
        full_turn, _ = car.mesh(env)
        np.testing.assert_allclose(still, full_turn, atol=1e-10)
    finally:
        env.close()
