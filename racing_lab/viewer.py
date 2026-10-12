"""Pygame presentation only: never changes simulator state."""

import numpy as np
import pygame


class Viewer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.offscreen = env.render_mode == "rgb_array"
        self.screen = (
            pygame.Surface((1200, 900)) if self.offscreen else pygame.display.set_mode((1200, 900))
        )
        if not self.offscreen:
            pygame.display.set_caption("Racing Lab | human demonstrations & machine learning")
        self.font = pygame.font.SysFont("consolas", 19)
        self.title = pygame.font.SysFont("consolas", 30, bold=True)
        self.note = ""
        self.zoom = True
        self.trail = []
        self.mode = env.view
        self.scene3d = None
        self.scene2d = None
        self.practice = None
        self.agent_ghost = None
        self.human_line = None
        self.show_lines = True
        self.race_phase = "ready"
        self.countdown = 3.0
        self.show_ghost = True
        self.camera_motion = False
        self.renderer_status = "2D"

    @property
    def ghost_reference(self):
        return self.agent_ghost or self.practice

    def ghost_pose(self):
        reference = self.ghost_reference
        pose = reference.ghost(self.env.steps * self.env.dt) if reference else None
        # A coincident SAC car causes depth flicker and can obscure the player's car.
        # Only presentation is hidden; its timing, map marker and line remain visible.
        if self.agent_ghost and pose is not None and np.linalg.norm(pose[:2] - self.env.pos) < 2:
            return None
        return pose

    def cycle_camera(self):
        if self.mode == "3d":
            self._scene().camera = (self._scene().camera + 1) % 3
        else:
            self.zoom = not self.zoom

    def _scene(self):
        if self.scene3d is None:
            if self.env.renderer != "software":
                try:
                    from .gpu_scene import GPUScene

                    self.scene3d = GPUScene()
                    self.renderer_status = "GPU"
                except Exception as exc:
                    if self.env.renderer == "gpu":
                        raise RuntimeError(f"GPU renderer unavailable: {exc}") from exc
                    import warnings

                    warnings.warn(f"GPU unavailable; using software 3D: {exc}", RuntimeWarning)
            if self.scene3d is None:
                from .scene3d import Scene3D

                self.scene3d = Scene3D()
                self.renderer_status = "SOFTWARE"
        return self.scene3d

    def draw(self, rgb=False):
        if self.mode == "3d":
            self._scene().draw(self)
            if not self.offscreen:
                pygame.display.flip()
            if rgb:
                return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
            return None
        return self.draw_2d(rgb)

    def draw_2d(self, rgb=False):
        if self.scene2d is None:
            from .scene2d import Scene2D

            self.scene2d = Scene2D()
        return self.scene2d.draw(self, rgb)

    def close(self):
        if self.scene3d and hasattr(self.scene3d, "close"):
            self.scene3d.close()
        pygame.quit()
