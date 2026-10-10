"""Procedural driving feedback. Audio never enters simulation or research state."""

import numpy as np


class DrivingAudio:
    def __init__(self, enabled=True, volume=0.25):
        import pygame

        if not np.isfinite(volume) or not 0 <= volume <= 1:
            raise ValueError("Volume must be between zero and one")
        self.enabled = enabled
        self.available = False
        self.volume = volume
        self.bin = None
        self.engine_channel = 0
        self.gear = None
        self.last_impact_step = -1
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.rate, _, self.channels = pygame.mixer.get_init()
            pygame.mixer.set_num_channels(8)
            self.sounds = {}
            for rpm in range(4500, 15251, 250):
                period = max(1, round(self.rate / (rpm / 20)))
                phase = np.arange(period * 30) * 2 * np.pi / period
                wave = sum(np.sin(phase * i) / i for i in (1, 2, 3, 5, 7)) * 0.18
                self.sounds[rpm] = self.sound(wave)
            rng = np.random.default_rng(313)
            noise = rng.normal(0, 0.22, self.rate // 2)
            self.scrub = self.sound(np.convolve(noise, np.ones(3) / 3, mode="same"))
            t = np.arange(self.rate // 6) / self.rate
            self.hit = self.sound(rng.normal(0, 0.35, len(t)) * np.exp(-35 * t))
            t = np.arange(self.rate // 20) / self.rate
            self.shift = self.sound(np.sin(2 * np.pi * 160 * t) * np.exp(-70 * t) * 0.2)
            pygame.mixer.Channel(2).play(self.scrub, loops=-1)
            pygame.mixer.Channel(2).set_volume(0)
            self.available = True
        except pygame.error:
            import warnings

            warnings.warn("Audio device unavailable; driving continues silently", RuntimeWarning)

    def sound(self, wave):
        import pygame

        samples = (np.clip(wave, -1, 1) * 32767).astype(np.int16)
        if self.channels == 2:
            samples = np.repeat(samples[:, None], 2, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(samples))

    def update(self, env, paused=False):
        import pygame

        if not self.available:
            return
        volume = self.volume if self.enabled and not paused else 0.0
        rpm = int(np.clip(round(env.vehicle.rpm / 250) * 250, 4500, 15250))
        if rpm != self.bin:
            pygame.mixer.Channel(self.engine_channel).fadeout(40)
            self.engine_channel = 1 - self.engine_channel
            pygame.mixer.Channel(self.engine_channel).play(self.sounds[rpm], loops=-1, fade_ms=40)
            self.bin = rpm
        pygame.mixer.Channel(self.engine_channel).set_volume(volume * (0.4 + 0.6 * env.action[1]))
        slip = max(
            float(np.max(np.abs(env.vehicle.slip))) - 0.1,
            (max(abs(env.telemetry["front_slip_deg"]), abs(env.telemetry["rear_slip_deg"])) - 4)
            / 12,
            0,
        )
        pygame.mixer.Channel(2).set_volume(volume * min(1.0, slip) * min(1.0, abs(env.speed) / 6))
        if volume and env.impact > 1 and env.steps != self.last_impact_step:
            self.last_impact_step = env.steps
            pygame.mixer.Channel(3).set_volume(volume * min(1.0, env.impact / 15))
            pygame.mixer.Channel(3).play(self.hit)
        if volume and self.gear is not None and env.vehicle.gear != self.gear:
            pygame.mixer.Channel(4).set_volume(volume)
            pygame.mixer.Channel(4).play(self.shift)
        self.gear = env.vehicle.gear

    def close(self):
        import pygame

        if self.available:
            pygame.mixer.stop()
