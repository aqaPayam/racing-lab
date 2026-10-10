"""Deterministic fixed-step bicycle model; intentionally not an F1 engineering simulator."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .track import Track

DT = 1 / 60
LOOKAHEAD = np.array([10, 25, 50, 90, 150, 230], dtype=float)
OBS_NAMES = ["speed/85", "lateral_speed/20", "yaw_rate/3", "steering/0.45",
             "lateral_offset/half_width", "sin_heading_error", "cos_heading_error",
             "half_width/20"] + [f"preview_{int(d)}m_{axis}/230" for d in LOOKAHEAD for axis in ("forward", "right")]


def wrap(x):
    return (x+np.pi) % (2*np.pi)-np.pi


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, track="monza", render_mode=None, max_seconds=300, reward_weights=None):
        self.track = Track(track)
        self.render_mode = render_mode
        self.max_steps = int(max_seconds/DT)
        if self.max_steps < 1:
            raise ValueError("max_seconds must allow at least one simulation step")
        self.weights = {"progress": 1., "time": .1, "offtrack": 2., "smoothness": .02}
        self.weights.update(reward_weights or {})
        if set(self.weights) != {"progress", "time", "offtrack", "smoothness"}:
            raise ValueError("Unknown reward feature")
        self.action_space = spaces.Box(np.array([-1, 0, 0], np.float32), np.ones(3, np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(OBS_NAMES),), np.float32)
        self.viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.track.sample(0).copy()
        _, _, self.heading = self.track.project(self.pos)
        jitter = (options or {}).get("lateral_jitter", 0.)
        self.pos += self.np_random.uniform(-jitter, jitter)*np.array([-np.sin(self.heading), np.cos(self.heading)])
        self.speed = self.lateral_speed = self.yaw_rate = self.steering = 0.
        self.steps = self.offtrack_steps = 0
        self.progress = self.last_s = self.smoothness = 0.
        self.last_s = self.track.project(self.pos)[0]
        self.next_checkpoint = 1
        self.action = np.zeros(3, np.float32)
        self.done = False
        return self.observe(), self.info()

    def state(self):
        return np.array([*self.pos, self.heading, self.speed, self.lateral_speed, self.yaw_rate, self.steering], dtype=np.float64)

    def observe(self):
        s, lateral, tangent = self.track.project(self.pos)
        h = self.heading
        delta = self.track.sample(s+LOOKAHEAD)-self.pos
        local = delta @ np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        return np.array([self.speed/85, self.lateral_speed/20, self.yaw_rate/3, self.steering/.45,
                         lateral/(self.track.width/2), np.sin(h-tangent), np.cos(h-tangent),
                         self.track.width/40, *(local/230).ravel()], dtype=np.float32)

    def info(self):
        _, lateral, _ = self.track.project(self.pos)
        success = self.progress >= self.track.length and self.next_checkpoint >= 41
        return {"track": self.track.name, "elapsed": self.steps*DT, "progress": self.progress/self.track.length,
                "offtrack": abs(lateral)>self.track.width/2, "offtrack_fraction": self.offtrack_steps/max(1,self.steps),
                "success": bool(success), "lap_time": self.steps*DT if success else None,
                "smoothness": self.smoothness/max(1,self.steps), "checkpoint": self.next_checkpoint-1}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended: call reset() before step().")
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError("Action must be three finite values: steering, throttle, brake")
        a = np.clip(a, self.action_space.low, self.action_space.high)
        change = float(np.sum((a-self.action)**2))
        self.smoothness += change
        self.action = a.copy()
        self.steering += np.clip(a[0]*.45-self.steering, -1.5*DT, 1.5*DT)
        _, lateral, _ = self.track.project(self.pos)
        off = abs(lateral)>self.track.width/2
        accel = a[1]*12-a[2]*24-.0017*self.speed**2-.12*self.speed
        if off:
            accel -= 1.8*self.speed
        self.speed = float(np.clip(self.speed+accel*DT, 0, 85))
        desired_yaw = self.speed/3.6*np.tan(self.steering)
        grip = (8 if off else 17) / max(self.speed, 2)
        target_yaw = np.clip(desired_yaw, -grip, grip)
        self.yaw_rate += (target_yaw-self.yaw_rate)*min(1, DT*9)
        self.lateral_speed += ((desired_yaw-self.yaw_rate)*self.speed*.07-self.lateral_speed)*DT*5
        self.lateral_speed = float(np.clip(self.lateral_speed, -15, 15))
        self.heading = wrap(self.heading+self.yaw_rate*DT)
        forward = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.pos += DT*(forward*self.speed+np.array([-forward[1], forward[0]])*self.lateral_speed)
        s, lateral, _ = self.track.project(self.pos)
        delta = (s-self.last_s+self.track.length/2) % self.track.length-self.track.length/2
        self.last_s = s
        off = abs(lateral)>self.track.width/2
        # Ignore projection jumps; require every gate in order on the road.
        valid_delta = delta if abs(delta)<max(5, self.speed*DT*3) else 0.
        self.progress += valid_delta
        gate = self.next_checkpoint*self.track.length/40
        if not off and gate <= self.progress <= gate+self.track.width:
            self.next_checkpoint += 1
        self.steps += 1
        self.offtrack_steps += int(off)
        features = {"progress": valid_delta, "time": -DT, "offtrack": -float(off)*DT, "smoothness": -change}
        reward = sum(self.weights[k]*v for k,v in features.items())
        info = self.info()
        info["reward_features"] = features
        terminated = info["success"]
        truncated = self.steps>=self.max_steps
        self.done = terminated or truncated
        if self.render_mode == "human":
            self.render()
        return self.observe(), float(reward), terminated, truncated, info

    def render(self):
        from .viewer import Viewer
        if self.viewer is None:
            self.viewer = Viewer(self)
        return self.viewer.draw(rgb=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
