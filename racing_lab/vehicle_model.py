"""Four-wheel planar model with a reduced-order sprung-body and eight-speed drive.

This is an inspectable generic model, not identified from Formula 1 tire data.
Wheel order is front-left, front-right, rear-left, rear-right (positive y is right).
"""

from dataclasses import dataclass, field, asdict
import math
import numpy as np
from .dynamics import PARAMETERS

RADIUS = 0.36
WHEEL_INERTIA = 1.5
GEAR_RATIOS = (18.0, 14.0, 11.5, 9.6, 8.3, 7.4, 6.7, 6.0)
WHEEL_X = np.array([1.8, 1.8, -1.6, -1.6])
WHEEL_Y = np.array([-0.75, 0.75, -0.75, 0.75])
VEHICLE_STATE_NAMES = [
    "gear",
    "rpm",
    "shift_timer",
    "shift_cooldown",
    "pitch",
    "roll",
    "pitch_rate",
    "roll_rate",
    "ax",
    "ay",
] + [f"{name}_{wheel}" for name in ("omega", "angle", "slip") for wheel in ("fl", "fr", "rl", "rr")]


@dataclass(frozen=True)
class DrivingConfig:
    automatic: bool = True
    traction_control: bool = True
    abs: bool = True

    def __post_init__(self):
        if not all(isinstance(v, bool) for v in asdict(self).values()):
            raise ValueError("DrivingConfig settings must be booleans")


@dataclass
class VehicleState:
    gear: int = 1
    rpm: float = 4500.0
    shift_timer: float = 0.0
    shift_cooldown: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    omega: np.ndarray = field(default_factory=lambda: np.zeros(4))
    angle: np.ndarray = field(default_factory=lambda: np.zeros(4))
    slip: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @classmethod
    def rolling(cls, speed=0):
        state = cls()
        state.omega[:] = speed / RADIUS
        if speed < 0:
            state.gear = -1
        else:
            state.gear = next(
                (
                    i + 1
                    for i, r in enumerate(GEAR_RATIOS)
                    if speed / RADIUS * r * 60 / (2 * math.pi) < 11000
                ),
                8,
            )
        state.rpm = max(4500.0, abs(speed) / RADIUS * state.ratio() * 60 / (2 * math.pi))
        return state

    def ratio(self):
        return 12.0 if self.gear < 0 else GEAR_RATIOS[self.gear - 1]

    def vector(self):
        return np.array(
            [
                self.gear,
                self.rpm,
                self.shift_timer,
                self.shift_cooldown,
                self.pitch,
                self.roll,
                self.pitch_rate,
                self.roll_rate,
                self.ax,
                self.ay,
                *self.omega,
                *self.angle,
                *self.slip,
            ],
            dtype=float,
        )

    def restore_vector(self, vector):
        if np.shape(vector) != (len(VEHICLE_STATE_NAMES),) or not np.isfinite(vector).all():
            raise ValueError("Invalid vehicle state")
        for name, value in zip(VEHICLE_STATE_NAMES[:10], vector[:10]):
            setattr(self, name, float(value))
        self.gear = int(self.gear)
        if self.gear not in (-1, 1, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError("Invalid gear")
        self.omega, self.angle, self.slip = [np.asarray(v).copy() for v in np.split(vector[10:], 3)]

    def shift(self, direction):
        if self.gear < 1 or self.shift_cooldown > 0 or not direction:
            return
        selected = int(np.clip(self.gear + direction, 1, 8))
        projected = (
            abs(float(self.omega[2:].mean())) * GEAR_RATIOS[selected - 1] * 60 / (2 * math.pi)
        )
        if selected != self.gear and projected < 14800:
            self.gear = selected
            self.shift_timer = 0.08
            self.shift_cooldown = 0.35


def advance_vehicle(
    u,
    v,
    yaw,
    steer,
    throttle,
    brake,
    offroad,
    dt,
    state,
    config,
    grade=0.0,
    bank=0.0,
    p=PARAMETERS,
):
    state.shift_timer = max(0.0, state.shift_timer - dt)
    state.shift_cooldown = max(0.0, state.shift_cooldown - dt)
    target_rpm = max(
        4500.0, abs(float(state.omega[2:].mean())) * state.ratio() * 60 / (2 * math.pi)
    )
    state.rpm += min(1.0, dt / 0.04) * (target_rpm - state.rpm)
    if config.automatic and state.gear > 0:
        if state.rpm > 12400:
            state.shift(1)
        elif state.rpm < 6500:
            state.shift(-1)
    # Broad torque curve, automatic launch clutch, torque interruption during shifts.
    torque = (
        float(
            np.interp(
                state.rpm, [4500, 7500, 10500, 12500, 14500, 15000], [240, 350, 430, 400, 300, 0]
            )
        )
        * throttle
    )
    torque = min(torque, p.power / max(1.0, state.rpm * 2 * math.pi / 60))
    if state.shift_timer > 0:
        torque *= 0.15
    if state.gear == -1 and u < -6:
        torque = 0.0
    drive = torque * state.ratio() * 0.94 * (1 if state.gear > 0 else -1)
    tc_factor = 1.0
    if config.traction_control:
        spinning = max(0.0, float(np.max(state.slip[2:] * (1 if state.gear > 0 else -1))) - 0.10)
        tc_factor = 1 / (1 + 12 * spinning)
        drive *= tc_factor
    engine_brake = (1 - throttle) * 65 * state.ratio() * min(1.0, abs(u) / 5) * np.sign(u)
    wheel_torque = np.array([0.0, 0.0, (drive - engine_brake) / 2, (drive - engine_brake) / 2])
    brake_torque = (
        brake
        * p.brake_force
        * RADIUS
        * np.array([p.brake_balance / 2] * 2 + [(1 - p.brake_balance) / 2] * 2)
    )
    abs_active = np.zeros(4, dtype=bool)
    if config.abs and abs(u) > 5:
        abs_active = state.slip * np.sign(u) < -0.13
        brake_torque[abs_active] *= 0.18
    aero = 0.5 * p.air_density * u * u
    downforce = aero * p.downforce_area
    gravity_normal = p.mass * 9.81 * math.cos(math.atan(grade)) * math.cos(bank)
    # Damped pitch/roll modes drive sprung load transfer. No chassis-height teleport.
    for name, target, frequency in (
        ("pitch", state.ax / 9.81 * 0.018, 11.0),
        ("roll", state.ay / 9.81 * 0.025, 13.0),
    ):
        value = getattr(state, name)
        rate = getattr(state, name + "_rate")
        rate += (
            frequency**2 * (float(np.clip(target, -0.12, 0.12)) - value)
            - 2 * 0.85 * frequency * rate
        ) * dt
        setattr(state, name + "_rate", rate)
        setattr(state, name, value + rate * dt)
    longitudinal_transfer = np.clip(
        state.pitch / 0.018 * p.mass * 9.81 * p.cg_height / p.wheelbase,
        -0.35 * gravity_normal,
        0.35 * gravity_normal,
    )
    front = gravity_normal * 1.6 / 3.4 + downforce * p.aero_front_fraction - longitudinal_transfer
    rear = gravity_normal - front + downforce
    side_transfer = state.roll / 0.025 * p.mass * 9.81 * p.cg_height / 1.5
    loads = np.maximum(
        80.0,
        np.array([front / 2, front / 2, rear / 2, rear / 2])
        + side_transfer * np.array([0.55, -0.55, 0.45, -0.45]),
    )
    mu = np.where(offroad, p.grass_mu, p.tyre_mu)
    capacities = mu * loads * (loads / (p.mass * 9.81 / 4)) ** (-0.08)
    angles = np.array([steer, steer, 0.0, 0.0])
    cs, sn = np.cos(angles), np.sin(angles)
    vx, vy = u - yaw * WHEEL_Y, v + yaw * WHEEL_X
    longitudinal = vx * cs + vy * sn
    transverse = -vx * sn + vy * cs
    slip_angle = np.arctan2(transverse, np.maximum(np.abs(longitudinal), 3.0))
    stiffness = np.array([p.front_stiffness / 2] * 2 + [p.rear_stiffness / 2] * 2)
    fy = -capacities * np.tanh(stiffness * slip_angle / capacities) * min(1.0, abs(u) / 3)
    # Semi-implicit compliant longitudinal tire: avoids low-speed wheel stiffness
    # instability while retaining independent angular speeds, spin and locking.
    free = state.omega + wheel_torque * dt / WHEEL_INERTIA
    braked = np.sign(free) * np.maximum(0.0, np.abs(free) - brake_torque * dt / WHEEL_INERTIA)
    longitudinal_stiffness = 80000.0
    denominator = (
        np.maximum(np.abs(longitudinal), 3.0)
        + longitudinal_stiffness * RADIUS**2 * dt / WHEEL_INERTIA
    )
    fx = longitudinal_stiffness * (RADIUS * braked - longitudinal) / denominator
    usage = np.maximum(1.0, np.hypot(fx, fy) / capacities)
    fx, fy = fx / usage, fy / usage
    state.omega = braked - fx * RADIUS * dt / WHEEL_INERTIA
    state.angle += state.omega * dt
    state.slip = np.clip(
        (RADIUS * state.omega - longitudinal) / np.maximum(np.abs(longitudinal), 3.0), -10, 10
    )
    bx, by = fx * cs - fy * sn, fx * sn + fy * cs
    rolling = p.mass * 9.81 * (0.015 + 0.045 * float(np.mean(offroad))) * math.tanh(u)
    gravity_x = -9.81 * math.sin(math.atan(grade))
    gravity_y = -9.81 * math.sin(bank)
    ax = (float(bx.sum()) - aero * p.drag_area * np.sign(u) - rolling) / p.mass + gravity_x
    ay = float(by.sum()) / p.mass + gravity_y
    state.ax += min(1.0, dt / 0.035) * (ax - state.ax)
    state.ay += min(1.0, dt / 0.035) * (ay - state.ay)
    new_u = float(np.clip(u + (ax + v * yaw) * dt, -8.0, 110.0))
    new_v = v + (ay - u * yaw) * dt
    new_yaw = yaw + float(np.sum(WHEEL_X * by - WHEEL_Y * bx)) / p.yaw_inertia * dt
    if abs(u) < 5:
        blend = np.clip((abs(u) - 1.5) / 3.5, 0.0, 1.0)
        blend = blend * blend * (3 - 2 * blend)
        geometric_yaw = new_u * math.tan(steer) / p.wheelbase
        new_yaw = float(blend * new_yaw + (1 - blend) * geometric_yaw)
        new_v = float(blend * new_v + (1 - blend) * geometric_yaw * 1.6)
    if abs(new_u) < 0.05 and throttle == 0 and brake > 0.05:
        new_u = new_v = new_yaw = 0.0
        state.omega[:] = 0.0
    telemetry = {
        "longitudinal_g": ax / 9.81,
        "lateral_g": ay / 9.81,
        "downforce_n": downforce,
        "front_slip_deg": float(np.degrees(slip_angle[:2].mean())),
        "rear_slip_deg": float(np.degrees(slip_angle[2:].mean())),
        "front_load_n": float(loads[:2].sum()),
        "rear_load_n": float(loads[2:].sum()),
        "wheel_load_n": loads.tolist(),
        "wheel_force_n": np.column_stack([fx, fy]).tolist(),
        "wheel_capacity_n": capacities.tolist(),
        "wheel_slip_ratio": state.slip.tolist(),
        "rpm": float(state.rpm),
        "gear": state.gear,
        "pitch": state.pitch,
        "roll": state.roll,
        "tc_active": bool(tc_factor < 0.99),
        "abs_active": bool(abs_active.any()),
        "grade": float(grade),
        "bank": float(bank),
    }
    return new_u, new_v, new_yaw, telemetry
