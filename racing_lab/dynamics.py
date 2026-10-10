"""Planar single-track force model, SI units; generic open-wheel parameters.

Nonlinear axle slip forces share a friction circle with braking/traction.
Quasi-static longitudinal load transfer and speed-squared aero are included.
This is not a calibrated model of a specific Formula 1 car.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VehicleParameters:
    mass: float = 800.0
    yaw_inertia: float = 1100.0
    wheelbase: float = 3.4
    cg_to_front: float = 1.8
    cg_height: float = 0.3
    power: float = 500000.0
    drive_force: float = 14500.0
    brake_force: float = 32000.0
    brake_balance: float = 0.58
    drag_area: float = 1.1
    downforce_area: float = 4.2
    aero_front_fraction: float = 0.45
    tyre_mu: float = 1.65
    grass_mu: float = 0.48
    front_stiffness: float = 105000.0
    rear_stiffness: float = 115000.0
    air_density: float = 1.225


PARAMETERS = VehicleParameters()


def parameter_dict():
    return asdict(PARAMETERS)


def clamp(value, low, high):
    return max(low, min(high, value))


def advance(speed, lateral, yaw, steering, throttle, brake, offroad, dt, p=PARAMETERS):
    """One force integration step; returns body velocities and diagnostic forces."""
    rear_arm = p.wheelbase - p.cg_to_front
    aero = 0.5 * p.air_density * speed * speed
    downforce = aero * p.downforce_area
    drag = aero * p.drag_area
    rolling = p.mass * 9.81 * (0.06 if offroad else 0.015) * math.tanh(speed)
    mu = p.grass_mu if offroad else p.tyre_mu
    drive = throttle * min(p.drive_force, p.power / max(speed, 10.0))
    braking = brake * p.brake_force
    requested_ax = (drive - braking - drag - rolling) / p.mass
    transfer = clamp(
        p.mass * requested_ax * p.cg_height / p.wheelbase,
        -p.mass * 9.81 * 0.35,
        p.mass * 9.81 * 0.35,
    )
    front_load = max(
        100.0, p.mass * 9.81 * rear_arm / p.wheelbase + downforce * p.aero_front_fraction - transfer
    )
    rear_load = max(
        100.0,
        p.mass * 9.81 * p.cg_to_front / p.wheelbase
        + downforce * (1 - p.aero_front_fraction)
        + transfer,
    )
    cap_front, cap_rear = mu * front_load, mu * rear_load
    fx_front = clamp(-braking * p.brake_balance, -cap_front, cap_front)
    fx_rear = clamp(drive - braking * (1 - p.brake_balance), -cap_rear, cap_rear)
    alpha_front = steering - math.atan2(lateral + p.cg_to_front * yaw, max(speed, 3.0))
    alpha_rear = -math.atan2(lateral - rear_arm * yaw, max(speed, 3.0))
    remaining_front = math.sqrt(max(0.0, cap_front**2 - fx_front**2))
    remaining_rear = math.sqrt(max(0.0, cap_rear**2 - fx_rear**2))
    fy_front = remaining_front * math.tanh(p.front_stiffness * alpha_front / max(cap_front, 1.0))
    fy_rear = remaining_rear * math.tanh(p.rear_stiffness * alpha_rear / max(cap_rear, 1.0))
    cosine, sine = math.cos(steering), math.sin(steering)
    body_fx = fx_front * cosine - fy_front * sine + fx_rear - drag - rolling
    body_fy = fy_front * cosine + fx_front * sine + fy_rear
    ax = body_fx / p.mass
    ay = body_fy / p.mass
    new_speed = clamp(speed + (ax + lateral * yaw) * dt, 0.0, 110.0)
    if speed < 3.0:
        # Avoid the bicycle model's slip singularity near rest.
        new_yaw = new_speed * math.tan(steering) / p.wheelbase
        new_lateral = new_yaw * rear_arm
    else:
        new_lateral = lateral + (ay - speed * yaw) * dt
        moment = p.cg_to_front * (fy_front * cosine + fx_front * sine) - rear_arm * fy_rear
        new_yaw = yaw + moment / p.yaw_inertia * dt
    if new_speed < 0.05 and throttle == 0:
        new_speed = new_lateral = new_yaw = 0.0
    telemetry = {
        "longitudinal_g": ax / 9.81,
        "lateral_g": ay / 9.81,
        "downforce_n": downforce,
        "front_slip_deg": math.degrees(alpha_front),
        "rear_slip_deg": math.degrees(alpha_rear),
        "front_load_n": front_load,
        "rear_load_n": rear_load,
        "front_force_n": (fx_front, fy_front),
        "rear_force_n": (fx_rear, fy_rear),
        "front_capacity_n": cap_front,
        "rear_capacity_n": cap_rear,
    }
    return new_speed, new_lateral, new_yaw, telemetry
