"""Shared chassis parameters in SI units; see vehicle_model for the current dynamics."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VehicleParameters:
    mass: float = 800.0
    yaw_inertia: float = 1100.0
    wheelbase: float = 3.4
    cg_to_front: float = 1.8
    cg_height: float = 0.3
    power: float = 500000.0
    brake_force: float = 32000.0
    brake_balance: float = 0.64
    drag_area: float = 1.1
    downforce_area: float = 4.2
    aero_front_fraction: float = 0.45
    tyre_mu: float = 1.65
    grass_mu: float = 0.48
    front_stiffness: float = 95000.0
    rear_stiffness: float = 150000.0
    air_density: float = 1.225


PARAMETERS = VehicleParameters()


def parameter_dict():
    from .vehicle_model import RADIUS, WHEEL_INERTIA, GEAR_RATIOS

    return {
        **asdict(PARAMETERS),
        "wheel_radius": RADIUS,
        "wheel_inertia": WHEEL_INERTIA,
        "gear_ratios": list(GEAR_RATIOS),
        "reverse_ratio": 12.0,
        "longitudinal_stiffness": 80000.0,
        "load_sensitivity_exponent": -0.08,
        "pitch_frequency": 11.0,
        "roll_frequency": 13.0,
        "suspension_damping": 0.85,
    }
