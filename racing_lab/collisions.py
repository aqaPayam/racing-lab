"""Local road-corridor contacts, resolved after every short integration substep."""

import numpy as np
from .vehicle import CAR_LENGTH, CAR_WIDTH
from .dynamics import PARAMETERS

CORNERS = np.array(
    [[x, y] for x in (-CAR_LENGTH / 2, CAR_LENGTH / 2) for y in (-CAR_WIDTH / 2, CAR_WIDTH / 2)]
)


def resolve(track, pos, heading, speed, lateral, yaw, hint):
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, s], [-s, c]])
    arms = CORNERS @ rotation
    velocity = np.array([speed, lateral]) @ rotation
    impact = 0.0
    contact = False
    for _ in range(8):
        _, offsets, angles = track.project_many(pos + arms, hint=hint)
        penetration = np.abs(offsets) - track.barrier_offset
        index = int(np.argmax(penetration))
        if penetration[index] < -0.003:
            break
        contact = True
        normal = np.array([-np.sin(angles[index]), np.cos(angles[index])]) * np.sign(offsets[index])
        arm = arms[index]
        point_velocity = velocity + yaw * np.array([-arm[1], arm[0]])
        closing = float(point_velocity @ normal)
        if closing > 0:
            impact = max(impact, closing)
            lever = arm[0] * normal[1] - arm[1] * normal[0]
            impulse = closing / (1 / PARAMETERS.mass + lever**2 / PARAMETERS.yaw_inertia)
            velocity -= impulse / PARAMETERS.mass * normal
            yaw -= impulse * lever / PARAMETERS.yaw_inertia
            # Dissipative tangential friction, without reversing the sliding motion.
            tangent = np.array([-normal[1], normal[0]])
            along = float(velocity @ tangent)
            velocity -= tangent * np.clip(
                along, -0.18 * impulse / PARAMETERS.mass, 0.18 * impulse / PARAMETERS.mass
            )
        if penetration[index] <= 1e-5:
            break
        pos = pos - normal * (penetration[index] + 1e-5)
    local_velocity = velocity @ rotation.T
    return pos, float(local_velocity[0]), float(local_velocity[1]), float(yaw), contact, impact
