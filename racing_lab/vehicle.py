"""Shared dimensions in metres; these are game parameters, not homologation data."""

SIMULATOR_VERSION = "0.4.0"
CAR_LENGTH = 5.5
CAR_WIDTH = 1.9
WHEELBASE = 3.4


def road_envelope(heading_error):
    """Half of the rectangular car's projection across a locally straight road."""
    import numpy as np

    return abs(np.cos(heading_error)) * CAR_WIDTH / 2 + abs(np.sin(heading_error)) * CAR_LENGTH / 2
