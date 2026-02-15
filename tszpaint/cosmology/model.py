import numpy as np

from tszpaint.y_profile.y_profile import Battaglia16ThermalSZProfile, angular_size


def get_angular_size_from_comoving(
    model: Battaglia16ThermalSZProfile,
    radius: np.ndarray,
    z: float,
):
    d_prop = radius / (1 + z)
    return angular_size(model, d_prop, z)
