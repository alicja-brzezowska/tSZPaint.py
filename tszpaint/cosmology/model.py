import numpy as np

from tszpaint.y_profile.y_profile import Battaglia16ThermalSZProfile, angular_size


def get_angular_size_from_comoving(
    model: Battaglia16ThermalSZProfile,
    radius: np.ndarray,
    z: float,
):
    # radius should be in physical Mpc (already converted from survey units if needed)
    d_prop = radius / (1 + z)
    return angular_size(model, d_prop, z)
