import numpy as np

from tszpaint.y_profile.y_profile import (
    Battaglia16ThermalSZProfile,
    angular_size,
    compute_R_delta,
)


def get_angular_size_from_comoving(
    model: Battaglia16ThermalSZProfile,
    radius: np.ndarray,
    z: float,
):
    # radius should be in physical Mpc (already converted from survey units if needed)
    d_prop = radius / (1 + z)
    return angular_size(model, d_prop, z)


def compute_theta_200(
    model: Battaglia16ThermalSZProfile,
    m_halos: np.ndarray,
    z: float,
    delta: int = 200,
):
    """Compute θ_200 (angular radius) from halo mass and redshift."""
    r_200 = compute_R_delta(model, m_halos, z, delta=delta)
    return angular_size(model, r_200, z)
