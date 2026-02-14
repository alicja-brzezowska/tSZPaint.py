import numpy as np


def convert_rad_to_cart(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Convert radial coordinates to Cartesian (unit vectors)."""
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=1
    )


def convert_cart_to_rad(xyz: np.ndarray):
    """Given cartesian coordinates, convert to radial."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)  # Ensure phi in [0, 2π]
    return theta, phi
