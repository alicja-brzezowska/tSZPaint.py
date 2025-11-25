import pickle
from pathlib import Path
import numpy as np

from astropy.cosmology import FlatLambdaCDM
import astropy.constants as aconst
import astropy.units as au
from scipy.integrate import quad_vec

from tszpaint.config import DATA_PATH  


class Battaglia16ThermalSZProfile:
    def __init__(self, f_b, cosmo):
        self.f_b = f_b
        self.cosmo = cosmo


def create_battaglia_profile(Omega_c=0.2589, Omega_b=0.0486, h=0.6774):
    """Create Battaglia cosmology + baryon fraction."""
    OmegaM = Omega_b + Omega_c
    f_b = Omega_b / OmegaM
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=OmegaM, Tcmb0=2.725)
    return Battaglia16ThermalSZProfile(f_b=f_b, cosmo=cosmo)


def get_params(M_200, z):
    z1 = z + 1.0
    m = M_200 / 1e14

    P0 = 18.1 * m**0.154 * z1 ** (-0.758)
    xc = 0.497 * m ** (-0.00865) * z1**0.731
    beta = 4.35 * m**0.0393 * z1**0.415
    alpha = 1.0
    gamma = -0.3
    beta = gamma - alpha * beta 

    return xc, alpha, beta, gamma, P0


def generalized_nfw(x, xc, alpha, beta, gamma):
    x_bar = x / xc
    return x_bar**gamma * (1 + x_bar**alpha) ** ((beta - gamma) / alpha)


def nfw_los(x, xc, alpha, beta, gamma, zmax=1e5, rtol=1e-6):
    """Line-of-sight integral of gNFW profile."""
    scale = 1e9

    def integrand(y):
        r_3d = np.sqrt(y**2 + x**2)
        return scale * generalized_nfw(r_3d, xc, alpha, beta, gamma)

    integral, _ = quad_vec(integrand, 0.0, zmax, epsrel=rtol)
    return 2.0 * integral / scale


def compute_R_delta(model, M_200, z, delta=200):
    """Compute R_delta in Mpc."""
    rho_crit = model.cosmo.critical_density(z).to(au.Msun / au.Mpc**3).value
    R = (3 * M_200 / (4 * np.pi * delta * rho_crit)) ** (1 / 3)
    return R


def angular_size(model, physical_size, z):
    """Convert physical size (Mpc) to angular size (radians)."""
    d_A = model.cosmo.angular_diameter_distance(z).to(au.Mpc).value
    return np.arctan(physical_size / d_A)


def dimensionless_P_profile_los(model, r, M_200, z):
    """Dimensionless pressure profile along LOS at angle r (radians)."""
    xc, alpha, beta, gamma, P0 = get_params(M_200, z)

    R_200 = compute_R_delta(model, M_200, z, delta=200)
    theta_200 = angular_size(model, R_200, z)
    x = r / theta_200

    integral = nfw_los(x, xc, alpha, beta, gamma)
    return P0 * integral


def P_th_los(model, r, M_200c, z):
    """Thermal pressure along LOS."""
    M_200c_kg = M_200c * aconst.M_sun.value
    rho_crit = model.cosmo.critical_density(z).to(au.kg / au.m**3).value
    norm = aconst.G.value * M_200c_kg * 200 * rho_crit * model.f_b / 2
    P_tilde = dimensionless_P_profile_los(model, r, M_200c, z)
    return norm * P_tilde


def P_e_los(model, r, M_200c, z):
    """Electron pressure along LOS."""
    return 0.5176 * P_th_los(model, r, M_200c, z)


P_E_FACTOR = aconst.sigma_T.value / (aconst.m_e.value * aconst.c.value**2)


def compton_y(model, r, M_200c, z):
    """Compton-y at angle r, mass M_200c, redshift z."""
    P_e = P_e_los(model, r, M_200c, z)
    return P_e * P_E_FACTOR



def build_python_grid(
    log_theta_min=-25.4,
    log_theta_max=11.5,
    N_log_theta=64,
    z_min=1e-3,
    z_max=5.0,
    N_z=32,
    log_M_min=11.0,
    log_M_max=15.7,
    N_log_M=64,
    outfile: Path | None = None,
):

    model = create_battaglia_profile()

    log_thetas = np.linspace(log_theta_min, log_theta_max, N_log_theta)
    redshifts = np.linspace(z_min, z_max, N_z)
    log_masses = np.linspace(log_M_min, log_M_max, N_log_M)

    thetas = np.exp(log_thetas)
    masses = 10.0**log_masses

    Y = np.zeros((N_log_theta, N_z, N_log_M))

    for i, theta in enumerate(thetas):
        for j, z in enumerate(redshifts):
            for k, M in enumerate(masses):
                y_val = compton_y(model, theta, M, z)
                Y[i, j, k] = max(y_val, 0.0)

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "wb") as f:
            pickle.dump(
                {
                    "log_thetas": log_thetas,
                    "redshifts": redshifts,
                    "log_masses": log_masses,
                    "prof_y": Y,
                },
                f,
            )
        print(f"Saved Python grid to {outfile}")

    return log_thetas, redshifts, log_masses, Y



if __name__ == "__main__":
    OUTFILE = DATA_PATH / "y_values_python.pkl"

    build_python_grid(
        N_log_theta=256,
        N_z=128,
        N_log_M=128,
        outfile=OUTFILE,
    )
