import os
from dataclasses import dataclass

import astropy.constants as aconst
import astropy.units as au
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad_vec



@dataclass
class Battaglia16ThermalSZProfile:
    f_b: float
    cosmo: FlatLambdaCDM


def create_battaglia_profile(Omega_c=0.2589, Omega_b=0.0486, h=0.6774): #TODO: check new parameters
    """Create a cosmology model based on parameters from Battaglia et al. 2016."""
    OmegaM = Omega_b + Omega_c
    f_b = Omega_b / OmegaM

    cosmo = FlatLambdaCDM(H0=h * 100, Om0=OmegaM, Tcmb0=2.725)
    return Battaglia16ThermalSZProfile(f_b=f_b, cosmo=cosmo)


def get_params(M_200, z):
    """given M_200 and z, return gNFW parameters for the halo taken from Battagalia et al. 2016."""
    z1 = z + 1.0
    m = M_200 / 1e14

    # From Battaglia et al. 2016
    P0 = 18.1 * m**0.154 * z1 ** (-0.758)
    xc = 0.497 * m ** (-0.00865) * z1**0.731  # dimensionless scaling factor 
    beta = 4.35 * m**0.0393 * z1**0.415  
    alpha = 1.0
    gamma = -0.3
    beta = gamma - alpha * beta  # Sigurd's conversion from Battaglia to standard NFW

    return {"xc": xc, "alpha": alpha, "beta": beta, "gamma": gamma, "P0": P0}


def generalized_nfw(x, xc, alpha, beta, gamma):
    """Generalized NFW profile equation"""

    x_bar = x / xc  
    return x_bar**gamma * (1 + x_bar**alpha) ** ((beta - gamma) / alpha)


def nfw_los(x, xc, alpha, beta, gamma, zmax=1e5, rtol=1e-8):
    """Obtain the gNFW distribution along the line of sight"""
    # We observe the 2D projection of the gNFW profile, integrate along the line of sight
    # check diff integration schemes, abel transform 
    
    # x represents the projected radius on the sky

    scale = 1e9

    def integrand(z):
        r_3d = np.sqrt(z**2 + x**2)
        return scale * generalized_nfw(r_3d, xc, alpha, beta, gamma)

    integral, error = quad_vec(integrand, 0.0, zmax, epsrel=rtol)
    return 2.0 * integral / scale


def compute_R_delta(model, M_200, z, delta=200):
    """Compute R_delta"""
    # R_delta - radius where the mean density is delta times the critical density
    rho_crit = model.cosmo.critical_density(z).to(au.Msun / au.Mpc**3).value
    R = (3 * M_200 / (4 * np.pi * delta * rho_crit)) ** (1 / 3)
    return R


def angular_size(model, physical_size, z):
    """Convert physical size to angular size"""
    d_A = model.cosmo.angular_diameter_distance(z).to(au.Mpc).value
    return np.arctan(physical_size / d_A)


def dimensionless_P_profile_los(model, r, M_200, z):
    """Dimensionless pressure profile"""
    params = get_params(M_200, z)
    R_200 = compute_R_delta(model, M_200, z, delta=200)
    theta_200 = angular_size(model, R_200, z)
    x = r / theta_200

    integral = nfw_los(
        x, params["xc"], params["alpha"], params["beta"], params["gamma"]
    )
    return params["P0"] * integral


def P_th_los(model, r, M_200c, z):
    """Obtain thermal pressure from dimensionless pressure"""
    M_200c_kg = M_200c * aconst.M_sun.value
    rho_crit = model.cosmo.critical_density(z).to(au.kg / au.m**3).value

    norm = aconst.G.value * M_200c_kg * 200 * rho_crit * model.f_b / 2
    P_tilde = dimensionless_P_profile_los(model, r, M_200c, z)

    return norm * P_tilde

def P_e_los(model, r, M_200c, z):
    """Electron pressure"""
    return 0.5176 * P_th_los(model, r, M_200c, z)  # under what assumptions?


# prefactor for compton_y : Thomson cross-section / (electron mass * c^2)
P_e_factor = aconst.sigma_T.value / (aconst.m_e.value * aconst.c.value**2)


def compton_y(model: Battaglia16ThermalSZProfile, r: float, M_200c: float, z: float):
    """Calculate the Compton-y parameter"""
    P_e = P_e_los(model, r, M_200c, z)
    return P_e * P_e_factor


# GRID
def profile_grid(model, log_thetas, redshifts, log_masses, verbose=True):
    """Build 3D grid of Compton-y values"""
    thetas = np.exp(log_thetas)
    masses = np.pow(10, log_masses)

    compton = compton_y(model, thetas, masses, redshifts)
    A = np.maximum(0.0, compton)

    return A


