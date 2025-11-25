import pickle
from pathlib import Path
from time import perf_counter


import jax
import jax.numpy as jnp
import numpy as np
import unxt as u
from jax.typing import ArrayLike 
from quadax import quadts # jax friendly quadrature integration
from wcosmo import comoving_distance, efunc  #jax friendly atropy.cosmology
from wcosmo.utils import disable_units  # wcosmo objects have defaults units; disable as jax arrays don't support units

import tszpaint.constants as const

jax.config.update("jax_enable_x64", True) #make float 64 
disable_units()

FILE = Path("y_values_jax_2.pkl")

G_VALUE = const.G_SI.value

DELTA = 200
SCALE = 1e9 
ZMAX = 1e5
RTOL = 1e-4 # saw that this was sufficient in tests, can change


def get_gnfw_params(mass: float, redshift: float):
    z1 = redshift + 1.0
    m = mass / 1e14

    # From Battaglia et al. 2016
    P0 = 18.1 * m**0.154 * z1 ** (-0.758)
    xc = 0.497 * m ** (-0.00865) * z1**0.731    # dimensionless scale radius
    beta = 4.35 * m**0.0393 * z1**0.415
    alpha = 1.0
    gamma = -0.3
    beta = gamma - alpha * beta  # Sigurd's conversion from Battaglia to standard NFW
    return xc, alpha, beta, gamma, P0


def critical_density(redshift: float, H0_cgs: float, Om0: float):
    # return value in g/cm3
    crd0 = (3 * H0_cgs**2) / (8 * jnp.pi * const.G_CGS)
    return crd0 * efunc(redshift, Om0) ** 2


def y_value(
    log_theta: float,
    redshift: float,
    log_mass: float,
    H0: float,
    H0_cgs: float,
    f_b: float,
    Om0: float,
):
    theta = jnp.exp(log_theta)
    mass = 10**log_mass
    rho_crit = critical_density(redshift, H0_cgs, Om0)
    rho_crit_kg_m3 = rho_crit * const.G_CM3_TO_KG_M3
    rho_crit_msun_mpc3 = rho_crit * const.G_CM3_TO_MSUN_MPC3

    R_200 = (3 * mass / (4 * jnp.pi * DELTA * rho_crit_msun_mpc3)) ** (1 / 3)

    d_A = comoving_distance(redshift, H0, Om0) / (
        redshift + 1.0
    )  
    theta_200 = jnp.arctan(R_200 / d_A)

    x_sq = (theta / theta_200) ** 2
    xc, alpha, beta, gamma, P0 = get_gnfw_params(mass, redshift)

    def integrand(y):
        r_3d = jnp.sqrt(y**2 + x_sq)
        x_bar = r_3d / xc
        generalized_nfw = x_bar**gamma * (1 + x_bar**alpha) ** ((beta - gamma) / alpha)
        return SCALE * generalized_nfw

    integral_value, _ = quadts(
        integrand,
        [0.0, ZMAX],  
        epsrel=RTOL,
    )

    integral = 2.0 * integral_value / SCALE
    P_tilde = P0 * integral

    mass_kg = mass * const.M_SUN
    norm = G_VALUE * mass_kg * 200 * rho_crit_kg_m3 * f_b / 2
    P_e = 0.5176 * norm * P_tilde
    compton = P_e * const.P_E_FACTOR

    return jnp.maximum(0.0, compton)


DIM = 256


def measure_y_values(
    Omega_c: float = 0.2589,
    Omega_b: float = 0.0486,
    h: float = 0.6774,
    N_log_theta: int = 2*DIM,
    log_theta_min: float = -25.4,
    log_theta_max: float = 11.5,
    z_min: float = 1e-3,
    z_max: float = 5.0,
    log_M_min: float = 11.0,
    log_M_max: float = 15.7,
    N_z: int = DIM, 
    N_log_M: int = DIM,
):
    OmegaM = Omega_b + Omega_c
    f_b = Omega_b / OmegaM
    H0 = h * u.Quantity(100, "km/s/Mpc")
    H0_cgs = H0.to("1/s")

    H0_value = float(H0.value)
    H0_cgs_value = float(H0_cgs.value)

    log_thetas = jnp.linspace(log_theta_min, log_theta_max, N_log_theta)
    redshifts = jnp.linspace(z_min, z_max, N_z)
    log_masses = jnp.linspace(log_M_min, log_M_max, N_log_M)

    theta_grid, z_grid, mass_grid = jnp.meshgrid(
        log_thetas, redshifts, log_masses, indexing="ij"
    )
    
    # flatten grids for vmap
    theta_flat = theta_grid.ravel()
    z_flat = z_grid.ravel()
    mass_flat = mass_grid.ravel()

    vmap_all = jax.vmap(y_value, in_axes=[0, 0, 0, None, None, None, None])
    vmap_all_jitted = jax.jit(vmap_all, static_argnames=("H0", "H0_cgs", "f_b", "Om0"))

    t = perf_counter()
    chunk_size = 65536  # tune based on RAM
    results = []
    for i in range(0, theta_flat.size, chunk_size):
        y_chunk = vmap_all_jitted(
            theta_flat[i : i + chunk_size],
            z_flat[i : i + chunk_size],
            mass_flat[i : i + chunk_size],
            H0_value,
            H0_cgs_value,
            f_b,
            OmegaM,
        ).block_until_ready()
        results.append(y_chunk)
    y_flat = jnp.concatenate(results)
    print(f"Chunks of {chunk_size}: {perf_counter() - t} seconds")
    t = perf_counter()

    # y_flat = vmap_all_jitted(
    #     theta_flat, z_flat, mass_flat, H0_value, H0_cgs_value, f_b, OmegaM
    # ).block_until_ready()
    # print("No chunking:", perf_counter() - t)

    # Reshape back to grid
    y_grid = y_flat.reshape(theta_grid.shape)
    return log_thetas, redshifts, log_masses, y_grid


def dump_to_file(
    log_thetas: ArrayLike,
    redshifts: ArrayLike,
    log_masses: ArrayLike,
    y_grid: ArrayLike,
):
    # TODO: save this instead
    log_prof_y_np = np.asarray(jnp.log(y_grid + 1e-100))
    prof_y_np = np.asarray(y_grid)
    log_thetas_np = np.asarray(log_thetas)
    redshifts_np = np.asarray(redshifts)
    log_masses_np = np.asarray(log_masses)

    with open(FILE, "wb") as f:
        pickle.dump(
            {
                "log_thetas": log_thetas_np,
                "redshifts": redshifts_np,
                "log_masses": log_masses_np,
                "prof_y": prof_y_np,
            },
            f,
        )


if __name__ == "__main__":
    dump_to_file(*measure_y_values())

