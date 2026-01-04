from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tszpaint.interpolator import BattagliaLogInterpolator
from tszpaint.config import DATA_PATH
from tszpaint.y_profile import (
    create_battaglia_profile,
    compton_y,
)

JAX_PATH = DATA_PATH / "y_values_jax_2.pkl"
JULIA_PATH = DATA_PATH / "battaglia_interpolation.jld2"
PYTHON_PATH = DATA_PATH / "y_values_python.pkl"


LOG_THETA_MIN = -16.5
LOG_THETA_MAX = 2.5
LOG_M_MIN = 11.0
LOG_M_MAX = 15.7


Z_SLICE = 0.5          
THETA_SLICE = 1e-4    
LOGM_SLICE = 14.0      


N_M = 200
N_THETA = 400


MODEL = create_battaglia_profile()

def compute_direct_Y_vs_M(theta, z, M_array):
    Y = np.empty_like(M_array, dtype=float)
    for i, M in enumerate(M_array):
        Y[i] = compton_y(MODEL, theta, M, z)
    return Y

def compute_direct_Y_vs_theta(theta_array, z, M_fixed):
    return compton_y(MODEL, theta_array, M_fixed, z)


def load_interpolators():
    jax_interp = BattagliaLogInterpolator.from_pickle(JAX_PATH)
    julia_interp = BattagliaLogInterpolator.from_jld2(JULIA_PATH)
    python_interp = BattagliaLogInterpolator.from_pickle(PYTHON_PATH)

    return jax_interp, julia_interp, python_interp


def plot_Y_vs_M(jax_interp, julia_interp, python_interp):

    z = Z_SLICE
    theta = THETA_SLICE

    logM = np.linspace(LOG_M_MIN, LOG_M_MAX, N_M)
    M = 10.0**logM  

    theta_arr = np.full_like(M, theta, dtype=float)
    z_arr = np.full_like(M, z, dtype=float)

    Y_jax = jax_interp.eval(theta_arr, z_arr, M)
    Y_julia = julia_interp.eval(theta_arr, z_arr, M)
    Y_python = python_interp.eval(theta_arr, z_arr, M)

    Y_ref = compute_direct_Y_vs_M(theta, z, M)

    max_diff_interp = np.max(np.abs(Y_jax - Y_julia))

    abs_err_py = np.abs(Y_python - Y_ref)
    abs_err_jl = np.abs(Y_julia - Y_ref)
    abs_err_jax = np.abs(Y_jax - Y_ref)

    denom = np.maximum(np.abs(Y_ref), 1e-30)  
    rel_err_py = abs_err_py / denom
    rel_err_jl = abs_err_jl / denom
    rel_err_jax = abs_err_jax / denom

    max_rel_err_jax = np.max(rel_err_jax)
    max_rel_err_py = np.max(rel_err_py)
    max_rel_err_jl = np.max(rel_err_jl)

    plt.figure(figsize=(7, 5))
    plt.loglog(M, Y_ref, "k-", label="Direct (compton_y)")
    plt.loglog(M, Y_jax, label="JAX grid")
    plt.loglog(M, Y_julia, "--", label="Julia grid")
    plt.loglog(M, Y_python,"--", label="Python grid")

    plt.xlabel(r"$M\ [M_\odot]$")
    plt.ylabel(r"$Y(\theta, z, M)$")
    plt.title(
        rf"$Y$ vs $M$ at $\theta={theta:.2e}$, $z={z}$"
    )
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.semilogx(M, rel_err_py, label="Python / direct − 1")
    plt.semilogx(M, rel_err_jl, "--", label="Julia / direct − 1")
    plt.semilogx(M, rel_err_jax, label= "JAX / direct - 1")

    plt.xlabel(r"$M\ [M_\odot]$")
    plt.ylabel(r"Relative error of interpolators")
    plt.title("Relative error at $\theta={theta:.2e}$, $z={z}$")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    return max_diff_interp, max_rel_err_py, max_rel_err_jl, max_rel_err_jax

def plot_Y_vs_theta(jax_interp, julia_interp, python_interp):
    z = Z_SLICE
    logM_fixed = LOGM_SLICE
    M_fixed = 10.0**logM_fixed  

    log_theta = np.linspace(LOG_THETA_MIN, LOG_THETA_MAX, N_THETA)
    theta = np.exp(log_theta)  

    z_arr = np.full_like(theta, z, dtype=float)
    M_arr = np.full_like(theta, M_fixed, dtype=float)

    Y_jax = jax_interp.eval(theta, z_arr, M_arr)
    Y_julia = julia_interp.eval(theta, z_arr, M_arr)
    Y_python = python_interp.eval(theta, z_arr, M_arr)

    Y_ref = compute_direct_Y_vs_theta(theta, z, M_fixed)

    max_diff_interp = np.max(np.abs(Y_jax - Y_julia))

    abs_err_py = np.abs(Y_python - Y_ref)
    abs_err_jl = np.abs(Y_julia - Y_ref)
    abs_err_jax = np.abs(Y_jax - Y_ref)

    denom = np.maximum(np.abs(Y_ref), 1e-30)
    rel_err_py = abs_err_py / denom
    rel_err_jl = abs_err_jl / denom
    rel_err_jax = abs_err_jax /denom

    max_rel_err_py = np.max(rel_err_py)
    max_rel_err_jl = np.max(rel_err_jl)
    max_rel_err_jax = np.max(rel_err_jax)

    plt.figure(figsize=(7, 5))
    plt.loglog(theta, Y_ref, "k-", label="Direct (compton_y)")
    plt.loglog(theta, Y_jax, label="JAX grid")
    plt.loglog(theta, Y_julia, "--", label="Julia grid")
    plt.loglog(theta, Y_python, "--", label="Python grid")

    plt.xlabel(r"$\theta\ \mathrm{[radians]}$")
    plt.ylabel(r"$Y(\theta, z, M)$")
    plt.title("$Y$ vs $\theta$ at $z={z}, M=10^{{{logM_fixed:.1f}}} M_\odot$")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.semilogx(theta, rel_err_py, label="Python / direct − 1")
    plt.semilogx(theta, rel_err_jl, "--", label="Julia / direct − 1")
    plt.semilogx(theta, rel_err_jax, "--", label="JAX / direct − 1")

    plt.xlabel(r"$\theta\ \mathrm{[radians]}$")
    plt.ylabel(r"relative error of interpolators")
    plt.title(
        rf"Relative error at $z={z}, M=10^{{{logM_fixed:.1f}}} M_\odot$")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

    return max_diff_interp, max_rel_err_py, max_rel_err_jl, max_rel_err_jax


def plot_multiple_Y_vs_theta():
    z = Z_SLICE
    logM_list = [13.0, 13.5, 14.0, 14.5, 15, 15.5, 16]

    log_theta = np.linspace(LOG_THETA_MIN, LOG_THETA_MAX, N_M)
    theta = np.exp(log_theta)

    plt.figure(figsize=(7, 5))

    for logM in logM_list:
        M = 10.0**logM
        # this helper is for Y(theta) at fixed M
        Y_ref = compute_direct_Y_vs_theta(theta, z, M)

        plt.loglog(
            theta,
            Y_ref,
            label=rf"$M = 10^{{{logM:.1f}}}\,M_\odot$"
        )


    plt.xlabel(r"$\theta\ \mathrm{[radians]}$")
    plt.ylabel(r"$Y(\theta, z, M)$")
    plt.title(rf"$Y$ vs $\theta$ at $z={z}$")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()

def plot_multiple_Y_vs_mass():
    z = Z_SLICE

    theta = THETA_SLICE  # already a linear angle in radians

    logM = np.linspace(LOG_M_MIN, LOG_M_MAX, N_M)
    M = 10.0**logM

    Y_ref = compute_direct_Y_vs_M(theta, z, M)

    plt.figure(figsize=(7, 5))
    plt.loglog(
        M,
        Y_ref,
        label=rf"$\theta = {theta:.2e}\ \mathrm{{rad}}$"
    )

    pivot_index = len(M) // 2
    pivot_M = M[pivot_index]
    pivot_Y = Y_ref[pivot_index]

    Y_refline = pivot_Y * (M / pivot_M)**(5/3)

    plt.loglog(M, Y_refline, 'k--', label=r"$\propto M^{5/3}$")
    plt.xlabel(r"$M\ [M_\odot]$")
    plt.ylabel(r"$Y(\theta, z, M)$")
    plt.title(rf"$Y$ vs $M$ at $z={z}$, $\theta={theta:.2e}$")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()



def integrated_Y_cyl(M, z, theta_min, theta_max, n_theta=512):
    """
    Cylindrically integrated Compton-Y:
        Y_cyl(<theta_max) = 2π ∫ y(theta) * theta dtheta
    """
    theta = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta)
    y_theta = compton_y(MODEL, theta, M, z)  # array of y(theta)

    integrand = y_theta * theta
    Ycyl = 2 * np.pi * np.trapz(integrand, theta)  # cylindrical integration

    return Ycyl


def plot_integrated_Y_vs_mass():
    z = Z_SLICE

    # Choose physical aperture: a few arcmin → ~1e-3 rad
    theta_min = 1e-7
    theta_max = 1e-2   # adjust if you want a different aperture

    logM = np.linspace(LOG_M_MIN, LOG_M_MAX, N_M)
    M = 10.0**logM

    # Compute integrated Y for each mass
    Y_cyl = np.array([
        integrated_Y_cyl(Mi, z, theta_min, theta_max)
        for Mi in M
    ])

    # --- Compute slope d(logY)/d(logM)
    logY = np.log(Y_cyl)
    logM_lin = np.log(M)
    slope = np.gradient(logY, logM_lin)
    mean_slope = np.mean(slope[10:-10])
    print(f"\nMean slope dlogY/dlogM = {mean_slope:.3f} (expected ~1.667)\n")

    # --- Reference M^(5/3) line (matched at pivot point)
    pivot = len(M) // 2
    Y_refline = Y_cyl[pivot] * (M / M[pivot])**(5/3)

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.loglog(M, Y_cyl, label=rf"Integrated $Y_{{\rm cyl}}(<{theta_max:.1e}\,{{\rm rad}})$")
    plt.loglog(M, Y_refline, 'k--', label=r"$\propto M^{5/3}$")

    plt.xlabel(r"$M\ [M_\odot]$")
    plt.ylabel(r"$Y_{\rm cyl}$")
    plt.title(rf"Integrated Compton-$Y$ vs Mass at $z={z}$")
    plt.legend()
    plt.grid(True, which='both', ls=":")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    jax_interp, julia_interp, python_interp = load_interpolators()

    max_diff_M, max_rel_py, max_rel_jl, max_rel_jax = plot_Y_vs_M(jax_interp, julia_interp, python_interp)
    max_diff_theta = plot_Y_vs_theta(jax_interp, julia_interp, python_interp)
    plot_masses = plot_multiple_Y_vs_theta()
    plot_masses_2 = plot_integrated_Y_vs_mass()


    print(f"Max deltaY for Y vs M (Python vs Julia):        {max_diff_M:.6e}")
    print(f"Max rel error vs direct (Python):            {max_rel_py:.6e}")
    print(f"Max rel error vs direct (Julia):             {max_rel_jl:.6e}")
    print(f"Max rel error vs direct (JAX):             {max_rel_jax:.6e}")
    print(f"Max deltaY for Y vs theta slice (interp-only): {max_diff_theta:.6e}")
