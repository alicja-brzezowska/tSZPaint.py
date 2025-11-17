import pickle
import numpy as np
from time import perf_counter
from scipy.interpolate import RegularGridInterpolator


from y_profile import profile_grid

# INTERPOLATOR
def build_interpolator(
    model,
    cache_file="",
    N_log_theta=128,
    log_theta_min=-16.5,
    log_theta_max=2.5,
    z_min=1e-3,
    z_max=5.0,
    log_M_min=11.0,
    log_M_max=15.7,
    N_z=64,
    N_log_M=64,
    overwrite=True,
    verbose=True,
):
    if not overwrite and cache_file and os.path.exists(cache_file):
        if verbose:
            print(f"Loading from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        log_thetas = data["log_thetas"]
        redshifts = data["redshifts"]
        log_masses = data["log_masses"]
        prof_y = data["prof_y"]
    else:
        if verbose:
            print("Building new interpolator")

        log_thetas = np.linspace(log_theta_min, log_theta_max, N_log_theta)
        redshifts = np.linspace(z_min, z_max, N_z)
        log_masses = np.linspace(log_M_min, log_M_max, N_log_M)

        xx, yy, zz = np.meshgrid(log_thetas, redshifts, log_masses, indexing="ij")

        t = perf_counter()
        prof_y = profile_grid(model, xx, yy, zz, verbose=verbose)

        print(f"profile_grid {perf_counter() - t} seconds")
        t = perf_counter()

        if cache_file:
            if verbose:
                print(f"Saving to {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "log_thetas": log_thetas,
                        "redshifts": redshifts,
                        "log_masses": log_masses,
                        "prof_y": prof_y,
                    },
                    f,
                )

        print(f"dumping to file took {perf_counter() - t} seconds")
        t = perf_counter()

    log_prof_y = np.log(prof_y + 1e-100)

    itp = RegularGridInterpolator(
        (log_thetas, redshifts, log_masses),
        log_prof_y,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )
    print(f"creating interpolator took {perf_counter() - t} seconds")

    return LogInterpolatorProfile(model, itp, log_thetas, redshifts, log_masses)


class LogInterpolatorProfile:
    """Wrapper for log-space interpolation"""

    def __init__(self, model, interpolator, log_thetas, redshifts, log_masses):
        self.model = model
        self.itp = interpolator
        self.log_thetas = log_thetas
        self.redshifts = redshifts
        self.log_masses = log_masses

    def __call__(self, theta, M, z):
        """Evaluate at (theta, M, z)"""
        log_theta = np.log(theta)
        log_M = np.log10(M)
        log_y = self.itp((log_theta, z, log_M))
        return np.exp(log_y)
