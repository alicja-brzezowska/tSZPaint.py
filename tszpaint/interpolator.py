import pickle
from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path
from loguru import logger

import h5py
import numpy as np
import pandas as pd
import jax.numpy as jnp


DEBUG = True
USE_JAX = True

if USE_JAX:
    from jax.scipy.interpolate import RegularGridInterpolator
else:
    from scipy.interpolate import RegularGridInterpolator


def linspace_from_julia_obj(julia_obj: dict[str, float | int]):
    return np.linspace(
        julia_obj["start"],
        julia_obj["stop"],
        julia_obj["len"],  
    )

@dataclass
class BattagliaLogInterpolator:
    interpolator: Callable
    use_jax: bool 

    @classmethod
    def from_matrices(
        cls,
        log_thetas: np.ndarray,
        redshifts: np.ndarray,
        log_masses: np.ndarray,
        prof_y: np.ndarray,
        use_jax: bool = USE_JAX
    ):
        log_prof_y = np.log(prof_y + 1e-100)
        logger.debug(
            f"Log thetas array contains {log_thetas.shape[0]} elements, smallest: {log_thetas.min()}, largest: {log_thetas.max()}"
        )
        logger.debug(
            f"Redshifts array contains {redshifts.shape[0]} elements, smallest: {redshifts.min()}, largest: {redshifts.max()}"
        )
        logger.debug(
            f"Log masses array contains {log_masses.shape[0]} elements, smallest: {log_masses.min()}, largest: {log_masses.max()}"
        )
        logger.debug(
            f"Largest value in prof-y: {prof_y.max()}, smallest: {prof_y.min()}"
        )

        if use_jax:
            lists = (jnp.asarray(log_thetas), jnp.asarray(redshifts), jnp.asarray(log_masses))
            log_prof_y = jnp.asarray(log_prof_y)
            interpolator = RegularGridInterpolator(lists, 
                                          log_prof_y, 
                                          method = "linear",
                                          bounds_error = False,
                                          fill_value = None)

        else:
            lists = (log_thetas, redshifts, log_masses)
            interpolator = RegularGridInterpolator( lists, 
                                          log_prof_y, 
                                          method = "linear",
                                          bounds_error = False,
                                          fill_value = None)

        return cls(interpolator = interpolator, use_jax = use_jax)

    @classmethod
    def from_csv(cls, path: Path):
        df = pd.read_csv(path)
        return cls.from_df(df)


    @classmethod
    def from_df(cls, df: pd.DataFrame, use_jax: bool = USE_JAX):
        log_thetas = df["log_thetas"].to_numpy()
        redshifts = df["redshifts"].to_numpy()
        log_masses = df["log_masses"].to_numpy()
        prof_y = df["prof_y"].to_numpy()
        log_prof_y = np.log(prof_y + 1e-100).to_numpy()


    @classmethod
    def from_jld2(cls, path: Path):
        with h5py.File(path, "r") as f:
            data = {k: f[k] for k in f.keys()}
            linspace_thetas = linspace_from_julia_obj(data["prof_logθs"])
            linspace_redshift = linspace_from_julia_obj(data["prof_redshift"])
            linspace_masses = linspace_from_julia_obj(data["prof_logMs"])
            prof_y_array = data["prof_y"][:].transpose(2, 1, 0)
            return cls.from_matrices(
                linspace_thetas,
                linspace_redshift,
                linspace_masses,
                prof_y_array,
            )

    @classmethod
    def from_pickle(cls, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls.from_matrices(
            np.array(data["log_thetas"]),
            np.array(data["redshifts"]),
            np.array(data["log_masses"]),
            np.array(data["prof_y"]),
        )

    def eval_for_logs(self, log_theta, z, log_M):
        if self.use_jax:
            lists = jnp.stack([log_theta, z, log_M], axis = -1) 
            log_y = self.interpolator(lists)
            return jnp.exp(log_y)
        else:
            log_y = self.interpolator((log_theta, z, log_M))
            return np.exp(log_y)

    def eval(self, theta, z, m):
        if self.use_jax:
            log_theta = jnp.log(theta)
            log_M = jnp.log10(m)
            lists = jnp.stack([log_theta, z, log_M], axis = -1)
            log_y = self.interpolator(lists)
            return jnp.exp(log_y)

        else: 
            log_theta = np.log(theta)
            log_M = np.log10(m)
            log_y = self.interpolator((log_theta, z, log_M))
            return np.exp(log_y)
    


