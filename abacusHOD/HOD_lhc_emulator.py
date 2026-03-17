"""
HOD GP emulator: one GP per aperture, trained on the HOD LHC stacked profiles.

  python HOD_lhc_emulator.py         # train and save emulator
  python HOD_lhc_emulator.py --check # also print in-sample RMS residuals
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler


NPZ_PATH = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/hod_lhc_stacked.npz")
PKL_PATH = NPZ_PATH.with_suffix(".pkl")


def subsample_maximin(params, n: int, seed: int = 0) -> np.ndarray:
    """Return indices of n points from params that maximise the minimum pairwise distance.

    Uses greedy farthest-point sampling on unit-scaled parameters.
    """
    rng = np.random.default_rng(seed)
    X = (params - params.min(0)) / (params.max(0) - params.min(0))  # scale to [0,1]^d
    N = len(X)
    chosen = [int(rng.integers(N))]
    # distance from each point to nearest chosen point
    min_dists = np.full(N, np.inf)
    for _ in range(n - 1):
        last = chosen[-1]
        d = np.sum((X - X[last]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, d)
        min_dists[chosen] = -1.0
        chosen.append(int(np.argmax(min_dists)))
    return np.array(chosen)


def load_data():
    """Load HOD-LHC training data from NPZ.
    """
    d = np.load(NPZ_PATH, allow_pickle=True)
    apertures    = d["apertures_arcmin"]
    profiles_raw = d["profiles"]
    # Convert mean y_CAP → Y_CAP [arcmin²] 
    if profiles_raw.max() < 1e-3:        
        profiles_ycap = profiles_raw * np.pi * apertures[np.newaxis, :] ** 2
    else:
        profiles_ycap = profiles_raw
    return d["params"], profiles_ycap, apertures, list(d["param_names"])


def _build_kernel(n_dims: int):
    amp = ConstantKernel(1.0, (1e-3, 1e3))
    ls  = np.ones(n_dims)
    return amp * Matern(ls, (1e-2, 1e2), nu=2.5)


def train(params, profiles, verbose: bool = False):
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(params)

    log_profiles = np.log10(profiles)

    gps, y_scalers = [], []
    for j in range(profiles.shape[1]):
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(log_profiles[:, j : j + 1]).ravel()
        gp_model = GaussianProcessRegressor(
            kernel=_build_kernel(params.shape[1]),
            n_restarts_optimizer=10,
            alpha=1e-10,
        )
        gp_model.fit(X, y)
        if verbose:
            print(f"  Aperture {j+1}: log-ML = {gp_model.log_marginal_likelihood_value_:.3f}")
        gps.append(gp_model)
        y_scalers.append(y_scaler)

    return gps, X_scaler, y_scalers


def predict(gps, X_scaler, y_scalers, params_new):
    """Return mean (M, N_ap) and std (M, N_ap) in linear (Y_CAP) units."""
    X_new = X_scaler.transform(params_new)
    means, stds = [], []
    for gp_model, ys in zip(gps, y_scalers):
        mu_s, sig_s = gp_model.predict(X_new, return_std=True)
        log_mu  = ys.inverse_transform(mu_s.reshape(-1, 1)).ravel()
        log_sig = sig_s * ys.scale_[0]
        mu_lin  = 10.0 ** log_mu
        sig_lin = mu_lin * np.log(10) * log_sig
        means.append(mu_lin)
        stds.append(sig_lin)
    return np.column_stack(means), np.column_stack(stds)


def load_emulator():
    with open(PKL_PATH, "rb") as f:
        d = pickle.load(f)
    return d["gps"], d["X_scaler"], d["y_scalers"], d["param_names"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",     action="store_true", help="Print in-sample RMS residuals")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Train on a maximin subsample of this many points")
    args = parser.parse_args()

    params, profiles, apertures, param_names = load_data()

    if args.subsample is not None:
        idx = subsample_maximin(params, args.subsample)
        params, profiles = params[idx], profiles[idx]
        print(f"Maximin subsample: {len(idx)} / {len(np.load(NPZ_PATH)['params'])} points")
    print(f"Training: {len(params)} points, {params.shape[1]} params, "
          f"{profiles.shape[1]} apertures")

    gps, X_scaler, y_scalers = train(params, profiles, verbose=True)

    with open(PKL_PATH, "wb") as f:
        pickle.dump(
            {"gps": gps, "X_scaler": X_scaler, "y_scalers": y_scalers,
             "param_names": param_names},
            f,
        )
    print(f"Saved → {PKL_PATH}")

    if args.check:
        mu, _ = predict(gps, X_scaler, y_scalers, params)
        print("\nIn-sample RMS residuals:")
        for j, ap in enumerate(apertures):
            rms = np.sqrt(np.mean((profiles[:, j] - mu[:, j]) ** 2))
            print(f"  {ap:.3f} arcmin: {rms:.3e}  (mean={profiles[:, j].mean():.3e})")
