"""
GP emulator: one GP per aperture, trained on the 125-point LHC grid
over Battaglia parameters (alpha, beta0, gamma, log10_P0).

  python gp_emulator.py          # train and save
  python gp_emulator.py --check  # also print in-sample RMS residuals
"""

import numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
import pickle
import argparse

NPZ_PATH = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/all_steps_stacked.npz")
PKL_PATH = NPZ_PATH.with_suffix(".pkl")


def load_data():
    d = np.load(NPZ_PATH, allow_pickle=True)
    apertures = d["apertures_arcmin"]
    # Convert mean y_CAP → Y_CAP [arcmin²]: Y = y × π θ²
    profiles_ycap = d["profiles"] * np.pi * apertures[np.newaxis, :] ** 2
    return d["params"], profiles_ycap, apertures, list(d["param_names"])


def _build_kernel(n_dims):
    amp = ConstantKernel(1.0, (1e-3, 1e3))
    ls  = np.ones(n_dims)
    return amp * Matern(ls, (1e-2, 1e2), nu=2.5)


def train(params, profiles, verbose=False):
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(params)

    log_profiles = np.log10(profiles)

    gps, y_scalers = [], []
    for j in range(profiles.shape[1]):
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(log_profiles[:, j : j + 1]).ravel()
        gp = GaussianProcessRegressor(
            kernel=_build_kernel(params.shape[1]),
            n_restarts_optimizer=10,
            alpha=1e-10,
        )
        gp.fit(X, y)
        if verbose:
            print(f"  Aperture {j+1}: log-ML = {gp.log_marginal_likelihood_value_:.3f}")
        gps.append(gp)
        y_scalers.append(y_scaler)

    return gps, X_scaler, y_scalers


def predict(gps, X_scaler, y_scalers, params_new):
    """Returns mean (M, N_ap) and std (M, N_ap) in linear (y) units."""
    X_new = X_scaler.transform(params_new)
    means, stds = [], []
    for gp, ys in zip(gps, y_scalers):
        mu_s, sig_s = gp.predict(X_new, return_std=True)
        # back to log10 space
        log_mu  = ys.inverse_transform(mu_s.reshape(-1, 1)).ravel()
        log_sig = sig_s * ys.scale_[0]
        # back to linear; propagate uncertainty: σ_y ≈ y * ln(10) * σ_log10
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
    parser.add_argument("--check", action="store_true", help="Print in-sample RMS residuals")
    args = parser.parse_args()

    params, profiles, apertures, param_names = load_data()
    print(f"Training: {len(params)} points, {params.shape[1]} params, {profiles.shape[1]} apertures")

    gps, X_scaler, y_scalers = train(params, profiles)

    with open(PKL_PATH, "wb") as f:
        pickle.dump({"gps": gps, "X_scaler": X_scaler, "y_scalers": y_scalers,
                     "param_names": param_names}, f)
    print(f"Saved → {PKL_PATH}")

    if args.check:
        mu, _ = predict(gps, X_scaler, y_scalers, params)
        print("\nIn-sample RMS residuals:")
        for j, ap in enumerate(apertures):
            rms = np.sqrt(np.mean((profiles[:, j] - mu[:, j]) ** 2))
            print(f"  {ap:.3f} arcmin: {rms:.3e}  (mean={profiles[:, j].mean():.3e})")
