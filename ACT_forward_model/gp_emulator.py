import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler


class GPEmulator:
    def __init__(self, params, profiles):
        """Train one GP per aperture on params (N,4) and profiles (N, N_ap)."""
        self.X_scaler = StandardScaler()
        X = self.X_scaler.fit_transform(params)
        log_profiles = np.log10(profiles)

        self.gps, self.y_scalers = [], []
        for j in range(profiles.shape[1]):
            ys = StandardScaler()
            y = ys.fit_transform(log_profiles[:, j:j+1]).ravel()
            ls_bounds = [(1e-2, 1e2)] * (params.shape[1] - 1) + [(1e-2, 1e4)]
            gp = GaussianProcessRegressor(
                kernel=Matern(np.ones(params.shape[1]), ls_bounds, nu=2.5),
                n_restarts_optimizer=15,
                alpha=1e-10,
            )
            gp.fit(X, y)
            self.gps.append(gp)
            self.y_scalers.append(ys)

    def predict(self, params_new):
        """Returns mean (M, N_ap) and std (M, N_ap) in linear Y units."""
        X_new = self.X_scaler.transform(params_new)
        means, stds = [], []
        for gp, ys in zip(self.gps, self.y_scalers):
            mu_s, sig_s = gp.predict(X_new, return_std=True)
            log_mu = ys.inverse_transform(mu_s.reshape(-1, 1)).ravel()
            log_sig = sig_s * ys.scale_[0]
            mu_lin = 10.0 ** log_mu
            means.append(mu_lin)
            stds.append(mu_lin * np.log(10) * log_sig)
        return np.column_stack(means), np.column_stack(stds)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    apertures = d["apertures_arcmin"]
    profiles_ycap = d["profiles"] * np.pi * apertures[np.newaxis, :] ** 2
    return d["params"], profiles_ycap, apertures, list(d["param_names"])


if __name__ == "__main__":
    stacked_prof_path = "/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/all_steps_stacked_new.npz"
    emulator_path = stacked_prof_path.replace(".npz", ".pkl")

    params, profiles, apertures, param_names = load_data(stacked_prof_path)
    em = GPEmulator(params, profiles)
    em.save(emulator_path)

