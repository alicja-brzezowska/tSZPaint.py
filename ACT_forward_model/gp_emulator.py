import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler


class GPEmulator:
    def __init__(self, params, profiles):
        """Train one GP per aperture on params (N,4) and profiles (N, N_ap).
        Uses arcsinh transform so negative ring-ring values are handled correctly."""
        self.X_scaler = StandardScaler()
        X = self.X_scaler.fit_transform(params)
        self._profile_scale = np.median(np.abs(profiles[profiles != 0]))
        transformed = np.arcsinh(profiles / self._profile_scale)

        self.gps, self.y_scalers = [], []
        for j in range(profiles.shape[1]):
            ys = StandardScaler()
            y = ys.fit_transform(transformed[:, j:j+1]).ravel()
            ls_bounds = [(1e-2, 1e2)] * (params.shape[1] - 1) + [(1e-2, 1e4)]
            gp = GaussianProcessRegressor(
                kernel=Matern(np.ones(params.shape[1]), ls_bounds, nu=2.5),
                n_restarts_optimizer=15,
                alpha=1e-10,
            )
            gp.fit(X, y)
            self.gps.append(gp)
            self.y_scalers.append(ys)

    def __getattr__(self, name):
        # backward compatibility: old pickles may be missing these attributes
        if name == "_profile_scale":
            return None
        if name == "_start_idx":
            return 0
        raise AttributeError(name)

    def predict(self, params_new):
        """Returns mean (M, N_ap) and std (M, N_ap) in linear Y units.
        Only aperture bins [_start_idx:] are returned (set at training time)."""
        X_new = self.X_scaler.transform(params_new)
        means, stds = [], []
        for gp, ys in zip(self.gps[self._start_idx:], self.y_scalers[self._start_idx:]):
            mu_s, sig_s = gp.predict(X_new, return_std=True)
            t_mu  = ys.inverse_transform(mu_s.reshape(-1, 1)).ravel()
            t_sig = sig_s * ys.scale_[0]
            if self._profile_scale is None:
                # legacy log10-trained pickle
                mu_lin = 10.0 ** t_mu
                means.append(mu_lin)
                stds.append(mu_lin * np.log(10) * t_sig)
            else:
                mu_lin = np.sinh(t_mu) * self._profile_scale
                means.append(mu_lin)
                stds.append(np.cosh(t_mu) * self._profile_scale * t_sig)
        return np.column_stack(means), np.column_stack(stds)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def load_data_cap(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    apertures = d["apertures_arcmin"]
    profiles_ycap = d["profiles"] * np.pi * apertures[np.newaxis, :] ** 2
    return d["params"], profiles_ycap, apertures, list(d["param_names"])


def load_data_ring_ring(npz_path, ring_width_arcmin=0.5):
    d = np.load(npz_path, allow_pickle=True)
    apertures = d["apertures_arcmin"]
    theta_0 = np.maximum(apertures - ring_width_arcmin, 0.0)
    ring_area = np.pi * (apertures**2 - theta_0**2)
    profiles_yrr = d["profiles"] * ring_area[np.newaxis, :]
    return d["params"], profiles_yrr, apertures, list(d["param_names"])


if __name__ == "__main__":
    stacked_prof_path = "/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/all_steps_stacked_ring_ring11.npz"
    emulator_path = stacked_prof_path.replace(".npz", ".pkl")

    params, profiles, apertures, param_names = load_data_ring_ring(stacked_prof_path)
    em = GPEmulator(params, profiles)
    em._start_idx = 2  # skip first two aperture bins; data has 9 points, emulator trains on 11
    em.save(emulator_path)

