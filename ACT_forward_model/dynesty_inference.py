import pickle
import numpy as np
import dynesty
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import matplotlib.pyplot as plt

from gp_emulator import GPEmulator


def make_prior_transform(bounds):
    """bounds: list of (lo, hi) per parameter."""
    def prior_transform(u):
        return np.array([lo + u[i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)])
    return prior_transform


def _gauss_loglike(mu, sig_gp, y_data, cov_data):
    n = len(y_data)
    cov_total = cov_data + np.diag(sig_gp ** 2)
    try:
        L = np.linalg.cholesky(cov_total)
    except np.linalg.LinAlgError:
        return -1e30
    delta = mu - y_data
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, delta))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (delta @ alpha + log_det + n * np.log(2 * np.pi))


def make_log_likelihood(emulator, y_data, cov_data):
    """Plain fit: emulator vs data on all aperture bins."""
    def log_likelihood(theta):
        mu, sig_gp = emulator.predict(np.array(theta).reshape(1, -1))
        return _gauss_loglike(mu[0], sig_gp[0], y_data, cov_data)
    return log_likelihood


def make_log_likelihood_anchored(emulator, y_data, cov_data):
    """Anchored fit: emulator rescaled to pass through last data point."""
    y_fit   = y_data[:-1]
    cov_fit = cov_data[:-1, :-1]
    def log_likelihood(theta):
        mu, sig_gp = emulator.predict(np.array(theta).reshape(1, -1))
        scale = y_data[-1] / mu[0][-1]
        return _gauss_loglike(mu[0][:-1] * scale, sig_gp[0][:-1] * scale, y_fit, cov_fit)
    return log_likelihood


def run_nested(log_likelihood, prior_transform, ndim, nlive=500, dlogz=0.5):
    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, ndim,
        nlive=nlive, bound="multi", sample="rwalk",
    )
    sampler.run_nested(dlogz=dlogz, print_progress=True)
    return sampler.results


def summarise(results, param_names):
    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights)
    print("\nPosterior summary:")
    for i, name in enumerate(param_names):
        lo, mid, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name}: {mid:.4f} +{hi-mid:.4f} -{mid-lo:.4f}")
    return samples


def plot_corner(results, param_names, out_path):
    n = len(param_names)
    fig, _ = dyplot.cornerplot(
        results, labels=param_names, show_titles=True,
        title_kwargs={"fontsize": 12},
        fig=plt.subplots(n, n, figsize=(3 * n, 3 * n)),
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Corner plot → {out_path}")


def _draw_band(equal, emulator, rng, n_samples):
    idx = rng.choice(len(equal), size=min(n_samples, len(equal)), replace=False)
    mus, band = [], []
    for theta in equal[idx]:
        mu, sig_gp = emulator.predict(theta.reshape(1, -1))
        mus.append(mu[0])
        draw = mu[0] + rng.normal(size=mu[0].shape) * sig_gp[0]
        band.append(np.where(draw > 0, draw, mu[0]))
    mu_med = np.median(mus, axis=0)
    lo68, hi68 = np.percentile(band, [16, 84], axis=0)
    return mu_med, lo68, hi68


def _save_fit_fig(ax, apertures, mu_med, lo68, hi68, y_data, y_err, out_path, title,
                  emulator_pred=None):
    ax.fill_between(apertures, lo68 * 1e6, hi68 * 1e6, color="lightblue", alpha=0.6)
    ax.plot(apertures, mu_med * 1e6, color="steelblue", ls="--", lw=2,
            label=r"Best fit $\pm 1\sigma$")
    if emulator_pred is not None:
        ax.plot(apertures, emulator_pred * 1e6, color="red", ls=":", lw=2,
                label="Emulator prediction")
    ax.errorbar(apertures, y_data * 1e6, yerr=y_err * 1e6,
                fmt="o", color="royalblue", lw=2, capsize=4, ms=6, label="Data")
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel(r"Aperture radius $\theta$ [arcmin]", fontsize=16)
    ax.set_ylabel(r"$Y_\mathrm{CAP}$ [arcmin$^2$] [$\times 10^{-6}$]", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Fit plot → {out_path}")


def plot_fit(results, emulator, apertures, y_data, y_err, out_path, n_samples=500,
             title=None, truth_params=None):
    """Plain fit: raw emulator output vs data."""
    weights = np.exp(results.logwt - results.logz[-1])
    equal   = resample_equal(results.samples, weights)
    mu_med, lo68, hi68 = _draw_band(equal, emulator, np.random.default_rng(0), n_samples)
    emulator_pred = None
    if truth_params is not None:
        mu_true, _ = emulator.predict(np.array(truth_params).reshape(1, -1))
        emulator_pred = mu_true[0]
    _save_fit_fig(plt.subplots(figsize=(7, 5))[1], apertures, mu_med, lo68, hi68,
                  y_data, y_err, out_path, title, emulator_pred=emulator_pred)


def plot_fit_anchored(results, emulator, apertures, y_data, y_err, out_path, n_samples=500,
                      title=None, truth_params=None):
    """Anchored fit: emulator rescaled to pass through last data point."""
    weights = np.exp(results.logwt - results.logz[-1])
    equal   = resample_equal(results.samples, weights)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(equal), size=min(n_samples, len(equal)), replace=False)
    mus, band = [], []
    for theta in equal[idx]:
        mu, sig_gp = emulator.predict(theta.reshape(1, -1))
        scale = y_data[-1] / mu[0][-1]
        mus.append(mu[0] * scale)
        draw = mu[0] * scale + rng.normal(size=mu[0].shape) * sig_gp[0] * scale
        band.append(np.where(draw > 0, draw, mu[0] * scale))
    mu_med = np.median(mus, axis=0)
    lo68, hi68 = np.percentile(band, [16, 84], axis=0)
    emulator_pred = None
    if truth_params is not None:
        mu_true, _ = emulator.predict(np.array(truth_params).reshape(1, -1))
        emulator_pred = mu_true[0]
    _save_fit_fig(plt.subplots(figsize=(7, 5))[1], apertures, mu_med, lo68, hi68,
                  y_data, y_err, out_path, title, emulator_pred=emulator_pred)


if __name__ == "__main__":
    # import pandas as pd
    from pathlib import Path

    emulator_path = "/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/all_steps_stacked_new.pkl"
    stacked_prof_path = emulator_path.replace(".pkl", ".npz")
    out_dir = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/inference")
    out_dir.mkdir(exist_ok=True)

    # data (Liu et al. 2025)
    # _df     = pd.read_csv(Path(__file__).parent / "fig4.csv").dropna()
    # ACT_AP  = _df["RApArcmin"].values
    # ACT_Y   = _df["pz2_act_dr6_Beta_1.6"].values
    # ACT_ERR = _df["pz2_act_dr6_Beta_1.6_err"].values
    ACT_AP  = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])
    ACT_Y   = np.array([8.9375901426e-08, 3.5286669084e-07, 7.0189522333e-07,
                        1.2090546136e-06, 1.6113528108e-06, 1.9116110451e-06,
                        2.3413244009e-06, 2.9703678477e-06, 3.6016090038e-06])
    ACT_ERR = np.array([9.4267502323e-09, 2.2639352683e-08, 3.6185523443e-08,
                        5.6818925154e-08, 8.2360102801e-08, 1.1498411717e-07,
                        1.4746886290e-07, 1.8369545709e-07, 2.1632720061e-07])
    _corr   = np.load(Path(__file__).parent / "fig6_pz2.npy")
    COV_DATA = np.diag(ACT_ERR) @ _corr @ np.diag(ACT_ERR)

    # prior bounds
    _npz = np.load(stacked_prof_path, allow_pickle=True)
    param_names  = list(_npz["param_names"])
    prior_bounds = [
        (float(np.percentile(_npz["params"][:, i], 5)), float(np.percentile(_npz["params"][:, i], 95)))
        for i in range(_npz["params"].shape[1])
    ]

    em = GPEmulator.load(emulator_path)

    log_likelihood  = make_log_likelihood_anchored(em, ACT_Y, COV_DATA)
    prior_transform = make_prior_transform(prior_bounds)

    results = run_nested(log_likelihood, prior_transform, ndim=len(param_names))

    with open(out_dir / "dynesty_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results → {out_dir / 'dynesty_results.pkl'}")

    summarise(results, param_names)
    plot_corner(results, param_names, out_dir / "corner_anchored.png")
    plot_fit_anchored(results, em, ACT_AP, ACT_Y, ACT_ERR, out_dir / "fit_plot_anchored.png")