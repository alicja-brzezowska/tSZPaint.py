
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dynesty.utils import resample_equal
import pandas as pd

from gp_emulator import load_emulator, predict

DYNESTY_PKL  = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/inference/dynesty_results.pkl")
EMULATOR_PKL = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/all_steps_stacked.pkl")
CORR_NPY     = Path(__file__).parent / "fig6_pz2.npy"
OUT_DEFAULT  = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/inference/dynesty_fit_plot.png")


emulator_path = "/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/all_steps_stacked_new.pkl"
stacked_prof_path = emulator_path.replace(".pkl", ".npz")
out_dir = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/inference")
out_dir.mkdir(exist_ok=True)

# data (Liu et al. 2025)
_df     = pd.read_csv(Path(__file__).parent / "data" / "fig4.csv").dropna()
ACT_AP  = _df["RApArcmin"].values
ACT_Y   = _df["pz2_act_dr6_Beta_fiducial"].values
ACT_ERR = _df["pz2_act_dr6_Beta_fiducial_err"].values
    #ACT_AP  = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])
    #ACT_Y   = np.array([8.9375901426e-08, 3.5286669084e-07, 7.0189522333e-07,
    #                    1.2090546136e-06, 1.6113528108e-06, 1.9116110451e-06,
    #                    2.3413244009e-06, 2.9703678477e-06, 3.6016090038e-06])
    #ACT_ERR = np.array([9.4267502323e-09, 2.2639352683e-08, 3.6185523443e-08,
    #                    5.6818925154e-08, 8.2360102801e-08, 1.1498411717e-07,
    #                    1.4746886290e-07, 1.8369545709e-07, 2.1632720061e-07])
_corr   = np.load(Path(__file__).parent / "data" / "fig4_cov.npy")
COV_DATA = np.diag(ACT_ERR) @ _corr @ np.diag(ACT_ERR)



def load_posterior_samples(n_samples):
    """Draw n_samples equal-weighted samples from the dynesty posterior."""
    with open(DYNESTY_PKL, "rb") as f:
        results = pickle.load(f)
    weights = np.exp(results.logwt - results.logz[-1])
    equal_samples = resample_equal(results.samples, weights)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(equal_samples), size=min(n_samples, len(equal_samples)), replace=False)
    return equal_samples[idx], results


def build_predictive_band(samples, gps, X_scaler, y_scalers, include_gp_unc=True):
    """
    For each posterior sample predict the emulator profile.
    Returns array of shape (n_samples, n_apertures) in units of Y_CAP [arcmin²].

    If include_gp_unc=True, add GP noise by drawing from N(mu, sig_gp) per aperture.
    """
    rng = np.random.default_rng(0)
    all_profiles = []
    for theta in samples:
        mu, sig_gp = predict(gps, X_scaler, y_scalers, theta.reshape(1, -1))
        mu    = mu[0]
        sig_gp = sig_gp[0]
        if include_gp_unc:
            draw = mu + rng.normal(size=mu.shape) * sig_gp
            # keep positive
            draw = np.where(draw > 0, draw, mu)
            all_profiles.append(draw)
        else:
            all_profiles.append(mu)
    return np.array(all_profiles)   # (n_samples, n_ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",       default=str(OUT_DEFAULT))
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of posterior draws for the predictive band")
    parser.add_argument("--no-gp-unc", action="store_true",
                        help="Exclude GP emulator uncertainty from the predictive band")
    args = parser.parse_args()


    gps, X_scaler, y_scalers, param_names = load_emulator()

    samples, results = load_posterior_samples(args.n_samples)

    # ── posterior median prediction ──────────────────────────────────────────
    weights_all = np.exp(results.logwt - results.logz[-1])
    equal_all   = resample_equal(results.samples, weights_all)
    median_params = np.median(equal_all, axis=0)
    mu_med, _ = predict(gps, X_scaler, y_scalers, median_params.reshape(1, -1))
    mu_med = mu_med[0]

    print("\nPosterior median parameters:")
    for name, val in zip(param_names, median_params):
        print(f"  {name} = {val:.4f}")
    print(f"\nMedian predicted Y_CAP × 1e6: {mu_med * 1e6}")

    # ── posterior predictive band (1σ parameter uncertainty + GP emulator unc) ─
    band = build_predictive_band(samples, gps, X_scaler, y_scalers,
                                 include_gp_unc=not args.no_gp_unc)

    lo68, hi68 = np.percentile(band, [16, 84], axis=0)

    # plot 
    fig, ax = plt.subplots(figsize=(7, 5))

    # 1sig band (parameter posterior + GP emulator uncertainty)
    ax.fill_between(ACT_AP, lo68 * 1e6, hi68 * 1e6,
                    color="lightblue", alpha=0.6)

    # Best-fit (posterior median)
    ax.plot(ACT_AP, mu_med * 1e6, color="lightblue", ls="--", lw=2, label="Best fit")

    # ACT DR6 data
    ax.errorbar(ACT_AP, ACT_Y * 1e6, yerr=ACT_ERR * 1e6,
                fmt="o-", color="royalblue", lw=2, capsize=4, ms=6,
                label="ACT DR6 (Liu+2025)")

    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel(r"Aperture radius $\theta$ [arcmin]", fontsize=13)
    ax.set_ylabel(r"$Y_\mathrm{CAP}$ [arcmin$^2$] [$\times 10^{-6}$]", fontsize=13)
    ax.set_title(r"DESI PZ2, $\bar{z} = 0.628$", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
