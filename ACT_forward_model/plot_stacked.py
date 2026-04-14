"""
Plot stacked CAP profiles from a saved .npz file.

Usage:
  python plot_stacked.py file1.npz file2.npz --out comparison.png
  python plot_stacked.py file1.npz file2.npz --log10-p0 1.0 --out comparison.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = ['tomato', 'seagreen', 'darkorange', 'purple', 'steelblue']
FMTS   = ['s--', '^--', 'D--', 'v--', 'o--']

HOD_LATEX = {
    'log_Mcut':   r'$\log M_\mathrm{cut}$',
    'log_M1':     r'$\log M_1$',
    'sigma':      r'$\sigma$',
    'alpha_hod':  r'$\alpha_\mathrm{HOD}$',
    'kappa':      r'$\kappa$',
}

DATA_DIR = Path(__file__).parent / "data"


def load_act_data():
    df = pd.read_csv(DATA_DIR / "fig4.csv").dropna()
    ap  = df["RApArcmin"].values
    y   = df["pz2_act_dr6_Beta_1.6"].values
    err = df["pz2_act_dr6_Beta_1.6_err"].values
    return ap, y, err


def load(npz_path):
    d = np.load(npz_path)
    param_names     = list(d['param_names'])
    hod_param_names = list(d['param_names_hod']) if 'param_names_hod' in d else []
    hod_params      = d['hod_params'] if 'hod_params' in d else None
    return d['apertures_arcmin'], d['profiles'], d['errors'], d['params'], param_names, hod_params, hod_param_names



def plot(files, out_path, log10_p0=1.0, anchored=False, row=None):
    act_ap, act_y, act_err = load_act_data()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(act_ap, act_y * 1e6, yerr=act_err * 1e6,
                fmt='o-', color='royalblue', capsize=4, lw=2,
                label=r'ACT DR6 (pz2, $\beta=1.6$)')

    title_str = None

    for k, fpath in enumerate(files):
        ap, profiles, errors, params, param_names, hod_params, hod_param_names = load(fpath)

        if row is not None:
            idx = row
        else:
            # find nearest-to-fiducial gNFW row
            fiducial_vals = [1.0, 4.5, -0.3, log10_p0]
            fiducial = np.array(fiducial_vals[:params.shape[1]])
            scale = params.std(axis=0)
            scale[scale == 0] = 1.0
            idx = np.argmin(np.linalg.norm((params - fiducial) / scale, axis=1))

        y     = profiles[idx]
        y_err = errors[idx]
        p     = params[idx]

        Y     = y     * np.pi * ap**2
        Y_err = y_err * np.pi * ap**2

        if anchored:
            anchor_scale = act_y[-1] / Y[-1]
            Y     = Y     * anchor_scale
            Y_err = Y_err * anchor_scale

        # title: gNFW params (same for all files, set once)
        if title_str is None:
            pn = param_names
            title_str = (
                r"$\alpha$=" + f"{p[0]:.2f}"
                + r",  $\beta_0$=" + f"{p[1]:.2f}"
                + r",  $\gamma$=" + f"{p[2]:.3f}"
            )
            if 'log10P0' in pn:
                title_str += r",  $\log_{10}P_0$=" + f"{p[3]:.2f}"
            if anchored:
                title_str += "  [anchored]"

        # legend: HOD params only (LaTeX names)
        if hod_params is not None:
            hod_str = ",  ".join(
                f"{HOD_LATEX.get(n, n)}={v:.3g}" for n, v in zip(hod_param_names, hod_params)
            )
        else:
            hod_str = Path(fpath).stem

        ax.errorbar(ap, Y * 1e6, yerr=Y_err * 1e6,
                    fmt=FMTS[k], color=COLORS[k], capsize=4, lw=1.5,
                    label=hod_str)

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('R [arcmin]', fontsize=14)
    ax.set_ylabel(r'$Y_\mathrm{CAP}$ [arcmin$^2$] [$\times 10^{-6}$]', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(title_str or '', fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='.npz file(s)')
    parser.add_argument('--out', default='stacked_profile_plot.png')
    parser.add_argument('--log10-p0', type=float, default=1.0)
    parser.add_argument('--anchored', action='store_true', help='Rescale each profile to pass through last ACT point')
    parser.add_argument('--row', type=int, default=None, help='Use a specific gNFW parameter row instead of nearest-fiducial')
    args = parser.parse_args()

    plot(args.files, args.out, log10_p0=args.log10_p0, anchored=args.anchored, row=args.row)


if __name__ == '__main__':
    main()
