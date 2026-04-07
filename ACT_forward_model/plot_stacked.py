"""
Plot stacked CAP profiles from a saved .npz file.
Reproduces the Liu+2025 Fig. 3 style.

Usage:
  python plot_stacked.py stacked_profiles/all_steps_stacked_no_gaussian.npz
  python plot_stacked.py file1.npz file2.npz --labels "no beam" "with beam"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple']
FMTS   = ['o--', 's--', '^--', 'D--', 'v--']

# Liu+2025 ACT DR6 fiducial, pz1 (z̄=0.470), apertures 1.0–6.0 arcmin
# Y_CAP [arcmin²]  (plot ×1e6)
ACT_AP  = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])
ACT_Y   = np.array([8.9375901426e-08, 3.5286669084e-07, 7.0189522333e-07,
                    1.2090546136e-06, 1.6113528108e-06, 1.9116110451e-06,
                    2.3413244009e-06, 2.9703678477e-06, 3.6016090038e-06])
ACT_ERR = np.array([9.4267502323e-09, 2.2639352683e-08, 3.6185523443e-08,
                    5.6818925154e-08, 8.2360102801e-08, 1.1498411717e-07,
                    1.4746886290e-07, 1.8369545709e-07, 2.1632720061e-07])


def load(npz_path):
    d = np.load(npz_path)
    param_names = list(d['param_names']) if 'param_names' in d else ['alpha', 'beta_mul', 'gamma']
    return d['apertures_arcmin'], d['profiles'], d['errors'], d['params'], param_names


def plot(files, labels, out_path, log10_p0=1.0):
    fig, ax = plt.subplots(figsize=(6, 4))

    # ACT data first (background reference)
    ax.errorbar(ACT_AP, ACT_Y * 1e6, yerr=ACT_ERR * 1e6,
                fmt='o-', color='royalblue', capsize=4, lw=2,
                label=r'ACT DR6 (fiducial, $\bar{z}=0.470$)')

    for k, (fpath, label) in enumerate(zip(files, labels)):
        ap, profiles, errors, params, param_names = load(fpath)

        # Build fiducial target vector matching the number of params in the file
        fiducial_vals = [1.0, 4.5, -0.3, log10_p0]
        fiducial = np.array(fiducial_vals[:params.shape[1]])

        # Normalise each column by its std so all parameters are weighted equally
        scale = params.std(axis=0)
        scale[scale == 0] = 1.0  # avoid divide-by-zero for degenerate columns
        idx = np.argmin(np.linalg.norm((params - fiducial) / scale, axis=1))

        y     = profiles[idx]
        y_err = errors[idx]
        p     = params[idx]

        # Convert to Y_CAP [arcmin²] = y_CAP × π θ²
        Y     = y     * np.pi * ap**2
        Y_err = y_err * np.pi * ap**2

        legend_str = f"{label}  (α={p[0]:.2f}, β_mul={p[1]:.2f}, γ={p[2]:.3f}"
        if 'log10_P0' in param_names:
            legend_str += f", log₁₀P₀={p[3]:.2f}"
        legend_str += ")"

        ax.errorbar(ap, Y * 1e6, yerr=Y_err * 1e6,
                    fmt=FMTS[k], color=COLORS[k], capsize=4, lw=1.5,
                    label=legend_str)

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('R [arcmin]')
    ax.set_ylabel(r'Compton Y-parameter [arcmin$^2$] [$\times 10^{-6}$]')
    ax.set_title(r'tSZPaint stacked profile, $\bar{z} \approx 0.52$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='.npz file(s)')
    parser.add_argument('--labels', nargs='*', default=None)
    parser.add_argument('--out', default='stacked_profile_plot.png')
    parser.add_argument('--log10-p0', type=float, default=1.0)
    args = parser.parse_args()

    labels = args.labels or [Path(f).stem for f in args.files]
    plot(args.files, labels, args.out, log10_p0=args.log10_p0)


if __name__ == '__main__':
    main()
