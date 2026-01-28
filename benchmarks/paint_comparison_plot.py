import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from tszpaint.config import DATA_PATH

OUTPUT_DIR = DATA_PATH / "comparison_output"


def load_maps():
    """Load Python (no weights), Python (weighted), and Julia maps."""
    python_map = hp.read_map(OUTPUT_DIR / "python_map.fits")
    julia_map = hp.read_map(OUTPUT_DIR / "julia_map.fits")

    weighted_path = OUTPUT_DIR / "python_map_weighted.fits"
    if weighted_path.exists():
        python_weighted = hp.read_map(weighted_path)
    else:
        python_weighted = None

    return python_map, julia_map, python_weighted


def plot_scatter_comparison(python_map, julia_map, save_path=None):
    """Scatter plot of pixel values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mask = (python_map > 0) | (julia_map > 0)
    py_vals = python_map[mask]
    jl_vals = julia_map[mask]

    # Linear scale
    ax = axes[0]
    ax.scatter(jl_vals, py_vals, alpha=0.3, s=1, c='steelblue')
    lims = [0, max(py_vals.max(), jl_vals.max()) * 1.1]
    ax.plot(lims, lims, 'r--', lw=2, label='1:1 line')
    ax.set_xlabel("Julia (XGPaint)")
    ax.set_ylabel("Python (tszpaint)")
    ax.set_title("Pixel-by-pixel comparison (linear)")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Log scale
    ax = axes[1]
    both_positive = (py_vals > 0) & (jl_vals > 0)
    ax.scatter(jl_vals[both_positive], py_vals[both_positive], alpha=0.3, s=1, c='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    lims_log = [min(py_vals[both_positive].min(), jl_vals[both_positive].min()) * 0.5,
                max(py_vals.max(), jl_vals.max()) * 2]
    ax.plot(lims_log, lims_log, 'r--', lw=2, label='1:1 line')
    ax.set_xlabel("Julia (XGPaint)")
    ax.set_ylabel("Python (tszpaint)")
    ax.set_title("Pixel-by-pixel comparison (log)")
    ax.legend()

    # Add correlation coefficient
    correlation = np.corrcoef(py_vals, jl_vals)[0, 1]
    fig.suptitle(f"Correlation: {correlation:.6f}", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved scatter plot to {save_path}")

    return fig


def plot_histogram_comparison(python_map, julia_map, save_path=None):
    """Histogram comparison of y values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    py_nonzero = python_map[python_map > 0]
    jl_nonzero = julia_map[julia_map > 0]

    # Linear histogram
    ax = axes[0]
    bins = np.linspace(0, max(py_nonzero.max(), jl_nonzero.max()), 50)
    ax.hist(py_nonzero, bins=bins, alpha=0.6, label=f'Python (n={len(py_nonzero)})', color='steelblue')
    ax.hist(jl_nonzero, bins=bins, alpha=0.6, label=f'Julia (n={len(jl_nonzero)})', color='coral')
    ax.set_xlabel("y value")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of y values (linear)")
    ax.legend()
    ax.set_yscale('log')

    # Log histogram
    ax = axes[1]
    log_bins = np.logspace(
        np.log10(min(py_nonzero.min(), jl_nonzero.min())),
        np.log10(max(py_nonzero.max(), jl_nonzero.max())),
        50
    )
    ax.hist(py_nonzero, bins=log_bins, alpha=0.6, label='Python', color='steelblue')
    ax.hist(jl_nonzero, bins=log_bins, alpha=0.6, label='Julia', color='coral')
    ax.set_xlabel("y value")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of y values (log bins)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved histogram to {save_path}")

    return fig


def print_statistics(python_map, julia_map):
    """Print comparison statistics."""
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)

    mask = (python_map > 0) | (julia_map > 0)

    print(f"\n{'Metric':<25} {'Python':<15} {'Julia':<15}")
    print("-"*55)
    print(f"{'Total signal':<25} {python_map.sum():<15.6e} {julia_map.sum():<15.6e}")
    print(f"{'Max':<25} {python_map.max():<15.6e} {julia_map.max():<15.6e}")
    print(f"{'Mean (non-zero)':<25} {python_map[mask].mean():<15.6e} {julia_map[mask].mean():<15.6e}")
    print(f"{'Non-zero pixels':<25} {(python_map > 0).sum():<15} {(julia_map > 0).sum():<15}")

    # Differences
    abs_diff = np.abs(python_map - julia_map)
    print(f"\n{'Differences:'}")
    print(f"{'  Mean abs diff':<25} {abs_diff[mask].mean():<15.6e}")
    print(f"{'  Max abs diff':<25} {abs_diff.max():<15.6e}")
    print(f"{'  Correlation':<25} {np.corrcoef(python_map[mask], julia_map[mask])[0,1]:<15.6f}")

    # Relative difference
    rel_diff_pct = 100 * np.abs(python_map.sum() - julia_map.sum()) / julia_map.sum()
    print(f"{'  Total signal diff':<25} {rel_diff_pct:<15.2f}%")


def main():
    """Generate all comparison plots."""
    print("Loading maps...")
    python_map, julia_map, python_weighted = load_maps()

    print_statistics(python_map, julia_map)

    if python_weighted is not None:
        print("\n" + "="*60)
        print("EFFECT OF WEIGHTING")
        print("="*60)
        print(f"  Total signal (no weights):  {python_map.sum():.6e}")
        print(f"  Total signal (weighted):    {python_weighted.sum():.6e}")
        print(f"  Ratio (weighted/no_wt):     {python_weighted.sum() / python_map.sum():.4f}")


    # Histogram
    plot_histogram_comparison(
        python_map, julia_map,
        save_path=OUTPUT_DIR / "comparison_histogram.png"
    )

    # Scatter plot 
    plot_scatter_comparison(
        python_map, julia_map,
        save_path=OUTPUT_DIR / "comparison_scatter.png"
    )

    print("\nDone! Generated plots:")
    print(f"  {OUTPUT_DIR / 'comparison_histogram.png'}")
    print(f"  {OUTPUT_DIR / 'comparison_scatter.png'}")

    plt.show()


if __name__ == "__main__":
    main()
