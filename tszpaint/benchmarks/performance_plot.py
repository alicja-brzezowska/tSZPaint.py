
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from tszpaint.y_profile import build_python_grid as y_vals_np
from tszpaint.y_profile_jax import measure_y_values as y_vals_jax

SAMPLE_DIMS = [8, 16]
ALL_DIMS = [16, 32, 64, 128]


def time_function(func, **kwargs: Any):
    from time import perf_counter

    start = perf_counter()
    print(f"Starting function {func.__name__} with args {kwargs}")
    func(**kwargs)
    end = perf_counter()
    print(f"Function {func.__name__} completed in {end - start} seconds")
    return end - start


def benchmark_grid(dims: list[int]):
    rv = {}

    for dim in dims:
        rv[dim] = [
            {
                "backend": "numpy",
                "time": time_function(y_vals_np, N_log_theta=dim, N_z=dim, N_log_M=dim),
            },
            {
                "backend": "jax",
                "time": time_function(
                    y_vals_jax, N_log_theta=dim, N_z=dim, N_log_M=dim
                ),
            },
        ]

    df = pd.DataFrame(
        [
            {"dimension": dim, "backend": entry["backend"], "time": entry["time"]}
            for dim in rv
            for entry in rv[dim]
        ]
    )
    df.to_csv("benchmark_y_profile2.csv", index=False)
    return df


def visualize_benchmark(df: pd.DataFrame):
    df = df.copy()
    df["npoints"] = df["dimension"] ** 3

    # Paper-style defaults
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Professional, muted palette
    styles = {
        "numpy": dict(color="#4C72B0", linestyle="-", marker="o"),
        "jax":   dict(color="#55A868", linestyle="-", marker="s"),
        "julia": dict(color="#CC3E43", linestyle="-", marker="^"),
    }

    for backend, style in styles.items():
        data = df[df["backend"] == backend]
        if data.empty:
            continue

        ax.loglog(
            data["npoints"],
            data["time"],
            label=backend.capitalize(),
            **style,
        )

    ax.set_xlabel(r"Grid resolution N ( total points: $N^3$)")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("y-profile computation benchmark")

    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(frameon=False)
    fig.savefig("benchmark_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig("benchmark_scaling.pdf", bbox_inches="tight")

    fig.tight_layout()
    plt.show()




if __name__ == "__main__":
    df = pd.read_csv("benchmark_y_profile.csv")
    # julia values go here
    julia_df = pd.DataFrame(
        [
            {"dimension": 16, "backend": "julia", "time": 1.13 },
            {"dimension": 32, "backend": "julia", "time": 2.07},
            {"dimension": 64, "backend": "julia", "time": 9.67},
            {"dimension": 128, "backend": "julia", "time": 73.13},
            {"dimension": 256, "backend": "julia", "time": 561.56},
            {"dimension": 512, "backend": "julia", "time": 4433.99},
        ]
    )
    visualize_benchmark(pd.concat([df, julia_df]))
