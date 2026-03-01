from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tszpaint.paint.abacus_loader import load_abacus_halos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect first 20 entries from load_abacus_halos output."
    )
    parser.add_argument("halo_file", type=Path, help="Path to halo ASDF file")
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of entries to print (default: 20)",
    )
    args = parser.parse_args()

    (
        positions,
        num_particles,
        particle_mass,
        redshift,
        radius,
        comoving_distance,
    ) = load_abacus_halos(args.halo_file)

    n = min(args.n, len(num_particles))

    np.set_printoptions(precision=6, suppress=True)
    print(f"redshift={redshift}")
    print(f"particle_mass={particle_mass}")
    print(f"N_halos={len(num_particles)}")
    print(f"positions shape={positions.shape}")
    print(f"radius shape={radius.shape}")
    print(f"comoving_distance shape={comoving_distance.shape}")
    print("\nFirst entries:")

    m_halos = num_particles.astype(np.float64) * particle_mass
    for i in range(n):
        print(
            f"[{i}] pos={positions[i]}  N={num_particles[i]}  "
            f"M={m_halos[i]:.3e}  r90={radius[i]:.3e}  chi={comoving_distance[i]:.3e}"
        )


if __name__ == "__main__":
    main()
