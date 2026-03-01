from dataclasses import dataclass


@dataclass
class PainterConfig:
    nside: int = 8192
    search_radius: float = (
        4  # Battaglia (2012) found that most tSZ signal is within 4*R_{200} critical
    )
    weight_bin_width: float = 2e-5  # typical halo has r_90 of 4.183e-04: 5% of that
