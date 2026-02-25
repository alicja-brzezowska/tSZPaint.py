from dataclasses import dataclass


@dataclass
class PainterConfig:
    nside: int = 8192
    search_radius: float = 2
    n_bins: int = 20
