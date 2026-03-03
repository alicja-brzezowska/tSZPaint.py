from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from tszpaint.cosmology.model import compute_theta_200
from tszpaint.paint.abacus_loader import SimulationData
from tszpaint.y_profile.interpolator import BattagliaLogInterpolator
from tszpaint.y_profile.y_profile import Battaglia16ThermalSZProfile


@dataclass
class RadialProfile:
    x_centers: np.ndarray
    y_mean: np.ndarray
    y_err: np.ndarray
    counts: np.ndarray
    num_samples: int
    x_ref: np.ndarray
    y_battaglia: np.ndarray
    mass_ref: float
    logM_center: float

    def as_dict(self):
        return {
            "x_centers": self.x_centers,
            "y_mean": self.y_mean,
            "y_err": self.y_err,
            "counts": self.counts,
            "num_samples": self.num_samples,
            "x_ref": self.x_ref,
            "y_battaglia": self.y_battaglia,
            "mass_ref": self.mass_ref,
            "logM_center": self.logM_center,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            x_centers=np.asarray(d["x_centers"]),
            y_mean=np.asarray(d["y_mean"]),
            y_err=np.asarray(d["y_err"]),
            counts=np.asarray(d["counts"]),
            num_samples=int(d["num_samples"]),
            x_ref=np.asarray(d["x_ref"]),
            y_battaglia=np.asarray(d["y_battaglia"]),
            mass_ref=float(d["mass_ref"]),
            logM_center=float(d["logM_center"]),
        )


@dataclass
class RadialProfileBuilderConfig:
    r_search: np.ndarray
    num_halos: int = 1000
    seed: int = 123
    log_m_centers: list[float] = field(
        default_factory=lambda: [12, 12.5, 13, 13.5, 14, 14.5]
    )
    log_m_halfwidth: float = 0.15
    num_bins: int = 20


@dataclass
class RadialProfileBuilder:
    cfg: RadialProfileBuilderConfig
    data: SimulationData
    pix_in_halos: np.ndarray
    halo_starts: np.ndarray
    halo_counts: np.ndarray
    distances: np.ndarray
    interpolator: BattagliaLogInterpolator
    model: Battaglia16ThermalSZProfile
    y_values: np.ndarray | None = None  # per-pair y values for isolated (pre-superimpose) profile

    def _common_x_grid(self):
        theta_200 = compute_theta_200(self.model, self.data.m_halos, self.data.redshift)
        ratio = self.cfg.r_search / theta_200
        good = np.isfinite(ratio) & (ratio > 0)

        if np.any(good):
            x_max = float(np.nanmax(ratio[good]))
        else:
            x_max = 1.0
        x_min = max(1e-4, x_max / 1e3)

        bin_edges = np.logspace(
            np.log10(x_min),
            np.log10(x_max),
            self.cfg.num_bins + 1,
        )
        x_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        x_ref = np.logspace(np.log10(x_min), np.log10(x_max), 400)
        return theta_200, bin_edges, x_centers, x_ref

    def _build_single(
        self,
        logm_center: float,
        y_pairs: np.ndarray,
        theta_200: np.ndarray,
        bin_edges: np.ndarray,
        x_centers: np.ndarray,
        x_ref: np.ndarray,
    ):
        """Build a radial profile for halos in a mass bin, given per-pair y values."""
        rng = np.random.default_rng(seed=self.cfg.seed)
        log_m = np.log10(self.data.m_halos)
        in_bin = np.abs(log_m - logm_center) <= self.cfg.log_m_halfwidth
        candidate_halos = np.flatnonzero(in_bin)

        num_samples = min(len(candidate_halos), self.cfg.num_halos)
        if len(candidate_halos) < self.cfg.num_halos:
            logger.warning(
                f"Not enough halos in mass bin centered at logM={logm_center} (found {len(candidate_halos)}, needed {self.cfg.num_halos})"
            )
        sample_halos = rng.choice(
            candidate_halos,
            size=min(self.cfg.num_halos, len(candidate_halos)),
            replace=False,
        )

        sum_m = np.zeros(self.cfg.num_bins, dtype=np.float64)
        sum_m2 = np.zeros(self.cfg.num_bins, dtype=np.float64)
        n_halos = np.zeros(self.cfg.num_bins, dtype=np.float64)

        for h in sample_halos:
            start = self.halo_starts[h]
            count = self.halo_counts[h]
            if count == 0:
                continue
            d = self.distances[start : start + count]
            y = y_pairs[start : start + count]
            keep = d <= self.cfg.r_search[h]
            x = d[keep] / theta_200[h]
            y = y[keep]
            bin_ids = np.minimum(
                np.searchsorted(bin_edges[1:], x, side="left"),  # pyright: ignore[reportUnknownArgumentType]
                self.cfg.num_bins - 1,
            )
            n = np.bincount(bin_ids, minlength=self.cfg.num_bins).astype(np.float64)
            has = n > 0
            mu = np.where(has, np.bincount(bin_ids, weights=y, minlength=self.cfg.num_bins) / np.where(has, n, 1.0), 0.0)
            sum_m += mu
            sum_m2 += mu**2
            n_halos += has

        with np.errstate(invalid="ignore", divide="ignore"):
            y_mean = sum_m / n_halos
            y_var = sum_m2 / n_halos - y_mean**2
            y_err = np.sqrt(np.maximum(y_var, 0.0)) / np.sqrt(n_halos)

        mass_ref = np.median(self.data.m_halos[sample_halos])
        theta_ref = compute_theta_200(
            self.model, np.array([mass_ref]), self.data.redshift
        )[0]
        theta_values = np.maximum(x_ref * theta_ref, 1e-40)  # pyright: ignore[reportAny, reportUnknownArgumentType]
        log_theta = np.log(theta_values)  # pyright: ignore[reportAny]
        log_M = np.full_like(log_theta, np.log10(mass_ref), dtype=np.float64)
        z_values = np.full_like(log_theta, self.data.redshift, dtype=np.float32)
        y_battaglia = np.asarray(
            self.interpolator.eval_for_logs(log_theta, z_values, log_M)  # pyright: ignore[reportUnknownArgumentType, reportCallIssue]
        )
        return RadialProfile(
            x_centers,
            y_mean,
            y_err,
            n_halos,
            num_samples,
            x_ref,
            y_battaglia,
            mass_ref,  # pyright: ignore[reportArgumentType]
            logm_center,
        )

    def build(self, y_map: np.ndarray):
        y_pairs = y_map[self.pix_in_halos]
        theta_200, bin_edges, x_centers, x_ref = self._common_x_grid()
        return [
            self._build_single(logm, y_pairs, theta_200, bin_edges, x_centers, x_ref)
            for logm in self.cfg.log_m_centers
        ]

    def build_isolated(self):
        assert self.y_values is not None, "y_values must be set to build isolated profiles"
        theta_200, bin_edges, x_centers, x_ref = self._common_x_grid()
        return [
            self._build_single(logm, self.y_values, theta_200, bin_edges, x_centers, x_ref)
            for logm in self.cfg.log_m_centers
        ]
