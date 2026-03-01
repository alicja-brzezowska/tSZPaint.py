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
        default_factory=lambda: [12.7, 13.0, 13.7, 14.0, 14.7, 15.0]
    )
    log_m_halfwidth: float = 0.2
    num_bins: int = 20


@dataclass
class RadialProfileBuilder:
    cfg: RadialProfileBuilderConfig
    data: SimulationData
    halo_starts: np.ndarray
    halo_counts: np.ndarray
    distances: np.ndarray
    interpolator: BattagliaLogInterpolator
    model: Battaglia16ThermalSZProfile

    def _build_single(self, logm_center: float, y_map: np.ndarray):
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
        theta_200 = compute_theta_200(self.model, self.data.m_halos, self.data.redshift)
        ratio = self.cfg.r_search[sample_halos] / theta_200[sample_halos]
        x_max = np.nanmax(ratio[np.isfinite(ratio)])
        x_min = max(1e-4, x_max / 1e3)
        bin_edges = np.logspace(  # pyright: ignore[reportUnknownVariableType]
            np.log10(x_min),  # pyright: ignore[reportAny]
            np.log10(x_max),  # pyright: ignore[reportAny]
            self.cfg.num_bins + 1,
        )
        x_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # pyright: ignore[reportAny, reportUnknownArgumentType]

        sum_y = np.zeros(self.cfg.num_bins, dtype=np.float64)
        sum_y2 = np.zeros(self.cfg.num_bins, dtype=np.float64)
        counts = np.zeros(self.cfg.num_bins, dtype=np.float64)

        for h in sample_halos:
            start = self.halo_starts[h]
            count = self.halo_counts[h]
            if count == 0:
                continue
            d = self.distances[start : start + count]
            y = y_map[start : start + count]
            keep = d <= self.cfg.r_search[h]
            if not np.any(keep):
                continue
            d = d[keep]
            y = y[keep]
            x = d / theta_200[h]
            bin_ids = np.searchsorted(bin_edges[1:], x, side="left")  # pyright: ignore[reportUnknownArgumentType]
            bin_ids = np.minimum(bin_ids, self.cfg.num_bins - 1)

            sum_y += np.bincount(bin_ids, weights=y, minlength=self.cfg.num_bins)
            sum_y2 += np.bincount(bin_ids, weights=y**2, minlength=self.cfg.num_bins)
            counts += np.bincount(bin_ids, minlength=self.cfg.num_bins)

        with np.errstate(invalid="ignore", divide="ignore"):
            y_mean = sum_y / counts
            y_var = sum_y2 / counts - y_mean**2
            y_std = np.sqrt(np.maximum(y_var, 0.0))
            y_err = y_std / np.sqrt(counts)

        mass_ref = np.median(self.data.m_halos[sample_halos])
        theta_ref = compute_theta_200(
            self.model, np.array([mass_ref]), self.data.redshift
        )[0]
        x_ref = np.logspace(
            np.log10(x_min),  # pyright: ignore[reportAny]
            np.log10(x_max),  # pyright: ignore[reportAny]
            400,
        )
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
            counts,
            num_samples,
            x_ref,
            y_battaglia,
            mass_ref,  # pyright: ignore[reportArgumentType]
            logm_center,
        )

    def build(self, y_map: np.ndarray):
        return [self._build_single(logm, y_map) for logm in self.cfg.log_m_centers]
