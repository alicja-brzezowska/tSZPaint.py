from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import warnings

import asdf
import numpy as np

from tszpaint.config import ABACUS_DATA_PATH, HALO_CATALOGS_PATH, HEALCOUNTS_TOTAL_PATH, OUTPUT_PATH


@dataclass
class HealpixFileInfo:
    file_path: Path
    chi_min: float
    chi_max: float
    z_min: float
    z_max: float


@dataclass
class HaloCatalogInfo:
    file_path: Path
    redshift: float
    chi_center: float
    chi_min: float
    chi_max: float


@dataclass
class MatchInfo:
    healpix: HealpixFileInfo

    catalogs: list[HaloCatalogInfo]


def parse_redshift_from_folder(folder_name: str) -> float:
    return float(folder_name[1:])


def get_healpix_chi(filepath: Path) -> HealpixFileInfo:
    with asdf.open(filepath) as af:
        headers = af["headers"]
        chis = np.asarray(
            [h["CoordinateDistanceHMpc"] for h in headers], dtype=np.float32
        )
        zs = np.asarray([h["Redshift"] for h in headers], dtype=np.float32)

    return HealpixFileInfo(
        file_path=filepath,
        chi_min=float(chis.min()),
        chi_max=float(chis.max()),
        z_min=float(zs.min()),
        z_max=float(zs.max()),
    )


def discover_healpix_shells(healcounts_dir: Path) -> list[HealpixFileInfo]:
    return [get_healpix_chi(fp) for fp in sorted(healcounts_dir.glob("*.asdf"))]


def get_halo_chi(filepath: Path) -> float:
    with asdf.open(filepath) as af:
        return float(af["header"]["CoordinateDistanceHMpc"])


def discover_halo_catalogs(
    halo_root: Path,
    halo_filename: str = "lightcone_halo_info_000.asdf",
) -> list[HaloCatalogInfo]:
    records: list[tuple[float, Path, float]] = []

    for z_dir in sorted(
        d for d in halo_root.iterdir() if d.is_dir() and d.name.startswith("z")
    ):
        halo_file = z_dir / halo_filename
        if halo_file.exists():
            z = parse_redshift_from_folder(z_dir.name)
            chi_center = get_halo_chi(halo_file)
            records.append((z, halo_file, chi_center))

    records.sort(key=lambda rec: rec[0])
    centers = np.asarray([rec[2] for rec in records], dtype=np.float64)

    if len(records) == 1:
        ranges = [(-np.inf, np.inf)]
    else:
        boundaries = 0.5 * (centers[:-1] + centers[1:])
        lower0 = centers[0] - 0.5 * (centers[1] - centers[0])
        upper_last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
        lowers = np.concatenate(([lower0], boundaries))
        uppers = np.concatenate((boundaries, [upper_last]))
        ranges = list(zip(lowers, uppers, strict=True))

    return [
        HaloCatalogInfo(
            file_path=fp,
            redshift=z,
            chi_center=chi_center,
            chi_min=float(chi_min),
            chi_max=float(chi_max),
        )
        for (z, fp, chi_center), (chi_min, chi_max) in zip(records, ranges, strict=True)
    ]


def _overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return max(a_min, b_min) <= min(a_max, b_max)


def match_healpix_to_halos(
    shells: list[HealpixFileInfo],
    catalogs: list[HaloCatalogInfo],
) -> list[MatchInfo]:
    return [
        MatchInfo(
            healpix=shell,
            catalogs=[
                cat
                for cat in catalogs
                if _overlap(shell.chi_min, shell.chi_max, cat.chi_min, cat.chi_max)
            ],
        )
        for shell in shells
    ]


def _mask_lightcone_by_chi(
    lightcone: dict, chi_min: float, chi_max: float
) -> tuple[dict, int]:
    chi = np.asarray(lightcone["Interpolated_ComovingDist"])
    mask = (chi >= chi_min) & (chi <= chi_max)
    n_rows = len(chi)

    masked: dict = {}
    for key, value in lightcone.items():
        arr = np.asarray(value) if hasattr(value, "shape") else None
        if arr is not None and arr.ndim > 0 and arr.shape[0] == n_rows:
            masked[key] = arr[mask]
        else:
            masked[key] = value

    return masked, int(mask.sum())


def _merge_masked_lightcones(masked_lightcones: list[dict]) -> dict:
    ref = masked_lightcones[0]
    row0 = len(np.asarray(ref["Interpolated_ComovingDist"]))
    merged: dict = {}

    for key, ref_value in ref.items():
        ref_arr = np.asarray(ref_value) if hasattr(ref_value, "shape") else None
        is_row_field = (
            ref_arr is not None and ref_arr.ndim > 0 and ref_arr.shape[0] == row0
        )

        if is_row_field:
            merged[key] = np.concatenate(
                [np.asarray(lightcone[key]) for lightcone in masked_lightcones], axis=0
            )
        else:
            merged[key] = ref_value

    return merged


def write_masked_halo_catalog(
    input_halo_file: Path,
    output_halo_file: Path,
    chi_min: float,
    chi_max: float,
) -> int:
    output_halo_file.parent.mkdir(parents=True, exist_ok=True)
    if output_halo_file.exists() or output_halo_file.is_symlink():
        output_halo_file.unlink()

    with asdf.open(input_halo_file) as af:
        header = af["header"]
        halo_timeslice = af["halo_timeslice"]
        masked_lightcone, n_kept = _mask_lightcone_by_chi(
            af["halo_lightcone"],
            chi_min=chi_min,
            chi_max=chi_max,
        )
        asdf.AsdfFile(
            {
                "header": header,
                "halo_timeslice": halo_timeslice,
                "halo_lightcone": masked_lightcone,
            }
        ).write_to(output_halo_file)

    return n_kept


def write_merged_masked_halo_catalog(
    input_halo_files: list[Path],
    output_halo_file: Path,
    chi_min: float,
    chi_max: float,
) -> int:
    output_halo_file.parent.mkdir(parents=True, exist_ok=True)
    if output_halo_file.exists() or output_halo_file.is_symlink():
        output_halo_file.unlink()

    masked_lightcones: list[dict] = []
    with asdf.open(input_halo_files[0]) as af0:
        header = af0["header"]
        halo_timeslice = af0["halo_timeslice"]

    for halo_file in input_halo_files:
        with asdf.open(halo_file) as af:
            masked, n_kept = _mask_lightcone_by_chi(
                af["halo_lightcone"], chi_min=chi_min, chi_max=chi_max
            )
            masked_lightcones.append(masked)

    merged_lightcone = _merge_masked_lightcones(masked_lightcones)
    merged_count = int(len(np.asarray(merged_lightcone["Interpolated_ComovingDist"])))

    asdf.AsdfFile(
        {
            "header": header,
            "halo_timeslice": halo_timeslice,
            "halo_lightcone": merged_lightcone,
        }
    ).write_to(output_halo_file)

    return merged_count


def _redshift_label(z_min: float, z_max: float) -> str:
    return f"z{z_min:.3f}_to_z{z_max:.3f}"


def _place_healpix_file(src: Path, dst: Path, mode: str = "symlink") -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def write_matched_outputs(
    matches: list[MatchInfo],
    output_root: Path,
    healpix_mode: str = "symlink",
    max_shells: int | None = None,
) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for match in matches:
        if max_shells is not None and len(manifest) >= max_shells:
            break

        shell = match.healpix
        if not match.catalogs:
            warnings.warn(
                (
                    "Skipping shell with no overlapping halo catalogs: "
                    f"{shell.file_path.name} "
                    f"(z=[{shell.z_min:.3f}, {shell.z_max:.3f}], "
                    f"chi=[{shell.chi_min:.3f}, {shell.chi_max:.3f}])"
                ),
                stacklevel=2,
            )
            continue

        shell_dir = output_root / _redshift_label(shell.z_min, shell.z_max)
        shell_dir.mkdir(parents=True, exist_ok=True)

        particle_counts_output = shell_dir / shell.file_path.name
        _place_healpix_file(shell.file_path, particle_counts_output, mode=healpix_mode)

        halo_catalog_outputs: list[Path] = []
        halo_rows_after_mask_total = 0
        for catalog in match.catalogs:
            halo_catalog_output = (
                shell_dir / f"{catalog.file_path.parent.name}_{catalog.file_path.name}"
            )
            n_kept = write_masked_halo_catalog(
                input_halo_file=catalog.file_path,
                output_halo_file=halo_catalog_output,
                chi_min=shell.chi_min,
                chi_max=shell.chi_max,
            )
            halo_rows_after_mask_total += n_kept
            halo_catalog_outputs.append(halo_catalog_output)

        manifest.append(
            {
                "z_min": shell.z_min,
                "z_max": shell.z_max,
                "chi_min": shell.chi_min,
                "chi_max": shell.chi_max,
                "particle_counts_file": str(particle_counts_output),
                "halo_catalog_files": [str(path) for path in halo_catalog_outputs],
                "halo_catalog_count": len(halo_catalog_outputs),
                "halo_rows_after_mask": halo_rows_after_mask_total,
            }
        )

    return manifest


def build_halo_catalog_index(
    halo_root: Path,
    output_json: Path,
    halo_filename: str = "lightcone_halo_info_000.asdf",
) -> list[HaloCatalogInfo]:
    """Discover all halo catalogs under halo_root, compute their chi ranges, and save to JSON.

    This only needs to be run once. The resulting JSON is then passed to
    load_abacus_for_painting at paint time so it can pick the right 1–2 files
    per healcounts shell without re-scanning the filesystem every run.
    """
    catalogs = discover_halo_catalogs(halo_root, halo_filename=halo_filename)
    records = [
        {
            "file_path": str(c.file_path),
            "redshift": c.redshift,
            "chi_center": c.chi_center,
            "chi_min": c.chi_min,
            "chi_max": c.chi_max,
        }
        for c in catalogs
    ]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return catalogs


def load_halo_catalog_index(json_path: Path) -> list[HaloCatalogInfo]:
    """Load a precomputed halo catalog index produced by build_halo_catalog_index."""
    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)
    return [
        HaloCatalogInfo(
            file_path=Path(r["file_path"]),
            redshift=r["redshift"],
            chi_center=r["chi_center"],
            chi_min=r["chi_min"],
            chi_max=r["chi_max"],
        )
        for r in records
    ]


def build_redshift_matched_catalogs(
    healcounts_dir: Path,
    halo_root: Path,
    output_root: Path,
    halo_filename: str = "lightcone_halo_info_000.asdf",
    healpix_mode: str = "symlink",
    max_shells: int | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[dict]:
    shells = discover_healpix_shells(healcounts_dir)
    if z_min is not None:
        shells = [s for s in shells if s.z_max >= z_min]
    if z_max is not None:
        shells = [s for s in shells if s.z_min <= z_max]
    catalogs = discover_halo_catalogs(halo_root, halo_filename=halo_filename)
    matches = match_healpix_to_halos(shells, catalogs)

    manifest = write_matched_outputs(
        matches=matches,
        output_root=output_root,
        healpix_mode=healpix_mode,
        max_shells=max_shells,
    )
    with (output_root / "match_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def summarize_matches(manifest: list[dict]) -> None:
    total_shells = len(manifest)
    total_halos = sum(m["halo_catalog_count"] for m in manifest)
    total_rows = sum(m["halo_rows_after_mask"] for m in manifest)
    print(
        f"Summary: {total_shells} healpix shells, {total_halos} halo catalogs, "
        f"{total_rows} masked halo rows"
    )


def main():
    output_json = OUTPUT_PATH / "halo_catalog_index.json"
    catalogs = build_halo_catalog_index(HALO_CATALOGS_PATH, output_json)
    print(f"Built index with {len(catalogs)} halo catalogs -> {output_json}")


if __name__ == "__main__":
    main()
