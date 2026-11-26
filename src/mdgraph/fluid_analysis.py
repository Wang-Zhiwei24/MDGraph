"""Property estimators for fixed-volume production of registered molecular fluids."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import dump_frames, parse_thermo, write_csv
from .artifacts import read_json
from .config import TaskConfig
from .molecular import MolecularModel
from .sampling import equilibration_check, windowed_msd

ANALYSIS_CHOICES = {
    "surface_tension": {
        "five_blocks": "Use five contiguous independent-time blocks and check an eight-block sensitivity",
        "eight_blocks": "Use eight contiguous time blocks and check a five-block sensitivity",
    },
    "self_diffusion": {
        "late_msd": "Fit molecular MSD between 35% and 80% of the block lag limit",
        "broad_msd": "Fit molecular MSD between 25% and 70% of the block lag limit",
    },
    "density_profile": {
        "five_blocks": "Wall-referenced molecular COM density in 0.25 Å bins, five time blocks"
    },
    "radial_density_profile": {
        "five_blocks": "Water COM mass divided by cylindrical shell volume, with five time blocks"
    },
}


def molecular_centers(frame, masses: np.ndarray) -> np.ndarray:
    _, _, types, _, positions, _, molecules = frame
    if np.any(molecules <= 0) or np.any(types < 1) or np.any(types > len(masses)):
        raise ValueError("Invalid molecule ID or atom type in molecular trajectory")
    weights = masses[types - 1]
    _, groups = np.unique(molecules, return_inverse=True)
    total = np.bincount(groups, weights)
    if not np.allclose(total, masses.sum()):
        raise ValueError("Trajectory molecule masses differ from the frozen force field")
    return np.column_stack([np.bincount(groups, weights * axis) / total for axis in positions.T])


def fit_diffusion(curve: np.ndarray, time_ps: np.ndarray, choice: str) -> dict:
    low, high = (0.35, 0.8) if choice == "late_msd" else (0.25, 0.7)
    mask = (time_ps >= time_ps[-1] * low) & (time_ps <= time_ps[-1] * high)
    x, y = time_ps[mask], curve[mask]
    slope, intercept = map(float, np.polyfit(x, y, 1))
    log_slope = float(np.polyfit(np.log(x), np.log(y), 1)[0]) if np.all(y > 0) else 0.0
    residual = np.sum((y - (slope * x + intercept)) ** 2)
    r2 = 1 - float(residual / max(np.sum((y - y.mean()) ** 2), 1e-20))
    return {
        "value_m2_s": slope / 6 * 1e-8,
        "r2": r2,
        "log_slope": log_slope,
        "window_ps": [float(x[0]), float(x[-1])],
    }


def diffusion(folder: Path, model: MolecularModel, dt_fs: float, choice: str) -> tuple[dict, dict]:
    masses = np.array([site[1] for site in model.sites])
    series = []
    steps = []
    first = None
    for frame in dump_frames(folder / "trajectory.lammpstrj", molecules=True):
        identities = np.column_stack((frame[1], frame[2], frame[6]))
        if first is None:
            first = identities, frame[5]
        if not np.array_equal(identities, first[0]) or not np.allclose(
            frame[5], first[1], rtol=1e-7
        ):
            raise ValueError(
                "Diffusion requires fixed volume and consistent molecule/atom identity"
            )
        series.append(molecular_centers(frame, masses))
        steps.append(frame[0])
    if len(steps) < 100 or np.any(np.diff(steps) != steps[1] - steps[0]) or steps[1] <= steps[0]:
        raise ValueError("Diffusion needs at least 100 evenly spaced production frames")
    coordinates = np.asarray(series)
    coordinates -= coordinates.mean(axis=1, keepdims=True)
    blocks = np.array_split(coordinates, 5)
    lag_count = min(map(len, blocks)) // 2
    lags = np.arange(lag_count) * (steps[1] - steps[0]) * dt_fs / 1000
    curves = np.array([windowed_msd(block)[:lag_count] for block in blocks])
    total = windowed_msd(coordinates)[:lag_count]
    estimate = fit_diffusion(total, lags, choice)
    other = fit_diffusion(total, lags, "broad_msd" if choice == "late_msd" else "late_msd")
    fits = [fit_diffusion(curve, lags, choice) for curve in curves]
    values = np.array([fit["value_m2_s"] for fit in fits])
    value = estimate["value_m2_s"]
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    relative_sem = sem / max(abs(value), 1e-30)
    window_deviation = abs(value - other["value_m2_s"]) / max(abs(value), 1e-30)
    write_csv(
        folder / "msd.csv",
        [{"lag_ps": float(t), "msd_angstrom2": float(y)} for t, y in zip(lags, total)],
    )
    write_csv(
        folder / "diffusion_blocks.csv",
        [{"block": i, **fit} for i, fit in enumerate(fits, start=1)],
    )
    metrics = {
        "self_diffusion_m2_s": value,
        "self_diffusion_block_sem_m2_s": sem,
        "msd_effective_max_lag_ps": float(lags[-1]),
        "molecules": coordinates.shape[1],
    }
    checks = {
        "passed": value > 0
        and all(v > 0 for v in values)
        and estimate["r2"] >= 0.98
        and 0.85 <= estimate["log_slope"] <= 1.15
        and relative_sem <= 0.20
        and window_deviation <= 0.20,
        "estimator": choice,
        "fit": estimate,
        "alternate_fit": other,
        "relative_block_sem": relative_sem,
        "fit_window_sensitivity": window_deviation,
        "blocks": fits,
    }
    return metrics, checks


def slab_density(folder: Path, model: MolecularModel, lengths: np.ndarray) -> dict:
    masses = np.array([row[1] for row in model.sites])
    edges = np.linspace(-lengths[2] / 2, lengths[2] / 2, 65)
    histogram = np.zeros(len(edges) - 1)
    count = 0
    for frame in dump_frames(folder / "trajectory.lammpstrj", molecules=True):
        z = molecular_centers(frame, masses)[:, 2]
        center = np.angle(np.exp(2j * np.pi * z / lengths[2]).mean()) * lengths[2] / (2 * np.pi)
        centered = (z - center + lengths[2] / 2) % lengths[2] - lengths[2] / 2
        histogram += np.histogram(centered, bins=edges)[0]
        count += 1
    if count == 0:
        raise ValueError("Surface tension requires a sampled slab density profile")
    density = (
        histogram
        * masses.sum()
        * 1.66053906660
        / (count * lengths[0] * lengths[1] * np.diff(edges))
    )
    z = (edges[1:] + edges[:-1]) / 2
    core = float(density[np.abs(z) < lengths[2] * 0.1].mean())
    vapor = float(density[np.abs(z) > lengths[2] * 0.4].mean())
    write_csv(
        folder / "density_profile.csv",
        [{"z_angstrom": float(x), "density_g_cm3": float(y)} for x, y in zip(z, density)],
    )
    return {
        "passed": core > 0 and vapor < core * 0.25,
        "core_density_g_cm3": core,
        "vapor_density_g_cm3": vapor,
        "frames": count,
    }


def confined_density(cfg: TaskConfig, folder: Path, model: MolecularModel) -> tuple[dict, dict]:
    """Normalize confined-fluid mass counts by planar bins or cylindrical shells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cfg.protocol
    structure = read_json(folder / "structure_build.json")["source"]
    radial = cfg.potential.target_property == "radial_density_profile"
    if radial:
        axis = np.array(structure["cnt_axis_xy_angstrom"])
        width = structure["cnt_radius_angstrom"]
        length = structure["cnt_length_angstrom"]
        lateral = structure["transverse_box_angstrom"]
    else:
        planes = np.array(structure["wall_z_angstrom"])
        width = float(np.diff(planes)[0])
    masses = np.array([row[1] for row in model.sites])
    water_mass = masses.sum() * 1.66053906660
    edges = np.linspace(0, width, round(width / 0.25) + 1)
    coordinate = (edges[:-1] + edges[1:]) / 2
    bin_volume = (
        np.pi * length * np.diff(edges**2)
        if radial
        else structure["area_angstrom2"] * np.diff(edges)
    )
    first_step = -(-p.equilibration_steps // p.dump_every) * p.dump_every
    expected_steps = range(first_step, p.equilibration_steps + p.steps + 1, p.dump_every)
    histograms = []
    first_identity = None
    for expected, frame in zip(
        expected_steps, dump_frames(folder / "trajectory.lammpstrj", molecules=True), strict=True
    ):
        step, ids, types, wrapped, _, cell, molecule_ids = frame
        identity = np.column_stack((ids, types, molecule_ids))
        if first_identity is None:
            first_identity = identity
        same_cell = (
            np.allclose(cell, np.diag([lateral, lateral, length]), rtol=1e-5)
            if radial
            else np.isclose(cell[0, 0] * cell[1, 1], structure["area_angstrom2"], rtol=1e-5)
        )
        if step != expected or not np.array_equal(identity, first_identity) or not same_cell:
            raise ValueError(
                "Density profile requires the complete fixed-geometry production trajectory"
            )
        water = molecule_ids > 0
        wall = wrapped[~water]
        wall_ok = (
            np.allclose(np.linalg.norm(wall[:, :2] - axis, axis=1), width, rtol=0, atol=1e-4)
            if radial
            else np.all(np.min(np.abs(wall[:, 2, None] - planes), axis=1) < 1e-4)
        )
        if len(wall) != structure["wall_atoms"] or not wall_ok:
            raise ValueError("Carbon wall atoms departed from the configured confinement geometry")
        subset = (
            step,
            ids[water],
            types[water],
            wrapped[water],
            frame[4][water],
            cell,
            molecule_ids[water],
        )
        centers = molecular_centers(subset, masses)
        distances = (
            np.linalg.norm(centers[:, :2] - axis, axis=1) if radial else centers[:, 2] - planes[0]
        )
        hist = np.histogram(distances, bins=edges)[0]
        if hist.sum() != len(centers) or len(centers) != structure["molecules"]:
            raise ValueError(
                "A water molecule left the carbon confinement or is missing from the trajectory"
            )
        histograms.append(hist)
    if len(histograms) < 10:
        raise ValueError("Density profile needs at least ten production frames")
    counts = np.asarray(histograms)
    density = counts.mean(axis=0) * water_mass / bin_volume
    blocks = np.array(
        [block.mean(axis=0) * water_mass / bin_volume for block in np.array_split(counts, 5)]
    )
    sem = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    half_a, half_b = (part.mean(axis=0) for part in np.array_split(counts, 2))
    half_difference = float(np.abs(half_a - half_b).sum() / counts.mean(axis=0).sum())
    key = "r_from_axis_angstrom" if radial else "z_from_lower_plate_angstrom"
    output = folder / ("radial_density_profile" if radial else "density_profile")
    write_csv(
        output.with_suffix(".csv"),
        [
            {
                key: float(x),
                "density_g_cm3": float(y),
                "block_sem_g_cm3": float(s),
                "bin_inner_angstrom": float(lo),
                "bin_outer_angstrom": float(hi),
            }
            for x, y, s, lo, hi in zip(coordinate, density, sem, edges[:-1], edges[1:])
        ],
    )
    write_csv(
        output.with_name(output.name + "_blocks.csv"),
        [
            {key: float(x), **{f"block_{i}_g_cm3": float(y) for i, y in enumerate(block, 1)}}
            for x, block in zip(coordinate, blocks.T)
        ],
    )
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
    ax.fill_between(
        coordinate,
        np.maximum(0, density - sem),
        density + sem,
        color="#19658b",
        alpha=0.22,
        linewidth=0,
        label="Block SEM",
    )
    ax.plot(coordinate, density, color="#124b67", lw=1.8, label="Water COM density")
    if radial:
        ax.axvline(width, color="#62666c", linestyle="--", lw=0.9, label="Carbon wall")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    else:
        for plane in (0, width):
            ax.axvline(plane, color="#62666c", linestyle="--", lw=0.9, clip_on=False)
    ax.set(
        xlabel="Radial distance from CNT axis (Å)"
        if radial
        else "Distance from lower graphene plane (Å)",
        ylabel=r"Water mass density (g cm$^{-3}$)",
        xlim=(0, width),
        ylim=(0, None),
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    mean = float(np.average(density, weights=bin_volume))
    tolerance = cfg.validation.sampling_block_relative_tolerance
    checks = {
        "passed": half_difference <= tolerance,
        "relative_half_difference": half_difference,
        "relative_tolerance": tolerance,
        "blocks": len(blocks),
        "bin_width_angstrom": float(edges[1] - edges[0]),
    }
    if radial:
        peaks = [int(np.argmax(density))]
        metrics = {
            "cnt_radius_angstrom": width,
            "cnt_length_angstrom": length,
            "cnt_geometric_average_water_density_g_cm3": mean,
        }
    else:
        peaks = [
            int(np.argmax(np.where(coordinate < width / 2, density, -1))),
            int(np.argmax(np.where(coordinate >= width / 2, density, -1))),
        ]
        metrics = {"slit_width_angstrom": width, "slit_average_water_density_g_cm3": mean}
        symmetry = float(np.abs(density - density[::-1]).sum() / density.sum())
        checks.update(
            passed=checks["passed"] and symmetry <= tolerance, relative_wall_asymmetry=symmetry
        )
    metrics.update(
        density_profile_frames=len(counts),
        water_molecules=structure["molecules"],
        density_peak_positions_angstrom=coordinate[peaks].tolist(),
        density_peak_heights_g_cm3=density[peaks].tolist(),
    )
    return metrics, checks


def surface_tension(
    folder: Path, model: MolecularModel, rows: list[dict], choice: str
) -> tuple[dict, dict]:
    lengths = np.array([[row["Lx"], row["Ly"], row["Lz"]] for row in rows])
    if np.any(lengths <= 0) or not np.allclose(lengths, lengths[0], atol=1e-6):
        raise ValueError("Pressure-tensor surface tension requires a fixed orthogonal slab cell")
    # Two liquid-vapor interfaces normal to z. atm * Å converts to 0.0101325 mN/m.
    gamma = np.array(
        [
            0.5 * lz * (row["Pzz"] - 0.5 * (row["Pxx"] + row["Pyy"])) * 0.0101325
            for lz, row in zip(lengths[:, 2], rows)
        ]
    )
    count = 5 if choice == "five_blocks" else 8
    blocks = np.array([block.mean() for block in np.array_split(gamma, count)])
    alternate = np.array([block.mean() for block in np.array_split(gamma, 13 - count)])
    mean = float(gamma.mean())
    sem = float(blocks.std(ddof=1) / np.sqrt(count))
    alternate_sem = float(alternate.std(ddof=1) / np.sqrt(len(alternate)))
    conservative_sem = max(sem, alternate_sem)
    half_difference = float(abs(gamma[: len(gamma) // 2].mean() - gamma[len(gamma) // 2 :].mean()))
    profile = slab_density(folder, model, lengths[0])
    write_csv(
        folder / "surface_tension_blocks.csv",
        [
            {"block": i, "surface_tension_mn_m": float(value)}
            for i, value in enumerate(blocks, start=1)
        ],
    )
    metrics = {"surface_tension_mn_m": mean, "surface_tension_block_sem_mn_m": conservative_sem}
    checks = {
        "passed": mean > 0
        and len(rows) >= 160
        and profile["passed"]
        and conservative_sem <= 0.15 * mean
        and half_difference <= max(0.20 * mean, 2 * conservative_sem),
        "estimator": choice,
        "block_means_mn_m": blocks.tolist(),
        "alternate_sem_mn_m": alternate_sem,
        "half_difference_mn_m": half_difference,
        "profile": profile,
    }
    return metrics, checks


def analyze_fluid(cfg: TaskConfig, folder: Path, atoms: int, choice: str) -> dict:
    """Check the fluid trajectory and evaluate the selected physical estimator."""
    p = cfg.protocol
    text = (folder / "log.lammps").read_text(encoding="utf-8", errors="replace")
    if (
        "ERROR:" in text
        or "MDGRAPH_COMPLETED" not in text
        or not (folder / "final.restart").is_file()
    ):
        raise ValueError("LAMMPS did not produce a complete molecular production run")
    rows = parse_thermo(text)
    confined = cfg.potential.target_property in {"density_profile", "radial_density_profile"}
    start = p.equilibration_steps if confined else 0
    if confined:
        equil_rows = [row for row in rows if row["Step"] <= start]
        rows = [row for row in rows if row["Step"] >= start]
    if (
        len(rows) < 20
        or not start <= rows[0]["Step"] < start + p.thermo_every
        or rows[-1]["Step"] != start + p.steps
    ):
        raise ValueError("Production thermo does not cover the planned sampling interval")
    if (
        any(row["Atoms"] != atoms for row in rows)
        or not np.isfinite([list(row.values()) for row in rows]).all()
    ):
        raise ValueError("Molecular production has nonfinite thermo or a changed atom count")
    model = MolecularModel.model_validate(read_json(folder / "molecular_model.json"))
    tail = rows[len(rows) // 2 :]
    mean_pressure = float(np.mean([r["Press"] for r in tail])) * 1.01325
    mean_temp = float(np.mean([r["Temp"] for r in tail]))
    metrics = {
        "duration_ps": p.steps * p.timestep_fs / 1000,
        "atoms": atoms,
        "tail_mean_temperature_k": mean_temp,
        "tail_mean_pressure_bar": mean_pressure,
        "tail_mean_density_g_cm3": float(np.mean([r["Density"] for r in tail])),
    }
    if cfg.potential.target_property == "surface_tension":
        values, checks = surface_tension(folder, model, rows, choice)
        limitations = [
            "A single periodic slab size and model; the block standard error does not quantify model or finite-size bias."
        ]
    elif confined:
        values, checks = confined_density(cfg, folder, model)
        metrics.pop("tail_mean_pressure_bar")
        metrics.pop("tail_mean_density_g_cm3")
        limitations = [
            "Fixed water loading and rigid carbon. Block uncertainty excludes force-field and finite-size effects."
        ]
        if cfg.potential.target_property == "radial_density_profile":
            limitations.append(
                "Axially periodic tube with transverse vacuum padding and 3D PPPM. There is no external water reservoir."
            )
    else:
        values, checks = diffusion(folder, model, p.timestep_fs, choice)
        limitations = [
            "A single bulk box and force field; the block standard error excludes finite-size and force-field biases."
        ]
        tolerance = (
            0.25 * p.pressure_bar
            if cfg.structure.molecular.phase == "gas"
            else max(30, 0.10 * p.pressure_bar)
        )
        pressure_ok = abs(mean_pressure - p.pressure_bar) <= tolerance
        checks.update(target_pressure_bar=p.pressure_bar, pressure_consistent=pressure_ok)
        checks["passed"] = checks["passed"] and pressure_ok
    if not confined:
        _, _, equil_log = text.partition("\nMDGRAPH_EQUILIBRATION_BEGIN\n")
        equil_log, _, _ = equil_log.partition("\nMDGRAPH_EQUILIBRATION_END\n")
        equil_rows = parse_thermo(equil_log)
    eq = equilibration_check(
        equil_rows,
        p.temperature_k,
        p.equilibration_ensemble or p.ensemble,
        temperature_tolerance=cfg.validation.equilibration_temperature_tolerance,
        volume_tolerance=cfg.validation.equilibration_volume_tolerance,
    )
    thermal_ok = abs(mean_temp - p.temperature_k) <= 0.05 * p.temperature_k
    sampling = {
        "mode": p.sampling_mode,
        "passed": p.sampling_mode == "production"
        and checks["passed"]
        and eq["passed"]
        and thermal_ok,
        "equilibration": eq,
        "production": checks,
        "temperature_consistent": thermal_ok,
    }
    issues = (
        []
        if sampling["passed"]
        else ["Sampling checks did not establish a production-quality estimate"]
    )
    return {
        "passed": True,
        "metrics": {**metrics, **values},
        "sampling": sampling,
        "issues": issues,
        "scope": cfg.potential.target_property,
        "limitations": limitations,
    }
