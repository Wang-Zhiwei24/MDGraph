from __future__ import annotations

import csv
import math
from collections import deque
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from .artifacts import atomic_json
from .config import TaskConfig
from .elastic import analyze_elastic
from .prepare import relaxation_cell_ok
from .sampling import analyze_windowed_msd, block_relative_error, equilibration_check
from .trajectory_tools import available_trajectory_backends, backend_info, pairs_within, save_plots


def parse_thermo(text: str, header=("Step", "Time", "Temp")) -> list[dict]:
    """Read the final thermo table matching the expected header prefix."""
    headers = None
    rows = []
    for line in text.splitlines():
        tokens = line.split()
        if tuple(tokens[: len(header)]) == header:
            headers = tokens
            rows = []  # final production run is the authoritative one
            continue
        if headers is None:
            continue
        if line.lstrip().startswith("WARNING:"):
            continue
        if len(tokens) != len(headers):
            if line.startswith(("Loop time", "ERROR")):
                headers = None
            continue
        try:
            values = [float(x) for x in tokens]
        except ValueError:
            continue
        rows.append(dict(zip(headers, values)))
    return rows


def dump_frames(path: Path, *, molecules: bool = False, metadata: bool = False):
    """Streaming reader for our atomic custom dump, including restricted triclinic cells."""
    with path.open(encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError("Malformed dump timestep header")
            step = int(f.readline())
            if f.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Malformed atom-count header")
            n = int(f.readline())
            if n < 2:
                raise ValueError("Invalid atom count")
            box_header = f.readline().split()
            if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("Malformed box header")
            bounds = [[float(v) for v in f.readline().split()] for _ in range(3)]
            if box_header[3:6] == ["xy", "xz", "yz"]:
                (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo, zhi, yz) = bounds
                xlo, xhi = xlo_b - min(0, xy, xz, xy + xz), xhi_b - max(0, xy, xz, xy + xz)
                ylo, yhi = ylo_b - min(0, yz), yhi_b - max(0, yz)
                cell = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
            else:
                cell = np.diag([b[1] - b[0] for b in bounds])
            fields = f.readline().split()[2:]
            required = ["id", "type", "x", "y", "z", "ix", "iy", "iz"]
            if not all(key in fields for key in required):
                raise ValueError("Dump must have id type x y z ix iy iz")
            data = np.array([[float(v) for v in f.readline().split()] for _ in range(n)])
            data = data[np.argsort(data[:, fields.index("id")])]
            ids = data[:, fields.index("id")].astype(int)
            types = data[:, fields.index("type")].astype(int)
            pos = data[:, [fields.index(k) for k in ("x", "y", "z")]]
            images = data[:, [fields.index(k) for k in ("ix", "iy", "iz")]]
            if len(np.unique(ids)) != n or not np.isfinite(data).all() or np.linalg.det(cell) <= 0:
                raise ValueError("Invalid/non-finite dump geometry")
            row = step, ids, types, pos, pos + images @ cell, cell
            if molecules:
                if "mol" not in fields:
                    raise ValueError("Molecular trajectory is missing molecule IDs")
                row += (data[:, fields.index("mol")].astype(int),)
            if metadata:
                flags = box_header[-3:]
                if len(flags) != 3 or any(
                    len(flag) != 2 or set(flag) - set("pfsm") for flag in flags
                ):
                    raise ValueError("Dump does not record the three boundary flags")
                origin = (
                    [xlo, ylo, zlo]
                    if box_header[3:6] == ["xy", "xz", "yz"]
                    else [b[0] for b in bounds]
                )
                row += (
                    np.array(origin),
                    np.array([flag == "pp" for flag in flags]),
                    data[:, fields.index("mol")].astype(int) if "mol" in fields else None,
                )
            yield row


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_relaxation(cfg: TaskConfig, folder: Path, expected_atoms: int) -> dict:
    """Validate the final zero-step snapshot and recover the conventional lattice parameter."""
    issues = []
    metrics = {}
    log = folder / "log.lammps"
    for name in ("final.data", "final.restart"):
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"Missing/empty output: {name}")
    if not log.is_file():
        issues.append("Missing log.lammps")
    else:
        text = log.read_text(encoding="utf-8", errors="replace")
        if any(line.startswith("ERROR") for line in text.splitlines()):
            issues.append("LAMMPS reported ERROR")
        if "MDGRAPH_COMPLETED" not in text.splitlines():
            issues.append("Missing final completion marker")
        columns = (
            "Step",
            "Atoms",
            "PotEng",
            "Press",
            "Pxx",
            "Pyy",
            "Pzz",
            "Pxy",
            "Pxz",
            "Pyz",
            "Lx",
            "Ly",
            "Lz",
            "Fmax",
        )
        rows = parse_thermo(text, columns)
        if len(rows) != 1 or not all(math.isfinite(v) for v in rows[0].values()):
            issues.append("Missing or non-finite final relaxation thermo snapshot")
        else:
            row = rows[0]
            if row["Atoms"] != expected_atoms:
                issues.append("Atom count changed")
            target = cfg.protocol.pressure_bar
            pressure_error = max(abs(row[key] - target) for key in ("Press", "Pxx", "Pyy", "Pzz"))
            pressure_error = max(pressure_error, *(abs(row[key]) for key in ("Pxy", "Pxz", "Pyz")))
            symmetry = cfg.protocol.relax_cell
            if symmetry == "hexagonal":
                constrained_error = max(
                    abs((row["Pxx"] + row["Pyy"]) / 2 - target), abs(row["Pzz"] - target)
                )
            elif len(cfg.structure.elements) > 1:
                # An ordered alloy can retain deviatoric stress under the imposed cubic constraint.
                constrained_error = abs(row["Press"] - target)
            else:
                constrained_error = pressure_error
            if constrained_error > cfg.validation.relax_pressure_tolerance_bar:
                issues.append("Residual stress exceeds relaxation tolerance")
            if not 0 <= row["Fmax"] <= cfg.validation.relax_force_tolerance_ev_per_angstrom:
                issues.append("Maximum force exceeds relaxation tolerance")
            lengths = [row[key] for key in ("Lx", "Ly", "Lz")]
            crystal = cfg.structure.crystal
            repeat = crystal.repeat if crystal else (1, 1, 1)
            angles = [row.get(k) for k in ("CellAlpha", "CellBeta", "CellGamma")]
            edges = [row.get(k) for k in ("Cella", "Cellb", "Cellc")]
            if symmetry == "cubic" and crystal and angles == [None] * 3 and edges == [None] * 3:
                # Version 2 checkpoints from before lattice geometry was logged, for cubic builders only.
                angles, edges = [90.0] * 3, lengths
            values = [edge / n for edge, n in zip(edges, repeat)] if None not in edges else []
            if not values or None in angles or not relaxation_cell_ok(values, angles, symmetry):
                issues.append("Final box is incompatible with the configured crystal symmetry")
            else:
                metrics["lattice_constant_angstrom"] = (
                    sum(values) / 3 if symmetry == "cubic" else sum(values[:2]) / 2
                )
            metrics.update(
                atoms=int(row["Atoms"]),
                temperature_k=0,
                box_lengths_angstrom=lengths,
                lattice_by_axis_angstrom=values,
                lattice_angles_deg=angles,
                pressure_bar=row["Press"],
                target_pressure_bar=target,
                max_stress_error_bar=pressure_error,
                constrained_stress_error_bar=constrained_error,
                max_force_ev_per_angstrom=row["Fmax"],
                potential_energy_ev_per_atom=row["PotEng"] / expected_atoms,
                potential_kind=cfg.potential.kind,
            )
            write_csv(folder / "relaxation.csv", [{**row, **metrics}])
    result = {
        "passed": not issues,
        "issues": issues,
        "metrics": metrics,
        "scope": "0 K static lattice relaxation with crystal symmetry constrained",
        "limitations": [
            "No finite-temperature sampling or independent reference-potential validation"
        ],
    }
    if cfg.protocol.elastic and not issues:
        try:
            elastic, elastic_issues = analyze_elastic(
                text, folder, expected_atoms, cfg.validation.relax_force_tolerance_ev_per_angstrom
            )
            metrics.update(elastic)
            issues.extend(elastic_issues)
        except (ValueError, OSError) as error:
            issues.append(str(error))
        result.update(
            passed=not issues,
            scope="0 K relaxed-ion cubic elastic tensor by central stress differences",
        )
    atomic_json(folder / "validation.json", result)
    return result


def analyze(
    cfg: TaskConfig, folder: Path, expected_atoms: int, *, trajectory_backend: str | None = None
) -> dict:
    if cfg.protocol.ensemble == "relax":
        return analyze_relaxation(cfg, folder, expected_atoms)
    issues = []
    log = folder / "log.lammps"
    if not log.is_file():
        return {"passed": False, "issues": ["Missing log.lammps"], "scope": "execution_only"}
    text = log.read_text(encoding="utf-8", errors="replace")
    if "ERROR" in text:
        issues.append("LAMMPS reported ERROR")
    if "MDGRAPH_COMPLETED" not in text:
        issues.append("Missing final completion marker")
    for name in ("final.data", "final.restart", "trajectory.lammpstrj"):
        if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
            issues.append(f"Missing/empty output: {name}")
    rows = parse_thermo(text)
    required = {"Step", "Time", "Temp", "PotEng", "KinEng", "TotEng", "Press", "Volume", "Atoms"}
    if not rows or not required.issubset(rows[0]):
        return {
            "passed": False,
            "issues": issues + ["Missing production thermo columns"],
            "scope": "execution_only",
        }
    if len(rows) < 5:
        issues.append("Insufficient production thermo samples")
    if any(not math.isfinite(v) for row in rows for v in row.values()):
        return {
            "passed": False,
            "issues": issues + ["Non-finite thermo values"],
            "scope": "execution_only",
        }
    if cfg.structure.water_repeat:
        # Normalize LAMMPS real units to the common workflow units.
        for row in rows:
            row["Time"] /= 1000
            row["Press"] *= 1.01325
            for energy in ("PotEng", "KinEng", "TotEng"):
                row[energy] *= 0.0433641153087705
    if cfg.protocol.equilibration_steps:
        # LAMMPS may carry elapsed time through reset_timestep even as Step starts at zero.
        production_start = rows[0]["Time"]
        for row in rows:
            row["Time"] -= production_start
    if rows[-1]["Step"] != cfg.protocol.steps or rows[0]["Step"] != 0:
        issues.append("Production step count differs from configured run")
    if any(b["Step"] <= a["Step"] for a, b in zip(rows, rows[1:])):
        issues.append("Non-monotonic production steps")
    if any(row["Atoms"] != expected_atoms for row in rows):
        issues.append("Atom count changed")
    if any(row["Volume"] <= 0 or row["Temp"] < 0 for row in rows):
        issues.append("Invalid temperature or volume")
    expected_time = cfg.protocol.steps * cfg.protocol.timestep_fs / 1000
    if not math.isclose(rows[-1]["Time"], expected_time, rel_tol=1e-5, abs_tol=1e-8):
        issues.append("Physical duration differs from configuration (ps)")
    tail = rows[len(rows) // 2 :]
    mean_temp = float(np.mean([r["Temp"] for r in tail]))
    temperature_ok = (
        abs(mean_temp - cfg.protocol.temperature_k) / cfg.protocol.temperature_k
        <= cfg.validation.temperature_relative_tolerance
    )
    if cfg.protocol.ensemble != "nve" and not temperature_ok:
        issues.append("Tail mean temperature outside configured tolerance")
    metrics = {
        "production_steps": rows[-1]["Step"],
        "duration_ps": rows[-1]["Time"],
        "atoms": expected_atoms,
        "tail_mean_temperature_k": mean_temp,
        "tail_mean_pressure_bar": float(np.mean([r["Press"] for r in tail])),
        "potential_kind": cfg.potential.kind,
    }
    if cfg.structure.water_repeat:
        mass_grams = expected_atoms / 3 * (15.9994 + 2 * 1.008) / 6.02214076e23
        metrics["tail_mean_density_g_cm3"] = float(
            np.mean([mass_grams / (r["Volume"] * 1e-24) for r in tail])
        )
    if cfg.protocol.ensemble == "nve" and len(rows) >= 2:
        t = np.array([r["Time"] for r in rows])
        e = np.array([r["TotEng"] / expected_atoms for r in rows])
        slope = abs(float(np.polyfit(t, e, 1)[0]))
        metrics["nve_energy_drift_ev_per_atom_ps"] = slope
        threshold = cfg.validation.max_nve_energy_drift_ev_per_atom_ps
        if threshold is not None and slope > threshold:
            issues.append("NVE total energy drift exceeds threshold")
    write_csv(folder / "thermo.csv", rows)
    trajectory = folder / "trajectory.lammpstrj"
    if trajectory.exists():
        try:
            metrics.update(
                analyze_trajectory(cfg, folder, expected_atoms, backend=trajectory_backend)
            )
        except (ValueError, IndexError, OSError, ImportError) as e:
            issues.append(f"Trajectory validation failed: {e}")
    result = {
        "passed": not issues,
        "issues": issues,
        "metrics": metrics,
        "scope": "MD execution and configured numerical checks; NOT potential accuracy certification",
        "limitations": [
            "No DFT reference accuracy check in this MD-only version",
            "Temperature tail mean does not establish statistical convergence",
            "MSD includes collective motion; NPT cell deformation can contribute",
            "MSD uses the first frame as the origin; no diffusion fit",
            "Partial RDFs describe the configured species pairs"
            if cfg.validation.species_resolved
            else "RDF is a total all-species RDF, not species-resolved",
        ],
    }
    if cfg.protocol.sampling_mode == "production":
        _, begin, tail_text = text.partition("\nMDGRAPH_EQUILIBRATION_BEGIN\n")
        eq_text, end, _ = tail_text.partition("\nMDGRAPH_EQUILIBRATION_END\n")
        eq_rows = parse_thermo(eq_text) if begin and end else []
        equilibrium = equilibration_check(
            eq_rows,
            cfg.protocol.temperature_k,
            cfg.protocol.equilibration_ensemble or cfg.protocol.ensemble,
            temperature_tolerance=cfg.validation.equilibration_temperature_tolerance,
            volume_tolerance=cfg.validation.equilibration_volume_tolerance,
        )
        consistency = metrics.get(
            "sampling_consistency",
            {"passed": False, "reason": "No production trajectory diagnostics"},
        )
        result["sampling"] = {
            "mode": "production",
            "passed": equilibrium["passed"] and consistency["passed"],
            "equilibration": equilibrium,
            "production_consistency": consistency,
        }
        if cfg.potential_scope()["target_property"] == "msd":
            result["limitations"] = [
                item for item in result["limitations"] if not item.startswith("MSD")
            ]
            result["limitations"].append(
                "Multiple-origin MSD uses one box and model; it does not estimate finite-size bias"
            )
    elif cfg.protocol.sampling_mode == "smoke":
        result["sampling"] = {
            "mode": "smoke",
            "passed": False,
            "reason": "Exploratory run; no convergence assessment",
        }
    atomic_json(folder / "validation.json", result)
    return result


def analyze_trajectory(
    cfg: TaskConfig, folder: Path, expected_atoms: int, *, backend: str | None = None
) -> dict:
    backend = backend or cfg.validation.trajectory_backend
    if backend == "auto":
        backend = available_trajectory_backends()[-1]
    if backend == "mdanalysis" and backend not in available_trajectory_backends():
        raise ValueError("MDAnalysis is unavailable; install .[trajectory] or select ASE")
    backend_metadata = backend_info(backend)
    recent = deque(maxlen=cfg.validation.max_analysis_frames)
    reference = None
    reference_ids = None
    reference_types = None
    msd_rows = []
    last_step = -1
    species = list(enumerate(cfg.structure.elements, start=1))
    split = cfg.validation.species_resolved
    water = cfg.structure.water_repeat is not None
    formal = cfg.protocol.sampling_mode == "production"
    formal_msd = formal and cfg.potential_scope()["target_property"] == "msd"
    rdf_frames = []
    for frame in dump_frames(folder / "trajectory.lammpstrj"):
        step, ids, types, pos, unwrapped, cell = frame
        if len(ids) != expected_atoms or step <= last_step:
            raise ValueError("Trajectory atom count changed or steps not increasing")
        if np.any(types < 1) or np.any(types > len(cfg.structure.elements)):
            raise ValueError("Trajectory type mapping invalid")
        if reference is None:
            reference, reference_ids = unwrapped.copy(), ids.copy()
            reference_types = types.copy()
        if not np.array_equal(ids, reference_ids):
            raise ValueError("Atom identities changed")
        if not np.array_equal(types, reference_types):
            raise ValueError("Atom types changed")
        squared_displacement = np.sum((unwrapped - reference) ** 2, axis=1)
        row = {
            "time_ps": step * cfg.protocol.timestep_fs / 1000,
            "msd_angstrom2": float(squared_displacement.mean()),
        }
        if split:
            for i, element in species:
                if np.any(types == i):
                    row[f"msd_{element}_angstrom2"] = float(squared_displacement[types == i].mean())
        msd_rows.append(row)
        recent.append(frame)
        last_step = step
    if len(msd_rows) < 2:
        raise ValueError("Need at least two complete dump frames")
    if water and (
        not np.array_equal(reference_ids, np.arange(1, expected_atoms + 1))
        or not np.array_equal(reference_types, np.tile([1, 2, 2], expected_atoms // 3))
    ):
        raise ValueError("Water trajectory does not match its molecule topology")
    expected_last = cfg.protocol.steps // cfg.protocol.dump_every * cfg.protocol.dump_every
    if last_step != expected_last:
        raise ValueError("Trajectory ends before configured dump schedule")
    bins = cfg.validation.rdf_bins
    # Sphere must fit within half the smallest perpendicular periodic cell height.
    heights = [1 / np.linalg.norm(np.linalg.inv(f[-1]), axis=0).max() for f in recent]
    rmax = min(cfg.validation.rdf_max_angstrom, min(heights) * 0.499)
    edges = np.linspace(0, rmax, bins + 1)
    shells = 4 * np.pi / 3 * np.diff(edges**3)
    histogram = np.zeros(bins)
    partials = {}
    if split:
        populations = {i: int(np.count_nonzero(reference_types == i)) for i, _ in species}
        for (i, a), (j, b) in combinations_with_replacement(species, 2):
            excluded = {"O": 1, "H": 2}[b] if water else int(i == j)
            normalization = populations[i] * (populations[j] - excluded)
            if normalization > 0:
                partials[(i, j, a, b)] = (normalization, np.zeros(bins))
    for _, ids, types, pos, _, cell in recent:
        # Keep the contact check in ASE (float64), independently of the RDF backend.
        minimum = cfg.validation.min_pair_distance_angstrom
        i, j, distances = pairs_within(
            "ase", pos, cell, max(rmax, minimum) if backend == "ase" else minimum
        )
        if np.any(distances < minimum):
            raise ValueError("Unphysical short interatomic separation")
        if backend != "ase":
            i, j, distances = pairs_within(backend, pos, cell, rmax)
        in_range = distances < rmax
        if water:
            molecule = (ids - 1) // 3
            in_range &= molecule[i] != molecule[j]
        counts, _ = np.histogram(distances[in_range], bins=edges)
        # The ordered-pair normalization excludes the reference atom or its whole water molecule.
        volume = np.linalg.det(cell)
        frame_total = (
            counts * volume / (expected_atoms * (expected_atoms - (3 if water else 1)) * shells)
        )
        histogram += frame_total
        frame_curves = {"g_r": frame_total}
        for (first, second, a, b), (norm, total) in partials.items():
            mask = (types[i] == first) & (types[j] == second) & in_range
            counts, _ = np.histogram(distances[mask], bins=edges)
            frame_partial = counts * volume / (norm * shells)
            total += frame_partial
            frame_curves[f"g_{a}_{b}"] = frame_partial
        if formal and not formal_msd:
            rdf_frames.append(frame_curves)
    histogram /= len(recent)
    consistency = None
    if formal_msd:
        positions = np.array([frame[4] for frame in recent])
        time = np.array([frame[0] * cfg.protocol.timestep_fs / 1000 for frame in recent])
        windowed_rows, consistency = analyze_windowed_msd(
            positions,
            time,
            molecular_water=water,
            tolerance=cfg.validation.sampling_block_relative_tolerance,
        )
        if water and cfg.protocol.ensemble != "nvt":
            consistency.update(
                passed=False, reason="Water MSD needs a fixed-volume production cell"
            )
        write_csv(folder / "msd.csv", windowed_rows)
    else:
        write_csv(folder / "msd.csv", msd_rows)
    if rdf_frames:
        tolerance = cfg.validation.sampling_block_relative_tolerance
        consistency = {
            "passed": False,
            "reason": "RDF needs at least 30 production frames for three blocks",
        }
        if len(rdf_frames) >= 30:
            errors = {}
            for key in rdf_frames[0]:
                curves = np.array([frame[key] for frame in rdf_frames])
                blocks = np.array([block.mean(axis=0) for block in np.array_split(curves, 3)])
                errors[key] = block_relative_error(blocks)
            consistency = {
                "passed": max(errors.values()) <= tolerance,
                "blocks": 3,
                "block_relative_error_by_curve": errors,
                "tolerance": tolerance,
            }
    radii = (edges[1:] + edges[:-1]) / 2
    rdf_rows = [{"r_angstrom": float(r), "g_r": float(g)} for r, g in zip(radii, histogram)]
    for (_, _, a, b), (_, total) in partials.items():
        for row, value in zip(rdf_rows, total / len(recent)):
            row[f"g_{a}_{b}"] = float(value)
    write_csv(folder / "rdf.csv", rdf_rows)
    if cfg.validation.trajectory_plots:
        curves = {f"{a}–{b}": total / len(recent) for (_, _, a, b), (_, total) in partials.items()}
        plot_rows = windowed_rows if formal_msd else msd_rows
        time_key = "lag_ps" if formal_msd else "time_ps"
        save_plots(
            folder,
            [r[time_key] for r in plot_rows],
            [r["msd_angstrom2"] for r in plot_rows],
            radii,
            histogram,
            partials=curves if water else None,
        )
    result = {
        "dump_frames": len(msd_rows),
        "rdf_frames": len(recent),
        "rdf_rmax_angstrom": rmax,
        "rdf_first_step": recent[0][0],
        "rdf_last_step": last_step,
        "trajectory_tool": backend_metadata,
        "trajectory_plot": "trajectory_summary.png" if cfg.validation.trajectory_plots else None,
        "rdf_pair_scope": "intermolecular" if water else "all atom pairs",
    }
    if consistency is not None:
        result["sampling_consistency"] = consistency
    if formal_msd and windowed_rows:
        result["msd_effective_max_lag_ps"] = windowed_rows[-1]["lag_ps"]
    elif not formal_msd:
        result["final_msd_angstrom2"] = msd_rows[-1]["msd_angstrom2"]
    return result
