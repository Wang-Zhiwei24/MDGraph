"""Registered molecular force fields, structure construction and LAMMPS setup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from ase import Atoms
from ase.build import graphene, make_supercell, nanotube
from ase.data import atomic_masses, atomic_numbers
from ase.io import write
from ase.neighborlist import neighbor_list
from pydantic import Field, model_validator
from scipy.spatial.transform import Rotation

from .artifacts import atomic_json, read_json, sha256
from .config import CNT_WATER_SIGMA_CO_ANGSTROM, MolecularFluid, StrictModel, TaskConfig

KB_KCAL = 0.00198720425864
MODELS_FILE = Path(__file__).with_name("molecular_models.yaml")
GRAPHENE_WATER = {
    "source": "https://doi.org/10.1021/jp0268112",
    "epsilon_kj_mol": 0.392,
    "sigma_angstrom": CNT_WATER_SIGMA_CO_ANGSTROM,
}


class MolecularModel(StrictModel):
    species: str
    source: str
    parameters_source: str | None = None
    evidence: str
    integration: Literal["rigid", "shake_bonds"]
    sites: list[tuple[str, float, float, float, float]] = Field(min_length=2, max_length=64)
    coordinates: list[tuple[float, float, float]]
    bonds: list[tuple[int, int, float]] = Field(min_length=1)
    constrained_bond_types: list[int] = Field(default_factory=list)
    angles: list[tuple[int, int, int, float, float]] = Field(default_factory=list)
    torsions: list[tuple[int, int, int, int, float, float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid(self):
        count = len(self.sites)
        if len(self.coordinates) != count or abs(sum(row[2] for row in self.sites)) > 1e-6:
            raise ValueError(
                "A molecular model requires matching coordinates and a neutral molecule"
            )
        if any(mass <= 0 or eps < 0 or sigma <= 0 for _, mass, _, eps, sigma in self.sites):
            raise ValueError("Invalid site masses or Lennard-Jones parameters")
        xyz = np.asarray(self.coordinates)
        for collection, indices in ((self.bonds, 2), (self.angles, 3), (self.torsions, 4)):
            if any(
                min(row[:indices]) < 0
                or max(row[:indices]) >= count
                or len(set(row[:indices])) != indices
                for row in collection
            ):
                raise ValueError("Invalid site index in molecular topology")
        for i, j, length in self.bonds:
            if length <= 0 or not np.isclose(np.linalg.norm(xyz[i] - xyz[j]), length, atol=1e-4):
                raise ValueError("Coordinates disagree with registered bond lengths")
        if self.integration == "shake_bonds" and (
            not self.constrained_bond_types
            or any(i < 1 or i > len(self.bonds) for i in self.constrained_bond_types)
        ):
            raise ValueError("SHAKE models require explicit valid bond types")
        return self


class MolecularSpecies(StrictModel):
    label: str
    elements: list[str]
    phases: list[Literal["gas", "liquid", "supercritical"]]
    initial_liquid_density_g_cm3: float | None = Field(default=None, gt=0, le=5)


class MolecularCatalog(StrictModel):
    species: dict[str, MolecularSpecies]
    models: dict[str, MolecularModel]

    @model_validator(mode="after")
    def valid(self):
        for model in self.models.values():
            if model.species not in self.species or set(self.species[model.species].elements) != {
                s[0] for s in model.sites
            }:
                raise ValueError("Molecular catalog species and site elements disagree")
        if set(self.species) != {model.species for model in self.models.values()}:
            raise ValueError("Each molecular species requires at least one sourced model")
        return self


def load_catalog() -> MolecularCatalog:
    path = Path(os.environ.get("MDGRAPH_MOLECULAR_CATALOG", str(MODELS_FILE)))
    return MolecularCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def select_molecular_model(cfg: TaskConfig, folder: Path, choose) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "potential_selection.json"
    if path.is_file():
        from .artifacts import verify_files

        selected = read_json(path)
        verify_files(folder, selected["files"])
        return selected
    catalog = load_catalog()
    fluid = cfg.structure.molecular
    if (
        fluid.species not in catalog.species
        or fluid.phase not in catalog.species[fluid.species].phases
    ):
        raise ValueError("No registered molecular model covers the requested species and phase")
    candidates = {
        key: model
        for key, model in catalog.models.items()
        if model.species == fluid.species and cfg.potential.candidate_id in {None, key}
    }
    confined = fluid.geometry in {"graphene_slit", "cnt"}
    if confined and cfg.potential.wall_interface is None:
        candidates = {key: model for key, model in candidates.items() if key == "spce"}
    if not candidates:
        raise ValueError("The specified model is unavailable for the requested species")
    summaries = [
        {
            "id": key,
            "kind": "molecular",
            "source": model.source,
            "evidence": model.evidence,
            "integration": model.integration,
        }
        for key, model in candidates.items()
    ]
    interaction = (
        cfg.potential.wall_interface.model_dump(mode="json")
        if cfg.potential.wall_interface
        else GRAPHENE_WATER
        if confined
        else None
    )
    if interaction:
        summaries[0]["interface"] = interaction
    decision = choose(summaries) if choose else None
    if decision is None and len(candidates) != 1:
        raise ValueError(
            "Select candidate_id explicitly, or enable molecular model selection by the Potential Agent"
        )
    key = decision["choice"] if decision else next(iter(candidates))
    model = candidates[key]
    model_file = folder / "molecular_model.json"
    atomic_json(model_file, model.model_dump(mode="json"))
    selected = {
        "kind": "molecular",
        "mode": "registered_model",
        "candidate_id": key,
        "source": model.source,
        "filename": model_file.name,
        "files": {model_file.name: sha256(model_file)},
    }
    if interaction:
        name = (
            "wall_interface_parameters.json"
            if cfg.potential.wall_interface
            else "graphene_water_parameters.json"
        )
        interface = folder / name
        atomic_json(interface, interaction)
        selected["files"][interface.name] = sha256(interface)
        selected["interface"] = interaction
        if cfg.potential.wall_interface:
            selected["source"] = "https://doi.org/" + cfg.potential.wall_interface.source_doi
    if decision:
        selected["decision"] = decision
    atomic_json(path, selected)
    return selected


def target_density(
    catalog: MolecularCatalog,
    species: str,
    phase: str,
    temperature: float,
    pressure_bar: float | None,
    density: float | None,
) -> float:
    if phase == "gas" and temperature > 0 and pressure_bar is not None and 0 < pressure_bar <= 5:
        model = next(model for model in catalog.models.values() if model.species == species)
        molar_mass = sum(row[1] for row in model.sites)
        return pressure_bar * 100 * molar_mass / (8.314462618 * temperature * 1000)
    if density is not None:
        return density
    reference = catalog.species[species].initial_liquid_density_g_cm3
    if phase == "liquid" and reference:
        return reference
    raise ValueError(
        "A non-dilute fluid needs a sourced initial density at the requested state point"
    )


def assign_sublattices(tube: Atoms, elements: tuple[str, ...], bond: float) -> None:
    if len(elements) == 1:
        tube.set_chemical_symbols([elements[0]] * len(tube))
        return
    left, right = neighbor_list("ij", tube, 1.2 * bond)
    neighbors = [[] for _ in tube]
    for i, j in zip(left, right):
        neighbors[i].append(j)
    if any(len(row) != 3 for row in neighbors):
        raise ValueError("The rolled honeycomb must have three neighbors per wall atom")
    labels = np.full(len(tube), -1, dtype=int)
    labels[0] = 0
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in neighbors[site]:
            if labels[neighbor] == -1:
                labels[neighbor] = 1 - labels[site]
                pending.append(neighbor)
            elif labels[neighbor] == labels[site]:
                raise ValueError("The requested roll cannot assign alternating sublattices")
    if np.any(labels < 0) or np.count_nonzero(labels) * 2 != len(tube):
        raise ValueError("Expected a connected, balanced two-sublattice wall")
    tube.set_chemical_symbols([elements[label] for label in labels])


def cnt_geometry(fluid: MolecularFluid, rng) -> tuple:
    recipe = fluid.honeycomb_tube
    spec = recipe or fluid.cnt_model
    bond = (
        recipe.bond_angstrom if recipe else spec.length_angstrom / (np.sqrt(3) * spec.axial_cells)
    )
    tube = nanotube(*spec.chirality, length=spec.axial_cells, bond=bond)
    if len(tube) > 12000:
        raise ValueError("The rolled honeycomb exceeds the wall atom limit")
    if recipe:
        assign_sublattices(tube, recipe.wall_elements, bond)
    length = float(tube.cell[2, 2])
    lateral = spec.transverse_box_angstrom
    axis = np.array([lateral / 2, lateral / 2])
    wall = tube.positions.copy()
    center = wall[:, :2].mean(axis=0)
    radius = float(np.linalg.norm(wall[:, :2] - center, axis=1).mean())
    if length < 22 or lateral - 2 * radius < 25 or radius <= 3:
        raise ValueError(
            "The tube needs enough axial length, transverse padding and accessible pore radius"
        )
    wall[:, :2] += axis - center
    cell = np.array([lateral, lateral, length])
    # Fit water centers on triangular layers and retain the requested loading.
    count = recipe.fluid_molecules if recipe else spec.water_molecules
    packing_radius = radius - 2.5
    nominal_spacing = float(
        np.clip((np.pi * packing_radius**2 * length / count) ** (1 / 3), 2.3, 3.1)
    )
    span = int(np.ceil(2 * packing_radius / nominal_spacing))
    i, j = np.mgrid[-span : span + 1, -span : span + 1].reshape(2, -1)
    planar = np.column_stack((i + 0.5 * j, np.sqrt(3) * j / 2)) * nominal_spacing
    radii = np.linalg.norm(planar, axis=1)
    ordering = np.argsort(radii + rng.uniform(0, 1e-7, len(radii)))
    sites = planar[ordering[radii[ordering] < packing_radius]]
    nz = max(round(length / nominal_spacing), int(np.ceil(count / len(sites))))
    if length / nz < 2.25:
        raise ValueError(
            "Requested water loading exceeds the supported nonoverlapping packing for this CNT"
        )
    per_layer = np.bincount(np.arange(count) % nz, minlength=nz)
    layers = []
    for layer, n_centers in enumerate(per_layer):
        angle = np.pi * (layer % 2) / 6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = sites[:n_centers] @ rotation + axis
        layers.append(np.column_stack((xy, np.full(n_centers, (layer + 0.5) * length / nz))))
    source = {
        "wall_atoms": len(wall),
        "cnt_axis_xy_angstrom": axis.tolist(),
        "cnt_radius_angstrom": radius,
        "cnt_length_angstrom": length,
        "transverse_box_angstrom": lateral,
        "packing_radius_angstrom": packing_radius,
    }
    return np.vstack(layers), wall, cell, nominal_spacing, source, tube.get_chemical_symbols()


def build_molecular_structure(
    cfg: TaskConfig, folder: Path, model: MolecularModel, packing: str
) -> dict:
    """Pack valid molecular geometries and write the force field's site topology."""
    fluid = cfg.structure.molecular
    slit = fluid.geometry == "graphene_slit"
    cnt = fluid.geometry == "cnt"
    confined = slit or cnt
    masses = np.array([row[1] for row in model.sites])
    symbols = [row[0] for row in model.sites]
    recipe = fluid.honeycomb_tube
    wall_elements = list(recipe.wall_elements) if recipe else ["C"] if confined else []
    if (
        set(symbols + wall_elements) != set(cfg.structure.elements)
        or model.species != fluid.species
    ):
        raise ValueError("Molecular model and requested structure disagree")
    rng = np.random.default_rng(cfg.protocol.seed)
    gap = fluid.vacuum_angstrom or 0
    wall = np.empty((0, 3))
    wall_symbols = []
    confinement_source = {}
    if slit:
        lattice = 2.46
        width = fluid.slit_width_angstrom
        assembly = make_supercell(
            graphene(a=lattice, vacuum=1), [[-1, -2, 0], [1, 0, 0], [0, 0, 1]]
        )
        assembly = assembly.repeat((fluid.repeat, fluid.repeat, 1))
        lx, ly = map(float, assembly.cell.lengths()[:2])
        cell = np.array([lx, ly, width + gap])
        planes = [gap / 2, gap / 2 + width]
        xy = assembly.get_scaled_positions()[:, :2] * [lx, ly]
        physical = fluid.slit_model
        stacking = physical.wall_stacking if physical else "aa"
        exclusion = physical.loading_exclusion_per_wall_angstrom if physical else 2.5
        upper_xy = xy.copy()
        if stacking == "ab":
            upper_xy[:, 0] = (upper_xy[:, 0] + lattice / np.sqrt(3)) % lx
        wall = np.vstack(
            (
                np.column_stack((xy, np.full(len(xy), planes[0]))),
                np.column_stack((upper_xy, np.full(len(xy), planes[1]))),
            )
        )
        wall_symbols = ["C"] * len(wall)
        accessible = width - 2 * exclusion
        count = round(lx * ly * accessible * fluid.density_g_cm3 / (masses.sum() * 1.66053906660))
        nx, ny = np.rint(np.array([lx, ly]) / 2.9).astype(int)
        repeats = (nx, ny, int(np.ceil(count / (nx * ny))))
        spacing = np.array([lx, ly, accessible]) / repeats
        grid = (np.indices(repeats).reshape(3, -1).T + 0.5) * spacing
        grid = grid[np.sort(rng.choice(len(grid), count, replace=False))]
        grid[:, 2] += planes[0] + exclusion
        confinement_source = {
            "wall_z_angstrom": planes,
            "wall_atoms": len(wall),
            "area_angstrom2": lx * ly,
            "wall_stacking": stacking,
            "loading_exclusion_per_wall_angstrom": exclusion,
            "loading_width_angstrom": accessible,
            "loading_molecules_nm2": 100 * count / (lx * ly),
        }
    elif cnt:
        grid, wall, cell, spacing, confinement_source, wall_symbols = cnt_geometry(fluid, rng)
        count = len(grid)
    else:
        count = fluid.repeat**3
        volume = count * masses.sum() * 1.66053906660 / fluid.density_g_cm3
        side = volume ** (1 / 3)
        if side <= 22:
            raise ValueError("Molecular box is too narrow for the 10 Å nonbonded cutoff")
        spacing = side / fluid.repeat
        grid = (np.indices((fluid.repeat,) * 3).reshape(3, -1).T + 0.5) * spacing
        grid[:, 2] += gap / 2
        cell = np.array([side, side, side + gap])
    if packing == "jittered_grid":
        grid += rng.uniform(-0.05, 0.05, grid.shape) * spacing
    elif packing != "regular_grid":
        raise ValueError("Unknown molecular packing method")
    geometry = np.asarray(model.coordinates)
    geometry -= np.average(geometry, axis=0, weights=masses)
    rotations = Rotation.random(count, random_state=rng).as_matrix()
    positions = np.einsum("mij,sj->msi", rotations, geometry) + grid[:, None, :]
    positions = np.vstack((positions.reshape(-1, 3), wall))
    atoms = Atoms(
        symbols * count + wall_symbols, positions=positions, cell=cell, pbc=[True, True, not slit]
    )
    size = len(symbols)
    molecule_ids = np.pad(np.repeat(np.arange(1, count + 1), size), (0, len(wall)))
    i, j = neighbor_list("ij", atoms, cfg.validation.min_pair_distance_angstrom)
    if np.any(molecule_ids[i] != molecule_ids[j]):
        raise ValueError("Generated molecular packing has overlapping molecules")
    atoms.new_array("molecule_id", molecule_ids)
    mass_by_element = {e: float(atomic_masses[atomic_numbers[e]]) for e in wall_elements}
    atoms.set_masses(
        np.concatenate((np.tile(masses, count), [mass_by_element[e] for e in wall_symbols]))
    )
    write(folder / "initial.extxyz", atoms)
    wrapped = np.mod(positions, cell)
    images = np.floor(positions / cell).astype(int)
    topologies = [
        ("bonds", model.bonds, 2),
        ("angles", model.angles, 3),
        ("dihedrals", model.torsions, 4),
    ]
    wall_types = {element: size + i for i, element in enumerate(wall_elements)}
    interface = cfg.potential.wall_interface
    charges = {site.element: site.charge_e for site in interface.sites} if interface else {}
    net_wall_charge = sum(charges.get(element, 0) for element in wall_symbols)
    if abs(net_wall_charge) > 1e-5:
        raise ValueError("The materialized wall is not charge-neutral")
    with (folder / "structure.data").open("w", encoding="ascii", newline="\n") as out:
        out.write(f"MDGraph molecular structure\n\n{len(atoms)} atoms\n")
        for name, rows, _ in topologies:
            out.write(f"{count * len(rows)} {name}\n")
        out.write(f"\n{size + len(wall_elements)} atom types\n")
        for name, rows, _ in topologies:
            if rows:
                out.write(f"{len(rows)} {name[:-1]} types\n")
        out.write("\n")
        for axis, length in zip("xyz", cell):
            out.write(f"0 {length:.14g} {axis}lo {axis}hi\n")
        out.write("\nMasses\n\n")
        for index, mass in enumerate(masses, start=1):
            out.write(f"{index} {mass}\n")
        for element, site in wall_types.items():
            out.write(f"{site + 1} {mass_by_element[element]}\n")
        out.write("\nAtoms # full\n\n")
        for index, (pos, image) in enumerate(zip(wrapped, images)):
            if molecule_ids[index]:
                site = index % size
                charge = model.sites[site][2]
            else:
                element = wall_symbols[index - count * size]
                site = wall_types[element]
                charge = charges.get(element, 0)
            values = " ".join(f"{x:.13g}" for x in pos) + " " + " ".join(map(str, image))
            out.write(f"{index + 1} {molecule_ids[index]} {site + 1} {charge} {values}\n")
        for name, rows, width in topologies:
            if rows:
                out.write(f"\n{name.capitalize()}\n\n")
                for molecule in range(count):
                    for index, row in enumerate(rows, start=1):
                        ids = " ".join(str(molecule * size + site + 1) for site in row[:width])
                        out.write(f"{molecule * len(rows) + index} {index} {ids}\n")
    info = {
        "mode": f"molecular_{fluid.geometry}",
        "atoms": len(atoms),
        "elements": cfg.structure.elements,
        "element_counts": {e: atoms.symbols.count(e) for e in cfg.structure.elements},
        "wrapped": True,
        "source": {
            "fluid": fluid.model_dump(),
            "molecules": count,
            "seed": cfg.protocol.seed,
            "atom_type_elements": symbols + wall_elements,
            "wall_charge_e": net_wall_charge,
            **confinement_source,
        },
        "tool": {
            "name": {
                "cnt": "ASE nanotube builder and molecular packing",
                "graphene_slit": "ASE graphene builder and molecular packing",
            }.get(fluid.geometry, "ASE molecular packing"),
            "operation": packing,
            "artifacts": [],
        },
        "files": {name: sha256(folder / name) for name in ("structure.data", "initial.extxyz")},
    }
    atomic_json(folder / "structure_build.json", info)
    return info


def molecular_setup(fluid: MolecularFluid, model: MolecularModel, interface=None) -> str:
    slit = fluid.geometry == "graphene_slit"
    cnt = fluid.geometry == "cnt"
    lines = [
        "# Registered molecular model, units kcal/mol, Angstrom, fs, atm",
        "units real",
        "atom_style full",
        "boundary p p f" if slit else "boundary p p p",
    ]
    if slit:
        lines.append("processors * * 1")
    lines.append("read_data structure.data")
    if fluid.geometry == "slab":
        lines += [
            "pair_style lj/long/coul/long long long 10.0",
            "kspace_style pppm/disp 1.0e-4",
            "kspace_modify mix/disp pair force/disp/real 1.0e-4 force/disp/kspace 1.0e-4",
        ]
    else:
        lines += ["pair_style lj/cut/coul/long 10.0", "kspace_style pppm 1.0e-4"]
    lines.append("pair_modify mix arithmetic")
    for index, (_, _, _, epsilon, sigma) in enumerate(model.sites, start=1):
        lines.append(f"pair_coeff {index} {index} {KB_KCAL * epsilon:.12g} {sigma}")
    if slit or cnt:
        first = len(model.sites) + 1
        if interface:
            sites = {site.element: site for site in interface.sites}
            coefficients = [
                (sites[e].epsilon_kcal_mol, sites[e].sigma_angstrom)
                for e in fluid.honeycomb_tube.wall_elements
            ]
        else:
            coefficients = [
                (GRAPHENE_WATER["epsilon_kj_mol"] / 4.184, GRAPHENE_WATER["sigma_angstrom"])
            ]
        for wall_type, (epsilon, sigma) in enumerate(coefficients, first):
            for index, site in enumerate(model.sites, 1):
                eps = epsilon if site[0] == "O" else 0
                lines.append(f"pair_coeff {index} {wall_type} {eps:.12g} {sigma:.12g}")
            for index in range(first, wall_type + 1):
                lines.append(f"pair_coeff {index} {wall_type} 0.0 1.0")
        if slit:
            lines.append("kspace_modify slab 3.0")
        wall_types = " ".join(map(str, range(first, first + len(coefficients))))
        lines += [
            f"group plates type {wall_types}",
            "group water subtract all plates",
            "fix fixed_plates plates setforce 0 0 0",
        ]
    lines.append("bond_style harmonic")
    for index, (_, _, length) in enumerate(model.bonds, start=1):
        lines.append(f"bond_coeff {index} 450.0 {length}")
    if model.angles:
        lines.append("angle_style harmonic")
        for index, (_, _, _, stiffness, angle) in enumerate(model.angles, start=1):
            lines.append(f"angle_coeff {index} {stiffness} {angle}")
    if model.torsions:
        lines.append("dihedral_style opls")
        for index, (_, _, _, _, k1, k2, k3) in enumerate(model.torsions, start=1):
            lines.append(
                f"dihedral_coeff {index} {2 * KB_KCAL * k1:.12g} {2 * KB_KCAL * k2:.12g} {2 * KB_KCAL * k3:.12g} 0"
            )
    # TraPPE-UA uses no intramolecular nonbonded interactions through 1-4.
    lines += [
        "special_bonds lj/coul 0 0 0",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
    ]
    return "\n".join(lines) + "\n"
