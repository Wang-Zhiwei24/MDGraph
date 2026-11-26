"""Compose a solvated periodic slit from a crystallographic slab."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from pydantic import Field
from scipy.spatial.transform import Rotation

from .config import StrictModel
from .molecular import load_catalog


class CrystallineSlitPreview(StrictModel):
    polymer: Literal["cellulose", "chitin"]
    allomorph: Literal["I-alpha", "I-beta", "II", "III-I", "alpha", "beta"]
    miller_face: tuple[int, int, Literal[0]]
    layers: int = Field(ge=2, le=8)
    chains_per_layer: int = Field(ge=4, le=16)
    degree_polymerization: int = Field(ge=4, le=20, multiple_of=2)
    width_angstrom: float = Field(ge=10, le=60)
    fluid: Literal["water"]
    reason: str = Field(min_length=5, max_length=550)


class PreviewDecision(StrictModel):
    recipe: CrystallineSlitPreview | None


def tool_inventory() -> dict:
    try:
        return {"cellulose_forge": version("cellulose-forge")}
    except PackageNotFoundError:
        return {}


def build_preview(recipe: CrystallineSlitPreview, directory: Path) -> dict:
    if recipe.polymer == "cellulose" and recipe.allomorph in {"alpha", "beta"}:
        raise ValueError("Select a cellulose allomorph")
    if recipe.polymer == "chitin" and recipe.allomorph not in {"alpha", "beta"}:
        raise ValueError("Select a chitin allomorph")
    if (
        recipe.miller_face == (0, 0, 0)
        or max(map(abs, recipe.miller_face)) > 4
        or recipe.layers * recipe.chains_per_layer * recipe.degree_polymerization > 800
    ):
        raise ValueError("Invalid or oversized crystallographic slab recipe")
    directory.mkdir(exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "cellulose_forge.cli",
        "build",
        "--polymer",
        recipe.polymer,
        "--allomorph",
        recipe.allomorph,
        "--shape",
        "slab",
        "--face",
        *map(str, recipe.miller_face),
        "--layers",
        str(recipe.layers),
        "--chains-per-layer",
        str(recipe.chains_per_layer),
        "--degree-polymerization",
        str(recipe.degree_polymerization),
        "--chain-boundary",
        "periodic",
        "--force-field",
        "none",
        "--vacuum",
        "40",
        "--output",
        "slab.pdb",
    ]
    result = subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    (directory / "builder.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError("The crystallographic builder failed. See preview/builder.log.")
    slab = read(directory / "slab.pdb")
    return pack_slit(slab, recipe.width_angstrom, directory)


def pack_slit(slab: Atoms, width: float, directory: Path) -> dict:
    cell = np.asarray(slab.cell)
    if not np.allclose(cell, np.diag(cell.diagonal()), atol=1e-5):
        raise ValueError("Slit packing requires an orthorhombic cell with the slab normal along z")
    lx, ly, old_lz = cell.diagonal()
    z = slab.positions[:, 2]
    slab_thickness = np.ptp(z)
    if min(lx, ly) < 22 or not 0 < slab_thickness < old_lz:
        raise ValueError("Insufficient in-plane dimensions or missing slab clearance")
    # A single slab in a periodic cell presents two facing surfaces across every gap.
    margin = 0.1
    lower, upper = margin + slab_thickness, margin + slab_thickness + width
    lz = slab_thickness + width
    shift = [0, 0, margin - z.min()]
    slab = Atoms(
        slab.get_chemical_symbols(), positions=slab.positions + shift, cell=[lx, ly, lz], pbc=True
    )
    slab.wrap()
    wall = len(slab)
    water = load_catalog().models["tip3p_ewald"]
    xyz = np.asarray(water.coordinates)
    xyz -= xyz[0].copy()  # Both TIP3P and original TIP4P use the same O–H geometry.
    masses = np.array([site[1] for site in water.sites])
    exclusion = 2.5
    accessible = width - 2 * exclusion
    initial_density = 0.997
    count = round(lx * ly * accessible * initial_density / (masses.sum() * 1.66053906660))
    nx, ny = np.rint(np.array([lx, ly]) / 2.9).astype(int)
    grid_shape = (nx, ny, max(1, int(np.ceil(count / (nx * ny)))))
    spacing = np.array([lx, ly, accessible]) / grid_shape
    if min(spacing) < 2.3:
        raise ValueError("Initial loading is too dense for the packing grid")
    rng = np.random.default_rng(2757)
    grid = (np.indices(grid_shape).reshape(3, -1).T + 0.5) * spacing
    grid = grid[np.sort(rng.choice(len(grid), count, replace=False))]
    grid[:, 2] += lower + exclusion
    ids = np.concatenate([np.zeros(wall, dtype=int), np.repeat(np.arange(1, count + 1), 3)])
    molecules = np.arange(count)
    positions = np.empty((count, 3, 3))
    slab += Atoms([site[0] for site in water.sites] * count)
    for _ in range(20):
        rotations = Rotation.random(len(molecules), random_state=rng).as_matrix()
        positions[molecules] = np.einsum("mij,sj->msi", rotations, xyz) + grid[molecules, None, :]
        slab.positions[wall:] = positions.reshape(-1, 3)
        slab.wrap()
        first, second = neighbor_list("ij", slab, 1.6)
        contacts = ids[first] != ids[second]
        if not np.any(contacts):
            break
        near = np.unique(ids[first[contacts]])
        molecules = near[near > 0] - 1
    else:
        raise ValueError("A fluid molecule overlaps another molecule or the solid")
    if not np.allclose(
        np.linalg.norm(positions[:, 1:] - positions[:, :1], axis=-1), 0.9572, atol=1e-4
    ) or not np.all((grid[:, 2] > lower) & (grid[:, 2] < upper)):
        raise ValueError("Water bond geometry or slit placement failed validation")
    slab.new_array("fluid_molecule_id", ids)
    write(directory / "initial.extxyz", slab)
    return {
        "status": "coordinates_checked",
        "geometry": "orthorhombic periodic stack with one water gap per cell",
        "width_definition": "distance between the facing outermost atomic z planes",
        "width_angstrom": float(width),
        "surface_planes_angstrom": [float(lower), float(upper)],
        "box_angstrom": list(map(float, (lx, ly, lz))),
        "area_angstrom2": float(lx * ly),
        "solid_atoms": wall,
        "water_molecules": count,
        "water_oxygen_accessible_width_angstrom": float(accessible),
        "initial_accessible_density_g_cm3": float(
            count * masses.sum() * 1.66053906660 / (lx * ly * accessible)
        ),
        "water_geometry": "three physical atomic sites; TIP4P virtual site belongs to the future force-field step",
        "packing_seed": 2757,
        "coordinate_file": str(directory / "initial.extxyz"),
    }
