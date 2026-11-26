"""Molecular water builder and vetted rigid water models for LAMMPS."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation

WATER_MODELS = {
    "spce": {
        "name": "SPC/E",
        "source": "https://docs.lammps.org/stable/Howto_spc.html",
        "description": "Rigid three-site liquid-water model; supports PPPM electrostatics",
        "oxygen_charge": -0.8476,
        "oxygen_epsilon": 0.1553,
        "oxygen_sigma": 3.166,
        "bond": 1.0,
        "angle": 109.47,
    },
    "tip3p_ewald": {
        "name": "TIP3P (Ewald)",
        "source": "https://docs.lammps.org/stable/Howto_tip3p.html",
        "description": "Rigid three-site TIP3P variant parameterized for long-range electrostatics",
        "oxygen_charge": -0.834,
        "oxygen_epsilon": 0.1020,
        "oxygen_sigma": 3.188,
        "bond": 0.9572,
        "angle": 104.52,
    },
}

WATER_CUTOFF_ANGSTROM = 8.5


def build_water_box(repeat: int, seed: int) -> Atoms:
    """Generate nonoverlapping, randomly oriented water on a grid near 1 g/cm³."""
    grid = (np.indices((repeat,) * 3).reshape(3, -1).T + 0.5) * 3.106
    angle = np.deg2rad(WATER_MODELS["spce"]["angle"] / 2)
    geometry = [[0, 0, 0], [np.sin(angle), np.cos(angle), 0], [-np.sin(angle), np.cos(angle), 0]]
    rotations = Rotation.random(len(grid), random_state=np.random.default_rng(seed)).as_matrix()
    positions = np.einsum("nij,aj->nai", rotations, geometry) + grid[:, None, :]
    return Atoms(
        "OH2" * len(grid), positions=positions.reshape(-1, 3), cell=[repeat * 3.106] * 3, pbc=True
    )


def write_water_data(path, atoms: Atoms) -> None:
    count = len(atoms) // 3
    with path.open("w", encoding="ascii", newline="\n") as out:
        out.write(
            f"MDGraph molecular water\n\n{len(atoms)} atoms\n{2 * count} bonds\n{count} angles\n\n"
            "2 atom types\n1 bond types\n1 angle types\n\n"
        )
        for axis, length in zip("xyz", atoms.cell.lengths()):
            out.write(f"0 {length:.12g} {axis}lo {axis}hi\n")
        out.write("\nMasses\n\n1 15.9994\n2 1.008\n\nAtoms # full\n\n")
        for index, (x, y, z) in enumerate(atoms.positions):
            out.write(
                f"{index + 1} {index // 3 + 1} {1 if index % 3 == 0 else 2} 0 "
                f"{x:.12g} {y:.12g} {z:.12g}\n"
            )
        out.write("\nBonds\n\n")
        for index in range(count):
            oxygen = 3 * index + 1
            out.write(
                f"{2 * index + 1} 1 {oxygen} {oxygen + 1}\n"
                f"{2 * index + 2} 1 {oxygen} {oxygen + 2}\n"
            )
        out.write("\nAngles\n\n")
        for index in range(count):
            oxygen = 3 * index + 1
            out.write(f"{index + 1} 1 {oxygen + 1} {oxygen} {oxygen + 2}\n")


def render_water_setup(model_id: str) -> str:
    model = WATER_MODELS[model_id]
    charge = model["oxygen_charge"]
    return (
        f"# {model['name']} rigid molecular water with long-range electrostatics\n"
        "units real\natom_style full\nboundary p p p\nread_data structure.data\n"
        f"set type 1 charge {charge}\nset type 2 charge {-charge / 2}\n"
        f"pair_style lj/cut/coul/long {WATER_CUTOFF_ANGSTROM}\n"
        f"pair_coeff 1 1 {model['oxygen_epsilon']} {model['oxygen_sigma']}\n"
        "pair_coeff 1 2 0.0 1.0\npair_coeff 2 2 0.0 1.0\n"
        "kspace_style pppm 1.0e-4\nbond_style harmonic\n"
        f"bond_coeff 1 450.0 {model['bond']}\n"
        f"angle_style harmonic\nangle_coeff 1 55.0 {model['angle']}\n"
        "special_bonds lj/coul 0.0 0.0 0.0\nneighbor 2.0 bin\n"
        "neigh_modify delay 0 every 1 check yes\n"
    )
