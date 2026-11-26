"""Fixed tool adapters. The model chooses a tool ID, never executable arguments."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase import __version__ as ase_version
from ase.build import bulk
from ase.io import read

from .config import Structure

ASE_TOOL = f"ASE {ase_version}"
ATOMSK_LATTICES = {"bcc", "fcc", "diamond", "hcp"}
SITES = {
    "sc": 1,
    "bcc": 2,
    "fcc": 4,
    "diamond": 8,
    "hcp": 2,
    "b2": 2,
    "rocksalt": 8,
    "zincblende": 8,
}


def available_builders(s: Structure) -> list[str]:
    supported = s.crystal and s.crystal.lattice in ATOMSK_LATTICES
    installed = supported and shutil.which(s.atomsk_executable) is not None
    if s.builder == "atomsk":
        if not installed:
            raise ValueError("Atomsk is required but its executable was not found")
        return ["atomsk"]
    return ["ase", "atomsk"] if s.builder == "auto" and installed else ["ase"]


def ensure_crystal_geometry(atoms, s: Structure) -> None:
    c = s.crystal
    target_lengths = np.array([c.a_angstrom, c.a_angstrom, c.c_angstrom or c.a_angstrom]) * c.repeat
    target_angles = [90, 90, 120 if c.lattice == "hcp" else 90]
    if len(atoms) != SITES[c.lattice] * int(np.prod(c.repeat)):
        raise ValueError("Builder returned an unexpected atom count")
    if set(atoms.get_chemical_symbols()) != set(s.elements):
        raise ValueError("Builder returned an unexpected chemical element")
    if not np.allclose(atoms.cell.lengths(), target_lengths, rtol=1e-7, atol=1e-7):
        raise ValueError("Builder changed the requested lattice constants or repeats")
    if not np.allclose(atoms.cell.angles(), target_angles, rtol=0, atol=1e-5):
        raise ValueError("Builder returned unexpected cell angles")


def _atomsk(s: Structure, folder: Path):
    c = s.crystal
    executable = shutil.which(s.atomsk_executable)
    if executable is None or c.lattice not in ATOMSK_LATTICES:
        raise ValueError("Atomsk executable or requested lattice is unavailable")
    output = "atomsk_structure.xsf"
    artifact = folder / output
    # Atomsk can prompt before overwriting; remove only its incomplete private output.
    artifact.unlink(missing_ok=True)
    args = [str(Path(executable).resolve()), "--create", c.lattice, str(c.a_angstrom)]
    if c.c_angstrom is not None:
        args.append(str(c.c_angstrom))
    args.extend([s.elements[0], "-duplicate", *map(str, c.repeat), output])
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        in {"PATH", "LD_LIBRARY_PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
    }
    env["OMP_NUM_THREADS"] = "1"
    log = folder / "atomsk.log"
    with log.open("wb") as stream:
        try:
            result = subprocess.run(
                args,
                cwd=folder,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("Atomsk timed out; inspect atomsk.log") from error
    if result.returncode or not artifact.is_file() or not artifact.stat().st_size:
        raise ValueError(
            f"Atomsk did not produce a structure (exit {result.returncode}); inspect atomsk.log"
        )
    atoms = read(artifact, format="xsf")
    # Record the generated structure and executable argv, not environment variables.
    return atoms, {
        "name": "Atomsk",
        "operation": "create and duplicate",
        "argv": args,
        "format_adapter": ASE_TOOL,
        "artifacts": [output, log.name],
    }


def build_crystal(s: Structure, folder: Path, builder: str):
    if builder not in available_builders(s):
        raise ValueError("Requested structure tool is outside the configured available tools")
    c = s.crystal
    if builder == "atomsk":
        atoms, tool = _atomsk(s, folder)
    else:
        if c.lattice == "b2":
            unit = Atoms(
                s.elements,
                scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
                cell=[c.a_angstrom] * 3,
                pbc=True,
            )
        else:
            unit = bulk(
                "".join(s.elements),
                c.lattice,
                a=c.a_angstrom,
                c=c.c_angstrom,
                cubic=c.lattice != "hcp",
            )
        atoms = unit.repeat(c.repeat)
        tool = {"name": ASE_TOOL, "operation": "bulk and repeat", "artifacts": []}
    ensure_crystal_geometry(atoms, s)
    return atoms, tool
