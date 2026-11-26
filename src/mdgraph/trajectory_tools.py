"""Numerical tools for validated LAMMPS trajectories (angstrom and ps)."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.neighborlist import neighbor_list


def available_trajectory_backends() -> list[str]:
    try:
        from MDAnalysis.lib.distances import self_capped_distance  # noqa: F401
    except (ImportError, OSError, ValueError):
        return ["ase"]
    return ["ase", "mdanalysis"]


def backend_info(name: str) -> dict:
    library = {"ase": "ASE", "mdanalysis": "MDAnalysis"}[name]
    return {
        "backend": name,
        "library": library,
        "version": version(library),
        "method": "ordered periodic neighbor list"
        if name == "ase"
        else "periodic self_capped_distance",
    }


def pairs_within(name, positions, cell, cutoff):
    """Return ordered pairs for consistent RDF normalization."""
    if name == "ase":
        return neighbor_list("ijd", Atoms(positions=positions, cell=cell, pbc=True), cutoff)
    from MDAnalysis.lib.distances import self_capped_distance

    pairs, distances = self_capped_distance(positions, cutoff, box=Cell(cell).cellpar())
    pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    return (
        np.concatenate((pairs[:, 0], pairs[:, 1])),
        np.concatenate((pairs[:, 1], pairs[:, 0])),
        np.tile(distances, 2),
    )


def save_plots(folder, time, msd, radii, rdf, *, partials=None):
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ValueError(
            "Trajectory plotting requires matplotlib; install .[trajectory]"
        ) from error
    fig = Figure(figsize=(8, 3.3), layout="constrained")
    msd_ax, rdf_ax = fig.subplots(1, 2)
    msd_ax.plot(time, msd, color="#136a7c")
    msd_ax.set(xlabel="Time (ps)", ylabel="MSD (Å²)")
    if partials:
        for label, values in partials.items():
            rdf_ax.plot(radii, values, label=label)
        rdf_ax.legend(frameon=False)
    else:
        rdf_ax.plot(radii, rdf, color="#27539a")
    rdf_ax.set(xlabel="Distance (Å)", ylabel="g(r)")
    fig.savefig(folder / "trajectory_summary.png", dpi=200)
