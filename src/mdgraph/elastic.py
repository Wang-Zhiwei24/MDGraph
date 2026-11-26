"""0 K cubic elastic tensor from relaxed-ion central stress differences."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from .artifacts import atomic_json

STRAINS = (0.001, 0.0005)
COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
DEFORMATIONS = ("x", "y", "z", "yz", "xz", "xy")
SCALE = ("lx", "ly", "lz", "lz", "lz", "ly")


def render_elastic(setup: str) -> str:
    chunks = []
    restart = setup.replace("read_data structure.data", "read_data final.data", 1)
    for level, eps in enumerate(STRAINS):
        for axis, sign in itertools.product(range(6), (-1, 1)):
            dim, length = DEFORMATIONS[axis], SCALE[axis]
            expression = f"{'0-' if sign < 0 else ''}{length}*{eps:.12g}"
            shift = "${displacement}"
            move = (
                f"change_box all {dim} delta 0 {shift} remap\n"
                if axis < 3
                else f"change_box all triclinic\nchange_box all {dim} delta {shift} remap\n"
            )
            variables = {f"stress_{item}": f"p{item}" for item in COMPONENTS}
            variables.update(maxforce="fmax", atomcount="atoms")
            definitions = "".join(
                f"variable {key} equal {value}\n" for key, value in variables.items()
            )
            values = " ".join("${" + key + "}" for key in variables)
            chunks.append(
                "clear\n"
                + restart
                + f"variable displacement equal {expression}\n"
                + move
                + "min_style cg\nthermo 100\n"
                "minimize 0.0 1e-10 3000 30000\n"
                "thermo_style custom step pxx pyy pzz pyz pxz pxy fmax atoms\nrun 0\n"
                + definitions
                + f'print "MDGRAPH_ELASTIC {level} {axis} {sign} {values}"\n'
            )
    return "".join(chunks)


def cubic_projection(tensor):
    c11 = float(np.diag(tensor)[:3].mean())
    c12 = float((tensor[:3, :3].sum() - np.diag(tensor)[:3].sum()) / 6)
    c44 = float(np.diag(tensor)[3:].mean())
    ideal = np.zeros((6, 6))
    ideal[:3, :3] = c12
    np.fill_diagonal(ideal, [c11] * 3 + [c44] * 3)
    return (c11, c12, c44), ideal


def analyze_elastic(
    text: str, folder: Path, atoms: int, force_limit: float
) -> tuple[dict, list[str]]:
    samples = {}
    for line in text.splitlines():
        if not line.startswith("MDGRAPH_ELASTIC "):
            continue
        fields = line.split()
        try:
            key = tuple(map(int, fields[1:4]))
            values = tuple(map(float, fields[4:]))
        except ValueError as exc:
            raise ValueError("Malformed elastic stress record") from exc
        if len(values) != 8 or not all(map(math.isfinite, values)) or key in samples:
            raise ValueError("Invalid or repeated elastic stress record")
        samples[key] = values
    keys = set(itertools.product(range(len(STRAINS)), range(6), (-1, 1)))
    if samples.keys() != keys:
        raise ValueError("Expected all 24 independent signed strain samples")
    issues = []
    if any(v[-1] != atoms or not 0 <= v[-2] <= force_limit for v in samples.values()):
        issues.append("Elastic samples have changed atom count or unconverged forces")
    matrices = []
    for level, eps in enumerate(STRAINS):
        # LAMMPS reports compression-positive pressure in bar; tensile stress is its negative.
        matrix = np.column_stack(
            [
                -1e-4 * (np.array(samples[level, j, 1][:6]) - samples[level, j, -1][:6]) / (2 * eps)
                for j in range(6)
            ]
        )
        matrices.append(matrix)
    constants, ideal = cubic_projection(matrices[-1])
    c11, c12, c44 = constants
    symmetry = float(np.max(np.abs(matrices[-1] - ideal)))
    delta = float(np.max(np.abs(matrices[0] - matrices[1])))
    if symmetry > max(0.75, 0.01 * abs(c11)):
        issues.append("Raw tensor exceeds cubic-symmetry tolerance")
    if delta > max(0.75, 0.015 * abs(c11)):
        issues.append("Elastic derivatives are sensitive to the selected strain amplitude")
    if min(c11 - c12, c11 + 2 * c12, c44) <= 0:
        issues.append("Cubic elastic stability conditions are not met")
    result = {
        "elastic_constants_gpa": dict(zip(("C11", "C12", "C44"), constants)),
        "bulk_modulus_gpa": (c11 + 2 * c12) / 3,
        "cubic_tensor_gpa": ideal.tolist(),
        "raw_tensor_gpa": matrices[-1].tolist(),
        "strain_amplitudes": list(STRAINS),
        "strain_samples": len(samples),
        "max_cubic_deviation_gpa": symmetry,
        "max_amplitude_difference_gpa": delta,
        "elastic_max_force_ev_per_angstrom": max(v[-2] for v in samples.values()),
    }
    atomic_json(
        folder / "elastic.json",
        {
            **result,
            "voigt_order": COMPONENTS,
            "samples": [
                {
                    "level": k[0],
                    "voigt_column": k[1],
                    "sign": k[2],
                    "pressure_bar": v[:6],
                    "fmax": v[-2],
                    "atoms": int(v[-1]),
                }
                for k, v in sorted(samples.items())
            ],
            "issues": issues,
        },
    )
    return result, issues
