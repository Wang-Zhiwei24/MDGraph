import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mdgraph.config import LLM, Execution, HoneycombTube, WallInterface, WallSite
from mdgraph.fluid_analysis import ANALYSIS_CHOICES, analyze_fluid, confined_density, fit_diffusion
from mdgraph.literature import normalize, verify_interface
from mdgraph.molecular import load_catalog, molecular_setup
from mdgraph.natural import SimulationPlan, compile_molecular, refine_tube_study
from mdgraph.prepare import build_structure, prepare

MODEL_CASES = [
    ("water", "liquid", "surface_tension", "pressure_tensor", "spce", 8, None, "nvt", "nvt", 50),
    ("co2", "gas", "self_diffusion", "molecular_msd_fit", "trappe_co2", 6, 1.0, "nvt", "nve", None),
    (
        "ethanol",
        "liquid",
        "self_diffusion",
        "molecular_msd_fit",
        "trappe_ua_ethanol",
        6,
        1.0,
        "npt",
        "nve",
        None,
    ),
]


@pytest.mark.parametrize("case", MODEL_CASES)
def test_registered_model_can_be_packed_and_rendered(tmp_path, case):
    species, phase, prop, method, model_id, repeat, pressure, equil, prod, gap = case
    assert load_catalog().models[model_id].species == species
    plan = SimulationPlan(
        action="run",
        system="molecular_fluid",
        species=species,
        phase=phase,
        target_property=prop,
        method=method,
        repeat=repeat,
        temperature_k=298.15,
        pressure_bar=pressure,
        equilibration_ensemble=equil,
        ensemble=prod,
        vacuum_angstrom=gap,
        timestep_fs=1,
        equilibration_steps=2000,
        production_steps=2000,
        reason="Run a short physical and executable smoke test.",
    )
    lmp = os.environ.get("MDGRAPH_TEST_LAMMPS")
    cfg = compile_molecular(
        plan, f"Measure {prop} of {species}", lmp or "lmp", LLM(), execution=Execution(), smoke=True
    )
    cfg.potential.candidate_id = model_id
    manifest = prepare(cfg, tmp_path)
    assert manifest["units"] == "real"
    assert manifest["atoms"] == repeat**3 * len(load_catalog().models[model_id].sites)
    text = (tmp_path / "in.lammps").read_text()
    assert "ix iy iz mol" in text and "pxx pyy pzz" in text
    assert ("kspace_style pppm/disp" in text) == (prop == "surface_tension")
    if species == "co2":
        assert cfg.structure.molecular.density_g_cm3 == pytest.approx(0.0017753, rel=1e-3)
        assert "rigid/nve/small molecule" in text and "shake" not in text
    if lmp:
        run = subprocess.run(
            [str(Path(lmp)), "-in", "in.lammps"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=420,
        )
        assert run.returncode == 0, run.stdout[-6000:] + run.stderr
        result = analyze_fluid(cfg, tmp_path, manifest["atoms"], next(iter(ANALYSIS_CHOICES[prop])))
        assert result["passed"]
        assert result["sampling"]["mode"] == "smoke"
        assert not result["sampling"]["passed"]


def test_diffusion_units_and_window():
    time = np.arange(200) * 0.5
    result = fit_diffusion(1.2 * time, time, "late_msd")
    assert result["value_m2_s"] == pytest.approx(2.0e-9)
    assert result["log_slope"] == pytest.approx(1)
    assert result["r2"] == pytest.approx(1)


def test_manager_slit_plan_builds_fixed_ase_walls_and_water_topology(tmp_path):
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    plan = SimulationPlan(
        action="run",
        system="molecular_fluid",
        species="water",
        phase="liquid",
        target_property="density_profile",
        method="wall_normal_histogram",
        repeat=9,
        temperature_k=298.15,
        slit_width_angstrom=24,
        vacuum_angstrom=30,
        slit_model={
            "wall_material": "graphene",
            "wall_count": 2,
            "wall_motion": "fixed",
            "wall_stacking": "ab",
            "fluid_representation": "explicit_water",
            "periodic_axes": "xy",
            "cross_interaction": "carbon_oxygen_lj",
            "loading_exclusion_per_wall_angstrom": 2.5,
        },
        equilibration_ensemble="nvt",
        ensemble="nvt",
        timestep_fs=2,
        equilibration_steps=2000,
        production_steps=2000,
        reason="Rigid graphene with explicit water and a wall-normal profile.",
    )
    cfg = compile_molecular(
        plan, "Wall-normal water density", "lmp", LLM(), execution=Execution(), smoke=True
    )
    result = build_structure(cfg, tmp_path, molecular_model=load_catalog().models["spce"])
    JsonPlusSerializer().dumps_typed(result)
    manifest = prepare(cfg, tmp_path)
    structure = result["source"]
    assert structure["wall_z_angstrom"] == [15, 39]
    assert structure["wall_stacking"] == "ab"
    assert manifest["atoms"] == 2 * 4 * 9**2 + 3 * structure["molecules"]
    data = (tmp_path / "structure.data").read_text()
    assert (
        f"{structure['molecules'] * 2} bonds" in data and f"{structure['molecules']} angles" in data
    )
    script = (tmp_path / "in.lammps").read_text()
    assert "boundary p p f" in script and "kspace_modify slab 3.0" in script
    assert "rigid/nvt/small molecule" in script and "dump_modify traj sort id delay 2000" in script
    assert script.count("\nrun ") == 1


def test_cnt_radial_profile_normalizes_by_cylindrical_shell_volume(tmp_path):
    from ase.io import read

    pytest.importorskip("matplotlib")
    plan = SimulationPlan(
        action="run",
        system="molecular_fluid",
        species="water",
        phase="liquid",
        target_property="radial_density_profile",
        method="cylindrical_mass_histogram",
        temperature_k=298.15,
        ensemble="nvt",
        equilibration_ensemble="nvt",
        timestep_fs=2,
        equilibration_steps=2000,
        production_steps=2000,
        reason="Build the tube and verify radial normalization.",
        cnt_model={
            "chirality": [12, 12],
            "axial_cells": 20,
            "length_angstrom": 49.2,
            "water_molecules": 221,
            "transverse_box_angstrom": 45,
        },
    )
    cfg = compile_molecular(
        plan,
        "Water density inside a (12,12) carbon nanotube",
        "lmp",
        LLM(),
        execution=Execution(),
        smoke=True,
    )
    model = load_catalog().models["spce"]
    structure = build_structure(cfg, tmp_path, molecular_model=model)
    assert structure["element_counts"] == {"O": 221, "H": 442, "C": 960}
    atoms = read(tmp_path / "initial.extxyz")
    cell = atoms.cell.lengths()
    mol = atoms.arrays["molecule_id"]
    types = {"O": 1, "H": 2, "C": 4}
    rows = [
        f"{i} {types[symbol]} {xyz[0]} {xyz[1]} {xyz[2]} 0 0 0 {molecule}\n"
        for i, (symbol, xyz, molecule) in enumerate(zip(atoms.symbols, atoms.positions, mol), 1)
    ]
    cfg.protocol.dump_every = 200
    with (tmp_path / "trajectory.lammpstrj").open("w") as out:
        for step in range(2000, 4001, cfg.protocol.dump_every):
            out.write(
                f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(atoms)}\nITEM: BOX BOUNDS pp pp pp\n"
            )
            out.writelines(f"0 {length}\n" for length in cell)
            out.write("ITEM: ATOMS id type x y z ix iy iz mol\n")
            out.writelines(rows)
    metrics, checks = confined_density(cfg, tmp_path, model)
    assert checks["passed"] and metrics["density_profile_frames"] == 11
    profile = np.genfromtxt(tmp_path / "radial_density_profile.csv", delimiter=",", names=True)
    shell_volumes = (
        np.pi * cell[2] * (profile["bin_outer_angstrom"] ** 2 - profile["bin_inner_angstrom"] ** 2)
    )
    expected_mass = 221 * sum(site[1] for site in model.sites) * 1.66053906660
    assert np.dot(profile["density_g_cm3"], shell_volumes) == pytest.approx(expected_mass)


def test_composed_binary_tube_resolves_wall_types_and_rejects_unsourced_coefficients(
    tmp_path, monkeypatch
):
    tube = HoneycombTube(
        wall_elements=["B", "N"],
        chirality=[12, 12],
        bond_angstrom=1.42,
        axial_cells=15,
        transverse_box_angstrom=45,
        fluid_molecules=162,
    )
    excerpt = "Test source: LJ in kcal/mol. Boron 0.1216 and nitrogen 0.1502. The common diameter is 3.26 Å."
    interface = WallInterface(
        source_doi="10.1234/test",
        fluid_model="spce",
        fluid_lj_element="O",
        wall_electrostatics="lj_only",
        evidence_quotes=[excerpt],
        scope="Rigid cylindrical pore test fixture.",
        sites=[
            {"element": element, "charge_e": 0, "epsilon_kcal_mol": epsilon, "sigma_angstrom": 3.26}
            for element, epsilon in (("B", 0.1216), ("N", 0.1502))
        ],
    )
    document = {"text": normalize(excerpt)}
    assert verify_interface(interface, tube, document) == [excerpt]
    mixed = WallSite.model_validate(
        {
            "element": "B",
            "charge_e": 0,
            "epsilon_kcal_mol": 0.1216,
            "sigma_mixing_inputs_angstrom": [3.36, 3.16],
        }
    )
    assert mixed.sigma_angstrom == pytest.approx(3.26)
    altered = interface.sites[0].model_copy(update={"epsilon_kcal_mol": 0.3})
    with pytest.raises(ValueError, match="epsilon"):
        verify_interface(
            interface.model_copy(update={"sites": [altered, interface.sites[1]]}), tube, document
        )

    plan = SimulationPlan(
        action="run",
        system="molecular_fluid",
        species="water",
        phase="liquid",
        temperature_k=298,
        target_property="radial_density_profile",
        method="cylindrical_mass_histogram",
        honeycomb_tube=tube,
        research_queries=["boron nitride nanotube confined water"],
        ensemble="nvt",
        equilibration_ensemble="nvt",
        timestep_fs=1,
        equilibration_steps=2000,
        production_steps=2000,
        reason="Test the reusable honeycomb roll and both wall pair coefficients.",
    )
    oversized = plan.model_copy(
        update={"honeycomb_tube": tube.model_copy(update={"fluid_molecules": 900})}
    )
    calls = []

    def planner(llm, role, evidence, schema, instruction, example):
        calls.append(dict(evidence))
        return oversized if len(calls) == 1 else plan

    monkeypatch.setattr("mdgraph.natural._invoke", planner)
    request = "Binary honeycomb tube"
    refined = refine_tube_study(oversized, request, {}, interface, excerpt, LLM())
    assert refined.honeycomb_tube == tube
    assert "holds 162 water molecules" in refined.reason
    assert "Loading 900 exceeds" in calls[1]["validation_feedback"]
    cfg = compile_molecular(
        plan,
        "Binary honeycomb tube",
        "lmp",
        LLM(),
        execution=Execution(),
        smoke=True,
        interface=interface,
    )
    model = load_catalog().models["spce"]
    result = build_structure(cfg, tmp_path, molecular_model=model)
    assert result["element_counts"] == {"O": 162, "H": 324, "B": 360, "N": 360}
    assert result["source"]["atom_type_elements"] == ["O", "H", "H", "B", "N"]
    script = molecular_setup(cfg.structure.molecular, model, interface)
    assert "pair_coeff 1 4 0.1216 3.26" in script
    assert "pair_coeff 1 5 0.1502 3.26" in script
    assert "group plates type 4 5" in script
