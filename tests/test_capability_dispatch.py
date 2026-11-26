import csv

import numpy as np
import pytest
from langchain_core.output_parsers import PydanticOutputParser

from mdgraph.analysis import analyze_trajectory
from mdgraph.config import TaskConfig
from mdgraph.natural import SimulationPlan, manager_request, task_from_request
from mdgraph.prepare import prepare
from mdgraph.sampling import analyze_windowed_msd, windowed_msd


def test_water_goal_compiles_molecular_input(monkeypatch, tmp_path):
    answer = SimulationPlan(
        action="run",
        system="water",
        target_property="rdf",
        method="equilibrated_md",
        phase="liquid",
        temperature_k=298,
        pressure_bar=1,
        repeat=6,
        ensemble="npt",
        timestep_fs=1,
        equilibration_steps=2000,
        production_steps=3000,
        water_model="spce",
        reason="Equilibrate molecular water before sampling its RDF.",
    )
    monkeypatch.setattr("mdgraph.natural._invoke", lambda *args: answer)
    monkeypatch.setattr("mdgraph.natural.shutil.which", lambda _: "/usr/bin/lmp")
    cfg = task_from_request("Calculate the RDF of bulk water")
    manifest = prepare(cfg, tmp_path)
    assert manifest["atoms"] == 648 and manifest["atom_style"] == "full"
    data = (tmp_path / "structure.data").read_text()
    assert "432 bonds" in data and "216 angles" in data
    script = (tmp_path / "in.lammps").read_text()
    assert "units real" in script and "kspace_style pppm" in script and " all shake " in script
    assert script.index("run 2000") < script.index("dump traj") < script.index("run 3000")
    assert cfg.validation.species_resolved


def test_manager_can_repair_an_incomplete_plan_without_changing_the_request(monkeypatch):
    request = "石墨烯板间距为 24 Å，计算受限水沿板法向的密度分布"
    answer = SimulationPlan(action="needs_input", reason="Confirm the initial water loading.")
    calls = []

    def invoke(llm, role, evidence, schema, instruction, example):
        calls.append(evidence)
        if len(calls) == 1:
            return PydanticOutputParser(pydantic_object=schema).parse(
                '{"action":"run","slit_width_angstrom":"missing","reason":"Prepare a slit model."}'
            )
        return answer

    monkeypatch.setattr("mdgraph.natural._invoke", invoke)
    plan, evidence, _ = manager_request(request)
    assert plan == answer
    assert calls[1]["request"] == request
    assert calls[1]["explicit_constraints"] == {"slit_width_angstrom": 24}
    assert calls[1]["validation_fields"][0]["field"] == "$.slit_width_angstrom"
    assert "validation_fields" not in evidence


def test_water_rdf_excludes_same_molecule_pairs(tmp_path):
    cfg = TaskConfig.model_validate(
        {
            "name": "water",
            "structure": {"water_repeat": 6, "elements": ["O", "H"]},
            "potential": {"kind": "water", "water_model": "spce"},
            "protocol": {"steps": 20, "thermo_every": 5, "dump_every": 20},
            "validation": {"rdf_max_angstrom": 2, "species_resolved": True},
        }
    )
    path = tmp_path / "trajectory.lammpstrj"
    geometry = (
        "1 1 1 1 1 0 0 0\n2 2 2 1 1 0 0 0\n3 2 1 2 1 0 0 0\n"
        "4 1 6 6 6 0 0 0\n5 2 7 6 6 0 0 0\n6 2 6 7 6 0 0 0\n"
    )
    path.write_text(
        "".join(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n6\nITEM: BOX BOUNDS pp pp pp\n"
            "0 12\n0 12\n0 12\nITEM: ATOMS id type x y z ix iy iz\n" + geometry
            for step in (0, 20)
        ),
        encoding="ascii",
    )
    metrics = analyze_trajectory(cfg, tmp_path, 6)
    assert metrics["rdf_pair_scope"] == "intermolecular"
    with (tmp_path / "rdf.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(float(row[key]) == 0 for row in rows for key in ("g_r", "g_O_O", "g_O_H", "g_H_H"))


def test_configured_system_can_reuse_provided_structure_and_potential(monkeypatch):
    cfg = TaskConfig.model_validate(
        {
            "name": "alloy",
            "structure": {"path": "provided.data", "elements": ["Cu", "Ni"]},
            "potential": {
                "kind": "eam/alloy",
                "model": "provided.eam.alloy",
                "metadata": "provided.json",
            },
            "protocol": {"ensemble": "nvt", "temperature_k": 400},
            "execution": {"command": ["/custom/lmp"]},
        }
    )
    answer = SimulationPlan(
        action="run",
        system="configured",
        target_property="msd",
        method="equilibrated_md",
        temperature_k=400,
        pressure_bar=0,
        reason="Use the supplied alloy protocol.",
    )
    monkeypatch.setattr("mdgraph.natural._invoke", lambda *args: answer)
    selected = task_from_request("Get the MSD of this Cu–Ni alloy", configured=cfg)
    assert selected.structure == cfg.structure and selected.potential == cfg.potential
    assert selected.protocol == cfg.protocol and selected.execution == cfg.execution


@pytest.mark.parametrize("smoke", [False, True])
def test_sampling_mode_preserves_manager_selected_production_time(tmp_path, monkeypatch, smoke):
    def plan(llm, role, evidence, schema, instruction, *_):
        assert evidence["sampling_mode"] == ("smoke" if smoke else "production")
        assert ("20–40 ps" in instruction) is smoke
        return SimulationPlan(
            action="run",
            system="water",
            target_property="msd",
            method="equilibrated_md",
            phase="liquid",
            temperature_k=298,
            pressure_bar=1,
            repeat=6,
            ensemble="nvt",
            equilibration_ensemble="npt",
            timestep_fs=1,
            equilibration_steps=10000 if smoke else 250000,
            production_steps=20000 if smoke else 1000000,
            reason="Use fixed volume for the displacement trajectory.",
        )

    monkeypatch.setattr("mdgraph.natural._invoke", plan)
    monkeypatch.setattr("mdgraph.natural.shutil.which", lambda _: "/usr/bin/lmp")
    cfg = task_from_request("Calculate the MSD of bulk water", smoke=smoke)
    assert cfg.protocol.steps == (20000 if smoke else 1000000)
    assert cfg.protocol.sampling_mode == ("smoke" if smoke else "production")
    cfg.potential.water_model = "spce"
    prepare(cfg, tmp_path)
    script = (tmp_path / "in.lammps").read_text()
    assert script.index("fix integrator all npt") < script.index("MDGRAPH_EQUILIBRATION_END")
    assert script.index("MDGRAPH_EQUILIBRATION_END") < script.index("fix integrator all nvt")


def test_windowed_msd_matches_explicit_time_origins():
    path = np.random.default_rng(7).normal(size=(17, 4, 3)).cumsum(axis=0)
    expected = [0.0] + [
        np.sum((path[lag:] - path[:-lag]) ** 2, axis=-1).mean() for lag in range(1, len(path))
    ]
    assert windowed_msd(path) == pytest.approx(expected, abs=1e-12)


def test_sampling_diagnostics_reject_ballistic_water_displacement():
    times = np.arange(90, dtype=float)
    centers = np.zeros((len(times), 2, 3))
    centers[:, 0, 0], centers[:, 1, 0] = times, -times
    rows, check = analyze_windowed_msd(
        np.repeat(centers, 3, axis=1), times, molecular_water=True, tolerance=0.15
    )
    assert rows[1]["msd_angstrom2"] == pytest.approx(1)
    assert check["block_log_slopes"] == pytest.approx([2, 2, 2])
    assert not check["passed"]
