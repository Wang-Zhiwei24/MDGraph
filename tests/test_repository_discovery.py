import pytest

from mdgraph.cli import create_run, drive
from mdgraph.config import Execution
from mdgraph.natural import SimulationPlan, default_llm, task_from_plan
from mdgraph.repository_discovery import validate_eam
from mdgraph.structure_tools import build_crystal


def test_nial_plan_builds_b2_before_live_selection(tmp_path):
    plan = SimulationPlan(
        action="run",
        system="ordered_crystal",
        elements=["Ni", "Al"],
        phase="b2",
        initial_a_angstrom=2.9,
        repeat=3,
        temperature_k=0,
        pressure_bar=0,
        target_property="lattice_constant",
        method="static_cell_relaxation",
        reason="Relax an ordered cubic B2 cell.",
    )
    task = task_from_plan(
        plan,
        {"request": "NiAl lattice constant", "explicit_constraints": {}},
        default_llm(),
        execution=Execution(),
    )
    assert task.potential.discover == "repositories"
    assert not task.potential.catalog and not task.potential.model
    atoms, _ = build_crystal(task.structure, tmp_path, "ase")
    assert len(atoms) == 54
    assert atoms.symbols.formula.count() == {"Ni": 27, "Al": 27}
    assert atoms.cell.lengths() == pytest.approx([8.7] * 3)


def test_binary_setfl_requires_cross_pair_table():
    data = (
        "comment\ncomment\ncomment\n2 Al Ni\n4 .1 4 .2 .8\n"
        "13 26.98 4.05 FCC\n0 0 0 0 0 0 0 0\n"
        "28 58.69 3.52 FCC\n0 0 0 0 0 0 0 0\n"
        "0 0 0 0 0 0 0 0 0 0 0 0"
    ).encode()
    validate_eam(data, "eam/alloy", ["Ni", "Al"])
    with pytest.raises(ValueError, match="shape"):
        validate_eam(data.rsplit(b" ", 4)[0], "eam/alloy", ["Ni", "Al"])
    with pytest.raises(ValueError, match="species"):
        validate_eam(data, "eam/alloy", ["Ti", "Al"])


def test_unregistered_molecule_search_reaches_potential_and_stops_before_submission(
    tmp_path, monkeypatch
):
    plan = SimulationPlan(
        action="run",
        system="molecular_fluid",
        species="methanol",
        elements=["C", "H", "O"],
        phase="liquid",
        repeat=7,
        temperature_k=298.15,
        pressure_bar=1,
        target_property="self_diffusion",
        method="molecular_msd_fit",
        ensemble="nve",
        equilibration_ensemble="npt",
        timestep_fs=1,
        equilibration_steps=200000,
        production_steps=1000000,
        reason="Look up a force field for the fluid before building it.",
    )
    plans = []
    task = task_from_plan(
        plan,
        {"request": "Methanol self-diffusion", "explicit_constraints": {}},
        default_llm(),
        execution=Execution(),
        on_plan=plans.append,
    )
    called = []

    def snapshot(cfg, folder, *, inventory):
        called.append((cfg.structure.research_species, inventory))
        folder.mkdir()
        return {
            "repositories": {"NIST": {"status": "searched"}, "OpenKIM": {"status": "searched"}},
            "shortlist": [],
        }

    monkeypatch.setattr("mdgraph.graph.discovery_snapshot", snapshot)
    monkeypatch.setattr(
        "mdgraph.graph.advise",
        lambda *args: {"summary": "Topology required", "evidence": [], "limitations": []},
    )
    monkeypatch.setattr(
        "mdgraph.graph.submission", lambda *args: pytest.fail("No topology: must not submit")
    )
    run = create_run(task, tmp_path, manager_plan=plans[0])
    result = drive(run)
    assert called == [("methanol", True)]
    assert result["status"] == "needs_input"
    assert (run / "attempt_00" / "potential_research_review.json").is_file()
