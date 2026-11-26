from argparse import Namespace
from threading import Event

import pytest

from mdgraph.artifacts import atomic_json, read_json
from mdgraph.chat import chat
from mdgraph.config import Execution
from mdgraph.natural import ManagerTurn, SimulationPlan


def test_background_review_takes_status_while_manager_is_busy(tmp_path, monkeypatch, capsys):
    started, resume = Event(), Event()
    inputs = iter(["Analyse water", "/status", "/confirm", "/wait", "/exit"])

    def manager(request, options, *, review, progress, **context):
        assert request == "Analyse water" and review
        progress("Research 1/3: retrieving full text")
        started.set()
        assert resume.wait(3)
        return ManagerTurn(action="reply", message="Research complete.")

    def user_input(_):
        message = next(inputs)
        if message == "/status":
            assert started.wait(3)
        if message == "/wait":
            output = capsys.readouterr().out
            assert "Research 1/3: retrieving full text" in output
            assert "A request is still being planned" in output
            resume.set()
        return message

    monkeypatch.setattr("builtins.input", user_input)
    monkeypatch.setattr("mdgraph.chat.natural_run", manager)
    chat(Namespace(run=None, review=True, background_planning=True, runs_root=tmp_path))
    assert "Research complete." in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "message, conditions",
    [
        ("calculate the lattice parameter of Cu", {}),
        ("在0 K和0 bar下计算铜的晶格常数", {"temperature_k": 0.0, "pressure_bar": 0.0}),
    ],
)
def test_manager_plans_and_handles_followups_in_chat(
    tmp_path, monkeypatch, capsys, message, conditions
):
    inputs = iter([message, "算完了吗？", "给我结果", "晶格常数怎么计算？", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("mdgraph.cli.execution_profile", lambda *args: Execution())
    requests, launches, watches = [], [], []

    def manager(llm, role, evidence, schema, *_):
        assert role == "manager" and schema is ManagerTurn
        requests.append(evidence["request"])
        context = evidence["conversation"]
        if evidence["request"] == message:
            assert context["active_run"] is None
            assert evidence["explicit_constraints"] == conditions
            plan = SimulationPlan(
                action="run",
                system="elemental_crystal",
                element="Cu",
                target_property="lattice_constant",
                method="static_cell_relaxation",
                phase="fcc",
                temperature_k=0,
                pressure_bar=0,
                initial_a_angstrom=3.615,
                repeat=3,
                reason="Relax the FCC cell at zero external pressure.",
            )
            return ManagerTurn(action="run", plan=plan)
        assert context["active_run"]["job_id"] == "12.admin"
        assert context["active_run"]["target_property"] == "lattice_constant"
        assert context["recent_turns"][0]["user"] == message
        if evidence["request"] == "晶格常数怎么计算？":
            assert context["recent_turns"][-1]["reply"].startswith("Running in PBS")
            return ManagerTurn(
                action="reply", message="Relax the cell and measure its equilibrium size."
            )
        return ManagerTurn(action="status" if evidence["request"] == "算完了吗？" else "result")

    def drive_fixture(path):
        launches.append(path)
        plan = read_json(path / "manager_plan.json")
        assert plan["evidence"]["request"] == message
        assert plan["proposal"]["method"] == "static_cell_relaxation"
        cfg = read_json(path / "task.json")
        assert cfg["goal"] == message and cfg["structure"]["elements"] == ["Cu"]
        receipt = {"job_id": "12.admin", "queue": "cpu"}
        atomic_json(path / "status.json", {"status": "submitted", "job": receipt, "attempt": 0})
        return {"status": "submitted", "receipt": receipt}

    monkeypatch.setattr("mdgraph.natural._invoke", manager)
    monkeypatch.setattr("mdgraph.chat.drive", drive_fixture)
    monkeypatch.setattr("mdgraph.chat.start_watch", watches.append)
    monkeypatch.setattr(
        "mdgraph.chat.get_backend",
        lambda _: Namespace(
            poll=lambda *_: {"status": "running", "scheduler_state": "R", "compute_host": "node43"}
        ),
    )
    chat(
        Namespace(
            run=None, system=None, execution=None, lammps=None, smoke=False, runs_root=tmp_path
        )
    )

    assert requests == [message, "算完了吗？", "给我结果", "晶格常数怎么计算？"]
    assert len(launches) == 1 and watches == launches
    assert list(tmp_path.iterdir()) == launches
    output = capsys.readouterr().out
    assert "Running in PBS (12.admin, cpu, node43)" in output
    assert "Relax the cell and measure its equilibrium size." in output


def test_review_mode_revises_before_creating_a_run(tmp_path, monkeypatch, capsys):
    initial = "Calculate a radial density profile for water in 1212CNT"
    correction = "将管长改为 4.92 nm"
    messages = iter([initial, correction, "/confirm", "/exit"])
    launched = []

    def user_input(_):
        message = next(messages)
        if message != "/exit":
            assert not list(tmp_path.iterdir())
        return message

    def manager(llm, role, evidence, schema, *_):
        revision = evidence["request"] == correction
        expected_pending = evidence["conversation"]["pending_plan"]
        assert bool(expected_pending) is revision
        tube = {
            "chirality": [12, 12],
            "length_angstrom": 49.2 if revision else 39.36,
            "transverse_box_angstrom": 50,
            "water_molecules": 221 if revision else 177,
            "axial_cells": 20 if revision else 16,
        }
        plan = SimulationPlan(
            action="run",
            system="molecular_fluid",
            species="water",
            phase="liquid",
            target_property="radial_density_profile",
            method="cylindrical_mass_histogram",
            cnt_model=tube,
            temperature_k=298.15,
            ensemble="nvt",
            equilibration_ensemble="nvt",
            timestep_fs=2,
            equilibration_steps=100000,
            production_steps=500000,
            reason="Build the specified CNT using fixed water loading.",
        )
        return ManagerTurn(action="revise" if revision else "run", plan=plan)

    def drive_fixture(path):
        task = read_json(path / "task.json")
        assert task["goal"] == initial
        assert task["structure"]["molecular"]["cnt_model"]["length_angstrom"] == 49.2
        launched.append(path)
        return {"status": "prepared"}

    monkeypatch.setattr("builtins.input", user_input)
    monkeypatch.setattr("mdgraph.natural._invoke", manager)
    monkeypatch.setattr("mdgraph.cli.execution_profile", lambda *args: Execution())
    monkeypatch.setattr("mdgraph.chat.drive", drive_fixture)
    chat(
        Namespace(
            run=None,
            system=None,
            execution=None,
            lammps=None,
            smoke=False,
            review=True,
            runs_root=tmp_path,
        )
    )
    assert len(launched) == 1
    output = capsys.readouterr().out
    assert "3.936 nm long" in output and "4.92 nm long" in output
