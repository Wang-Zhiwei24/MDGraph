import numpy as np
import pytest

from mdgraph.artifacts import atomic_json, read_json
from mdgraph.cli import create_run, drive
from mdgraph.elastic import STRAINS, analyze_elastic, cubic_projection
from mdgraph.natural import SimulationPlan, task_from_request


def test_natural_gold_uses_manager_conditions_and_refuses_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr("mdgraph.natural.shutil.which", lambda _: "/usr/bin/lmp")

    def propose(llm, role, evidence, *_):
        assert role == "manager"
        assert "request" in evidence and "methods" in evidence
        return SimulationPlan(
            action="run",
            element="Au",
            target_property="elastic_constants",
            method="static_finite_strain",
            phase="fcc",
            temperature_k=0,
            pressure_bar=0,
            initial_a_angstrom=4.17,
            repeat=3,
            reason="Static solid stiffness protocol.",
        )

    monkeypatch.setattr("mdgraph.natural._invoke", propose)
    plans = []
    task = task_from_request("Calculate the elastic constants of gold", on_plan=plans.append)
    assert task.structure.elements == ["Au"] and task.protocol.elastic
    assert task.protocol.temperature_k == 0 and task.protocol.pressure_bar == 0
    assert task.potential.discover == "repositories" and task.potential.catalog is None
    assert task.potential_scope() == {"phase": "fcc", "target_property": "elastic_constants"}
    assert task.structure.crystal.a_angstrom == 4.17 and task.structure.crystal.repeat == (3, 3, 3)
    assert plans[0]["proposal"]["reason"] == "Static solid stiffness protocol."
    assert plans[0]["evidence"]["explicit_constraints"] == {}
    with pytest.raises(ValueError, match="changed explicit"):
        task_from_request("Calculate the elastic constants of gold at 300 K")

    def unexpected_review(*args, **kwargs):
        pytest.fail("The validated manager plan should reach structure without another review")

    def structure_fixture(state):
        assert state["manager_plan"] == plans[0]
        assert state["config"]["structure"]["crystal"]["a_angstrom"] == 4.17
        raise ValueError("structure fixture reached")

    monkeypatch.setattr("mdgraph.graph.decide", unexpected_review)
    monkeypatch.setattr("mdgraph.graph.structure_agent", structure_fixture)
    monkeypatch.setattr("mdgraph.graph.advise", lambda *args: {"summary": "fixture"})
    root = create_run(task, tmp_path, manager_plan=plans[0])
    result = drive(root)
    assert result["validation"]["issues"] == ["structure: structure fixture reached"]
    assert read_json(root / "manager.json")["source"] == "manager_plan.json"
    assert read_json(root / "result.json")["manager_plan"] == plans[0]
    atomic_json(root / "manager_plan.json", {})
    with pytest.raises(ValueError, match="Frozen manager plan"):
        drive(root)


@pytest.mark.parametrize(
    "request_text", ["Calculate the elastic constants of Ag", "用 LAMMPS 算一下块体银的弹性常数"]
)
def test_natural_request_uses_manager_to_identify_material_and_property(request_text, monkeypatch):
    monkeypatch.setattr("mdgraph.natural.shutil.which", lambda _: "/usr/bin/lmp")

    def propose(llm, role, evidence, *_):
        assert role == "manager" and evidence["request"] == request_text
        return SimulationPlan(
            action="run",
            element="Ag",
            target_property="elastic_constants",
            method="static_finite_strain",
            phase="fcc",
            temperature_k=0,
            pressure_bar=0,
            initial_a_angstrom=4.09,
            repeat=3,
            reason="FCC silver stiffness.",
        )

    monkeypatch.setattr("mdgraph.natural._invoke", propose)
    task = task_from_request(request_text)
    assert task.structure.elements == ["Ag"] and task.protocol.elastic
    assert task.potential.discover == "repositories"


def test_natural_request_reports_manager_reason_for_unsupported_physics(monkeypatch):
    def propose(llm, role, evidence, *_):
        assert evidence["request"] == "Calculate the RDF of bulk water"
        return SimulationPlan(action="unsupported", reason="Molecular water RDF is not supported.")

    monkeypatch.setattr("mdgraph.natural._invoke", propose)
    monkeypatch.setattr(
        "mdgraph.natural.shutil.which", lambda _: pytest.fail("No solver should be needed")
    )
    with pytest.raises(ValueError, match="Molecular water RDF"):
        task_from_request("Calculate the RDF of bulk water")


def test_elastic_parser_pressure_sign_shear_and_sample_integrity(tmp_path):
    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = 85
    np.fill_diagonal(stiffness, [120] * 3 + [40] * 3)
    lines = []
    for level, eps in enumerate(STRAINS):
        for mode in range(6):
            for sign in (-1, 1):
                pressure = -10000 * stiffness[:, mode] * eps * sign + 2
                lines.append(
                    " ".join(map(str, ["MDGRAPH_ELASTIC", level, mode, sign, *pressure, 0, 256]))
                )
    metrics, issues = analyze_elastic("\n".join(lines), tmp_path, 256, 1e-6)
    assert not issues
    assert metrics["elastic_constants_gpa"] == pytest.approx({"C11": 120, "C12": 85, "C44": 40})
    assert metrics["bulk_modulus_gpa"] == pytest.approx((120 + 2 * 85) / 3)
    assert cubic_projection(stiffness)[1] == pytest.approx(stiffness)
    with pytest.raises(ValueError, match="24"):
        analyze_elastic("\n".join(lines[:-1]), tmp_path, 256, 1e-6)
