import numpy as np
import pytest

from mdgraph.analysis_worker import compile_analysis
from mdgraph.config import Execution
from mdgraph.generated_analysis import execute_program, stage_trajectory
from mdgraph.natural import SimulationPlan, default_llm, task_from_plan


@pytest.mark.parametrize(
    "body",
    [
        "import os",
        "return open('some_file')",
        "return np.load('some_file')",
        "return data['positions'].__class__",
    ],
)
def test_script_rejects_non_numerical_access(body):
    with pytest.raises(ValueError):
        compile_analysis("def analyze(data):\n    " + body)


def test_generated_analysis_uses_periodic_vectors_and_preserves_boundaries(tmp_path):
    dump = tmp_path / "trajectory"
    dump.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp ff
0 10
0 10
0 10
ITEM: ATOMS id type x y z ix iy iz
1 1 0.5 0 1 0 0 0
2 1 9.5 0 9 0 0 0
""",
        encoding="utf-8",
    )
    data = tmp_path / "data.npz"
    metadata = stage_trajectory(dump, 2, 1, data)
    assert metadata["periodic_axes"] == [(True, True, False)]
    script = tmp_path / "analysis.py"
    script.write_text(
        "def analyze(data):\n"
        "    delta = data['positions'][0, 1] - data['positions'][0, 0]\n"
        "    _, length = ase.find_mic(delta, data['cell'][0], data['pbc'][0])\n"
        "    return {'columns': {'distance': [length]}, 'units': {'distance': 'angstrom'}}\n",
        encoding="utf-8",
    )
    assert execute_program(script, data)["columns"]["distance"] == pytest.approx([np.sqrt(65)])


def test_custom_trajectory_plan_preserves_original_observable():
    objective = "Mean oxygen-oxygen coordination number within 3.5 angstrom"
    plan = SimulationPlan(
        action="run",
        system="water",
        target_property="custom_trajectory",
        analysis_objective=objective,
        method="equilibrated_md",
        temperature_k=298,
        pressure_bar=1,
        phase="liquid",
        ensemble="nvt",
        equilibration_ensemble="npt",
        repeat=6,
        timestep_fs=1,
        equilibration_steps=10000,
        production_steps=20000,
        water_model="spce",
        reason="Collect a test trajectory for the requested coordination-number estimator.",
    )
    config = task_from_plan(
        plan,
        {"explicit_constraints": {}, "request": objective},
        default_llm(),
        execution=Execution(),
        smoke=True,
    )
    assert config.analysis_objective == objective
    assert config.potential.target_property == "rdf"
