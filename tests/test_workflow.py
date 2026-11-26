import json

import numpy as np
import pytest
from ase.io import read

from mdgraph.analysis import analyze, dump_frames, parse_thermo
from mdgraph.artifacts import atomic_json, sha256
from mdgraph.backends import AmbiguousSubmission, LocalBackend, job_script
from mdgraph.cli import create_run, drive, run_lock
from mdgraph.config import Execution, TaskConfig
from mdgraph.prepare import prepare, render_input


@pytest.fixture
def cfg():
    return TaskConfig.model_validate(
        {
            "name": "test_md",
            "structure": {"demo_argon": True, "elements": ["Ar"]},
            "potential": {"kind": "lj_demo"},
            "protocol": {
                "steps": 40,
                "thermo_every": 10,
                "dump_every": 10,
                "temperature_k": 50,
                "timestep_fs": 1,
            },
        }
    )


def simulated_outputs(folder, cfg, complete=True, production_start=0):
    """Synthetic solver output only for workflow tests, never used as physical evidence."""
    atoms = read(folder / "initial.extxyz")
    n = len(atoms)
    steps = [0, 10, 20, 30, 40] if complete else [0, 10, 20]
    text = "Step Time Temp PotEng KinEng TotEng Press Volume Atoms\n"
    for step in steps:
        text += f"{step} {production_start + step / 1000} 50 -10 1 -9 0 {atoms.get_volume()} {n}\n"
    text += "Loop time of 0.1\n" + ("MDGRAPH_COMPLETED\n" if complete else "")
    (folder / "log.lammps").write_text(text)
    (folder / "final.data").write_text("TEST FIXTURE ONLY")
    (folder / "final.restart").write_bytes(b"TEST FIXTURE ONLY")
    chunks = []
    for step in steps:
        chunks += [
            "ITEM: TIMESTEP",
            str(step),
            "ITEM: NUMBER OF ATOMS",
            str(n),
            "ITEM: BOX BOUNDS pp pp pp",
        ]
        chunks += [f"0 {length}" for length in atoms.cell.lengths()]
        chunks += ["ITEM: ATOMS id type x y z ix iy iz"]
        chunks += [
            f"{i + 1} 1 {x + step * 0.0001} {y} {z} 0 0 0"
            for i, (x, y, z) in enumerate(atoms.positions)
        ]
    (folder / "trajectory.lammpstrj").write_text("\n".join(chunks) + "\n")


def test_units_and_integrator(cfg):
    source = render_input(cfg)
    assert "timestep 0.001" in source
    assert "nvt temp 50.0 50.0 0.1" in source
    assert "thermo_modify flush yes lost error" in source


def test_no_implicit_classical_fallback():
    with pytest.raises(ValueError, match="LJ is confined"):
        TaskConfig.model_validate(
            {
                "name": "bad",
                "structure": {"path": "a.xyz", "elements": ["C"]},
                "potential": {"kind": "lj_demo"},
            }
        )


def test_finite_and_unknown_config(cfg):
    raw = cfg.model_dump()
    raw["protocol"]["temperature_k"] = float("nan")
    with pytest.raises(ValueError):
        TaskConfig.model_validate(raw)
    raw = cfg.model_dump()
    raw["protocol"]["unknown"] = 1
    with pytest.raises(ValueError):
        TaskConfig.model_validate(raw)


def test_preparation_hash_and_mapping(cfg, tmp_path):
    manifest = prepare(cfg, tmp_path)
    assert manifest["atoms"] == 256
    original = (tmp_path / "in.lammps").read_text()
    (tmp_path / "in.lammps").write_text(original + "# changed")
    with pytest.raises(ValueError, match="changed"):
        prepare(cfg, tmp_path)


def test_model_metadata_type_order(tmp_path):
    model = tmp_path / "model.pb"
    model.write_bytes(b"test model")
    meta = tmp_path / "meta.json"
    atomic_json(meta, {"model_sha256": sha256(model), "elements": ["Si"], "units": "metal"})
    cfg = TaskConfig.model_validate(
        {
            "name": "badmap",
            "structure": {"path": "missing.xyz", "elements": ["C"]},
            "potential": {"kind": "deepmd", "model": str(model), "metadata": str(meta)},
        }
    )
    with pytest.raises(ValueError, match="mapping|mismatch"):
        prepare(cfg, tmp_path / "run")


def test_production_table_separation():
    text = (
        "Step Temp E_pair\n0 50 -10\nLoop time\nStep Time Temp\n0 0 50\n10 0.01 51\n"
        "WARNING: harmless diagnostic\n20 0.02 50\nLoop time\n"
    )
    rows = parse_thermo(text)
    assert len(rows) == 3 and rows[-1]["Step"] == 20


def test_partial_run_not_accepted(cfg, tmp_path):
    prepare(cfg, tmp_path)
    simulated_outputs(tmp_path, cfg, complete=False)
    result = analyze(cfg, tmp_path, 256)
    assert result["passed"] is False
    assert any("completion" in issue for issue in result["issues"])
    assert any("step count" in issue for issue in result["issues"])


@pytest.mark.parametrize("production_start", [0, 10])
def test_numerical_checks_and_trajectory(cfg, tmp_path, production_start):
    cfg.protocol.equilibration_steps = int(production_start * 1000)
    prepare(cfg, tmp_path)
    simulated_outputs(tmp_path, cfg, production_start=production_start)
    result = analyze(cfg, tmp_path, 256)
    assert result["passed"], result
    assert result["metrics"]["dump_frames"] == 5
    assert result["metrics"]["duration_ps"] == pytest.approx(0.04)
    assert result["metrics"]["final_msd_angstrom2"] == pytest.approx(0.004**2)
    assert (tmp_path / "rdf.csv").is_file()


def test_nan_and_lost_atoms_fail(cfg, tmp_path):
    prepare(cfg, tmp_path)
    simulated_outputs(tmp_path, cfg)
    p = tmp_path / "log.lammps"
    p.write_text(p.read_text().replace(" 50 ", " nan ", 1))
    assert not analyze(cfg, tmp_path, 256)["passed"]
    simulated_outputs(tmp_path, cfg)
    p.write_text(p.read_text().replace(" 256\n", " 255\n", 1))
    assert not analyze(cfg, tmp_path, 256)["passed"]


def test_dump_unwrap_triclinic(tmp_path):
    path = tmp_path / "dump"
    path.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS xy xz yz pp pp pp\n"
        "0 13 2\n0 13 1\n0 10 3\nITEM: ATOMS id type x y z ix iy iz\n"
        "1 1 1 1 1 1 1 0\n2 1 2 2 2 0 0 0\n"
    )
    frame = next(dump_frames(path))
    np.testing.assert_allclose(frame[-1], [[10, 0, 0], [2, 10, 0], [1, 3, 10]])
    np.testing.assert_allclose(frame[4][0], [13, 11, 1])


def test_scheduler_scripts_and_config():
    slurm = Execution(
        backend="ssh",
        scheduler="slurm",
        host="h",
        username="u",
        remote_root="/work space/md",
        command=["srun", "/opt/my lmp"],
        gpus=1,
    )
    script = job_script(slurm, "mdg_test", "/work space/md/run")
    assert "#SBATCH --gpus=1" in script
    assert "'/opt/my lmp'" in script
    assert "trap finish EXIT" in script
    assert "cd -- '/work space/md/run'" in script
    pbs = slurm.model_copy(update={"scheduler": "pbs", "gpus": 0})
    assert "#PBS -l select=" in job_script(pbs, "mdg_test", "/work")
    with pytest.raises(ValueError):
        Execution(backend="ssh", scheduler="slurm", host="h", username="u", remote_root="relative")


def test_launch_receipt_prevents_duplicate(cfg, tmp_path, monkeypatch):
    backend = LocalBackend(cfg.execution)
    atomic_json(tmp_path / "job.json", {"job_id": "123", "backend": "local"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing launch must never create a process")

    monkeypatch.setattr("mdgraph.backends.subprocess.Popen", forbidden)
    assert backend.submit(tmp_path, "test")["job_id"] == "123"


def test_ambiguous_local_launch_refused(cfg, tmp_path):
    (tmp_path / ".submission_started").touch()
    with pytest.raises(AmbiguousSubmission):
        LocalBackend(cfg.execution).submit(tmp_path, "test")


def test_checkpoint_resume_queries_existing_job(cfg, tmp_path, monkeypatch):
    class Backend:
        submissions = 0
        ready = False

        def submit(self, folder, run_id):
            self.submissions += 1
            return {"job_id": "test-fixture", "backend": "test"}

        def poll(self, folder, receipt):
            if not self.ready:
                return {"status": "running"}
            simulated_outputs(folder, cfg)
            return {"status": "completed", "returncode": 0}

        def collect(self, folder, receipt):
            pass

    backend = Backend()
    monkeypatch.setattr("mdgraph.graph.get_backend", lambda _: backend)
    root = create_run(cfg, tmp_path)
    pending = drive(root)
    assert pending["status"] == "submitted"
    backend.ready = True
    finished = drive(root)
    assert finished["status"] == "completed"
    assert backend.submissions == 1
    assert drive(root)["status"] == "completed"
    assert backend.submissions == 1


def test_frozen_task_changes_rejected(cfg, tmp_path):
    root = create_run(cfg, tmp_path)
    data = json.loads((root / "task.json").read_text())
    data["protocol"]["steps"] = 100
    atomic_json(root / "task.json", data)
    with pytest.raises(ValueError, match="Frozen"):
        drive(root)


def test_lock_excludes_second_controller(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another controller"):
            with run_lock(tmp_path):
                pass


def test_bounded_infrastructure_retry_keeps_attempts(cfg, tmp_path, monkeypatch):
    class Backend:
        submissions = 0

        def submit(self, folder, run_id):
            self.submissions += 1
            return {"job_id": str(self.submissions), "backend": "test"}

        def poll(self, folder, receipt):
            if folder.name == "attempt_00":
                return {"status": "failed", "scheduler_state": "NODE_FAIL", "retryable": True}
            simulated_outputs(folder, cfg)
            return {"status": "completed", "returncode": 0}

        def collect(self, folder, receipt):
            pass

    backend = Backend()
    monkeypatch.setattr("mdgraph.graph.get_backend", lambda _: backend)
    root = create_run(cfg, tmp_path)
    result = drive(root)
    assert result["status"] == "completed" and result["attempt"] == 1
    assert backend.submissions == 2
    assert (root / "attempt_00" / "diagnosis.json").is_file()
    assert sha256(root / "attempt_00" / "in.lammps") == sha256(root / "attempt_01" / "in.lammps")


def test_llm_enabled_needs_explicit_credentials(cfg, monkeypatch):
    from mdgraph.agents import advise

    cfg.llm.enabled = True
    cfg.llm.model = "test"
    monkeypatch.delenv("MDGRAPH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM enabled"):
        advise(cfg, "manager", {})
