from subprocess import CompletedProcess

import pytest

from mdgraph.backends import get_backend, job_script
from mdgraph.config import Execution


def test_pbs_records_busy_nodes_and_submits_once(tmp_path, monkeypatch):
    cfg = Execution(backend="cluster", scheduler="pbs", partition="cpu", pbs_resources="nodes")
    engine = get_backend(cfg)
    folder = tmp_path / "attempt_00"
    folder.mkdir()
    queue = (
        "Queue: cpu\n    enabled = True\n    started = True\n    acl_hosts = node41,node42\n"
        "    state_count = Queued:2 Running:9\n    max_running = 10\n"
    )
    nodes = (
        "node41\n    state = job-exclusive\n    np = 64\n\n"
        "node42\n    state = job-exclusive\n    np = 64\n\n"
        "unrelated\n    state = free\n    np = 64\n"
    )
    replies = {
        ("qstat", "-Qf", "cpu"): queue,
        ("pbsnodes", "node41", "node42"): nodes,
        ("qstat", "-f", "123.admin"): "Job Id: 123.admin\n    job_state = Q\n",
    }
    monkeypatch.setattr(
        engine, "query", lambda args: CompletedProcess(args, 0, replies[tuple(args)], "")
    )
    snapshot = engine.probe(folder)
    assert snapshot["idle_nodes"] == [] and len(snapshot["nodes"]) == 2
    assert snapshot["job_counts"] == {"queued": 2, "running": 9}
    calls = []

    def qsub(args, *, cwd, stdout, **kwargs):
        calls.append(args)
        assert cwd == folder
        stdout.write(b"123.admin\n")
        return CompletedProcess(args, 0)

    monkeypatch.setattr("mdgraph.pbs_cluster.subprocess.run", qsub)
    receipt = engine.submit(folder, "some_run")
    assert receipt["backend"] == "cluster" and receipt["queue"] == "cpu"
    (folder / "job.json").unlink()
    assert engine.submit(folder, "some_run") == receipt
    assert calls == [["qsub", "job.sh"]]
    script = (folder / "job.sh").read_text()
    assert "#PBS -q cpu" in script and "#PBS -l nodes=1:ppn=1" in script
    assert not (folder / "worker.json").exists()
    assert engine.poll(folder, receipt)["scheduler_state"] == "Q"
    (folder / "compute_host.txt").write_text("node41\n")
    (folder / ".exit_code").write_text("0\n")
    assert engine.poll(folder, receipt) == {
        "status": "completed",
        "returncode": 0,
        "compute_host": "node41",
    }


@pytest.mark.parametrize(
    ("cpu_state", "testcpu_limit", "selected"),
    [
        ("free", 32, "cpu"),
        ("job-exclusive", 32, "testcpu"),
        ("job-exclusive", 1, "cpu"),
    ],
)
def test_pbs_fallback_requires_capacity_and_resource_limits(
    tmp_path, monkeypatch, cpu_state, testcpu_limit, selected
):
    cfg = Execution(
        backend="cluster",
        scheduler="pbs",
        partition="cpu",
        fallback_partitions=["testcpu"],
        cpus_per_task=2,
    )
    engine = get_backend(cfg)
    queried = []

    def query(args):
        queried.append(tuple(args))
        if args[0] == "qstat":
            queue = args[-1]
            node = "node41" if queue == "cpu" else "node43"
            limit = 128 if queue == "cpu" else testcpu_limit
            text = (
                f"Queue: {queue}\n    enabled = True\n    started = True\n    acl_hosts = {node}\n"
                f"    resources_max.ncpus = {limit}\n    max_running = 10\n    state_count = Running:0\n"
            )
        else:
            node = args[-1]
            state = cpu_state if node == "node41" else "free"
            text = f"{node}\n    state = {state}\n    np = 64\n"
        return CompletedProcess(args, 0, text, "")

    monkeypatch.setattr(engine, "query", query)
    result = engine.probe(tmp_path)
    assert result["queue"] == selected
    assert (("qstat", "-Qf", "testcpu") in queried) == (cpu_state != "free")


@pytest.mark.parametrize(
    ("pbs_exit", "expected"), [("0", "completed"), ("-1", "failed"), ("", "unknown")]
)
def test_pbs_terminal_status_before_shared_marker(tmp_path, monkeypatch, pbs_exit, expected):
    engine = get_backend(Execution(backend="cluster", scheduler="pbs", partition="testcpu"))
    result = f"Job Id: 123.admin\n    job_state = C\n    exec_host = node43/0\n    exit_status = {pbs_exit}\n"
    monkeypatch.setattr(engine, "query", lambda args: CompletedProcess(args, 0, result, ""))
    assert engine.poll(tmp_path, {"job_id": "123.admin"})["status"] == expected


def test_pbs_mpi_uses_requested_process_count():
    cfg = Execution(
        backend="cluster",
        scheduler="pbs",
        pbs_resources="nodes",
        partition="cpu",
        tasks=32,
        mpi_launcher=["/mpi/mpiexec", "-launcher", "fork"],
        command=["/solver/lmp"],
    )
    script = job_script(cfg, "mdg_test", "/work/run")
    assert "#PBS -l nodes=1:ppn=32" in script
    assert "export OMP_NUM_THREADS=1" in script
    assert "/mpi/mpiexec -launcher fork -n 32 /solver/lmp -in in.lammps" in script
