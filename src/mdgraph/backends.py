from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

from .artifacts import atomic_json, read_json
from .config import Execution

OUTPUTS = [
    "log.lammps",
    "screen.log",
    "runner.log",
    "trajectory.lammpstrj",
    "final.data",
    "final.restart",
]


class AmbiguousSubmission(RuntimeError):
    """A job may already exist; automatically resubmitting would duplicate work."""


def job_script(cfg: Execution, job_name: str, remote_dir: str) -> str:
    if cfg.scheduler == "slurm":
        headers = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --nodes={cfg.nodes}",
            f"#SBATCH --ntasks={cfg.tasks}",
            f"#SBATCH --cpus-per-task={cfg.cpus_per_task}",
            f"#SBATCH --time={cfg.walltime}",
            f"#SBATCH --mem={cfg.memory_gb}G",
            "#SBATCH --output=scheduler.out",
            "#SBATCH --error=scheduler.err",
        ]
        if cfg.partition:
            headers.append(f"#SBATCH --partition={cfg.partition}")
        if cfg.account:
            headers.append(f"#SBATCH --account={cfg.account}")
        if cfg.gpus:
            headers.append(f"#SBATCH --gpus={cfg.gpus}")
    else:
        if cfg.tasks % cfg.nodes:
            raise ValueError("PBS: tasks must be divisible by nodes")
        cores = cfg.cpus_per_task * cfg.tasks // cfg.nodes
        if cfg.pbs_resources == "nodes":
            if cfg.gpus:
                raise ValueError("Legacy PBS nodes/ppn profile supports CPU jobs only")
            resources = [f"nodes={cfg.nodes}:ppn={cores}", f"mem={cfg.memory_gb * cfg.nodes}gb"]
        else:
            resource = f"select={cfg.nodes}:ncpus={cores}:mpiprocs={cfg.tasks // cfg.nodes}:mem={cfg.memory_gb}gb"
            if cfg.gpus:
                if cfg.gpus % cfg.nodes:
                    raise ValueError("PBS: GPUs must be divisible by nodes")
                resource += f":ngpus={cfg.gpus // cfg.nodes}"
            resources = [resource]
        headers = [
            f"#PBS -N {job_name}",
            *(f"#PBS -l {item}" for item in resources),
            f"#PBS -l walltime={cfg.walltime}",
            "#PBS -o scheduler.out",
            "#PBS -e scheduler.err",
        ]
        if cfg.partition:
            headers.append(f"#PBS -q {cfg.partition}")
        if cfg.account:
            headers.append(f"#PBS -A {cfg.account}")
    argv = cfg.command
    if cfg.mpi_launcher:
        argv = [*cfg.mpi_launcher, "-n", str(cfg.tasks), *argv]
    command = shlex.join(argv + ["-in", "in.lammps", "-log", "log.lammps", "-screen", "screen.log"])
    compute_record = (
        f"export OMP_NUM_THREADS={cfg.cpus_per_task}\nhostname > compute_host.txt\n"
        if cfg.backend == "cluster"
        else ""
    )
    return (
        "#!/bin/bash\n" + "\n".join(headers) + "\nset -euo pipefail\n"
        f"cd -- {shlex.quote(remote_dir)}\n"
        "finish() { rc=$?; trap - EXIT; printf '%s\\n' \"$rc\" > .exit_code.tmp; "
        'mv -- .exit_code.tmp .exit_code; exit "$rc"; }\ntrap finish EXIT\n'
        + compute_record
        + "\n".join(cfg.setup)
        + "\n"
        + f"{command} > runner.log 2>&1\n"
    )


class LocalBackend:
    def __init__(self, cfg: Execution):
        self.cfg = cfg

    def submit(self, folder: Path, run_id: str) -> dict:
        receipt = folder / "job.json"
        if receipt.exists():
            return read_json(receipt)
        try:
            fd = os.open(folder / ".submission_started", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError as e:
            raise AmbiguousSubmission(
                "Local launch started without receipt; inspect process before recovery"
            ) from e
        command = self.cfg.command + [
            "-in",
            "in.lammps",
            "-log",
            "log.lammps",
            "-screen",
            "screen.log",
        ]
        if command[0] == "python":
            command[0] = sys.executable
        atomic_json(
            folder / "worker.json", {"command": command, "timeout": self.cfg.timeout_seconds}
        )
        with (folder / "worker.log").open("wb") as output:
            proc = subprocess.Popen(
                [sys.executable, "-m", "mdgraph.worker", str(folder.resolve())],
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        data = {"job_id": str(proc.pid), "backend": "local", "run_id": run_id}
        atomic_json(receipt, data)
        return data

    def poll(self, folder: Path, receipt: dict) -> dict:
        path = folder / "exit.json"
        if path.exists():
            result = read_json(path)
            return {"status": "completed" if result["returncode"] == 0 else "failed", **result}
        return {"status": "running", "job_id": receipt["job_id"]}

    def collect(self, folder: Path, receipt: dict):
        return


class SSHBackend:
    def __init__(self, cfg: Execution):
        self.cfg = cfg

    def connect(self):
        import paramiko

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(
                hostname=self.cfg.host,
                port=self.cfg.port,
                username=self.cfg.username,
                key_filename=self.cfg.key_filename,
                timeout=20,
                banner_timeout=20,
                auth_timeout=20,
                allow_agent=True,
                look_for_keys=True,
            )
        except Exception:
            client.close()
            raise
        return client

    @staticmethod
    def command(client, text: str) -> tuple[int, str, str]:
        _, out, err = client.exec_command(text, timeout=45)
        stdout, stderr = (
            out.read().decode("utf-8", "replace"),
            err.read().decode("utf-8", "replace"),
        )
        return out.channel.recv_exit_status(), stdout.strip(), stderr.strip()

    def remote_dir(self, run_id: str, attempt: str) -> str:
        return str(PurePosixPath(self.cfg.remote_root) / run_id / attempt)

    def submit(self, folder: Path, run_id: str) -> dict:
        receipt_path = folder / "job.json"
        if receipt_path.exists():
            return read_json(receipt_path)
        remote = self.remote_dir(run_id, folder.name)
        short = "mdg_" + run_id[-8:] + "_" + folder.name[-2:]
        script = job_script(self.cfg, short, remote)
        (folder / "job.sh").write_text(script, encoding="utf-8", newline="\n")
        with self.connect() as client:
            code, _, err = self.command(client, f"mkdir -p -- {shlex.quote(remote)}")
            if code:
                raise RuntimeError(f"Remote mkdir failed: {err}")
            with client.open_sftp() as ftp:
                try:
                    with ftp.open(remote + "/.job_id") as f:
                        existing = f.read().decode().strip()
                except FileNotFoundError:
                    existing = None
                if existing:
                    return self._save_receipt(receipt_path, existing, remote)
                # Do not rewrite a possibly queued/running job after a previous uncertain submission.
                try:
                    ftp.stat(remote + "/.submit_lock")
                    raise AmbiguousSubmission(
                        f"Uncertain remote submission in {remote}; inspect scheduler and .job_id"
                    )
                except FileNotFoundError:
                    pass
                manifest = read_json(folder / "manifest.json")
                for name in [*manifest["files"], "manifest.json", "job.sh"]:
                    ftp.put(str(folder / name), remote + "/" + name)
            submit = "sbatch --parsable job.sh" if self.cfg.scheduler == "slurm" else "qsub job.sh"
            # Atomic remote claim prevents duplicates even if Python dies after scheduler submission.
            command = (
                f"cd -- {shlex.quote(remote)} && "
                "if test -s .job_id; then cat .job_id; "
                "elif mkdir .submit_lock 2>/dev/null; then "
                f"if {submit} > .job_id.tmp 2> .submit_error; then "
                "mv -- .job_id.tmp .job_id; cat .job_id; "
                "else cat .submit_error >&2; touch .submit_failed; exit 71; fi; "
                "else echo 'Submission already started; inspect remote directory' >&2; exit 72; fi"
            )
            code, output, err = self.command(client, command)
            if code:
                raise AmbiguousSubmission(
                    f"Scheduler submission not confirmed ({code}): {err}; remote={remote}"
                )
            return self._save_receipt(receipt_path, output, remote)

    def _save_receipt(self, path, raw, remote):
        job_id = raw.strip().split(";")[0]
        if not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9_.-]+)?", job_id):
            raise AmbiguousSubmission(f"Unrecognized scheduler receipt; inspect {remote}/.job_id")
        receipt = {
            "job_id": job_id,
            "backend": "ssh",
            "scheduler": self.cfg.scheduler,
            "remote_dir": remote,
        }
        atomic_json(path, receipt)
        return receipt

    def poll(self, folder: Path, receipt: dict) -> dict:
        remote = receipt["remote_dir"]
        with self.connect() as client:
            code, output, _ = self.command(client, f"cat -- {shlex.quote(remote + '/.exit_code')}")
            if code == 0:
                rc = int(output)
                return {"status": "completed" if rc == 0 else "failed", "returncode": rc}
            job_id = receipt["job_id"]
            if self.cfg.scheduler == "slurm":
                code, state, err = self.command(client, f"squeue -h -j {shlex.quote(job_id)} -o %T")
                if code:
                    raise RuntimeError(f"Cannot query Slurm: {err}")
                if state:
                    return {"status": "running", "scheduler_state": state}
                code, state, _ = self.command(
                    client, f"sacct -n -P -j {shlex.quote(job_id)} --format=State"
                )
                states = [s.split("|")[0].split()[0] for s in state.splitlines() if s.strip()]
                transient = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
                if states and states[0] in transient:
                    return {"status": "failed", "scheduler_state": states[0], "retryable": True}
                if states and states[0] in {
                    "FAILED",
                    "TIMEOUT",
                    "CANCELLED",
                    "OUT_OF_MEMORY",
                    "COMPLETED",
                }:
                    return {
                        "status": "failed",
                        "scheduler_state": states[0],
                        "retryable": False,
                        "reason": "Scheduler terminal but execution completion marker missing",
                    }
            else:
                code, text, err = self.command(client, f"qstat -f {shlex.quote(job_id)}")
                if code == 0:
                    match = re.search(r"job_state\s*=\s*(\w+)", text)
                    state = match.group(1) if match else "unknown"
                    if state in {"F", "C"}:
                        return {
                            "status": "failed",
                            "scheduler_state": state,
                            "reason": "PBS finished without execution completion marker",
                        }
                    return {"status": "running", "scheduler_state": state}
                if "Unknown Job" in err or "Unknown Job" in text:
                    return {
                        "status": "failed",
                        "reason": "PBS job missing and no exit marker; inspect cluster",
                    }
                raise RuntimeError(f"Cannot query PBS: {err or text}")
        # Accounting delays are possible. Keep pending; controller's watch timeout is bounded.
        return {
            "status": "unknown",
            "reason": "No active job, completion marker or accounting record yet",
        }

    def collect(self, folder: Path, receipt: dict):
        with self.connect() as client, client.open_sftp() as ftp:
            for name in OUTPUTS + ["scheduler.out", "scheduler.err", ".submit_error"]:
                try:
                    ftp.get(receipt["remote_dir"] + "/" + name, str(folder / name))
                except FileNotFoundError:
                    continue


def get_backend(cfg: Execution):
    if cfg.backend == "cluster":
        from .pbs_cluster import PBSClusterBackend

        return PBSClusterBackend(cfg)
    return SSHBackend(cfg) if cfg.backend == "ssh" else LocalBackend(cfg)
