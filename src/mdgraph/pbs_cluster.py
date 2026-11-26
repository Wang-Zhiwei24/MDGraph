"""Submit to PBS from a login node with a shared project filesystem."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_json, read_json
from .backends import AmbiguousSubmission, job_script
from .config import Execution


def pbs_attributes(text: str) -> dict[str, str]:
    attrs = {}
    key = None
    for line in text.splitlines():
        match = re.match(r"\s+([\w.]+)\s*=\s*(.*)", line)
        if match:
            key, value = match.groups()
            attrs[key] = value
        elif key and line.startswith("\t"):
            attrs[key] += line.strip()
    return attrs


def integer(value):
    return int(value) if value and value.isdecimal() else None


def walltime_seconds(value: str) -> int | None:
    if not re.fullmatch(r"\d+:\d{2}:\d{2}", value):
        return None
    hours, minutes, seconds = map(int, value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def inspect_nodes(text: str, queue: str, hinted: set[str], cores: int) -> list[dict]:
    nodes = []
    for match in re.finditer(r"(?m)^([\w.-]+)[ \t]*\n((?:[ \t]+[^\n]*\n?)*)", text):
        name, block = match.groups()
        attrs = pbs_attributes(block)
        if attrs.get("queue") not in {queue, None} or ("queue" not in attrs and name not in hinted):
            continue
        state = attrs.get("state", "unknown")
        capacity = integer(attrs.get("resources_available.ncpus", attrs.get("np")))
        assigned = integer(attrs.get("resources_assigned.ncpus"))
        jobs = [item for item in attrs.get("jobs", "").split(",") if item.strip()]
        if assigned is None and capacity is not None:
            assigned = len(jobs)
        available = None if capacity is None or assigned is None else max(0, capacity - assigned)
        free_state = set(state.split(",")) == {"free"}
        nodes.append(
            {
                "node": name,
                "state": state,
                "free_cpu_slots": available if free_state else 0,
                "idle": free_state and assigned == 0,
                "fits_cpu_request": free_state and (available or 0) >= cores,
                "source": "node.queue" if attrs.get("queue") else "queue.acl_hosts (hint)",
            }
        )
    return nodes


class PBSClusterBackend:
    def __init__(self, cfg: Execution):
        self.cfg = cfg

    @staticmethod
    def query(args: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, errors="replace", timeout=25, check=False
            )
        except subprocess.TimeoutExpired as error:
            raise OSError(f"{args[0]} did not respond within 25 seconds") from error

    def probe(self, folder: Path) -> dict:
        """Prefer a queue with capacity and persist the chosen queue for this attempt."""
        path = folder / "queue_probe.json"
        if path.exists():
            return read_json(path)
        observations = []
        immediate = None
        for queue in [self.cfg.partition, *self.cfg.fallback_partitions]:
            try:
                observation = self.probe_queue(queue)
            except (OSError, ValueError) as error:
                observation = {"queue": queue, "accepts_request": False, "rejections": [str(error)]}
            observations.append(observation)
            if observation.get("ready_candidate"):
                immediate = observation
                break
        selected = immediate or next(
            (item for item in observations if item["accepts_request"]), observations[0]
        )
        result = {
            **selected,
            "queues_checked": observations,
            "selection": (
                "capacity_available"
                if immediate
                else "queue_for_resources"
                if selected["accepts_request"]
                else "no_eligible_queue"
            ),
        }
        atomic_json(path, result)
        return result

    def probe_queue(self, queue: str) -> dict:
        info = self.query(["qstat", "-Qf", queue])
        if info.returncode:
            raise ValueError(
                f"Cannot query PBS queue {queue}: {(info.stderr or info.stdout).strip()}"
            )
        attrs = pbs_attributes(info.stdout)
        hinted = {
            name.strip()
            for name in attrs.get("acl_hosts", "").split(",")
            if re.fullmatch(r"[A-Za-z0-9][\w.-]*", name.strip())
        }
        try:
            node_info = self.query(["pbsnodes", *sorted(hinted)] if hinted else ["pbsnodes", "-a"])
            node_error = (
                (node_info.stderr or node_info.stdout).strip() if node_info.returncode else None
            )
        except OSError as error:
            node_error = str(error)
        cores = self.cfg.tasks * self.cfg.cpus_per_task // self.cfg.nodes
        nodes = [] if node_error else inspect_nodes(node_info.stdout, queue, hinted, cores)
        counts = {
            key.lower(): int(count)
            for key, count in re.findall(r"(\w+):(\d+)", attrs.get("state_count", ""))
        }
        max_running = integer(attrs.get("max_running"))
        enabled = attrs.get("enabled", "").lower() == "true"
        started = attrs.get("started", "").lower() == "true"
        max_cores = integer(attrs.get("resources_max.ncpus"))
        max_walltime = walltime_seconds(attrs.get("resources_max.walltime", ""))
        rejections = []
        if not (enabled and started):
            rejections.append("Queue is not enabled and started")
        if max_cores is not None and self.cfg.tasks * self.cfg.cpus_per_task > max_cores:
            rejections.append(f"Request exceeds CPU limit of {max_cores}")
        if max_walltime is not None and walltime_seconds(self.cfg.walltime) > max_walltime:
            rejections.append(
                f"Request exceeds the walltime limit of {attrs['resources_max.walltime']}"
            )
        has_job_slot = max_running is None or (
            "running" in counts and counts["running"] < max_running
        )
        candidates = [node["node"] for node in nodes if node["fits_cpu_request"]]
        result = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "scheduler": "pbs",
            "queue": queue,
            "enabled": enabled,
            "started": started,
            "accepts_request": not rejections,
            "rejections": rejections,
            "ready_candidate": not rejections
            and has_job_slot
            and len(candidates) >= self.cfg.nodes,
            "job_counts": counts,
            "queue_max_running": max_running,
            "requested_cores_per_node": cores,
            "requested_nodes": self.cfg.nodes,
            "nodes": nodes,
            "idle_nodes": [node["node"] for node in nodes if node["idle"]] if nodes else None,
            "cpu_capacity_candidates": candidates if nodes else None,
            "node_query_error": node_error,
            "placement": "PBS schedules the final placement subject to resources and queue policy.",
        }
        return result

    def submit(self, folder: Path, run_id: str) -> dict:
        receipt_path = folder / "job.json"
        if receipt_path.exists():
            return read_json(receipt_path)
        selection = self.probe(folder)
        queue = selection["queue"]
        if not selection.get(
            "accepts_request", selection.get("enabled") and selection.get("started")
        ):
            raise ValueError("No configured PBS queue accepts this job; see queue_probe.json")
        job_id_path = folder / ".job_id"
        if not job_id_path.exists():
            # An ambiguous qsub result requires reconciliation before resubmission.
            try:
                (folder / ".submit_lock").mkdir()
            except FileExistsError as error:
                raise AmbiguousSubmission(
                    f"PBS submission was already started. Inspect {folder}/.job_id* before recovery"
                ) from error
            short = "mdg_" + run_id[-8:] + "_" + folder.name[-2:]
            selected_cfg = self.cfg.model_copy(update={"partition": queue})
            script = job_script(selected_cfg, short, str(folder.resolve()))
            (folder / "job.sh").write_text(script, encoding="utf-8", newline="\n")
            pending = folder / ".job_id.tmp"
            error_log = folder / ".submit_error"
            with pending.open("wb") as output, error_log.open("wb") as err:
                try:
                    result = subprocess.run(
                        ["qsub", "job.sh"],
                        cwd=folder,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=err,
                        timeout=45,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    raise AmbiguousSubmission(
                        f"PBS submission was not confirmed. Inspect {pending} before recovery"
                    ) from error
            if result.returncode:
                raise ValueError(f"qsub failed: {error_log.read_text(errors='replace').strip()}")
            pending.replace(job_id_path)
        job_id = job_id_path.read_text().strip()
        if not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9_.-]+)?", job_id):
            raise AmbiguousSubmission(f"Unrecognized PBS job ID in {job_id_path}")
        receipt = {
            "job_id": job_id,
            "backend": "cluster",
            "scheduler": "pbs",
            "queue": queue,
            "run_id": run_id,
            "workdir": str(folder.resolve()),
        }
        atomic_json(receipt_path, receipt)
        return receipt

    def poll(self, folder: Path, receipt: dict) -> dict:
        marker = folder / ".exit_code"
        host_path = folder / "compute_host.txt"
        host = host_path.read_text().strip() if host_path.is_file() else None
        if marker.is_file():
            rc = int(marker.read_text())
            return {
                "status": "completed" if rc == 0 else "failed",
                "returncode": rc,
                "compute_host": host,
            }
        try:
            info = self.query(["qstat", "-f", receipt["job_id"]])
        except OSError as error:
            return {"status": "unknown", "reason": str(error)}
        if marker.is_file():
            return self.poll(folder, receipt)
        if info.returncode:
            detail = (info.stderr or info.stdout).strip()
            if "unknown job" in detail.lower():
                return {
                    "status": "failed",
                    "reason": "PBS job disappeared without an execution marker",
                }
            return {"status": "unknown", "reason": detail}
        attrs = pbs_attributes(info.stdout)
        state = attrs.get("job_state", "unknown")
        observation = {
            "scheduler_state": state,
            "exec_host": attrs.get("exec_host"),
            "compute_host": host,
        }
        if state in {"C", "F"}:
            # Scheduler completion can become visible before the shared filesystem exit marker.
            raw = attrs.get("exit_status", attrs.get("Exit_status", ""))
            if re.fullmatch(r"-?\d+", raw):
                rc = int(raw)
                return {
                    "status": "completed" if rc == 0 else "failed",
                    "returncode": rc,
                    **observation,
                }
            return {
                "status": "unknown",
                "reason": "Waiting for the PBS exit status or shared execution marker",
                **observation,
            }
        return {"status": "running", **observation}

    def collect(self, folder: Path, receipt: dict) -> None:
        # PBS writes into the same attempt directory on the shared filesystem.
        return
