"""Interactive terminal session for planning, submitting, and inspecting MD jobs."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from . import __version__
from .artifacts import read_json
from .backends import get_backend
from .cli import RunDraft, drive, launch_draft, natural_run
from .config import TaskConfig
from .natural import ManagerTurn
from .research import ResearchDraft, describe_research


@contextmanager
def turn_progress():
    started = monotonic()
    finished = Event()
    stage = "Preparing the request"

    def show(next_stage):
        nonlocal stage
        stage = next_stage
        print(f"Working> {stage}", flush=True)

    def heartbeat():
        while not finished.wait(30):
            elapsed = int(monotonic() - started)
            print(f"Working> {stage} ({elapsed} s elapsed)", flush=True)

    watcher = Thread(target=heartbeat, daemon=True)
    watcher.start()
    try:
        yield show
    finally:
        finished.set()
        watcher.join()


class ReviewPlanning:
    """Keep the terminal available while preparing a single review-only reply."""

    def __init__(self, message, options, conversation, reference_evidence, closed):
        self.message = message
        self.future = Future()
        self.waiting = Event()
        self.stage = "Preparing the request"
        self.started = monotonic()
        self.preview = None

        def progress(stage):
            self.stage = stage

        def publish(draft):
            previous = self.preview
            self.preview = draft
            if closed.is_set():
                return
            if draft.record["proposal"] and (previous is None or not previous.record["proposal"]):
                print(
                    "\nAgent> The first reviewed draft is available. Use /plan while the remaining checks run.",
                    flush=True,
                )
            elif previous is None:
                print(
                    "\nAgent> Early research findings are available. Use /plan to read them while checks continue.",
                    flush=True,
                )

        def run():
            try:
                reply = natural_run(
                    message,
                    options,
                    conversation=conversation,
                    review=True,
                    reference_evidence=reference_evidence,
                    progress=progress,
                    on_research_preview=publish,
                )
            except Exception as error:
                self.future.set_exception(error)
            else:
                self.future.set_result(reply)
            if not closed.is_set() and not self.waiting.is_set():
                print("\nAgent> Planning returned. Press Enter to read the reply.", flush=True)

        Thread(target=run, name="mdgraph-review", daemon=True).start()

    def status(self):
        elapsed = int(monotonic() - self.started)
        return f"{self.stage} ({elapsed} s elapsed)"


def attach(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not (root / "run_manifest.json").is_file() or not (root / "task.json").is_file():
        raise ValueError(f"Not an MDGraph run directory: {root}")
    return root


def snapshot(root: Path, *, live: bool = False) -> dict:
    status_file, result_file = root / "status.json", root / "result.json"
    task = read_json(root / "task.json")
    state = read_json(status_file) if status_file.is_file() else {}
    report = read_json(result_file) if result_file.is_file() else {}
    receipt = state.get("job") or report.get("job") or {}
    folder = root / f"attempt_{state.get('attempt', 0) or 0:02d}"
    observation = state.get("execution_result") or {}
    if live and receipt and not report:
        cfg = TaskConfig.model_validate(task)
        observation = get_backend(cfg.execution).poll(folder, receipt)
    validation = report.get("validation", {})
    metrics = validation.get("metrics", {})
    fields = (
        "elastic_constants_gpa",
        "lattice_constant_angstrom",
        "duration_ps",
        "rdf_frames",
        "msd_effective_max_lag_ps",
        "self_diffusion_m2_s",
        "self_diffusion_block_sem_m2_s",
        "surface_tension_mn_m",
        "surface_tension_block_sem_mn_m",
        "density_profile_frames",
        "slit_average_water_density_g_cm3",
    )
    return {
        "goal": task["goal"],
        "protocol": task["protocol"],
        "system": task["structure"]["elements"],
        "crystal": task["structure"].get("crystal"),
        "target_property": task["potential"].get("target_property"),
        "molecular": task["structure"].get("molecular"),
        "status": report.get("status")
        or observation.get("status")
        or state.get("status")
        or "prepared",
        "job_id": receipt.get("job_id"),
        "queue": receipt.get("queue"),
        "scheduler_state": observation.get("scheduler_state"),
        "compute_host": observation.get("compute_host"),
        "issues": validation.get("issues", []),
        "metrics": {key: metrics[key] for key in fields if key in metrics},
        "sampling": validation.get("sampling"),
        "run_dir": str(root),
        "generated_analysis": validation.get("generated_analysis"),
        "report": str(root / "report.md") if report else None,
        "results": [
            str(folder / name)
            for name in (
                "msd.csv",
                "rdf.csv",
                "elastic.json",
                "relaxation.csv",
                "diffusion_blocks.csv",
                "surface_tension_blocks.csv",
                "density_profile.csv",
                "density_profile_blocks.csv",
                "density_profile.png",
                "density_profile.pdf",
                "radial_density_profile.csv",
                "radial_density_profile_blocks.csv",
                "radial_density_profile.png",
                "radial_density_profile.pdf",
            )
            if (folder / name).is_file()
        ],
    }


def start_watch(root: Path) -> None:
    with (root / "controller-watch.log").open("ab") as output:
        subprocess.Popen(
            [sys.executable, "-m", "mdgraph.cli", "resume", str(root), "--watch"],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def describe_run(data: dict) -> str:
    labels = {
        "prepared": "Prepared",
        "submitted": "Submitted",
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "sampling_unverified": "Completed with sampling unverified",
        "unknown": "Status unavailable",
    }
    status = {"Q": "Queued in PBS", "H": "Held in PBS", "R": "Running in PBS"}.get(
        data["scheduler_state"], labels.get(data["status"], data["status"])
    )
    location = ", ".join(
        value for value in (data["job_id"], data["queue"], data["compute_host"]) if value
    )
    lines = [f"{status} ({location})" if location else status]
    metrics = data["metrics"]
    for key, value in metrics.items():
        if key == "elastic_constants_gpa":
            lines.append(
                "Elastic constants "
                + ", ".join(f"{name} = {number:.3f} GPa" for name, number in value.items())
            )
        elif key == "self_diffusion_m2_s":
            lines.append(
                f"Self-diffusion coefficient {value:.4g} ± {metrics['self_diffusion_block_sem_m2_s']:.2g} m²/s"
            )
        elif key == "surface_tension_mn_m":
            lines.append(
                f"Surface tension {value:.3f} ± {metrics['surface_tension_block_sem_mn_m']:.3f} mN/m"
            )
        elif not key.endswith("_sem_m2_s") and not key.endswith("_sem_mn_m"):
            formats = {
                "lattice_constant_angstrom": "Lattice constant {:.5f} Å",
                "duration_ps": "Production {:g} ps",
                "rdf_frames": "RDF frames {:g}",
                "msd_effective_max_lag_ps": "MSD maximum lag {:g} ps",
                "density_profile_frames": "Density profile frames {:g}",
                "slit_average_water_density_g_cm3": "Mean water density across the slit {:.4f} g/cm³",
            }
            lines.append(formats[key].format(value))
    if data["issues"]:
        lines.extend(data["issues"])
    if data["report"]:
        lines.append(f"Report {data['report']}")
    generated = data.get("generated_analysis")
    if generated:
        lines.append(f"Generated analysis {generated['status']}: {generated['objective']}")
        lines.append(str(Path(generated["directory"]) / generated.get("csv", "analysis.json")))
    lines.extend(data["results"])
    return "\n".join(lines)


def describe_draft(draft: RunDraft) -> str:
    task = draft.task
    protocol = task.protocol
    fluid = task.structure.molecular
    lines = (
        [f"Proposed trajectory analysis: {task.analysis_objective}"]
        if task.analysis_objective
        else [f"Proposed {task.potential_scope()['target_property']} calculation"]
    )
    if fluid and fluid.cnt_model:
        tube = fluid.cnt_model
        n, m = tube.chirality
        lines.append(
            f"Rigid ({n},{m}) single-wall CNT, {tube.length_angstrom / 10:g} nm long, "
            f"{tube.water_molecules} neutral water molecules (SPC/E)."
        )
    if fluid and fluid.honeycomb_tube:
        tube = fluid.honeycomb_tube
        n, m = tube.chirality
        interface = task.potential.wall_interface
        lines.append(
            f"Rigid ({n},{m}) single-wall {'/'.join(tube.wall_elements)} honeycomb tube with "
            f"{tube.axial_cells} axial cells and {tube.fluid_molecules} neutral {interface.fluid_model} waters."
        )
        lines.append(f"Wall force field DOI {interface.source_doi}. {interface.scope}")
        preflight = draft.manager_plan.get("preflight")
        if preflight:
            geometry = preflight["sources"]
            lines.append(
                f"Tube length {geometry['cnt_length_angstrom'] / 10:.4f} nm, "
                f"radius {geometry['cnt_radius_angstrom']:.3f} Å. LAMMPS run 0 passed."
            )
    lines.append(f"{protocol.temperature_k:g} K, {protocol.ensemble.upper()}.")
    if protocol.equilibration_steps:
        scale = protocol.timestep_fs / 1000
        eq = (protocol.equilibration_ensemble or protocol.ensemble).upper()
        lines.append(
            f"{protocol.equilibration_steps * scale:g} ps {eq} equilibration, "
            f"{protocol.steps * scale:g} ps production, {protocol.timestep_fs:g} fs time step."
        )
    lines.extend(
        [
            draft.manager_plan["proposal"]["reason"],
            "Enter a change or a question, /plan to see this proposal, or /confirm to submit.",
        ]
    )
    return "\n".join(lines)


def chat(options) -> None:
    current = attach(options.run) if options.run else None
    force_background = getattr(options, "background_planning", False)
    review = getattr(options, "review", False) or force_background
    pending = None
    draft_path = getattr(options, "draft", None)
    if draft_path:
        saved = read_json(draft_path)
        pending = (
            ResearchDraft(saved, draft_path.resolve())
            if saved.get("status") == "research_only"
            else RunDraft(TaskConfig.model_validate(saved["task"]), saved["manager_plan"])
        )
        review = True
    background = review and (force_background or sys.stdin.isatty())
    planning = None
    closed = Event()
    history = deque(maxlen=6)
    confirmations = {"/confirm", "确认运行", "confirm run"}

    def show_reply(decision, message):
        # The foreground owns draft state and is the only path to job submission.
        nonlocal pending, current
        if isinstance(decision, ResearchDraft):
            pending = decision
            action = "research"
            answer = describe_research(decision)
        elif isinstance(decision, RunDraft):
            pending = decision
            action = "draft"
            answer = describe_draft(decision)
        elif isinstance(decision, Path):
            action = "run"
            current = decision
            state = drive(current)
            if state.get("receipt") and state.get("status") in {"submitted", "running"}:
                start_watch(current)
            answer = snapshot(current)
        else:
            action = decision.action
            answer = decision.message
            if action in {"status", "result"}:
                answer = (
                    snapshot(current, live=True)
                    if current
                    else "Submit a calculation or attach an existing run with /use."
                )
                if (
                    current
                    and action == "result"
                    and not answer["report"]
                    and answer["job_id"]
                    and answer["status"] in {"completed", "failed"}
                ):
                    drive(current)
                    answer = snapshot(current)
        rendered = describe_run(answer) if isinstance(answer, dict) else answer
        print("Agent> " + rendered)
        history.append({"user": message[:1000], "action": action, "reply": rendered[:1000]})

    print(
        f"MDGraph {__version__} chat. Enter a calculation request or a question in English or Chinese."
    )
    print(
        "Commands: /status and /result for the current run, /use RUN_DIR to attach a run, /exit to quit."
    )
    if review:
        print("Review mode: /plan shows the draft, /confirm submits it, and /discard clears it.")
    if background:
        print(
            "Planning runs in the background. Use /status for its current stage or /wait to read the reply when ready."
        )
    if current:
        print(f"Current run: {current}")
    if pending:
        print(
            "Agent> "
            + (
                describe_research(pending)
                if isinstance(pending, ResearchDraft)
                else describe_draft(pending)
            )
        )
    while True:
        try:
            message = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            closed.set()
            return
        if message in {"/exit", "exit", "quit", "退出"}:
            closed.set()
            return
        try:
            if planning and (planning.future.done() or message == "/wait"):
                job = planning
                job.waiting.set()
                try:
                    answer = job.future.result()
                except KeyboardInterrupt:
                    job.waiting.clear()
                    print(
                        "\nAgent> Planning continues in the background. Use /status or /wait.",
                        flush=True,
                    )
                    continue
                finally:
                    if job.future.done():
                        planning = None
                show_reply(answer, job.message)
                if message in confirmations:
                    print("Agent> Review the reply above before confirming a runnable plan.")
                    continue
                if (
                    message == "/wait"
                    or message == "/status"
                    and not current
                    or message == "/plan"
                    and isinstance(answer, (RunDraft, ResearchDraft))
                ):
                    continue
            if not message:
                continue
            if planning:
                if message == "/status":
                    print(f"Agent> Planning in progress. {planning.status()}", flush=True)
                    continue
                if message == "/plan" and planning.preview:
                    print("Agent> " + describe_research(planning.preview), flush=True)
                    continue
                if message != "/result" or not current:
                    print(
                        "Agent> A request is still being planned. Use /status or /wait first.",
                        flush=True,
                    )
                    continue
            if message == "/wait" and background:
                print("Agent> There is no request waiting for planning.")
                continue
            if message.startswith("/use "):
                current = attach(Path(message[5:].strip()))
                history.clear()
                print(f"Agent> Current run: {current}")
                continue
            if review and message in {"/plan", "/discard", *confirmations}:
                if not pending:
                    print("Agent> There is no plan awaiting review.")
                    continue
                if message == "/plan":
                    print(
                        "Agent> "
                        + (
                            describe_research(pending)
                            if isinstance(pending, ResearchDraft)
                            else describe_draft(pending)
                        )
                    )
                    continue
                if message == "/discard":
                    pending = None
                    print("Agent> Draft cleared.")
                    continue
                if isinstance(pending, ResearchDraft):
                    print(
                        "Agent> A research draft requires validated inputs before it can be submitted."
                    )
                    continue
                decision = launch_draft(pending, options.runs_root)
                pending = None
            elif message in {"/status", "/result"}:
                decision = ManagerTurn(action=message[1:])
            else:
                conversation = {
                    "active_run": snapshot(current) if current else None,
                    "recent_turns": list(history),
                    "pending_plan": (
                        {"goal": pending.task.goal, "proposal": pending.manager_plan["proposal"]}
                        if isinstance(pending, RunDraft)
                        else None
                    ),
                    "pending_research": (
                        {
                            "goal": pending.record["request"],
                            "target": pending.record["target"],
                            "explicit_conditions": pending.record["explicit_conditions"],
                            "proposal": pending.record["proposal"],
                        }
                        if isinstance(pending, ResearchDraft)
                        else None
                    ),
                }
                reference_evidence = (
                    pending.manager_plan["evidence"].get("literature_evidence")
                    if isinstance(pending, RunDraft)
                    else None
                )
                if background:
                    print(
                        "Agent> Planning in background. Use /status to check the current stage.",
                        flush=True,
                    )
                    planning = ReviewPlanning(
                        message, options, conversation, reference_evidence, closed
                    )
                    continue
                with turn_progress() as progress:
                    decision = natural_run(
                        message,
                        options,
                        conversation=conversation,
                        review=review,
                        reference_evidence=reference_evidence,
                        progress=progress,
                    )
            show_reply(decision, message)
        except KeyboardInterrupt:
            print("\nAgent> Interrupted. Use /plan to inspect any existing draft.", flush=True)
        except Exception as error:
            # Keep the session available after an API or workflow error.
            print(f"Agent> {type(error).__name__}: {error}", file=sys.stderr)
