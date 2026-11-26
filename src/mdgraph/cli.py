from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import __version__
from .artifacts import atomic_json, read_frozen_task, read_json, sha256
from .config import Execution, TaskConfig, load_config
from .graph import build_graph
from .research import (
    ResearchDraft,
    describe_research,
    recent_research,
    research_system,
    save_research,
)


@contextlib.contextmanager
def run_lock(root: Path):
    """OS-level exclusive lock, automatically released if controller crashes."""
    with (root / "controller.lock").open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError("Another controller holds this run; use status instead") from e
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def create_run(cfg: TaskConfig, runs_root: Path, *, manager_plan: dict | None = None) -> Path:
    """Freeze the task and plan before the first graph node can run."""
    root = runs_root.resolve() / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + cfg.name + "_" + uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "task.json", cfg.model_dump())
    manifest = {"task_sha256": sha256(root / "task.json"), "schema": 2}
    if manager_plan is not None:
        plan_path = root / "manager_plan.json"
        atomic_json(plan_path, manager_plan)
        manifest["manager_plan_sha256"] = sha256(plan_path)
    atomic_json(root / "run_manifest.json", manifest)
    return root


def execution_profile(path: Path | None, cfg: TaskConfig | None = None) -> Execution | None:
    if path is None:
        if cfg and cfg.execution.backend != "local":
            return None
        defaults = (
            Path(sys.prefix).parent / "mdgraph-execution.yaml",
            Path("mdgraph-execution.yaml"),
        )
        path = next((candidate for candidate in defaults if candidate.is_file()), None)
    return (
        Execution.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))) if path else None
    )


@dataclass(frozen=True)
class RunDraft:
    task: TaskConfig
    manager_plan: dict


def launch_draft(draft: RunDraft, runs_root: Path) -> Path:
    root = create_run(draft.task, runs_root, manager_plan=draft.manager_plan)
    print(
        json.dumps({"manager_plan": draft.manager_plan["proposal"]}, ensure_ascii=False), flush=True
    )
    print(f"RUN_DIR={root}", flush=True)
    return root


def write_manager_validation(
    runs_root: Path, review_directory: Path | None, events: list[dict], final_action: str | None
) -> None:
    folder = (
        review_directory.parent
        if review_directory
        else runs_root
        / (datetime.now().strftime("%Y%m%d_%H%M%S_manager_validation_") + uuid4().hex[:8])
    )
    path = folder.resolve() / "manager_validation.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, {"role": "manager", "final_action": final_action, "attempts": events})
    except OSError as error:
        print(f"Could not write Manager validation log: {error}", file=sys.stderr)
        return
    print(f"MANAGER_VALIDATION_FILE={path}", file=sys.stderr, flush=True)


def natural_run(
    request: str,
    options: argparse.Namespace,
    *,
    conversation: dict | None = None,
    review: bool = False,
    reference_evidence: list[dict] | None = None,
    review_directory: Path | None = None,
    progress=None,
    on_research_preview=None,
):
    """Route the user request to a reply, a research draft, or a validated run."""
    from .natural import manager_request, task_from_plan

    configured = load_config(options.system) if options.system else None
    context = conversation or {}
    active_research = context.get("pending_research")
    if (
        review
        and not configured
        and not review_directory
        and not reference_evidence
        and not getattr(options, "refresh_research", False)
        and not context.get("active_run")
        and not context.get("pending_plan")
        and (not context.get("recent_turns") or active_research)
        and (not active_research or active_research["goal"] == request)
    ):
        prior = recent_research(request, options.runs_root)
        if prior:
            return prior
    execution = execution_profile(options.execution, configured)
    if execution and options.lammps:
        execution = Execution.model_validate(
            {**execution.model_dump(), "command": [options.lammps]}
        )
    plans = []
    validation_events = []
    turn = None
    try:
        if progress:
            progress(
                "Manager: interpreting the request and checking the available simulation methods"
            )
        turn, evidence, llm = manager_request(
            request,
            configured=configured,
            execution=execution,
            smoke=options.smoke,
            conversation=conversation,
            reference_evidence=reference_evidence,
            validation_events=validation_events,
        )
    finally:
        if validation_events:
            write_manager_validation(
                options.runs_root,
                review_directory,
                validation_events,
                turn.action if turn else None,
            )
    if turn.action in {"research", "revise_research"}:
        if configured:
            raise ValueError("Research cannot replace an explicitly configured simulation system")
        context = None
        goal = evidence["request"]
        conditions = evidence["explicit_constraints"]
        if turn.action == "revise_research":
            pending = (conversation or {}).get("pending_research")
            if not pending:
                raise ValueError("There is no research draft to revise")
            context = {
                "prior_goal": pending["goal"],
                "prior_target": pending["target"],
                "clarification": goal,
            }
            goal = f"{pending['goal']}\nAuthor clarification: {goal}"
            conditions = {**pending.get("explicit_conditions", {}), **conditions}
        folder = (
            review_directory.parent
            if review_directory
            else options.runs_root
            / (datetime.now().strftime("%Y%m%d_%H%M%S_research_") + uuid4().hex[:8])
        )
        reporting = {"progress": progress} if progress else {}
        if on_research_preview:
            preview_path = folder.resolve() / "research_preview.json"
            preview_path.parent.mkdir(parents=True, exist_ok=True)

            def publish(record):
                atomic_json(preview_path, record)
                on_research_preview(ResearchDraft(record, preview_path))

            reporting["on_preview"] = publish
        return save_research(
            research_system(goal, turn.research, conditions, llm, context=context, **reporting),
            folder,
            llm,
        )
    plan = turn
    if conversation is not None:
        if turn.action not in {"run", "revise"}:
            return turn
        plan = turn.plan
    if progress:
        progress("Manager: checking the proposed simulation plan")
    cfg = task_from_plan(
        plan,
        evidence,
        llm,
        options.lammps or "lmp",
        on_plan=plans.append,
        configured=configured,
        execution=execution,
        smoke=options.smoke,
    )
    if turn.action == "revise":
        pending = conversation.get("pending_plan")
        if not review or not pending:
            raise ValueError("There is no pending draft to revise")
        cfg = cfg.model_copy(update={"goal": pending["goal"]})
    if review and cfg.potential.wall_interface:
        from .prepare import prepare

        if review_directory:
            manifest = prepare(cfg, review_directory)
        else:
            with TemporaryDirectory(prefix="mdgraph-review-") as folder:
                manifest = prepare(cfg, Path(folder))
        plans[0]["preflight"] = {key: manifest[key] for key in ("sources", "wall_preflight")}
    draft = RunDraft(cfg, plans[0])
    return draft if review else launch_draft(draft, options.runs_root)


def drive(
    root: Path,
    watch: bool = False,
    poll_seconds: float | None = None,
    *,
    prepare_only: bool = False,
) -> dict:
    """Resume the frozen run from its checkpoint and publish scheduler observations."""
    root = root.resolve()
    manifest, task = read_frozen_task(root)
    cfg = TaskConfig.model_validate(task)
    manager_plan = None
    if "manager_plan_sha256" in manifest:
        plan_path = root / "manager_plan.json"
        if not plan_path.is_file() or sha256(plan_path) != manifest["manager_plan_sha256"]:
            raise ValueError("Frozen manager plan missing or changed; create a new run")
        manager_plan = read_json(plan_path)
    deadline = time.monotonic() + cfg.execution.max_watch_seconds
    with run_lock(root), SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")) as saver:
        graph = build_graph(saver, legacy=manifest["schema"] == 1, stop_before_submit=prepare_only)
        graph_cfg = {"configurable": {"thread_id": root.name}, "recursion_limit": 1000}
        snapshot = graph.get_state(graph_cfg)
        if not snapshot.values:
            value = {
                "run_id": root.name,
                "run_dir": str(root),
                "config": cfg.model_dump(),
                "attempt": 0,
            }
            if manager_plan is not None:
                value["manager_plan"] = manager_plan
        elif not snapshot.next:
            return snapshot.values
        else:
            interrupted = any(task.interrupts for task in snapshot.tasks)
            value = Command(resume={"poll": True}) if interrupted else None
        while True:
            for event in graph.stream(value, graph_cfg, stream_mode="updates"):
                for node, data in event.items():
                    update = {
                        "node": node,
                        "status": data.get("status") if isinstance(data, dict) else "waiting",
                    }
                    if isinstance(data, dict) and "next_agent" in data:
                        update["next_agent"] = data["next_agent"]
                    if isinstance(data, dict):
                        receipt = data.get("receipt", {})
                        if receipt.get("scheduler"):
                            update["job"] = receipt
                        if "queue_probe" in data:
                            probe = data["queue_probe"]
                            update["queue_probe"] = {
                                key: probe[key] for key in ("queue", "idle_nodes", "job_counts")
                            }
                        for key in ("scheduler_state", "exec_host", "compute_host"):
                            if data.get("execution_result", {}).get(key) is not None:
                                update[key] = data["execution_result"][key]
                    print(json.dumps(update, ensure_ascii=False), flush=True)
            snapshot = graph.get_state(graph_cfg)
            atomic_json(
                root / "status.json",
                {
                    "status": snapshot.values.get("status"),
                    "next": list(snapshot.next),
                    "job": snapshot.values.get("receipt"),
                    "attempt": snapshot.values.get("attempt"),
                    "execution_result": snapshot.values.get("execution_result"),
                },
            )
            if not snapshot.next or not watch:
                return snapshot.values
            if time.monotonic() >= deadline:
                print(
                    "Watch budget reached; job is not cancelled. Resume this directory later.",
                    flush=True,
                )
                return snapshot.values
            time.sleep(poll_seconds or cfg.execution.poll_seconds)
            value = Command(resume={"poll": True})


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph MD workflow (LAMMPS, local/SSH Slurm/PBS)"
    )
    parser.add_argument("--version", action="version", version=f"MDGraph {__version__}")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        cmd = sub.add_parser(action)
        cmd.add_argument("task", type=Path)
        cmd.add_argument("--runs-root", type=Path, default=Path("runs"))
        cmd.add_argument("--execution", type=Path, help="Override the default execution profile")
        if action == "run":
            cmd.add_argument("--watch", action="store_true")
    cmd = sub.add_parser("resume")
    cmd.add_argument("run_dir", type=Path)
    cmd.add_argument("--watch", action="store_true")
    cmd = sub.add_parser("status")
    cmd.add_argument("run_dir", type=Path)
    cmd = sub.add_parser(
        "analyze", help="Generate a Python estimator for a completed MD trajectory"
    )
    cmd.add_argument("run_dir", type=Path)
    cmd.add_argument("objective")
    cmd.add_argument("--output-root", type=Path, default=Path("analysis_runs"))
    sub.add_parser("graph")
    for action in ("ask", "chat", "plan"):
        cmd = sub.add_parser(
            action,
            help="Start an interactive session"
            if action == "chat"
            else "Run a supported natural-language materials task",
        )
        if action in {"ask", "plan"}:
            cmd.add_argument("request")
            if action == "plan":
                cmd.add_argument(
                    "--output",
                    type=Path,
                    required=True,
                    help="Create a new review artifact directory",
                )
        else:
            cmd.add_argument("--run", type=Path, help="Attach an existing run")
            cmd.add_argument(
                "--review",
                action="store_true",
                help="Review and confirm a proposed plan before submitting",
            )
            cmd.add_argument(
                "--background-planning",
                action="store_true",
                help="Enable background review even when input is redirected",
            )
            cmd.add_argument(
                "--refresh-research",
                action="store_true",
                help="Search again for an exact request that has a recent saved review",
            )
            cmd.add_argument(
                "--draft", type=Path, help="Open a saved MDGraph plan for author review"
            )
        solver = cmd.add_mutually_exclusive_group()
        solver.add_argument("--lammps")
        solver.add_argument(
            "--system",
            type=Path,
            help="Use a vetted task YAML with an existing structure/potential/protocol",
        )
        cmd.add_argument(
            "--execution",
            type=Path,
            help="Execution-only YAML; default: mdgraph-execution.yaml when present",
        )
        cmd.add_argument(
            "--smoke",
            action="store_true",
            help="Request an exploratory short trajectory instead of a production plan",
        )
        cmd.add_argument("--runs-root", type=Path, default=Path("runs"))
    args = parser.parse_args(sys.argv[1:] or ["chat"])
    try:
        if args.action == "chat":
            from .chat import chat

            chat(args)
            return
        if args.action == "analyze":
            from .generated_analysis import analyze_existing

            result = analyze_existing(
                args.run_dir,
                args.objective,
                args.output_root,
                progress=lambda step: print(f"Working> {step}", flush=True),
            )
            print(json.dumps(result, ensure_ascii=False))
            if result["status"] != "computed":
                raise SystemExit(1)
            return
        if args.action == "plan":
            from .chat import describe_draft

            args.output.mkdir(parents=True, exist_ok=False)
            draft = natural_run(
                args.request, args, review=True, review_directory=args.output / "preview"
            )
            if isinstance(draft, ResearchDraft):
                print(describe_research(draft))
                return
            path = args.output / "pending_plan_validated.json"
            atomic_json(
                path,
                {"manager_plan": draft.manager_plan, "task": draft.task.model_dump(mode="json")},
            )
            print(describe_draft(draft))
            print(f"PLAN_FILE={path.resolve()}")
            return
        if args.action == "graph":
            from langgraph.checkpoint.memory import InMemorySaver

            print(build_graph(InMemorySaver()).get_graph().draw_mermaid())
            return
        if args.action == "status":
            path = args.run_dir / "status.json"
            print(
                json.dumps(
                    read_json(path) if path.exists() else {"status": "prepared/not started"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if args.action in {"run", "prepare", "ask"}:
            if args.action == "ask":
                root = natural_run(args.request, args)
                if isinstance(root, ResearchDraft):
                    print(describe_research(root))
                    return
                args.watch = True
            else:
                cfg = load_config(args.task)
                execution = execution_profile(args.execution, cfg)
                if execution:
                    cfg = TaskConfig.model_validate(
                        {**cfg.model_dump(), "execution": execution.model_dump()}
                    )
                root = create_run(cfg, args.runs_root)
                print(f"RUN_DIR={root}", flush=True)
            if args.action == "prepare":
                result = drive(root, prepare_only=True)
                if result.get("status") != "prepared":
                    print(
                        json.dumps(
                            {"status": "failed", "errors": result.get("errors", [])},
                            ensure_ascii=False,
                        )
                    )
                    raise SystemExit(1)
                print(
                    json.dumps(
                        {"status": "prepared", "manifest": result["manifest"]},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                print("No job submitted. Launch later with: mdgraph resume RUN_DIR --watch")
                return
        else:
            root = args.run_dir
        result = drive(root, watch=args.watch)
        summary = {"status": result.get("status"), "report": str(root / "report.md")}
        if summary["status"] in {"failed", "needs_input"}:
            summary["issues"] = result.get("validation", {}).get("issues", result.get("errors", []))
        elif summary["status"] == "sampling_unverified":
            summary["sampling"] = result["validation"]["sampling"]
        if result.get("validation", {}).get("generated_analysis"):
            summary["analysis"] = result["validation"]["generated_analysis"]
        print(json.dumps(summary, ensure_ascii=False))
        if summary["status"] in {
            "failed",
            "needs_input",
            "sampling_unverified",
            "analysis_incomplete",
        }:
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
