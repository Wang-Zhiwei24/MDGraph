from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents import advise, decide
from .analysis import analyze
from .artifacts import atomic_json, read_json
from .backends import get_backend
from .config import TaskConfig
from .fluid_analysis import ANALYSIS_CHOICES, analyze_fluid
from .molecular import MolecularModel, select_molecular_model
from .potentials import retrieve_potential
from .prepare import build_structure, prepare, verified_manifest, write_input
from .repository_discovery import discovery_snapshot
from .structure_tools import available_builders
from .trajectory_tools import available_trajectory_backends


class MDState(TypedDict, total=False):
    run_id: str
    run_dir: str
    config: dict
    manager_plan: dict
    attempt: int
    status: str
    manifest: dict
    receipt: dict
    queue_probe: dict
    execution_result: dict
    validation: dict
    advice: dict
    errors: list[str]
    retry: bool
    stage: str
    next_agent: str
    structure: dict
    potential: dict


def cfg_of(state):
    return TaskConfig.model_validate(state["config"])


def attempt_dir(state):
    return Path(state["run_dir"]) / f"attempt_{state.get('attempt', 0):02d}"


def decision_at(state, role, evidence, choices, *, compare_candidates=False, action=None):
    path = attempt_dir(state) / f"{action or role}_decision.json"
    if path.is_file():
        saved = read_json(path)
        if saved["allowed_choices"] != choices or saved["choice"] not in choices:
            raise ValueError(f"Saved {role} decision does not match the current choices")
        return saved
    result = decide(cfg_of(state), role, evidence, choices, compare_candidates=compare_candidates)
    saved = {**result, "allowed_choices": choices}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, saved)
    return saved


def checked_stage(label, fn):
    try:
        return fn()
    except (ValueError, OSError) as error:
        return stage_failure(label, error)


def planner(state: MDState):
    cfg = cfg_of(state)
    advice = advise(
        cfg,
        "manager",
        {
            "goal": cfg.goal,
            "protocol": cfg.protocol.model_dump(),
            "potential": cfg.potential.model_dump(),
            "backend": cfg.execution.backend,
            "max_retries": cfg.execution.max_retries,
        },
    )
    atomic_json(Path(state["run_dir"]) / "manager.json", advice)
    return {"status": "planned", "advice": {"manager": advice}, "errors": []}


def potential_plan(cfg: TaskConfig) -> dict:
    if cfg.structure.molecular:
        return {
            "mode": "registered molecular models",
            "system": cfg.structure.molecular.model_dump(),
            "property": cfg.potential.target_property,
        }
    if cfg.potential.discover:
        names = "NIST and OpenKIM" if cfg.potential.discover == "repositories" else "NIST"
        return {
            "mode": f"live {names} repository search",
            **cfg.potential_scope(),
            "note": "The potential Agent searches by composition and target, then selects a downloadable model.",
        }
    if cfg.potential.catalog:
        if cfg.potential.application:
            return {"mode": "catalog by application", "application": cfg.potential.application}
        return {
            "mode": "materials catalog",
            **cfg.potential_scope(),
            "note": "No application label or preselected file is required; potential agent will match the configured catalog after structure preparation.",
        }
    return {
        "mode": "built-in demonstration"
        if cfg.potential.kind == "lj_demo"
        else "explicit potential"
    }


def preparation(state: MDState):
    cfg = cfg_of(state)
    try:
        manifest = prepare(cfg, attempt_dir(state))
        review = advise(
            cfg,
            "designer",
            {"goal": cfg.goal, "manifest": manifest, "protocol": cfg.protocol.model_dump()},
        )
        atomic_json(attempt_dir(state) / "designer.json", review)
        return {
            "manifest": manifest,
            "status": "prepared",
            "advice": {**state.get("advice", {}), "designer": review},
        }
    except (ValueError, OSError) as e:
        return {
            "status": "failed",
            "errors": [str(e)],
            "validation": {"passed": False, "issues": [str(e)]},
        }


def stage_failure(stage: str, error: Exception) -> dict:
    message = f"{stage}: {error}"
    return {
        "stage": stage,
        "status": "failed",
        "errors": [message],
        "validation": {"passed": False, "issues": [message]},
    }


def structure_agent(state: MDState):
    """Choose a supported builder and validate the structure before potential setup."""
    cfg = cfg_of(state)
    review = None
    if cfg.structure.molecular:
        folder = attempt_dir(state)
        model = MolecularModel.model_validate(read_json(folder / state["potential"]["filename"]))
        choices = {
            "regular_grid": "Seeded random orientations on regular cells at the requested phase density",
            "jittered_grid": "Seeded orientations and small separation-preserving jitter of molecule centers",
        }
        evidence = {
            "goal": cfg.goal,
            "geometry": cfg.structure.molecular.model_dump(),
            "potential": state["potential"]["candidate_id"],
            "explicit_periodicity": "XY only"
            if cfg.structure.molecular.geometry == "graphene_slit"
            else "3D",
            "task": "Choose the packing algorithm while preserving the planned box, interfaces and composition.",
        }
        review = decision_at(state, "structure", evidence, choices) if cfg.llm.decisions else None
        structure = build_structure(
            cfg,
            folder,
            molecular_model=model,
            builder=review["choice"] if review else "regular_grid",
        )
        return {
            "stage": "structure",
            "status": "structure_ready",
            "structure": structure,
            "advice": {
                **state["advice"],
                "structure": review or advise(cfg, "structure", evidence),
            },
        }
    builders = available_builders(cfg.structure)
    builder = builders[0]
    if cfg.llm.decisions:
        source = cfg.structure.model_dump(exclude={"path", "atomsk_executable"})
        evidence = {
            "goal": cfg.goal,
            "structure_request": source,
            "has_imported_structure": bool(cfg.structure.path),
            "boundary": "3D periodic",
            "requested_cell": cfg.protocol.relax_cell,
        }
        choices = {
            "preserve": f"Use {builder.upper()} and preserve the generated/imported positions",
            "wrap": f"Use {builder.upper()} and wrap positions without changing geometry",
            "stop": "Reject for a specific goal/structure mismatch",
        }
        if len(builders) > 1:
            choices.update(
                atomsk_preserve="Use Atomsk to create and duplicate, then preserve positions",
                atomsk_wrap="Use Atomsk to create and duplicate, then wrap positions",
            )
        review = decision_at(state, "structure", evidence, choices)
        if review["choice"] == "stop":
            return {
                **stage_failure("structure", ValueError(review["reason"])),
                "advice": {**state["advice"], "structure": review},
            }
        if review["choice"].startswith("atomsk_"):
            builder = "atomsk"
    try:
        structure = build_structure(
            cfg,
            attempt_dir(state),
            builder=builder,
            wrap=bool(review and review["choice"].endswith("wrap")),
        )
    except (ValueError, OSError) as error:
        return stage_failure("structure", error)
    evidence = {
        "goal": cfg.goal,
        "structure": structure,
        "checks": "Periodic cell, element mapping, overlap and requested cell symmetry checked",
    }
    review = review or advise(cfg, "structure", evidence)
    return {
        "stage": "structure",
        "status": "structure_ready",
        "structure": structure,
        "advice": {**state["advice"], "structure": review},
    }


def potential_agent(state: MDState):
    """Resolve documented potential candidates for the requested model and property."""
    cfg = cfg_of(state)

    if cfg.structure.research_species:
        snapshot = discovery_snapshot(cfg, attempt_dir(state), inventory=True)
        evidence = {
            "goal": cfg.goal,
            "species": cfg.structure.research_species,
            "property_scope": cfg.potential_scope(),
            "repositories": snapshot["repositories"],
            "candidates": snapshot["shortlist"],
            "task": "Review which candidates actually describe this molecule and thermodynamic state. "
            "List missing topology, parameter files or evidence. Do not approve a simulation.",
        }
        review = advise(cfg, "potential", evidence)
        atomic_json(attempt_dir(state) / "potential_research_review.json", review)
        issue = "A sourced force field and compatible molecular topology are required before submitting dynamics."
        return {
            "stage": "potential",
            "status": "needs_input",
            "validation": {"passed": False, "issues": [issue]},
            "potential": {
                "kind": "research",
                "discovery": snapshot,
                "research_review": review,
                "tool": {"name": "NIST and OpenKIM live repository search"},
            },
            "advice": {**state["advice"], "potential": review},
        }

    def choose(candidates, rejected=()):
        action = f"potential_retry_{len(rejected)}" if rejected else "potential"
        evidence = {
            "goal": cfg.goal,
            "structure": state.get("structure") or cfg.structure.model_dump(),
            "protocol": cfg.protocol.model_dump(),
            "property_scope": cfg.potential_scope(),
            "candidates": candidates,
        }
        if cfg.potential.discover == "repositories":
            evidence["task"] = (
                "Use the repository evidence to choose what to download for these species, phase, "
                "conditions and observable. Pick none only if the evidence gives a concrete "
                "applicability blocker. The selected table must pass a separate LAMMPS test."
            )
            evidence["rejected"] = [{"id": row["id"], "reason": row["reason"]} for row in rejected]
        return decision_at(
            state,
            "potential",
            evidence,
            {entry["id"]: entry["kind"] for entry in candidates},
            compare_candidates=True,
            action=action,
        )

    try:
        if cfg.structure.molecular:
            potential = select_molecular_model(
                cfg, attempt_dir(state), choose if cfg.llm.decisions else None
            )
        else:
            potential = retrieve_potential(
                cfg,
                attempt_dir(state),
                choose if cfg.llm.decisions else None,
                element_counts=state["structure"].get("element_counts"),
            )
    except (ValueError, OSError) as error:
        return stage_failure("potential", error)
    review = potential.get("decision") or advise(
        cfg,
        "potential",
        {"goal": cfg.goal, "selection": potential, "protocol": cfg.protocol.model_dump()},
    )
    return {
        "stage": "potential",
        "status": "potential_ready",
        "potential": potential,
        "advice": {**state["advice"], "potential": review},
    }


def input_agent(state: MDState):
    """Render inputs from the validated structure, potential and protocol."""
    cfg = cfg_of(state)
    review = None
    passes = 3
    if cfg.llm.decisions:
        choices = {
            "standard": "Use validated template and configured physics",
            "stop": "Stop for a concrete incompatibility",
        }
        if cfg.protocol.ensemble == "relax":
            choices["extra_relaxation"] = (
                "Use same physics with one extra bounded minimization pass"
            )
        selected = state["potential"]
        model_checks = {
            "lj_demo": "This explicitly configured argon LJ demonstration uses fixed coefficients embedded in the "
            "versioned template; no external model file or independent accuracy claim is needed.",
            "water": "The water model uses documented parameters recorded in hashed metadata and embedded in the input; "
            "there is no external model file.",
            "molecular": "The selected force field includes topology, parameters and source in immutable molecular_model.json. "
            "Free-liquid slabs use PPPM electrostatics and dispersion. Graphene slits use slab-corrected "
            "PPPM electrostatics, explicit C–O Lennard-Jones coefficients and rigid plates. "
            "Cylindrical confinement uses a rigid wall, 3D PPPM with lateral vacuum and its selected "
            "wall–water coefficients. Composed walls have an explicit element map and solver run-0 check. "
            "Equilibration and production use "
            "the manager's approved ensembles and conditions.",
        }
        review = decision_at(
            state,
            "designer",
            {
                "goal": cfg.goal,
                "potential": {
                    k: selected.get(k)
                    for k in ("kind", "mode", "candidate_id", "source", "model_sha256")
                },
                "selection_reason": (selected.get("decision") or {}).get("reason"),
                "model_checks": model_checks.get(
                    selected["kind"],
                    "The model file's SHA256 and type mapping were checked during staging.",
                ),
                "review_scope": "Check concrete incompatibility with the accepted goal. For an explicit exploratory "
                "protocol, lack of independent accuracy evidence or a separate equilibration segment "
                "is a report limitation, not a reason to prevent the requested test.",
                "plan_reason": state.get("manager_plan", {}).get("proposal", {}).get("reason"),
                "structure": {
                    k: state["structure"].get(k) for k in ("elements", "element_counts", "atoms")
                },
                "protocol": cfg.protocol.model_dump(),
            },
            choices,
        )
        if review["choice"] == "stop":
            return {
                **stage_failure("input", ValueError(review["reason"])),
                "advice": {**state["advice"], "designer": review},
            }
        passes = 4 if review["choice"] == "extra_relaxation" else 3
    try:
        manifest = write_input(
            cfg, attempt_dir(state), state["structure"], state["potential"], relax_passes=passes
        )
    except (ValueError, OSError) as error:
        return stage_failure("input", error)
    review = review or advise(
        cfg,
        "designer",
        {"goal": cfg.goal, "manifest": manifest, "protocol": cfg.protocol.model_dump()},
    )
    atomic_json(attempt_dir(state) / "designer.json", review)
    return {
        "stage": "input",
        "status": "prepared",
        "manifest": manifest,
        "advice": {**state["advice"], "designer": review},
    }


def submission(state: MDState):
    """Use the configured backend and record the receipt for idempotent recovery."""
    cfg = cfg_of(state)
    # Recheck checksums on every resume before crossing the execution boundary.
    folder = attempt_dir(state)
    manifest = verified_manifest(folder)
    queue = None
    backend = None
    if cfg.execution.backend == "cluster":
        backend = get_backend(cfg.execution)
        queue = backend.probe(folder)
        if not queue.get("accepts_request", queue.get("enabled") and queue.get("started")):
            return stage_failure(
                "submission",
                ValueError("No configured PBS queue accepts this job; see queue_probe.json"),
            )
    update = {}
    if queue is not None:
        update["queue_probe"] = queue
    if cfg.llm.decisions:
        review = decision_at(
            state,
            "operator",
            {
                "goal": cfg.goal,
                "atoms": manifest["atoms"],
                "potential_kind": manifest.get("potential_kind"),
                "checks": "Input hashes verified",
                "backend": cfg.execution.backend,
                "scheduler": cfg.execution.scheduler,
                "queue_probe": queue,
                "queue_policy": (
                    "Submit to the selected queue even when nodes are busy. PBS will hold the job "
                    "until a node is available. A busy queue is not a blocker."
                    if queue
                    else None
                ),
                "timeout_seconds": cfg.execution.timeout_seconds,
                "protocol": cfg.protocol.model_dump(),
            },
            {
                "submit": "Submit verified input through the configured backend",
                "stop": "Stop for a concrete execution or resource blocker",
            },
        )
        update["advice"] = {**state["advice"], "operator": review}
        if review["choice"] == "stop":
            return {**stage_failure("submission", ValueError(review["reason"])), **update}
    receipt = (backend or get_backend(cfg.execution)).submit(folder, state["run_id"])
    return {"stage": "submission", "receipt": receipt, "status": "submitted", **update}


def monitor(state: MDState):
    """Pause the graph until the controller requests another scheduler observation."""
    cfg = cfg_of(state)
    result = get_backend(cfg.execution).poll(attempt_dir(state), state["receipt"])
    if result["status"] in {"running", "unknown"}:
        interrupt(
            {
                "kind": "job_pending",
                "job": state["receipt"],
                "observation": result,
                "message": "Resume to re-query the existing job, not to submit a new one.",
            }
        )
        # A resume re-enters this node and queries external state again.
        # If still pending the next conditional route returns here for another checkpointed pause.
        result = get_backend(cfg.execution).poll(attempt_dir(state), state["receipt"])
    return {"stage": "monitor", "execution_result": result, "status": result["status"]}


def collection(state: MDState):
    cfg = cfg_of(state)
    get_backend(cfg.execution).collect(attempt_dir(state), state["receipt"])
    return {"stage": "collection", "status": "collected"}


def validator(state: MDState):
    """Validate completed outputs and dispatch the requested trajectory analysis."""
    cfg = cfg_of(state)
    folder = attempt_dir(state)
    backend_decision = None
    if (
        not cfg.structure.molecular
        and cfg.protocol.ensemble != "relax"
        and cfg.validation.trajectory_backend == "auto"
        and cfg.llm.decisions
        and state["execution_result"]["status"] == "completed"
        and (folder / "trajectory.lammpstrj").is_file()
        and "mdanalysis" in available_trajectory_backends()
    ):
        backend_decision = decision_at(
            state,
            "analyst",
            {
                "task": "Choose only the periodic pair-distance library used for RDF, keeping geometric and MD checks unchanged.",
                "goal": cfg.goal,
                "atoms": state["manifest"]["atoms"],
                "species_resolved": cfg.validation.species_resolved,
            },
            {
                "ase": "ASE float64 neighbor distances",
                "mdanalysis": "MDAnalysis periodic capped distances",
            },
            action="trajectory_tool",
        )
    if cfg.structure.molecular:
        choices = ANALYSIS_CHOICES[cfg.potential.target_property]
        estimator = (
            decision_at(
                state,
                "analyst",
                {
                    "goal": cfg.goal,
                    "protocol": cfg.protocol.model_dump(),
                    "geometry": cfg.structure.molecular.model_dump(),
                },
                choices,
                action="molecular_estimator",
            )
            if cfg.llm.decisions
            else None
        )
        try:
            result = analyze_fluid(
                cfg,
                folder,
                state["manifest"]["atoms"],
                estimator["choice"] if estimator else next(iter(choices)),
            )
        except (ValueError, KeyError, OSError) as error:
            result = {
                "passed": False,
                "issues": [str(error)],
                "scope": cfg.potential.target_property,
            }
        if estimator:
            result["estimator_decision"] = estimator
    else:
        result = analyze(
            cfg,
            folder,
            state["manifest"]["atoms"],
            trajectory_backend=backend_decision["choice"] if backend_decision else None,
        )
    if backend_decision:
        result["trajectory_tool_decision"] = backend_decision
    if "potential_kind" in state["manifest"] and "potential_kind" in result.get("metrics", {}):
        result["metrics"]["potential_kind"] = state["manifest"]["potential_kind"]
    if state["execution_result"]["status"] != "completed":
        result["passed"] = False
        result["issues"] = result.get("issues", []) + ["Execution did not complete successfully"]
    if cfg.analysis_objective and state["execution_result"]["status"] == "completed":
        from .generated_analysis import generate_analysis

        result["generated_analysis"] = generate_analysis(
            cfg, folder, state["manifest"], cfg.analysis_objective, folder / "analysis"
        )
    atomic_json(attempt_dir(state) / "validation.json", result)
    update = {}
    if cfg.llm.decisions:
        choices = (
            {
                "report": "Report the passing result",
                "report_with_caveat": "Report the passing result and highlight a material limitation",
            }
            if result["passed"]
            else {"diagnose": "Report numerical failure and request diagnosis"}
        )
        review = decision_at(state, "analyst", {"goal": cfg.goal, "validation": result}, choices)
        update["advice"] = {**state["advice"], "analyst": review}
    return {
        "stage": "analysis",
        "validation": result,
        "status": "validated" if result["passed"] else "failed",
        **update,
    }


def diagnosis(state: MDState):
    cfg = cfg_of(state)
    logs = {}
    for name in ("runner.log", "screen.log", "scheduler.err"):
        path = attempt_dir(state) / name
        if path.exists():
            with path.open("rb") as f:
                f.seek(max(0, path.stat().st_size - 6000))
                logs[name] = f.read().decode("utf-8", "replace")
    evidence = {
        "execution": state["execution_result"],
        "validation": state["validation"],
        "logs": logs,
    }
    # Only explicitly observed scheduler infrastructure failures permit an automatic retry.
    # No temperature, timestep, potential or scientific acceptance threshold changes.
    retry = (
        bool(state["execution_result"].get("retryable"))
        and state["attempt"] < cfg.execution.max_retries
    )
    if cfg.llm.decisions:
        choices = {"stop": "Report failure without rerunning"}
        if retry:
            choices["retry"] = "Retry identical inputs for this confirmed infrastructure failure"
        advice = decision_at(state, "diagnostician", evidence, choices)
        retry = retry and advice["choice"] == "retry"
    else:
        advice = advise(cfg, "diagnostician", evidence)
    atomic_json(attempt_dir(state) / "diagnosis.json", advice)
    return {
        "stage": "diagnosis",
        "retry": retry,
        "advice": {**state.get("advice", {}), "diagnostician": advice},
    }


def retry_attempt(state: MDState):
    old = attempt_dir(state)
    attempt = state["attempt"] + 1
    new = Path(state["run_dir"]) / f"attempt_{attempt:02d}"
    new.mkdir(exist_ok=True)
    manifest = read_json(old / "manifest.json")
    # Identical immutable inputs; preserve old outputs and never reuse their completion markers.
    for name in [*manifest["files"], "manifest.json"]:
        shutil.copyfile(old / name, new / name)
    return {
        "stage": "retry",
        "attempt": attempt,
        "status": "prepared",
        "receipt": {},
        "execution_result": {},
        "retry": False,
    }


def report(state: MDState):
    cfg = cfg_of(state)
    validation = state.get("validation", {"passed": False, "issues": state.get("errors", [])})
    advice = state.get("advice", {}).get("analyst") or advise(
        cfg,
        "analyst",
        {
            "goal": cfg.goal,
            "validation": validation,
            "execution": state.get("execution_result", {}),
        },
    )
    status = (
        "needs_input"
        if state.get("status") == "needs_input"
        else ("completed" if validation.get("passed") else "failed")
    )
    sampling = validation.get("sampling", {})
    if status == "completed" and sampling.get("mode") == "production" and not sampling["passed"]:
        status = "sampling_unverified"
    generated = validation.get("generated_analysis")
    if (
        status in {"completed", "sampling_unverified"}
        and generated
        and generated["status"] != "computed"
    ):
        status = "analysis_incomplete"
    result = {
        "run_id": state["run_id"],
        "status": status,
        "attempt": state.get("attempt", 0),
        "job": state.get("receipt", {}),
        "validation": validation,
        "analyst": advice,
        "note": "Completed means configured numerical checks passed, not that the potential is scientifically validated.",
    }
    if state.get("execution_result"):
        result["execution_result"] = state["execution_result"]
    if "potential" in state:
        result["potential_selection"] = state["potential"]
    if "structure" in state:
        result["structure"] = state["structure"]
    if state.get("next_agent"):
        result["agent_reviews"] = {**state.get("advice", {}), "analyst": advice}
    root = Path(state["run_dir"])
    plan_file = root / "manager_plan.json"
    if "manager_plan" in state:
        result["manager_plan"] = state["manager_plan"]
    elif plan_file.is_file():
        result["manager_plan"] = read_json(plan_file)
    trajectory_tool = validation.get("metrics", {}).get("trajectory_tool")
    tool_trace = {
        "structure": state.get("structure", {}).get("tool"),
        "potential": state.get("potential", {}).get("tool"),
        "input": {"name": "MDGraph LAMMPS template renderer"} if state.get("manifest") else None,
        "execution": {
            "name": "LAMMPS",
            "backend": cfg.execution.backend,
            "scheduler": cfg.execution.scheduler,
        }
        if state.get("receipt")
        else None,
        "analysis": {"name": "MDGraph thermo parser", "libraries": ["NumPy"]}
        if state.get("execution_result")
        else None,
    }
    if trajectory_tool:
        tool_trace["analysis"]["libraries"] += ["ASE"]
        if trajectory_tool["library"] != "ASE":
            tool_trace["analysis"]["libraries"].append(trajectory_tool["library"])
        if validation["metrics"].get("trajectory_plot"):
            tool_trace["analysis"]["libraries"].append("Matplotlib")
        tool_trace["analysis"]["rdf"] = trajectory_tool
    result["tools"] = {role: info for role, info in tool_trace.items() if info is not None}
    atomic_json(root / "result.json", result)
    summary = []
    if generated:
        summary.append(
            f"- Generated analysis status `{generated['status']}`. "
            f"See `{generated['directory']}/analysis.json`."
        )
        if generated.get("csv"):
            summary.append(f"- Observable data in `{generated['directory']}/{generated['csv']}`.")
    if result["job"].get("queue"):
        summary.append(f"- PBS job `{result['job']['job_id']}` in queue `{result['job']['queue']}`")
    if state.get("execution_result", {}).get("compute_host"):
        summary.append(f"- Compute node `{state['execution_result']['compute_host']}`")
    if "manager_plan" in result:
        summary.append(
            f"- Manager's protocol rationale. {result['manager_plan']['proposal']['reason']}"
        )
    if tool_trace["structure"]:
        summary.append(f"- Structure builder {tool_trace['structure']['name']}")
    selected = state.get("potential", {})
    if selected.get("research_review"):
        review = selected["research_review"]
        summary.append(f"- Potential Agent's repository review. {review['summary']}")
        summary.extend(f"  - {item}" for item in [*review["evidence"], *review["limitations"]])
        summary.append(
            "- Candidate IDs and source URLs are in `attempt_00/potential_discovery.json`."
        )
    if selected.get("decision"):
        decision = selected["decision"]
        summary.extend(
            [
                f"- Selected potential `{selected['candidate_id']}` ({selected['kind']})",
                f"- Selection rationale. {decision['reason']}",
            ]
        )
        summary.extend(
            f"  - `{item['candidate_id']}` — {item['assessment']}"
            for item in decision["comparisons"]
        )
    lattice = validation.get("metrics", {}).get("lattice_constant_angstrom")
    if cfg.protocol.ensemble == "relax" and validation.get("passed") and lattice is not None:
        summary.append(f"- Equilibrium lattice constant at 0 K is **{lattice:.8f} Å**.")
    elastic = validation.get("metrics", {}).get("elastic_constants_gpa")
    if elastic and validation.get("passed"):
        summary.append(
            "- Elastic constants " + ", ".join(f"{k} = {v:.3f} GPa" for k, v in elastic.items())
        )
    metrics = validation.get("metrics", {})
    if "self_diffusion_m2_s" in metrics:
        summary.append(
            f"- Self-diffusion coefficient **{metrics['self_diffusion_m2_s']:.4g} ± "
            f"{metrics['self_diffusion_block_sem_m2_s']:.2g} m²/s** (block standard error)."
        )
    if "surface_tension_mn_m" in metrics:
        summary.append(
            f"- Liquid-vapor surface tension **{metrics['surface_tension_mn_m']:.3f} ± "
            f"{metrics['surface_tension_block_sem_mn_m']:.3f} mN/m** (block standard error)."
        )
    if "density_profile_frames" in metrics:
        radial = cfg.potential.target_property == "radial_density_profile"
        profile = "radial_density_profile" if radial else "density_profile"
        label = "Radial" if radial else "Wall-normal"
        summary.append(
            f"- {label} water density from {metrics['density_profile_frames']} production frames. "
            f"See {profile}.csv, {profile}_blocks.csv and {profile}.pdf in the attempt directory."
        )
    if sampling.get("mode") == "production":
        p = cfg.protocol
        summary.append(
            f"- Equilibrated in {(p.equilibration_ensemble or p.ensemble).upper()} for "
            f"{p.equilibration_steps * p.timestep_fs / 1000:g} ps and sampled in "
            f"{p.ensemble.upper()} for {p.steps * p.timestep_fs / 1000:g} ps."
        )
        summary.append(
            "- Equilibration and block sampling checks "
            + (
                "passed."
                if sampling["passed"]
                else "had insufficient evidence. See the sampling diagnostics."
            )
        )
    elif sampling.get("mode") == "smoke":
        summary.append("- This exploratory run did not check sampling convergence.")
    if "msd_effective_max_lag_ps" in metrics:
        summary.append(
            f"- MSD data in `attempt_{state.get('attempt', 0):02d}/msd.csv` use multiple time origins "
            f"with a maximum lag of {metrics['msd_effective_max_lag_ps']:.3f} ps."
        )
    if validation.get("passed") and metrics.get("rdf_frames"):
        summary.append(
            f"- RDF data in `attempt_{state.get('attempt', 0):02d}/rdf.csv` use "
            f"{metrics['rdf_frames']} frames with a maximum radius of {metrics['rdf_rmax_angstrom']:.3f} Å."
        )
        if cfg.structure.water_repeat:
            summary.append("- The O–O, O–H, and H–H RDFs count only intermolecular pairs.")
    lines = [
        f"# MD report for {cfg.name}",
        "",
        f"- Goal — {cfg.goal}",
        f"- Status **{status}**",
        f"- Potential type `{selected.get('kind', cfg.potential.kind)}`",
        f"- Run ID `{state['run_id']}`",
        f"- Attempts {state.get('attempt', 0) + 1}",
        *summary,
        "",
        "## Calculation evidence",
        "",
        "```json",
        json.dumps(validation, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Analyst's interpretation",
        "",
        str(advice.get("reason") or advice.get("summary", "")),
        "",
        "Validation covers execution and configured numerical checks. Assessing potential accuracy requires independent reference data.",
        "",
        "Each attempt directory contains the original input, SHA256 hashes, job receipt, logs, and analysis files.",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "stage": "report",
        "status": status,
        "advice": {**state.get("advice", {}), "analyst": advice},
    }


def supervisor(state: MDState):
    """Route specialist decisions and artifacts through a single checkpointed state."""
    stage = state.get("stage")
    if stage is None:
        cfg = cfg_of(state)
        first_agent = (
            "potential_agent"
            if cfg.structure.molecular or cfg.structure.research_species
            else "structure_agent"
        )
        if "manager_plan" in state:
            plan = state["manager_plan"]
            review = {
                "role": "manager",
                "mode": "llm",
                "model": plan["model"],
                "choice": "proceed",
                "reason": plan["proposal"]["reason"],
                "source": "manager_plan.json",
            }
            atomic_json(Path(state["run_dir"]) / "manager.json", review)
            return {
                "status": "planned",
                "advice": {"manager": review},
                "errors": [],
                "next_agent": first_agent,
            }
        if cfg.llm.decisions:
            update = checked_stage(
                "planning",
                lambda: {
                    "decision": decision_at(
                        state,
                        "manager",
                        {
                            "goal": cfg.goal,
                            "protocol": cfg.protocol.model_dump(),
                            "structure_source": "provided"
                            if cfg.structure.path
                            else "configured builder",
                            "potential_plan": potential_plan(cfg),
                            "backend": cfg.execution.backend,
                        },
                        {
                            "proceed": "Run the predefined dependency order",
                            "stop": "Stop for a concrete plan blocker",
                        },
                    )
                },
            )
            if update.get("status") == "failed":
                return {**update, "next_agent": "report"}
            review = update["decision"]
            atomic_json(Path(state["run_dir"]) / "manager.json", review)
            if review["choice"] == "stop":
                return {
                    **stage_failure("planning", ValueError(review["reason"])),
                    "advice": {"manager": review},
                    "next_agent": "report",
                }
            return {
                "status": "planned",
                "advice": {"manager": review},
                "errors": [],
                "next_agent": first_agent,
            }
        return {**planner(state), "next_agent": first_agent}
    if stage in {"planning", "structure", "potential", "input", "submission"} and state[
        "status"
    ] in {"failed", "needs_input"}:
        return {"next_agent": "report"}
    if stage == "monitor":
        return {
            "next_agent": "monitor" if state["status"] in {"running", "unknown"} else "collector"
        }
    if stage == "analysis":
        return {"next_agent": "report" if state["validation"]["passed"] else "diagnostician"}
    if stage == "diagnosis":
        return {"next_agent": "retry_execution" if state["retry"] else "report"}
    if stage in {"structure", "potential"} and cfg_of(state).structure.molecular:
        # Molecular topology comes from the selected force field, before packing.
        return {"next_agent": "structure_agent" if stage == "potential" else "input_agent"}
    next_agents = {
        "structure": "potential_agent",
        "potential": "input_agent",
        "input": "operator",
        "submission": "monitor",
        "collection": "analysis_agent",
        "retry": "operator",
    }
    return {"next_agent": next_agents[stage]}


def build_graph(checkpointer, *, legacy=False, stop_before_submit=False):
    """Compile the active supervisor graph while keeping old checkpoints resumable."""
    if legacy:
        return build_legacy_graph(checkpointer)
    graph = StateGraph(MDState)
    graph.add_node("manager", supervisor)
    workers = {
        "structure_agent": lambda s: checked_stage("structure", lambda: structure_agent(s)),
        "potential_agent": potential_agent,
        "input_agent": lambda s: checked_stage("input", lambda: input_agent(s)),
        "operator": lambda s: checked_stage("submission", lambda: submission(s)),
        "monitor": monitor,
        "collector": collection,
        "analysis_agent": validator,
        "diagnostician": diagnosis,
        "retry_execution": retry_attempt,
        "report": report,
    }
    for name, worker in workers.items():
        graph.add_node(name, worker)
        graph.add_edge(name, END if name == "report" else "manager")
    graph.add_edge(START, "manager")
    graph.add_conditional_edges("manager", lambda state: state["next_agent"], list(workers))
    return graph.compile(
        checkpointer=checkpointer, interrupt_before=["operator"] if stop_before_submit else None
    )


def build_legacy_graph(checkpointer):
    """Keep pending version-1 checkpoints on their original node names."""
    graph = StateGraph(MDState)
    for name, fn in {
        "manager": planner,
        "designer": preparation,
        "operator": submission,
        "monitor": monitor,
        "collector": collection,
        "validator": validator,
        "diagnostician": diagnosis,
        "retry_execution": retry_attempt,
        "report": report,
    }.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "manager")
    graph.add_edge("manager", "designer")
    graph.add_conditional_edges(
        "designer",
        lambda s: "report" if s["status"] == "failed" else "operator",
        ["report", "operator"],
    )
    graph.add_edge("operator", "monitor")
    graph.add_conditional_edges(
        "monitor",
        lambda s: (
            "monitor" if s["execution_result"]["status"] in {"running", "unknown"} else "collector"
        ),
        ["monitor", "collector"],
    )
    graph.add_edge("collector", "validator")
    graph.add_conditional_edges(
        "validator",
        lambda s: "report" if s["validation"]["passed"] else "diagnostician",
        ["report", "diagnostician"],
    )
    graph.add_conditional_edges(
        "diagnostician",
        lambda s: "retry_execution" if s["retry"] else "report",
        ["retry_execution", "report"],
    )
    graph.add_edge("retry_execution", "operator")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
