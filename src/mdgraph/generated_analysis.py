"""Plan, run and record a generated numerical trajectory analysis."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

import numpy as np
from langchain_core.exceptions import OutputParserException
from openai import OpenAIError
from pydantic import Field

from .agents import _invoke
from .analysis import dump_frames, write_csv
from .analysis_worker import BUILTINS, LIBRARIES, METHODS
from .artifacts import atomic_json, read_frozen_task, read_json, sha256
from .config import StrictModel, TaskConfig


class AnalysisProgram(StrictModel):
    action: Literal["run", "needs_data"]
    method: str = Field(min_length=1, max_length=2000)
    limitations: list[str] = Field(default_factory=list, max_length=8)
    script: str = Field(default="", max_length=14000)


def stage_trajectory(trajectory: Path, atoms: int, timestep_fs: float, target: Path) -> dict:
    """Convert solver output to bounded, typed arrays for the numerical worker."""
    steps, wrapped, unwrapped, cells, origins, boundaries = [], [], [], [], [], []
    ids = types = molecules = None
    for step, frame_ids, frame_types, pos, unwrapped_pos, cell, origin, pbc, mol in dump_frames(
        trajectory, metadata=True
    ):
        if len(frame_ids) != atoms or steps and steps[-1] >= step:
            raise ValueError("Trajectory atom count changed or timesteps are out of order")
        if not steps:
            ids, types, molecules = frame_ids, frame_types, mol
        elif (
            not np.array_equal(ids, frame_ids)
            or not np.array_equal(types, frame_types)
            or not np.array_equal(molecules, mol)
        ):
            raise ValueError("Atom IDs, types or molecule IDs changed across the trajectory")
        if (len(steps) + 1) * (pos.nbytes + unwrapped_pos.nbytes + 136) > 96 * 1024**2:
            raise ValueError("Trajectory exceeds the 96 MiB numerical input limit")
        steps.append(step)
        wrapped.append(pos)
        unwrapped.append(unwrapped_pos)
        cells.append(cell)
        origins.append(origin)
        boundaries.append(pbc)
    if not steps:
        raise ValueError("Trajectory contains no frames")
    step_array = np.array(steps)
    arrays = {
        "step": step_array,
        "time_ps": step_array * (timestep_fs / 1000),
        "positions": np.asarray(wrapped),
        "unwrapped_positions": np.asarray(unwrapped),
        "cell": np.asarray(cells),
        "origin": np.asarray(origins),
        "pbc": np.asarray(boundaries),
        "atom_ids": ids,
        "atom_types": types,
    }
    if molecules is not None:
        arrays["molecule_ids"] = molecules
    np.savez(target, **arrays)
    return {
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "type_counts": {int(key): int(np.count_nonzero(types == key)) for key in np.unique(types)},
        "first_step": steps[0],
        "last_step": steps[-1],
        "regular_spacing": len(set(np.diff(steps))) <= 1,
        "first_cell": cells[0].tolist(),
        "periodic_axes": sorted({tuple(flags.tolist()) for flags in boundaries}),
    }


def execute_program(script: Path, data: Path) -> dict:
    """Run the estimator in a separate process without inherited API credentials."""
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    worker = Path(__file__).with_name("analysis_worker.py").resolve()
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(worker),
        str(script.resolve()),
        str(data.resolve()),
    ]
    with TemporaryDirectory(prefix="mdgraph-analysis-worker-") as folder:
        run = subprocess.run(
            command,
            cwd=folder,
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=50,
            check=False,
        )
    if run.returncode:
        raise ValueError(run.stderr.strip()[-1200:] or f"Analysis worker returned {run.returncode}")
    return json.loads(run.stdout)


def generate_analysis(
    cfg: TaskConfig,
    attempt: Path,
    manifest: dict,
    objective: str,
    output_root: Path,
    *,
    progress=None,
) -> dict:
    from .natural import default_llm

    output = output_root.resolve() / (
        datetime.now().strftime("%Y%m%d_%H%M%S_trajectory_analysis_") + uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    llm = cfg.llm if cfg.llm.enabled else default_llm()
    record = {
        "status": "failed",
        "objective": objective,
        "directory": str(output),
        "attempts": [],
        "model": llm.settings_for("analyst")["model"],
    }
    trajectory = attempt / "trajectory.lammpstrj"
    instruction = (
        "You may write one restricted numerical Python function to calculate the requested trajectory observable. "
        "Use the data keys exactly as supplied. Lengths and cell rows are in angstrom; time_ps is in ps. "
        "Cell vectors are rows; pbc is boolean per frame and axis. Distinguish wrapped and unwrapped coordinates. "
        "Use a correct periodic distance, bin volume, normalization, and selection where applicable. "
        "Only report what can be determined from these inputs. For unavailable velocities, forces, charges, "
        "topology, reference values, or sampling required for a defensible calculation, return needs_data and explain. "
        "The function must have the form def analyze(data): and return a dict with exactly two top-level keys, "
        "columns and units. Put the named output arrays inside columns, and matching unit strings inside units. "
        "Every column must be a non-empty 1-D numeric array (use one-element lists for scalars); give units. "
        "Use only the listed library attributes, builtins and methods. Libraries are already bound. "
        "No imports, files, helper definitions, lambda, f-strings, exceptions or type annotations. "
        "Basic loops, comprehensions, slicing, conditionals, arithmetic and calls to supplied functions are supported. "
        "Explain estimator choices, assumptions and convergence limitations."
    )
    try:
        with TemporaryDirectory(prefix="mdgraph-trajectory-") as staging:
            data = Path(staging) / "trajectory.npz"
            metadata = stage_trajectory(
                trajectory, manifest["atoms"], cfg.protocol.timestep_fs, data
            )
            sources = manifest.get("sources", {})
            elements = sources.get("atom_type_elements")
            if not elements and cfg.structure.water_repeat:
                elements = ["O", "H"]
            if (
                not elements
                and len(cfg.structure.elements) == 1
                and len(metadata["type_counts"]) == 1
            ):
                elements = cfg.structure.elements
            mapping = dict(enumerate(elements, 1)) if elements else {}
            evidence = {
                "objective": objective,
                "trajectory": metadata,
                "element_by_type_id": mapping,
                "protocol": cfg.protocol.model_dump(),
                "original_goal": cfg.goal,
                "libraries": {name: list(functions) for name, functions in LIBRARIES.items()},
                "builtins": list(BUILTINS),
                "methods": sorted(METHODS),
            }
            record["trajectory"] = {
                "path": str(trajectory.resolve()),
                "sha256": sha256(trajectory),
                "element_by_type_id": mapping,
                **metadata,
            }
            for index in range(1, 3):
                if progress:
                    progress(
                        f"Analysis {index}/2: generating the Python estimator with {record['model']}"
                    )
                try:
                    program = _invoke(
                        llm,
                        "analyst",
                        evidence,
                        AnalysisProgram,
                        instruction,
                        '{"action":"run","method":"Estimator and normalization","limitations":[],'
                        "\"script\":\"def analyze(data):\\n    return {'columns': {'frame_count': "
                        "[len(data['step'])]}, 'units': {'frame_count': '1'}}\"} "
                        'or {"action":"needs_data","method":"Explain missing inputs","limitations":[]}',
                    )
                    record.update(method=program.method, limitations=program.limitations)
                    if program.action == "needs_data":
                        record["status"] = "needs_data"
                        break
                    script = output / f"analysis_{index:02d}.py"
                    script.write_text(program.script.rstrip() + "\n", encoding="utf-8")
                    if progress:
                        progress(f"Analysis {index}/2: running the generated estimator")
                    result = execute_program(script, data)
                except (
                    OSError,
                    ValueError,
                    OpenAIError,
                    OutputParserException,
                    subprocess.TimeoutExpired,
                ) as error:
                    message = type(error).__name__
                    if not isinstance(error, OpenAIError):
                        message += f": {str(error)[:1200]}"
                    record["attempts"].append({"attempt": index, "error": message})
                    evidence["previous_error"] = message
                    continue
                columns = result["columns"]
                write_csv(
                    output / "result.csv",
                    [dict(zip(columns, row)) for row in zip(*columns.values())],
                )
                record.update(
                    status="computed",
                    script=script.name,
                    csv="result.csv",
                    units=result["units"],
                    rows=len(next(iter(columns.values()))),
                )
                break
    except (OSError, ValueError, KeyError) as error:
        record["error"] = str(error)
    atomic_json(output / "analysis.json", record)
    return record


def analyze_existing(root: Path, objective: str, output_root: Path, *, progress=None) -> dict:
    from .natural import default_llm

    root = root.resolve()
    _, task = read_frozen_task(root)
    cfg = TaskConfig.model_validate(task).model_copy(update={"llm": default_llm()})
    result = read_json(root / "result.json")
    if (
        result.get("status") not in {"completed", "sampling_unverified"}
        or result.get("execution_result", {}).get("status") != "completed"
    ):
        raise ValueError("Reanalysis requires a completed simulation trajectory")
    attempt = root / f"attempt_{int(result['attempt']):02d}"
    return generate_analysis(
        cfg,
        attempt,
        read_json(attempt / "manifest.json"),
        objective,
        output_root,
        progress=progress,
    )
