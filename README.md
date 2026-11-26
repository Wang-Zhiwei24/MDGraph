# MDGraph

MDGraph is a LangGraph workflow for molecular dynamics with LAMMPS. It turns a natural-language request or a YAML task into a simulation plan, prepares inputs, runs the job, and analyzes the results.

## Quick start

Use Python 3.11 or newer. Install LAMMPS separately for simulation runs.

```bash
python -m pip install -e '.[llm]'
export MDGRAPH_API_KEY='your-deepseek-api-key'
mdgraph chat --review
```

Enter a request such as `Calculate the lattice parameter of Cu` or `Calculate the RDF of bulk water`. English and Chinese input are supported. In review mode, use `/status` to check planning, `/plan` to view the draft, and `/confirm` to submit a runnable plan.

To prepare the Cu example without an API key or submitting a job, run `mdgraph prepare examples/cu_eam_lattice.yaml`.

## Cluster jobs

Copy [the PBS example](examples/mdgraph-execution.cpu-pbs.yaml) to `mdgraph-execution.yaml` and set the LAMMPS executable, MPI launcher, and resources for your cluster. The CLI also accepts a profile with `--execution`. Use `mdgraph status RUN_DIR` to check a run. To resume monitoring a submitted job, use `mdgraph resume RUN_DIR --watch`.

## Project layout

- `src/mdgraph/` contains the importable Python package.
- `natural.py` interprets requests and `graph.py` coordinates the workflow.
- `prepare.py` builds simulation inputs. `backends.py` and `pbs_cluster.py` handle execution.
- `analysis.py`, `fluid_analysis.py`, and `generated_analysis.py` process simulation outputs.
- `examples/` contains task and cluster profiles. `tests/` contains regression tests.

See [the design notes](docs/design.md) for details. To run the tests, install `.[dev,llm]` and run `python -m pytest`.
