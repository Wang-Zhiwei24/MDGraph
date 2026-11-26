from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Crystal(StrictModel):
    lattice: Literal["sc", "bcc", "fcc", "diamond", "hcp", "b2", "rocksalt", "zincblende"]
    a_angstrom: float = Field(gt=0)
    c_angstrom: float | None = Field(default=None, gt=0)
    repeat: tuple[int, int, int] = (1, 1, 1)

    @model_validator(mode="after")
    def valid(self):
        if any(n < 1 for n in self.repeat) or math.prod(self.repeat) * 8 > 250000:
            raise ValueError("Invalid repeat or supercell exceeds structure-builder limit")
        if (self.lattice == "hcp") != (self.c_angstrom is not None):
            raise ValueError("Provide c_angstrom for hcp only")
        return self


class SlitPhysicalModel(StrictModel):
    wall_material: Literal["graphene"]
    wall_count: Literal[2]
    wall_motion: Literal["fixed"]
    wall_stacking: Literal["aa", "ab"]
    fluid_representation: Literal["explicit_water"]
    periodic_axes: Literal["xy"]
    cross_interaction: Literal["carbon_oxygen_lj"]
    loading_exclusion_per_wall_angstrom: float = Field(ge=2.25, le=3.25)


CNT_WATER_SIGMA_CO_ANGSTROM = 3.19


class NanotubeModel(StrictModel):
    chirality: tuple[int, int]
    length_angstrom: float = Field(ge=24, le=150)
    transverse_box_angstrom: float = Field(ge=35, le=100)
    water_molecules: int = Field(ge=20, le=2500)
    axial_cells: int = Field(ge=5, le=75)

    @model_validator(mode="after")
    def valid(self):
        n, m = self.chirality
        if not 8 <= n <= 24 or m != n:
            raise ValueError("The CNT-water builder accepts armchair (n,n) tubes with 8 <= n <= 24")
        bond = self.length_angstrom / (math.sqrt(3) * self.axial_cells)
        if not 1.39 <= bond <= 1.46:
            raise ValueError("CNT length and axial-cell count must match a carbon bond near 1.42 Å")
        diameter = 3 * bond * n / math.pi
        if self.transverse_box_angstrom - diameter < 25:
            raise ValueError(
                "Use at least 25 Å between the periodic tube walls in the transverse plane"
            )
        effective_volume = (
            math.pi * ((diameter - CNT_WATER_SIGMA_CO_ANGSTROM) / 2) ** 2 * self.length_angstrom
        )
        max_molecules = int(1.2 * effective_volume / 29.915)
        if self.water_molecules > max_molecules:
            raise ValueError(
                f"CNT liquid loading is too high ({self.water_molecules} waters). "
                f"For this geometry use about {round(effective_volume / 29.915)} near-ambient "
                f"waters and at most {max_molecules} for the supported packing."
            )
        return self


class HoneycombTube(StrictModel):
    """Recipe for rolling a one- or two-sublattice lattice and packing its pore."""

    wall_elements: tuple[str, ...] = Field(min_length=1, max_length=2)
    chirality: tuple[int, int]
    bond_angstrom: float = Field(ge=1, le=2.7)
    axial_cells: int = Field(ge=1, le=80)
    transverse_box_angstrom: float = Field(ge=35, le=100)
    fluid_molecules: int = Field(ge=20, le=2500)

    @model_validator(mode="after")
    def valid(self):
        n, m = self.chirality
        if not 3 <= n <= 36 or not 0 <= m <= n:
            raise ValueError("Use a honeycomb roll index with 3 <= n <= 36 and 0 <= m <= n")
        if len(set(self.wall_elements)) != len(self.wall_elements) or any(
            not re.fullmatch(r"[A-Z][a-z]?", symbol) for symbol in self.wall_elements
        ):
            raise ValueError("Provide one wall element or two distinct sublattice elements")
        return self


class WallSite(StrictModel):
    element: str = Field(pattern=r"^[A-Z][a-z]?$")
    charge_e: float = Field(ge=-2, le=2)
    epsilon_kcal_mol: float = Field(gt=0, le=10)
    sigma_angstrom: float | None = Field(default=None, ge=2, le=7)
    sigma_mixing_inputs_angstrom: tuple[float, float] | None = None

    @model_validator(mode="after")
    def resolve_sigma(self):
        inputs = self.sigma_mixing_inputs_angstrom
        if inputs:
            mean = sum(inputs) / 2
            if (
                any(not math.isfinite(item) or not 1 <= item <= 8 for item in inputs)
                or not 2 <= mean <= 7
            ):
                raise ValueError("Invalid diameters for the documented mixing rule")
            if self.sigma_angstrom is not None and not math.isclose(
                mean, self.sigma_angstrom, abs_tol=1e-4
            ):
                raise ValueError(
                    "Mixed sigma must be the arithmetic mean of the two source diameters"
                )
            self.sigma_angstrom = mean
        elif self.sigma_angstrom is None:
            raise ValueError("Supply a published cross diameter or both mixing inputs")
        return self


class WallInterface(StrictModel):
    source_doi: str = Field(pattern=r"^10\.\d{4,9}/[^\s]+$")
    fluid_model: Literal["spce", "tip3p_ewald"]
    fluid_lj_element: Literal["O"]
    sites: list[WallSite] = Field(min_length=1, max_length=2)
    evidence_quotes: list[str] = Field(min_length=1, max_length=8)
    scope: str = Field(min_length=10, max_length=700)
    wall_electrostatics: Literal["lj_only", "fixed_partial_charges"]

    @model_validator(mode="after")
    def valid(self):
        if len({site.element for site in self.sites}) != len(self.sites):
            raise ValueError("There must be exactly one parameter row per wall element")
        if not math.isclose(sum(site.charge_e for site in self.sites), 0, abs_tol=1e-6):
            raise ValueError("The unit wall composition must be neutral")
        if (self.wall_electrostatics == "lj_only") != all(
            site.charge_e == 0 for site in self.sites
        ):
            raise ValueError("Wall charge settings disagree with the chosen electrostatic model")
        return self


class MolecularFluid(StrictModel):
    species: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    phase: Literal["gas", "liquid", "supercritical"]
    geometry: Literal["bulk", "slab", "graphene_slit", "cnt"]
    repeat: int | None = Field(default=None, ge=5, le=12)
    density_g_cm3: float = Field(gt=0, le=5)
    vacuum_angstrom: float | None = Field(default=None, ge=30, le=120)
    slit_width_angstrom: float | None = Field(default=None, ge=10, le=60)
    slit_model: SlitPhysicalModel | None = None
    cnt_model: NanotubeModel | None = None
    honeycomb_tube: HoneycombTube | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.geometry == "cnt":
            if (
                self.species != "water"
                or self.phase != "liquid"
                or (self.cnt_model is None) == (self.honeycomb_tube is None)
                or self.repeat is not None
                or self.vacuum_angstrom is not None
            ):
                raise ValueError(
                    "Cylindrical confinement requires one tube recipe and explicit liquid water"
                )
        elif self.repeat is None or self.cnt_model is not None or self.honeycomb_tube is not None:
            raise ValueError(
                "Use repeat for a bulk, slab or slit geometry and cnt_model only for a CNT"
            )
        if self.geometry == "slab" and (self.phase != "liquid" or self.vacuum_angstrom is None):
            raise ValueError("A slab requires a liquid with an explicit vapor gap")
        if self.geometry == "bulk" and self.vacuum_angstrom is not None:
            raise ValueError("A bulk fluid does not take a vapor gap")
        if self.geometry == "graphene_slit":
            if (
                self.species != "water"
                or self.phase != "liquid"
                or self.slit_width_angstrom is None
                or self.vacuum_angstrom is None
                or self.repeat < 9
            ):
                raise ValueError(
                    "Graphene slit requires liquid water, a wall separation, vacuum and repeat >= 9"
                )
        elif self.slit_width_angstrom is not None or self.slit_model is not None:
            raise ValueError("The plate configuration is only used for a graphene slit")
        return self


class Structure(StrictModel):
    path: str | None = None
    format: str | None = None
    demo_argon: bool = False
    crystal: Crystal | None = None
    water_repeat: int | None = Field(default=None, ge=6, le=9)
    molecular: MolecularFluid | None = None
    research_species: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,32}$")
    elements: list[str] = Field(default_factory=list)
    builder: Literal["ase", "atomsk", "auto"] = "ase"
    atomsk_executable: str = "atomsk"

    @model_validator(mode="after")
    def valid(self):
        if (
            sum(
                (
                    bool(self.path),
                    self.demo_argon,
                    self.crystal is not None,
                    self.water_repeat is not None,
                    self.molecular is not None,
                    self.research_species is not None,
                )
            )
            != 1
        ):
            raise ValueError(
                "Provide exactly one structure source or a species for force-field discovery"
            )
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise ValueError("Explicit, unique elements in LAMMPS type order are required")
        if any(not re.fullmatch(r"[A-Z][a-z]?", e) for e in self.elements):
            raise ValueError("Invalid chemical element symbol")
        if self.demo_argon and self.elements != ["Ar"]:
            raise ValueError("demo_argon requires elements: [Ar]")
        if self.crystal:
            sites = 2 if self.crystal.lattice in {"b2", "rocksalt", "zincblende"} else 1
            if len(self.elements) != sites:
                raise ValueError(
                    f"{self.crystal.lattice} requires {sites} chemical elements in basis order"
                )
        if self.water_repeat and self.elements != ["O", "H"]:
            raise ValueError("The water builder requires elements: [O, H] in that type order")
        if self.builder == "atomsk" and (
            not self.crystal or self.crystal.lattice not in {"bcc", "fcc", "diamond", "hcp"}
        ):
            raise ValueError("Atomsk requires a configured bcc, fcc, diamond or hcp crystal")
        if (
            not self.atomsk_executable.strip()
            or "\x00" in self.atomsk_executable
            or "\n" in self.atomsk_executable
        ):
            raise ValueError("Invalid Atomsk executable path")
        return self


PotentialStyle = Literal["deepmd", "pace", "eam", "eam/alloy", "eam/fs", "sw", "tersoff"]


class Potential(StrictModel):
    kind: PotentialStyle | Literal["auto", "lj_demo", "water", "molecular"]
    water_model: Literal["auto", "spce", "tip3p_ewald"] | None = None
    model: str | None = None
    plugin: str | None = None
    metadata: str | None = None
    catalog: str | None = None
    discover: Literal["nist", "repositories"] | None = None
    application: str | None = None
    candidate_id: str | None = None
    wall_interface: WallInterface | None = None
    allowed_kinds: list[PotentialStyle] = Field(default_factory=list)
    phase: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_\-]{0,39}$")
    target_property: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_\-]{0,59}$")

    @model_validator(mode="after")
    def valid(self):
        if self.kind == "molecular":
            if any(
                (
                    self.water_model,
                    self.model,
                    self.plugin,
                    self.metadata,
                    self.catalog,
                    self.discover,
                    self.application,
                    self.allowed_kinds,
                    self.phase,
                )
            ):
                raise ValueError(
                    "Molecular potentials use a registered model, optionally selected by candidate_id"
                )
            if self.target_property not in {
                "self_diffusion",
                "surface_tension",
                "density_profile",
                "radial_density_profile",
            }:
                raise ValueError("Select a molecular observable")
            return self
        if self.wall_interface is not None:
            raise ValueError("A wall interface requires a molecular potential")
        if self.kind == "water":
            if (
                self.water_model is None
                or any(
                    (
                        self.model,
                        self.plugin,
                        self.metadata,
                        self.catalog,
                        self.discover,
                        self.application,
                        self.candidate_id,
                        self.allowed_kinds,
                        self.phase,
                    )
                )
                or self.target_property not in {None, "rdf", "msd"}
            ):
                raise ValueError("Water potential requires only kind: water and water_model")
            return self
        if self.water_model is not None:
            raise ValueError("water_model requires kind: water")
        if self.discover:
            if self.kind != "auto":
                raise ValueError("Live discovery requires kind: auto")
            if self.discover == "nist" and set(self.allowed_kinds) != {"eam/alloy"}:
                raise ValueError("Legacy NIST discovery requires allowed_kinds: [eam/alloy]")
            if self.discover == "repositories" and set(self.allowed_kinds) - {
                "eam",
                "eam/alloy",
                "eam/fs",
                "sw",
                "tersoff",
            }:
                raise ValueError("Requested styles have no live parameter adapter")
            if any(
                (
                    self.catalog,
                    self.model,
                    self.metadata,
                    self.plugin,
                    self.application,
                    self.candidate_id,
                )
            ):
                raise ValueError(
                    "Live discovery uses repository evidence and requires no preselected model or catalog"
                )
            return self
        if self.kind == "auto":
            if not self.catalog or not self.allowed_kinds or self.plugin:
                raise ValueError(
                    "Automatic style selection requires a catalog and allowed_kinds, without a plugin"
                )
        elif self.allowed_kinds:
            raise ValueError("allowed_kinds is only used with kind: auto")
        if self.kind in {"eam", "eam/alloy", "eam/fs", "sw", "tersoff"} and self.plugin:
            raise ValueError("Built-in LAMMPS potential styles do not take a plugin")
        if self.kind == "lj_demo":
            if any(
                (
                    self.model,
                    self.plugin,
                    self.metadata,
                    self.catalog,
                    self.application,
                    self.candidate_id,
                    self.phase,
                    self.target_property,
                )
            ):
                raise ValueError("lj_demo does not take model selection settings")
        elif self.catalog:
            if self.model or self.metadata:
                raise ValueError("Catalog selection excludes model/metadata")
            if self.application is not None and not self.application.strip():
                raise ValueError("Omit application for a general catalog search")
        elif not (self.model and self.metadata) or any(
            (self.application, self.candidate_id, self.phase, self.target_property)
        ):
            raise ValueError("Provide model and metadata, or a catalog and application")
        return self


class Protocol(StrictModel):
    ensemble: Literal["nvt", "npt", "nve", "relax"] = "nvt"
    temperature_k: float = Field(default=300, ge=0)
    pressure_bar: float = 0
    timestep_fs: float = Field(default=0.5, gt=0, le=5)
    steps: int = Field(default=10000, ge=10, le=1000000000)
    equilibration_steps: int = Field(default=0, ge=0, le=1000000000)
    equilibration_ensemble: Literal["nvt", "npt"] | None = None
    sampling_mode: Literal["production", "smoke"] | None = None
    thermo_every: int = Field(default=100, ge=1)
    dump_every: int = Field(default=100, ge=1)
    tdamp_fs: float = Field(default=100, gt=0)
    pdamp_fs: float = Field(default=1000, gt=0)
    seed: int = Field(default=20261117, ge=1, le=900000000)
    minimize: bool = True
    relax_cell: Literal["cubic", "hexagonal"] = "cubic"
    elastic: bool = False

    @model_validator(mode="after")
    def valid(self):
        if (
            self.equilibration_ensemble or self.sampling_mode == "production"
        ) and not self.equilibration_steps:
            raise ValueError("This protocol requires an equilibration segment")
        if self.elastic and (
            self.ensemble != "relax" or self.relax_cell != "cubic" or self.pressure_bar != 0
        ):
            raise ValueError(
                "Elastic finite differences require static, zero-pressure cubic relaxation"
            )
        if self.ensemble == "relax":
            if self.temperature_k != 0 or not self.minimize or self.equilibration_steps:
                raise ValueError(
                    "Static relax requires temperature_k: 0, minimize: true and no equilibration steps"
                )
            return self
        if self.temperature_k <= 0:
            raise ValueError("MD requires positive temperature for velocity initialization")
        if self.thermo_every > self.steps // 4:
            raise ValueError("Need at least five thermo records including initial state")
        if self.dump_every > self.steps:
            raise ValueError("dump_every must not exceed steps")
        return self


class Execution(StrictModel):
    backend: Literal["local", "cluster", "ssh"] = "local"
    scheduler: Literal["shell", "slurm", "pbs"] = "shell"
    pbs_resources: Literal["select", "nodes"] = "select"
    command: list[str] = Field(default_factory=lambda: ["lmp"])
    mpi_launcher: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=3600, ge=10)
    poll_seconds: float = Field(default=15, ge=0.1)
    max_watch_seconds: int = Field(default=86400, ge=10)
    host: str | None = None
    username: str | None = None
    port: int = Field(default=22, ge=1, le=65535)
    key_filename: str | None = None
    remote_root: str | None = None
    partition: str | None = None
    fallback_partitions: list[str] = Field(default_factory=list)
    account: str | None = None
    nodes: int = Field(default=1, ge=1)
    tasks: int = Field(default=1, ge=1)
    cpus_per_task: int = Field(default=1, ge=1)
    gpus: int = Field(default=0, ge=0)
    memory_gb: int = Field(default=4, ge=1)
    walltime: str = "01:00:00"
    production_walltime: str | None = None
    setup: list[str] = Field(default_factory=list)
    max_retries: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def valid(self):
        if not self.command or any(not x or "\n" in x or "\x00" in x for x in self.command):
            raise ValueError("command must be a nonempty argv list")
        if self.mpi_launcher:
            if self.scheduler == "shell" or self.tasks < 2:
                raise ValueError("A scheduler MPI launcher requires at least two tasks")
            if any(not x or "\n" in x or "\x00" in x for x in self.mpi_launcher):
                raise ValueError("mpi_launcher must be a valid argv list")
        if self.backend == "ssh":
            if self.scheduler == "shell":
                raise ValueError("Remote execution requires Slurm or PBS")
            if not self.host or not self.username or not self.remote_root:
                raise ValueError("SSH requires host, username, remote_root")
            if (
                not PurePosixPath(self.remote_root).is_absolute()
                or ".." in PurePosixPath(self.remote_root).parts
            ):
                raise ValueError("remote_root must be an absolute POSIX path without '..'")
        elif self.backend == "cluster":
            if self.scheduler != "pbs" or not self.partition:
                raise ValueError("Cluster backend requires PBS and a named queue")
        elif self.scheduler != "shell":
            raise ValueError("Local backend uses scheduler=shell")
        for value in (self.walltime, self.production_walltime):
            if value is not None and not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d", value):
                raise ValueError("walltime must be HH:MM:SS")
        for name in ("partition", "account"):
            v = getattr(self, name)
            if v and not re.fullmatch(r"[\w.\-]+", v, flags=re.ASCII):
                raise ValueError(f"Invalid scheduler {name}")
        if self.fallback_partitions:
            if self.backend != "cluster":
                raise ValueError("Fallback queues require the cluster PBS backend")
            queues = [self.partition, *self.fallback_partitions]
            if len(set(queues)) != len(queues) or not all(
                re.fullmatch(r"[\w.\-]+", q, flags=re.ASCII) for q in queues
            ):
                raise ValueError("PBS queue names must be valid and unique")
        # Admin-owned shell setup only. Never populated by an LLM.
        if any("\n" in line or "\x00" in line for line in self.setup):
            raise ValueError("Each setup entry must be one shell line")
        return self


class Validation(StrictModel):
    temperature_relative_tolerance: float = Field(default=0.3, gt=0)
    max_nve_energy_drift_ev_per_atom_ps: float | None = Field(default=None, gt=0)
    rdf_bins: int = Field(default=100, ge=10, le=1000)
    rdf_max_angstrom: float = Field(default=5, gt=0)
    min_pair_distance_angstrom: float = Field(default=0.3, gt=0)
    max_analysis_frames: int = Field(default=200, ge=2)
    trajectory_backend: Literal["ase", "mdanalysis", "auto"] = "ase"
    species_resolved: bool = False
    trajectory_plots: bool = False
    relax_pressure_tolerance_bar: float = Field(default=0.1, gt=0)
    relax_force_tolerance_ev_per_angstrom: float = Field(default=1e-6, gt=0)
    sampling_block_relative_tolerance: float = Field(default=0.15, gt=0, lt=1)
    equilibration_temperature_tolerance: float = Field(default=0.05, gt=0, lt=1)
    equilibration_volume_tolerance: float = Field(default=0.02, gt=0, lt=1)


AgentRole = Literal[
    "manager", "structure", "potential", "designer", "operator", "analyst", "diagnostician"
]
ReasoningEffort = Literal["none", "low", "high", "max"]


class RoleLLM(StrictModel):
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    direct_connection: bool | None = None
    reasoning_effort: ReasoningEffort | None = None


class LLM(StrictModel):
    enabled: bool = False
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "MDGRAPH_API_KEY"
    direct_connection: bool = False
    reasoning_effort: ReasoningEffort = "none"
    decisions: bool = False
    roles: dict[AgentRole, RoleLLM] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self):
        if self.decisions and not self.enabled:
            raise ValueError("LLM decisions require llm.enabled")
        return self

    def settings_for(self, role: AgentRole) -> dict:
        fields = ("model", "base_url", "api_key_env", "direct_connection", "reasoning_effort")
        settings = {key: getattr(self, key) for key in fields}
        if role in self.roles:
            settings.update(self.roles[role].model_dump(exclude_none=True))
        return settings


class TaskConfig(StrictModel):
    name: str
    goal: str = "Run a validated molecular dynamics simulation"
    structure: Structure
    potential: Potential
    protocol: Protocol = Field(default_factory=Protocol)
    execution: Execution = Field(default_factory=Execution)
    validation: Validation = Field(default_factory=Validation)
    llm: LLM = Field(default_factory=LLM)
    analysis_objective: str | None = Field(default=None, min_length=5, max_length=2000)

    def potential_scope(self) -> dict:
        return {
            "phase": self.potential.phase
            or (
                self.structure.crystal.lattice
                if self.structure.crystal
                else "liquid"
                if self.structure.water_repeat
                else self.structure.molecular.phase
                if self.structure.molecular
                else None
            ),
            "target_property": self.potential.target_property
            or (
                "elastic_constants"
                if self.protocol.elastic
                else "lattice_constant"
                if self.protocol.ensemble == "relax"
                else None
            ),
        }

    @model_validator(mode="after")
    def valid(self):
        if self.analysis_objective and self.protocol.ensemble == "relax":
            raise ValueError("Generated trajectory analysis requires a dynamic sampling protocol")
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,60}", self.name):
            raise ValueError("name must use 1-60 ASCII letters, digits, '_' or '-'")
        if self.structure.research_species:
            if (
                self.potential.discover != "repositories"
                or not self.llm.decisions
                or self.potential.phase not in {"liquid", "gas", "supercritical"}
                or self.potential.target_property not in {"self_diffusion", "surface_tension"}
            ):
                raise ValueError(
                    "A structure-free molecular search requires repositories, a fluid phase and target observable"
                )
            return self
        if self.potential.kind == "lj_demo" and not self.structure.demo_argon:
            raise ValueError("LJ is confined to the explicit argon demo, never a fallback for ML")
        if self.structure.demo_argon and self.potential.kind != "lj_demo":
            raise ValueError("Argon demo requires lj_demo")
        if bool(self.structure.water_repeat) != (self.potential.kind == "water"):
            raise ValueError(
                "A generated molecular water box and a water potential must be used together"
            )
        if self.structure.water_repeat:
            if self.protocol.ensemble not in {"nvt", "npt"} or self.protocol.timestep_fs > 2:
                raise ValueError("Rigid water requires NVT or NPT with a time step of at most 2 fs")
            if self.potential.water_model == "auto" and not self.llm.decisions:
                raise ValueError("Automatic water model selection requires LLM decisions")
        fluid = self.structure.molecular
        if bool(fluid) != (self.potential.kind == "molecular"):
            raise ValueError("A molecular fluid needs a matching registered molecular potential")
        if fluid:
            p = self.protocol
            eq = p.equilibration_ensemble or p.ensemble
            if p.ensemble not in {"nvt", "nve"} or eq not in {"nvt", "npt"} or p.timestep_fs > 2:
                raise ValueError(
                    "Molecular fluids need NVT or NPT equilibration and fixed-volume production at up to 2 fs"
                )
            if (
                (fluid.geometry == "slab") != (self.potential.target_property == "surface_tension")
                or fluid.geometry == "slab"
                and (p.ensemble != "nvt" or eq != "nvt")
            ):
                raise ValueError(
                    "Surface tension needs a slab equilibrated and sampled at fixed volume"
                )
            if (fluid.geometry == "graphene_slit") != (
                self.potential.target_property == "density_profile"
            ):
                raise ValueError("The wall-normal density profile requires a graphene slit")
            if (fluid.geometry == "cnt") != (
                self.potential.target_property == "radial_density_profile"
            ):
                raise ValueError("The radial density profile requires a CNT")
            if fluid.geometry in {"graphene_slit", "cnt"}:
                if p.ensemble != "nvt" or eq != "nvt":
                    raise ValueError("Rigid molecular confinement requires NVT for both segments")
                interface = self.potential.wall_interface
                if fluid.honeycomb_tube:
                    wall = fluid.honeycomb_tube.wall_elements
                    if (
                        interface is None
                        or set(wall) != {site.element for site in interface.sites}
                        or self.structure.elements != ["O", "H", *wall]
                        or self.potential.candidate_id != interface.fluid_model
                    ):
                        raise ValueError(
                            "Recipe elements, sourced wall interactions and water-model selection must agree"
                        )
                elif (
                    interface is not None
                    or self.structure.elements != ["O", "H", "C"]
                    or self.potential.candidate_id not in {None, "spce"}
                ):
                    raise ValueError("The legacy rigid carbon builder uses SPC/E and [O, H, C]")
            elif self.potential.wall_interface:
                raise ValueError("Sourced wall interactions require a confinement recipe")
            if not p.equilibration_steps or not p.sampling_mode:
                raise ValueError(
                    "A molecular fluid needs an explicit equilibration and sampling plan"
                )
        if self.protocol.elastic:
            if (
                not self.structure.crystal
                or self.structure.crystal.lattice not in {"fcc", "bcc"}
                or self.potential_scope()["target_property"] != "elastic_constants"
            ):
                raise ValueError(
                    "Elastic task requires a generated elemental FCC/BCC conventional crystal"
                )
        if self.potential.catalog and self.potential.application is None:
            if not all(self.potential_scope().values()):
                raise ValueError(
                    "General catalog search requires phase and target_property when they cannot be inferred"
                )
        if self.potential.discover:
            scope = self.potential_scope()
            if not self.llm.decisions or not scope["phase"]:
                raise ValueError("Live discovery requires LLM decisions and a defined phase")
            if self.potential.discover == "nist" and (
                len(self.structure.elements) != 1 or scope["phase"] not in {"fcc", "bcc"}
            ):
                raise ValueError("Legacy NIST discovery requires elemental FCC/BCC")
            if self.protocol.ensemble == "relax":
                if (
                    scope["target_property"]
                    != ("elastic_constants" if self.protocol.elastic else "lattice_constant")
                    or self.protocol.pressure_bar != 0
                ):
                    raise ValueError(
                        "Static NIST discovery requires zero pressure and a matching static target"
                    )
            elif (
                scope["target_property"] not in {"rdf", "msd"}
                or not self.protocol.equilibration_steps
            ):
                raise ValueError(
                    "Dynamic NIST discovery requires RDF/MSD and an explicit equilibration segment"
                )
        if (
            self.potential.phase
            and self.structure.crystal
            and self.potential.phase != self.structure.crystal.lattice
        ):
            raise ValueError("potential.phase does not match structure.crystal.lattice")
        if self.protocol.ensemble == "relax":
            c = self.structure.crystal
            hcp = self.protocol.relax_cell == "hexagonal"
            if not (c or self.structure.path):
                raise ValueError("Static relaxation requires a crystal or an imported unit cell")
            if c and (c.lattice == "hcp") != hcp:
                raise ValueError("Match relax_cell to the selected crystal symmetry")
        return self


def load_config(path: str | Path) -> TaskConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = TaskConfig.model_validate(raw)
    for obj, key in (
        (cfg.structure, "path"),
        (cfg.potential, "model"),
        (cfg.potential, "metadata"),
        (cfg.potential, "catalog"),
        (cfg.potential, "plugin"),
        (cfg.execution, "key_filename"),
    ):
        value = getattr(obj, key)
        if value:
            setattr(obj, key, str((path.parent / Path(value).expanduser()).resolve()))
    executable = cfg.structure.atomsk_executable
    if "/" in executable or "\\" in executable:
        cfg.structure.atomsk_executable = str(
            (path.parent / Path(executable).expanduser()).resolve()
        )
    return cfg
