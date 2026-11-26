"""Interpret natural-language goals and compile them against installed workflow capabilities."""

from __future__ import annotations

import json
import re
import shutil
from math import ceil, gcd, pi, sqrt
from typing import Literal

from ase.data import atomic_numbers
from ase.formula import Formula
from langchain_core.exceptions import OutputParserException
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .agents import _invoke
from .config import (
    LLM,
    Execution,
    HoneycombTube,
    NanotubeModel,
    SlitPhysicalModel,
    StrictModel,
    TaskConfig,
    WallInterface,
)
from .molecular import load_catalog, target_density
from .research import ResearchRequest

METHODS = {
    "elastic_constants": "static_finite_strain",
    "lattice_constant": "static_cell_relaxation",
    "rdf": "equilibrated_md",
    "msd": "equilibrated_md",
    "custom_trajectory": "equilibrated_md",
    "self_diffusion": "molecular_msd_fit",
    "surface_tension": "pressure_tensor",
    "density_profile": "wall_normal_histogram",
    "radial_density_profile": "cylindrical_mass_histogram",
}
MOLECULAR_GEOMETRIES = {
    "molecular_msd_fit": "bulk",
    "pressure_tensor": "slab",
    "wall_normal_histogram": "graphene_slit",
    "cylindrical_mass_histogram": "cnt",
}
LEGACY_PROPERTIES = ["elastic_constants", "lattice_constant", "rdf", "msd"]
CAPABILITIES = {
    "elemental_crystal": {
        "builder": "ASE/Atomsk elemental cubic supercell (FCC, BCC, SC, diamond)",
        "potential": "Live NIST and OpenKIM search by the actual species and target property",
        "properties": LEGACY_PROPERTIES,
    },
    "ordered_crystal": {
        "builder": "ASE ordered two-species cubic cell (B2/CsCl, rocksalt, zincblende)",
        "potential": "Live NIST and OpenKIM search for cross-species interactions",
        "properties": ["lattice_constant", "rdf", "msd"],
    },
    "water": {
        "builder": "Periodic box of explicit H2O molecules",
        "potential": "Rigid SPC/E or TIP3P Ewald",
        "properties": ["rdf", "msd"],
    },
    "configured": {
        "builder": "Use supplied task YAML with its structure, potential and solver settings",
        "properties": LEGACY_PROPERTIES,
    },
}


class SimulationPlan(StrictModel):
    action: Literal["run", "research", "needs_input", "unsupported"]
    research: ResearchRequest | None = None
    system: (
        Literal["elemental_crystal", "ordered_crystal", "water", "configured", "molecular_fluid"]
        | None
    ) = None
    element: str | None = None
    elements: list[str] | None = Field(default=None, min_length=1, max_length=8)
    species: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,32}$")
    target_property: (
        Literal[
            "elastic_constants",
            "lattice_constant",
            "rdf",
            "msd",
            "self_diffusion",
            "surface_tension",
            "density_profile",
            "radial_density_profile",
            "custom_trajectory",
        ]
        | None
    ) = None
    analysis_objective: str | None = Field(default=None, min_length=5, max_length=2000)
    method: (
        Literal[
            "static_finite_strain",
            "static_cell_relaxation",
            "equilibrated_md",
            "finite_temperature",
            "molecular_msd_fit",
            "pressure_tensor",
            "wall_normal_histogram",
            "cylindrical_mass_histogram",
            "other",
        ]
        | None
    ) = None
    phase: (
        Literal[
            "fcc",
            "bcc",
            "sc",
            "diamond",
            "b2",
            "rocksalt",
            "zincblende",
            "liquid",
            "gas",
            "supercritical",
            "other",
        ]
        | None
    ) = None
    temperature_k: float | None = Field(default=None, ge=0)
    pressure_bar: float | None = None
    initial_a_angstrom: float | None = Field(default=None, ge=1.5, le=8)
    repeat: int | None = Field(default=None, ge=2, le=12)
    density_g_cm3: float | None = Field(default=None, gt=0, le=5)
    vacuum_angstrom: float | None = Field(default=None, ge=30, le=120)
    slit_width_angstrom: float | None = Field(default=None, ge=10, le=60)
    slit_model: SlitPhysicalModel | None = None
    cnt_model: NanotubeModel | None = None
    honeycomb_tube: HoneycombTube | None = None
    research_queries: list[str] | None = Field(default=None, min_length=1, max_length=3)
    ensemble: Literal["nvt", "npt", "nve"] | None = Field(
        default=None, validation_alias=AliasChoices("ensemble", "production_ensemble")
    )
    equilibration_ensemble: Literal["nvt", "npt"] | None = None
    timestep_fs: float | None = Field(default=None, ge=0.25, le=2)
    equilibration_steps: int | None = Field(default=None, ge=2000, le=10000000)
    production_steps: int | None = Field(default=None, ge=2000, le=10000000)
    water_model: Literal["auto", "spce", "tip3p_ewald"] | None = None
    reason: str = Field(min_length=5, max_length=1200)

    @field_validator(
        "action",
        "system",
        "target_property",
        "method",
        "phase",
        "ensemble",
        "equilibration_ensemble",
        "water_model",
        mode="before",
    )
    @classmethod
    def normalize_enum(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def complete(self):
        if (self.action == "research") != (self.research is not None):
            raise ValueError("A research action requires a research request")
        if self.action != "run":
            return self
        if any(value is None for value in (self.target_property, self.method, self.temperature_k)):
            raise ValueError("A run plan must specify the property, method and temperature")
        if self.target_property == "custom_trajectory" and not self.analysis_objective:
            raise ValueError("A custom trajectory plan must state the requested analysis_objective")
        if self.analysis_objective and self.method not in {
            "equilibrated_md",
            *MOLECULAR_GEOMETRIES,
        }:
            raise ValueError("A custom analysis requires a dynamic trajectory")
        if self.system == "configured":
            return self
        if self.system == "molecular_fluid":
            fields = (
                self.species,
                self.phase,
                self.ensemble,
                self.equilibration_ensemble,
                self.timestep_fs,
                self.equilibration_steps,
                self.production_steps,
            )
            if any(value is None for value in fields) or self.method not in MOLECULAR_GEOMETRIES:
                raise ValueError(
                    "Molecular calculations require a species, phase, cell and two sampling segments"
                )
            if self.method != "cylindrical_mass_histogram" and self.repeat is None:
                raise ValueError("Provide the geometry's repeat as an integer")
            if self.method == "pressure_tensor":
                if self.phase != "liquid" or self.vacuum_angstrom is None:
                    raise ValueError("Plan a liquid slab and an explicit vapor gap")
            elif self.method == "wall_normal_histogram":
                if (
                    self.species != "water"
                    or self.phase != "liquid"
                    or self.slit_width_angstrom is None
                    or self.slit_model is None
                    or self.vacuum_angstrom is None
                    or self.repeat < 9
                    or self.pressure_bar is not None
                    or self.ensemble != "nvt"
                    or self.equilibration_ensemble != "nvt"
                ):
                    raise ValueError(
                        "Plan liquid water in a fixed graphene slit, with a gap and NVT for both segments"
                    )
            elif self.method == "cylindrical_mass_histogram":
                if (
                    self.species != "water"
                    or self.phase != "liquid"
                    or (self.cnt_model is None) == (self.honeycomb_tube is None)
                    or any(
                        x is not None
                        for x in (self.repeat, self.vacuum_angstrom, self.pressure_bar)
                    )
                    or self.ensemble != "nvt"
                    or self.equilibration_ensemble != "nvt"
                ):
                    raise ValueError(
                        "Cylindrical confinement requires one tube recipe, NVT and null repeat/vacuum/pressure"
                    )
                if self.honeycomb_tube is not None and (
                    not self.research_queries
                    or any(not 8 <= len(query) <= 240 for query in self.research_queries)
                ):
                    raise ValueError(
                        "Supply up to three concise English literature search queries for the declared interface"
                    )
            elif self.pressure_bar is None:
                raise ValueError("Bulk molecular diffusion requires a stated pressure")
            return self
        if self.pressure_bar is None and (
            self.method != "equilibrated_md"
            or self.ensemble == "npt"
            or self.equilibration_ensemble == "npt"
        ):
            raise ValueError("Static and NPT plans must specify external pressure")
        if self.system == "water":
            if (
                self.phase != "liquid"
                or self.element
                or self.initial_a_angstrom is not None
                or not 6 <= (self.repeat or 0) <= 9
            ):
                raise ValueError("Water requires a liquid molecular box with a repeat of 6 to 9")
        elif self.system == "ordered_crystal":
            if (
                self.element is not None
                or len(self.elements or []) != 2
                or self.phase not in {"b2", "rocksalt", "zincblende"}
                or self.initial_a_angstrom is None
                or self.repeat is None
            ):
                raise ValueError(
                    "An ordered binary crystal needs two basis species, a supported prototype, lattice constant and repeat"
                )
        elif any(
            value is None
            for value in (self.element, self.phase, self.initial_a_angstrom, self.repeat)
        ):
            raise ValueError(
                "An elemental crystal needs a symbol, phase, initial lattice constant and repeat"
            )
        if self.method == "equilibrated_md" and any(
            value is None
            for value in (
                self.ensemble,
                self.timestep_fs,
                self.equilibration_steps,
                self.production_steps,
            )
        ):
            raise ValueError(
                "MD needs an ensemble, time step and separate equilibration and production lengths"
            )
        return self


class MolecularIdentity(StrictModel):
    formula: str = Field(pattern=r"^(?:[A-Z][a-z]?[0-9]*){1,24}$")


class ManagerTurn(StrictModel):
    action: Literal[
        "run",
        "revise",
        "research",
        "revise_research",
        "status",
        "result",
        "reply",
        "needs_input",
        "unsupported",
    ]
    plan: SimulationPlan | None = Field(default=None, validate_default=True)
    research: ResearchRequest | None = Field(default=None, validate_default=True)
    message: str | None = Field(default=None, min_length=1, max_length=1600, validate_default=True)

    @field_validator("plan")
    @classmethod
    def check_plan(cls, plan, info: ValidationInfo):
        action = info.data.get("action")
        if action in {"run", "revise"} and (plan is None or plan.action != "run"):
            raise ValueError(
                "This action requires an executable simulation plan with plan.action set to run"
            )
        if action and action not in {"run", "revise"} and plan is not None:
            raise ValueError("Only a new calculation can carry a simulation plan")
        return plan

    @field_validator("research")
    @classmethod
    def check_research(cls, research, info: ValidationInfo):
        action = info.data.get("action")
        if action and (action in {"research", "revise_research"}) != (research is not None):
            raise ValueError(
                "The research object is required for a research action and omitted for other actions"
            )
        return research

    @field_validator("message")
    @classmethod
    def check_message(cls, message, info: ValidationInfo):
        if info.data.get("action") in {"reply", "needs_input", "unsupported"} and not message:
            raise ValueError("A reply must include a message")
        return message


class ResearchContinuation(StrictModel):
    model_config = ConfigDict(extra="ignore")

    research: ResearchRequest | None


class ResearchRecovery(ManagerTurn):
    action: Literal["research", "revise_research", "needs_input", "reply"]


def manager_validation_fields(error: OutputParserException | ValidationError) -> list[dict]:
    cause = error
    while cause is not None:
        if isinstance(cause, ValidationError):
            return [
                {
                    "field": "$"
                    + "".join(
                        f"[{part}]" if isinstance(part, int) else f".{part}"
                        for part in detail["loc"]
                    ),
                    "type": detail["type"],
                    "message": detail["msg"],
                }
                for detail in cause.errors(
                    include_url=False, include_context=False, include_input=False
                )
            ]
        if isinstance(cause, json.JSONDecodeError):
            return [
                {
                    "field": "$",
                    "type": "json_invalid",
                    "message": cause.msg,
                    "line": cause.lineno,
                    "column": cause.colno,
                }
            ]
        cause = cause.__cause__
    return [
        {
            "field": "$",
            "type": "parser_error",
            "message": "The provider returned no usable JSON object.",
        }
    ]


def recover_explanatory_reason(error: OutputParserException | ValidationError, schema: type):
    """Accept the reason/message alias only on non-executable chat actions."""
    if (
        schema is not ManagerTurn
        or not isinstance(error, OutputParserException)
        or not error.llm_output
    ):
        return None
    try:
        data = json.loads(error.llm_output)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("reason"), str):
        return None
    if data.get("action") not in (
        "research",
        "revise_research",
        "needs_input",
        "reply",
        "unsupported",
    ):
        return None
    reason = data.pop("reason")
    if not data.get("message"):
        data["message"] = reason
    try:
        return ManagerTurn.model_validate(data)
    except ValidationError:
        return None


def default_llm() -> LLM:
    return LLM(
        enabled=True,
        decisions=True,
        model="deepseek-v4-pro",
        reasoning_effort="high",
        base_url="https://api.deepseek.com",
        direct_connection=True,
    )


def manager_request(
    request: str,
    *,
    configured: TaskConfig | None = None,
    execution: Execution | None = None,
    smoke: bool = False,
    conversation: dict | None = None,
    reference_evidence: list[dict] | None = None,
    validation_events: list[dict] | None = None,
) -> tuple[SimulationPlan | ManagerTurn, dict, LLM]:
    request = request.strip()
    if not request:
        raise ValueError("Give a nonempty simulation request")
    if configured and smoke:
        raise ValueError(
            "--smoke applies to automatically planned protocols; configure the supplied YAML directly"
        )
    conditions = {}
    for field, unit in (("temperature_k", "k|开尔文"), ("pressure_bar", "bar|巴")):
        values = sorted(
            {
                float(v)
                for v in re.findall(
                    rf"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(?:{unit})(?!\w)", request, re.I | re.ASCII
                )
            }
        )
        if values:
            conditions[field] = values
    slit_widths = re.findall(
        r"(?:slit width|plate separation|板间距|受限高度|通道高度)\s*(?:of\s+|为\s*|=\s*)?"
        r"(\d+(?:\.\d+)?)\s*(nm|纳米|Å|angstroms?|埃)",
        request,
        re.I,
    )
    if slit_widths:
        conditions["slit_width_angstrom"] = sorted(
            {
                float(value) * (10 if unit.lower() in {"nm", "纳米"} else 1)
                for value, unit in slit_widths
            }
        )
    constraints = {key: values[0] for key, values in conditions.items() if len(values) == 1}
    multiple = {key: values for key, values in conditions.items() if len(values) > 1}
    if multiple and conversation is None:
        raise ValueError(
            f"The automatic runner needs a single explicit {next(iter(multiple))} per run"
        )
    llm = configured.llm if configured and configured.llm.enabled else default_llm()
    events = validation_events if validation_events is not None else []
    evidence = {
        "request": request,
        "explicit_constraints": constraints,
        "methods": {**METHODS, "static_reference": "https://docs.lammps.org/Howto_elastic.html"},
        "material_evidence": {
            "Au": "NIST JARVIS-FF JLMP-1109 reports the elemental FCC conventional cell."
        },
        "sampling_mode": "smoke" if smoke else "production",
        "generated_analysis": "An Analyst can implement a new estimator in Python from sampled positions, "
        "unwrapped positions, cell and origin, boundary flags, atom IDs/types, time, "
        "and molecule IDs when recorded. Velocities, forces, and charges are unavailable.",
        "runner_capabilities": {
            key: value
            for key, value in CAPABILITIES.items()
            if (key == "configured") == (configured is not None)
        },
    }
    if reference_evidence:
        evidence["literature_evidence"] = reference_evidence
    if re.search(r"CNT|BNNT|nanotube|纳米管", request, re.I):
        indices = re.search(r"\(\s*(\d+)\s*[,，]\s*(\d+)\s*\)", request)
        shorthand = re.search(r"(?<!\d)(\d{1,2})\1\s*CNT\b", request, re.I)
        if indices:
            evidence["explicit_cnt_chirality"] = [int(i) for i in indices.groups()]
        elif shorthand:
            evidence["explicit_cnt_chirality"] = [int(shorthand[1])] * 2
    if multiple:
        evidence["multiple_explicit_conditions"] = multiple
    if configured:
        evidence["provided_system"] = {
            "goal": configured.goal,
            "elements": configured.structure.elements,
            "crystal": configured.structure.crystal.model_dump()
            if configured.structure.crystal
            else None,
            "is_argon_demo": configured.structure.demo_argon,
            "has_imported_structure": configured.structure.path is not None,
            "is_water": bool(configured.structure.water_repeat),
            "protocol": configured.protocol.model_dump(),
            "potential": configured.potential.kind,
        }
        instruction = (
            "The user supplied the system and protocol. Decide whether they can produce the requested "
            "observable without changing them. A dynamic NVT/NPT/NVE protocol can produce RDF, MSD, "
            "or a custom_trajectory observable. For the last case also set analysis_objective to the "
            "requested property and its species, conditions or analysis parameters as stated. "
            "A supplied protocol without a separate equilibration can still produce exploratory data; "
            "mention the sampling limitation. For a runnable plan set system to configured, repeat the "
            "provided temperature and pressure, and use the method ID from evidence.methods. "
            "If there is a mismatch describe it and return needs_input or unsupported. "
            "Do not choose any other system or propose model, structure, or protocol changes."
        )
        example = (
            '{"action":"run","system":"configured","target_property":"rdf",'
            '"method":"equilibrated_md","temperature_k":50,"pressure_bar":null,'
            '"reason":"The supplied protocol can produce a trajectory for RDF analysis."}'
        )
    else:
        catalog = load_catalog()
        evidence["molecular_tools"] = {
            "species_registry": {key: item.model_dump() for key, item in catalog.species.items()},
            "methods": MOLECULAR_GEOMETRIES,
            "analysis": {
                "pressure_tensor": "planar liquid-vapor slab, 3D periodic cell, two interfaces; "
                "block pressure-anisotropy averaging with long-range dispersion",
                "molecular_msd_fit": "unwrapped molecular center-of-mass MSD, linear regime, block uncertainty",
                "wall_normal_histogram": "fixed graphene monolayers with SPC/E water and published C–O LJ; "
                "mass density is binned from water molecular centers relative to the lower plate",
                "cylindrical_mass_histogram": "rigid honeycomb nanotube, molecular water and verified "
                "wall junction parameters; 3D PPPM with lateral vacuum; divide mass by cylindrical shell volumes",
            },
            "slit_model_schema": SlitPhysicalModel.model_json_schema(),
            "cnt_model_schema": NanotubeModel.model_json_schema(),
            "honeycomb_tube_schema": HoneycombTube.model_json_schema(),
            "assembly_tools": [
                "ASE roll a honeycomb lattice",
                "assign one or two lattice sublattices by ASE neighbors",
                "repeat axially and add transverse padding",
                "pack rigid molecules inside the pore",
                "resolve wall parameters from primary literature",
                "LAMMPS run 0 validation",
            ],
            "configuration": "For registered species, the specialists choose packing and a documented force field. "
            "For an unregistered species provide a concise ASCII species name and its unique element "
            "symbols in elements. The potential Agent will search the repositories before a topology "
            "can be considered. No assumed topology or coefficients in the plan.",
        }
        evidence["sampling_references"] = {
            "water_msd": "https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html",
            "water_diffusion_finite_size": "https://doi.org/10.1021/jp0477147",
            "rigid_water_transport_example": {
                "source": "https://doi.org/10.1088/0953-8984/24/28/284117",
                "observation": "This study used 1–10 ns trajectories with 1 fs steps for rigid-water transport. "
                "The published lengths are an example for that observable and setup, not bounds or "
                "evidence that a new plan is converged.",
            },
        }
        if execution is not None:
            evidence["execution_budget"] = {
                "cores": execution.tasks * execution.cpus_per_task,
                "walltime": execution.walltime
                if smoke
                else execution.production_walltime or execution.walltime,
            }
        sampling = (
            "In smoke mode use short segments for pipeline verification, for example 10–20 ps equilibration and "
            "20–40 ps production unless the user provides a different duration. Label the output exploratory. "
            if smoke
            else "Plan a production study. Choose and justify the two physical durations for this property and material, "
            "with no generic short-run duration. Explain equilibration indicators, time correlation and statistical "
            "block consistency, and any resource limitation. Use reference data only when applicable. "
            "For pressurized liquid-water MSD equilibrate in NPT then sample in NVT in the equilibrated cell. "
            "Implemented checks are blockwise temperature and NPT volume in equilibration and block consistency of "
            "the target observable in production. Distinguish proposed checks outside the workflow from implemented ones. "
            "For explicit water, repeat is the number of molecule centers along one side: one box of repeat cubed "
            "molecules, never independent boxes or replicates. Separate inferred conditions from those given by the user. "
            "A single box and model cannot independently validate a diffusion coefficient. "
        )
        instruction = (
            "Interpret the original wording and select compatible tools from runner_capabilities. Preserve explicit "
            "constraints, material, and observable. Use the canonical element symbol for elemental crystals. "
            "Otherwise choose and justify temperature, pressure, method, and cell for the request. "
            "Automatic static elastic supports elemental FCC/BCC at 0 K and 0 bar. Cubic lattice relaxation also "
            "supports elemental SC/diamond and ordered b2, rocksalt and zincblende alloys/compounds. "
            "For RDF/MSD select equilibrated_md, a supported phase, NVT or NPT, a 0.25–2 fs step, and separate "
            "equilibration and production lengths, each at most 10 million steps. If pressure is specified, "
            "use NPT in equilibration or production. Set equilibration_ensemble when it differs. "
            "For an unregistered analysis that uses the available trajectory fields and a supported bulk "
            "system, set target_property to custom_trajectory, method to equilibrated_md, and "
            "analysis_objective to the exact requested observable with selection, units or parameters if "
            "specified. Justify a sampling length and save cadence adequate for the observable. "
            "For an already supported molecular setup select its matching method and add analysis_objective "
            "for a new observable calculated from that trajectory. An unsupported material, interface or "
            "required trajectory field cannot be made available by changing the observable. "
            + sampling
            + "For liquid H2O use system water, phase liquid, no element or lattice constant, repeat 6–9, "
            "and water_model auto to delegate comparison unless a supported specific model is requested. "
            "For elemental crystals use canonical element and an explicit cubic prototype, and repeat 2–6. "
            "For an ordered two-species crystal (including materials absent from all local directories) set "
            "system ordered_crystal, elements to the two canonical symbols in basis order, element null, "
            "phase b2, rocksalt or zincblende as appropriate, and repeat 2–6. Supply an initial lattice constant, "
            "never a force-field ID or file. The potential Agent will query both repositories. For B2 map "
            "CsCl-type/B2 to b2, not bcc. If either segment uses npt or the plan is static, supply numeric pressure_bar. For nvt without "
            "an imposed pressure, use pressure_bar null. Use lowercase enum keys. "
            "Return needs_input for scientific ambiguity or required user data. "
            "For unsupported tasks name the missing builder, potential, protocol, or analysis."
        )
        instruction += (
            " The molecular_tools registry provides a separate composable molecular capability. Use system molecular_fluid "
            "for self_diffusion, surface_tension, density_profile and radial_density_profile, with element and water_model null. "
            "For water inside an armchair CNT requesting radial density, use radial_density_profile and "
            "cylindrical_mass_histogram (a tube-axis density profile). This model is axially periodic, rigid and filled "
            "at fixed N, without reservoirs. Use species water, phase liquid, NVT for both segments, and null repeat, "
            "vacuum_angstrom and pressure_bar. Provide the nested cnt_model JSON object following its schema, with "
            "chirality, axial_cells, length_angstrom, transverse_box_angstrom and water_molecules. Preserve exact "
            "user-supplied geometry and loading; when unspecified choose and explain them. Estimate the loading "
            "numerically from D = 3*1.42*n/pi in angstrom, V = pi*((D-3.19)/2)^2*L, and N = V/29.915 at 1 g/cm³. "
            "For (12,12) this gives approximately 4.50 water molecules per angstrom of tube, not hundreds in a "
            "single nanometer. Consider whether a longer tube is appropriate for reducing axial finite-size effects. "
            "For a radial structural "
            "observable choose equilibration and production suitable for time-block comparison, typically hundreds of "
            "ps for equilibration and 1–2 ns for an initial production trajectory at ambient conditions. "
            "For water confined between graphene plates and a density curve normal to the plates, use "
            "density_profile and wall_normal_histogram; never use a free liquid-vapor slab for this request. "
            "Set species water, phase liquid, repeat 9–12 (in-plane graphene cell repeats), numeric "
            "slit_width_angstrom (carbon-plane separation) and vacuum_angstrom (exterior gap), "
            "NVT in both segments and pressure_bar null. State the inferred slit width and choose "
            "a near-liquid loading density unless explicitly specified. A graphene-slit plan MUST contain the exact "
            'numeric key "slit_width_angstrom" in JSON as well as "vacuum_angstrom"; e.g. '
            '{"slit_width_angstrom": 20, "vacuum_angstrom": 30}. '
            "For this method provide a nested slit_model object matching evidence.molecular_tools.slit_model_schema. "
            'For example {"wall_material":"graphene","wall_count":2,"wall_motion":"fixed","wall_stacking":"aa",'
            '"fluid_representation":"explicit_water","periodic_axes":"xy","cross_interaction":"carbon_oxygen_lj",'
            '"loading_exclusion_per_wall_angstrom":2.5}. Choose AA or AB and explain inferred stacking and '
            "loading. The Potential Agent applies validated SPC/E and sourced C–O coefficients. "
            "Use the registered species ID when it "
            "matches. Otherwise supply a concise ASCII species name and every unique chemical element in elements for "
            "potential discovery. A molecular plan must return all these keys: target_property, method, species, phase, "
            "repeat (integer), temperature_k, "
            "pressure_bar (number or null), vacuum_angstrom (number for a slab, null for bulk), ensemble "
            "(the production ensemble), equilibration_ensemble, timestep_fs, equilibration_steps, and production_steps. "
            "Write the production ensemble under ensemble. Give numeric repeat and, for the slab, numeric vacuum_angstrom; "
            "putting the quantity in reason alone leaves the plan incomplete. Map a diffusion coefficient request for "
            "a pure molecular substance to self_diffusion, not to the MSD "
            "curve alone. For a bare request assume a pure substance and infer a conventional temperature and physically "
            "consistent phase/pressure; explain assumed conditions. At room conditions pure CO2 near 1 bar is a dilute gas; "
            "ethanol is a liquid. For dilute gas up to 5 bar plan NVT equilibration with ideal gas density initialized by the "
            "runner and pressure checked, followed by NVE production. For liquid bulk use NPT equilibration and fixed-volume "
            "NVE or NVT production. For the planar liquid-vapor surface set phase liquid, NVT for both segments, "
            "pressure_bar null unless the user mentions pressure, and a 45–80 Å gap. Use repeat 8–10 for the slab, "
            "and 7–10 for bulk. Start the slab and liquid bulk from a registry near-liquid density; dense gas/supercritical "
            "requires a sourced initial density. Choose a stable timestep (1–2 fs for rigid registered models, "
            "at most 1 fs for ethanol with a stiff flexible bond); "
            "use a production campaign suited to the observable (typically 0.2–0.6 ns equilibration and 1–3 ns production "
            "for first ambient liquid estimates, and gas diffusion needs multiple molecular collision times, typically "
            "2–5 ns at 1 bar). Use longer sampling as needed when diagnostic checks fail. Never claim convergence from duration. "
            "For smoke only, choose short exploratory times to test the workflow. For unregistered species, choose the "
            "molecular plan to route discovery, while explaining that a sourced compatible topology is required to run MD. "
            "Do not replace a requested species or observable."
        )
        instruction += (
            " A single-wall nanotube with other honeycomb wall elements (including alternating two-element tubes) "
            "can use the same cylindrical_mass_histogram method and radial_density_profile. Set cnt_model=null "
            "and provide honeycomb_tube with wall_elements in sublattice order, chirality, bond_angstrom, "
            "axial_cells, transverse_box_angstrom, fluid_molecules. Use explicit liquid water, NVT both segments, "
            "null repeat, vacuum_angstrom and pressure_bar. Include research_queries, 2–3 English bibliographic "
            "queries for the exact wall material, water model/interaction parameters, and observable. "
            "Keep one broad high-recall query (wall material spelled out, nanotube, confined water); keep another "
            "query focused on primary simulation papers about structure/density. "
            "The Potential Agent finds coefficient evidence and can block an unsupported interface after research. "
            "Do not invent force-field values or reject an alternating honeycomb wall solely because it is new."
        )
        instruction += (
            " Keep reason in two or three short sentences, at most 650 characters. For example, a molecular plan can "
            'contain {"action":"run","system":"molecular_fluid","target_property":"surface_tension",'
            '"method":"pressure_tensor","species":"water","phase":"liquid","temperature_k":298.15,'
            '"pressure_bar":null,"repeat":8,"vacuum_angstrom":60,"ensemble":"nvt","equilibration_ensemble":"nvt",'
            '"timestep_fs":2,"equilibration_steps":250000,"production_steps":1000000,'
            '"reason":"Pure water at an inferred ambient temperature. Use a vapor-liquid slab and sample the pressure tensor."}. '
            "Give target_property and method explicitly, even when obvious from the request."
        )
        instruction += (
            " For a physically meaningful calculation that needs a new geometry, bonded topology, "
            'force field or analysis adapter, use action "research" and a separate "research" object, '
            'e.g. {"system":"requested material and fluid","geometry":"requested geometry",'
            '"observable":"requested observable","queries":["material geometry fluid molecular dynamics",'
            '"material fluid force field interface"]}. This starts real literature searches. Preserve '
            "the original system and observable even if existing builders only support a different material. "
            "Do not put missing adapters in needs_input or unsupported. Supply search_terms with the exact "
            "material including its phase or morphology, the fluid or other medium (null if absent), and "
            "the most specific geometry, using concise English. Prefer conventional short literature terms such as "
            '"slit pores" or "nanotubes" when they match. Distinguish a planar slit from a cylinder and '
            "a bulk material shaped as a slab from a genuinely two-dimensional crystal. "
            "Infer obvious spelling errors from the physics (e.g. 法相密度 means 法向密度 for a wall), "
            "and keep genuine alternatives distinct. English queries should include accurate synonyms "
            "for the geometry and property instead of replacing them with general interface terms. "
            "For research put the descriptive system name only inside the research object. Omit all "
            "simulation fields, including the top-level system field. For non-chat research return reason as well."
        )
        example = (
            '{"action":"run","system":"elemental_crystal","element":"Ag","target_property":"lattice_constant",'
            '"method":"static_cell_relaxation","phase":"fcc","temperature_k":0,"pressure_bar":0,'
            '"initial_a_angstrom":4.09,"repeat":3,"reason":"Relax the FCC cell to determine its equilibrium size."}'
        )
    schema = SimulationPlan
    pending_research = (conversation or {}).get("pending_research")
    research_example = (
        '{"system":"named material and fluid","geometry":"requested geometry","observable":"observed property",'
        '"search_terms":{"material":"named material and phase","medium":"named fluid","geometry":"distinct geometry"},'
        '"queries":["named material water confinement","named material fluid observable molecular dynamics",'
        '"named material fluid force field interface"]}'
    )
    if pending_research and not configured:
        context = {
            "request": request,
            "active_research": pending_research["target"],
            "original_goal": pending_research["goal"],
            "prior_explicit_conditions": pending_research.get("explicit_conditions", {}),
            "new_explicit_conditions": constraints,
        }
        try:
            continuation = _invoke(
                llm,
                "manager",
                context,
                ResearchContinuation,
                "Decide only whether the latest user message revises or clarifies the active research study. "
                "For a correction or a request to advance the same study, put the full updated research target "
                "in research. Preserve the original observable and user conditions except where the author changes them. "
                "Apply the latest correction immediately. When the user asks you to choose conditions from papers, "
                "those conditions are research outputs; start discovery and do not ask the user to supply them. "
                "This only prepares an author-review draft. Fill search_terms with concise English material, "
                "medium and the specific geometry. For a separate task, a status or an explanation-only question, "
                "return research null. The only top-level key is research.",
                '{"research":' + research_example + '} or {"research":null}',
            )
        except (OutputParserException, ValidationError) as error:
            events.append(
                {
                    "stage": "research_continuation",
                    "schema": ResearchContinuation.__name__,
                    "errors": manager_validation_fields(error),
                    "handling": "continue_to_manager",
                }
            )
        else:
            if continuation.research:
                return (
                    ManagerTurn(action="revise_research", research=continuation.research),
                    evidence,
                    llm,
                )
    if conversation is not None:
        schema = ManagerTurn
        evidence["conversation"] = conversation
        directive = (
            "You are the main agent facing the user. Interpret the original request (English or Chinese) in context "
            "and return one action. Use run plus a complete nested plan for an explicit new calculation, including "
            "static lattice relaxation and finite strain, and for an explicit rerun with changes. For a rerun use "
            "the active run to resolve references such as 'it', preserving relevant conditions except requested changes. "
            "Use status for monitoring the active job and result for retrieving its output. The runner obtains actual "
            "scheduler status and results for these actions. Use reply with an English message for conceptual, how-to, "
            "or reference-value questions, even if the question is in Chinese, unless it explicitly requests another "
            "output language. Answer only the question asked. Do not volunteer a job status or result in a general reply. "
            "Use needs_input or unsupported with a message "
            "when a requested calculation is consequentially ambiguous with no defensible inference or is not an MD task. "
            "For a clear MD goal outside the ready-made tools choose research when available. "
            "Use only action, plan, research and message as top-level keys. Put any chat explanation in message; "
            "reason belongs inside an executable plan. For needs_input put one focused question in message. "
            "Only run and revise carry a plan. Do not invent a job, status, measurement, or result. "
            "Apply the following scientific instructions only when planning a calculation. "
        )
        if conversation.get("pending_plan"):
            directive += (
                "A pending plan has been presented for author review. Use revise plus the entire revised plan when "
                "the author changes its parameters. Preserve the objective and all unaffected parameters. "
                "For questions about that proposal use reply. Use run only for a separate new objective. "
                "Submission is handled by the chat's explicit confirmation command."
            )
        if pending_research:
            directive += (
                "A research draft is active. For a clarification, an answer to a pending assumption, or a request "
                "to refine sources or choices for that same objective, use revise_research with the complete updated "
                "research object. Carry forward the prior material, geometry, observable and explicit conditions "
                "except for changes requested by the author; the latest author wording controls. Keep a research "
                "question in reply if it only asks for an explanation. Use research for a distinct new study. "
                "Apply an explicit correction immediately. When the user asks the agent to choose parameters from "
                "literature, the unchosen parameters are research deliverables. Start research without asking "
                "the user to choose them or to restate the observable. Research is an author-review step."
            )
        instruction = directive + instruction
        example = '{"action":"run","plan":' + example + ',"message":null}'
        if configured is None:
            example += (
                ' or, for discovery only, {"action":"research","research":'
                + research_example
                + '} (use "revise_research" for an update to the active research draft)'
            )
    elif configured is None:
        example += (
            ' or, for discovery only, {"action":"research","research":'
            + research_example
            + ',"reason":"A new model needs primary-source evidence."}'
        )
    prompt_evidence = evidence
    for stage in ("initial", "retry"):
        try:
            return (
                _invoke(llm, "manager", prompt_evidence, schema, instruction, example),
                evidence,
                llm,
            )
        except (OutputParserException, ValidationError) as error:
            fields = manager_validation_fields(error)
            recovered = recover_explanatory_reason(error, schema)
            events.append(
                {
                    "stage": stage,
                    "schema": schema.__name__,
                    "errors": fields,
                    "handling": "reason_to_message"
                    if recovered
                    else "regenerate"
                    if stage == "initial"
                    else "rejected",
                }
            )
            if recovered is not None:
                return recovered, evidence, llm
            # Keep diagnostics ahead of the potentially large capability catalog in the prompt.
            prompt_evidence = {"validation_fields": fields, **evidence}
            instruction += (
                " Correct the listed fields and keep the original task and explicit conditions."
            )
    if conversation is not None and configured is None:
        try:
            recovered = _invoke(
                llm,
                "manager",
                {
                    "request": request,
                    "explicit_constraints": constraints,
                    "pending_research": pending_research,
                    "validation_fields": fields,
                },
                ResearchRecovery,
                "Recover a non-executable response to the original task. For a clear MD research goal, put the "
                "same material, geometry and observable in research. For a clarification of an active research "
                "draft, use revise_research and carry forward its unaffected conditions. If materially different "
                "physical models fit the request, ask one focused question with action needs_input. Use reply for "
                "an explanation. Interpret obvious typos in context, such as 法相密度 for the density normal to a wall. "
                "Output only action, research and message; never include plan or reason.",
                '{"action":"research","research":'
                + research_example
                + '} or {"action":"needs_input","message":"Which physical model do you mean?"}',
            )
        except (OutputParserException, ValidationError) as error:
            fields = manager_validation_fields(error)
            events.append(
                {
                    "stage": "research_recovery",
                    "schema": ResearchRecovery.__name__,
                    "errors": fields,
                    "handling": "rejected",
                }
            )
        else:
            events.append(
                {
                    "stage": "research_recovery",
                    "schema": ResearchRecovery.__name__,
                    "errors": [],
                    "handling": recovered.action,
                }
            )
            return recovered, evidence, llm
    summary = "; ".join(
        f"{item['field']} ({item['type']}): {item['message']}" for item in fields[:4]
    )
    raise ValueError(f"The Manager could not produce a valid response. {summary}")


def task_from_plan(
    plan: SimulationPlan,
    evidence: dict,
    llm: LLM,
    lammps: str = "lmp",
    *,
    on_plan=None,
    configured: TaskConfig | None = None,
    execution: Execution | None = None,
    smoke: bool = False,
) -> TaskConfig:
    """Preserve explicit user constraints when compiling the Manager's proposal."""
    if plan.action != "run":
        raise ValueError(f"Manager plan: {plan.action}. {plan.reason}")
    multiple = evidence.get("multiple_explicit_conditions")
    if multiple:
        raise ValueError(
            f"The automatic runner needs a single explicit {next(iter(multiple))} per run"
        )
    constraints = evidence["explicit_constraints"]
    request = evidence["request"]
    if any(getattr(plan, key) != val for key, val in constraints.items()):
        raise ValueError(
            "Manager proposal changed explicit user conditions; no calculation was launched"
        )
    explicit_cnt = evidence.get("explicit_cnt_chirality")
    tube = plan.honeycomb_tube or plan.cnt_model
    if explicit_cnt and (tube is None or list(tube.chirality) != explicit_cnt):
        raise ValueError(
            "Manager proposal changed the explicit CNT chirality; no calculation was launched"
        )
    if plan.method != METHODS[plan.target_property]:
        raise ValueError("The selected method cannot produce the requested observable")
    if (plan.method in MOLECULAR_GEOMETRIES) != (plan.system == "molecular_fluid"):
        raise ValueError("The chosen method must match the registered molecular system capability")
    if configured and plan.system != "configured":
        raise ValueError("A supplied system requires a compatible configured plan")
    if "pressure_bar" in constraints and plan.method == "equilibrated_md":
        ensembles = (
            {configured.protocol.ensemble, configured.protocol.equilibration_ensemble}
            if configured
            else {plan.ensemble, plan.equilibration_ensemble}
        )
        if "npt" not in ensembles:
            raise ValueError("A stated pressure requires an NPT plan")
    research = None
    if plan.system == "configured":
        task = compile_configured(plan, configured, request, llm)
        if execution is not None:
            task = TaskConfig.model_validate(
                {**task.model_dump(), "execution": execution.model_dump()}
            )
    elif plan.system == "molecular_fluid":
        interface = None
        if plan.honeycomb_tube is not None:
            from .literature import resolve_interface

            interface, research = resolve_interface(
                request, plan.honeycomb_tube, plan.research_queries, llm
            )
            plan = refine_tube_study(
                plan, request, constraints, interface, research.pop("planning_evidence"), llm
            )
        task = compile_molecular(
            plan, request, lammps, llm, execution=execution, smoke=smoke, interface=interface
        )
    else:
        task = compile_generated(plan, request, lammps, llm, execution=execution, smoke=smoke)
    if plan.analysis_objective:
        task = TaskConfig.model_validate(
            {**task.model_dump(), "analysis_objective": plan.analysis_objective}
        )
    if on_plan is not None:
        record = {
            "role": "manager",
            "model": llm.model,
            "evidence": evidence,
            "proposal": plan.model_dump(),
        }
        if research:
            record["potential_research"] = research
        on_plan(record)
    return task


def tube_loading_estimate(tube: HoneycombTube, interface: WallInterface) -> dict:
    n, m = tube.chirality
    roll = sqrt(n * n + n * m + m * m)
    diameter = sqrt(3) * tube.bond_angstrom * roll / pi
    length = 3 * tube.bond_angstrom * roll / gcd(2 * n + m, 2 * m + n) * tube.axial_cells
    exclusion = min(site.sigma_angstrom for site in interface.sites)
    volume = pi * ((diameter - exclusion) / 2) ** 2 * length
    return {
        "geometric_diameter_angstrom": diameter,
        "periodic_length_angstrom": length,
        "excluded_volume_loading_g_cm3": 29.915 * tube.fluid_molecules / volume,
        "loading_at_1_g_cm3": round(volume / 29.915),
        "loading_at_1_25_g_cm3": round(1.25 * volume / 29.915),
    }


def composed_plan_reason(plan: SimulationPlan, interface: WallInterface) -> str:
    tube = plan.honeycomb_tube
    estimate = tube_loading_estimate(tube, interface)
    ps_per_step = plan.timestep_fs / 1000
    return (
        f"The proposed axial period is {estimate['periodic_length_angstrom']:.3f} Å and holds "
        f"{tube.fluid_molecules} water molecules; the loading estimate is "
        f"{estimate['excluded_volume_loading_g_cm3']:.3f} g/cm³ after excluding one cross-site diameter "
        f"from the geometric pore diameter. Use the wall interaction model sourced to {interface.source_doi}. "
        f"After {plan.equilibration_steps * ps_per_step:g} ps of NVT equilibration, sample "
        f"{plan.production_steps * ps_per_step:g} ps and compare five consecutive time blocks."
    )


def refine_tube_study(
    plan: SimulationPlan,
    request: str,
    constraints: dict,
    interface: WallInterface,
    source_text: str,
    llm: LLM,
) -> SimulationPlan:
    tube = plan.honeycomb_tube
    estimate = tube_loading_estimate(tube, interface)
    example = plan.model_dump(mode="json", exclude_none=True)
    example["honeycomb_tube"]["fluid_molecules"] = estimate["loading_at_1_g_cm3"]
    example["reason"] = (
        "Use the evidence-supported geometry and loading. Sample the profile in five time blocks."
    )
    evidence = {
        "request": request,
        "user_supplied_conditions": constraints,
        "wall_sublattices": tube.wall_elements,
        "chirality": tube.chirality,
        "selected_doi": interface.source_doi,
        "verified_interface": interface.scope,
        "initial_geometry_estimate_only": estimate,
        "primary_source_methods": source_text,
    }
    instruction = (
        "Return the complete SimulationPlan for this exact material and radial observable. User-supplied conditions "
        "are only those stated by the user in the request, not earlier internally chosen geometry or water count. "
        "Prefer directly reported geometry and loading for the exact chirality in the source if the user did not "
        "specify them. Keep the same research queries, NVT both segments, and null imposed pressure. State in at "
        "most three short sentences (under 480 characters) what was taken from the study and why the sampling "
        "duration is suitable for comparing five time blocks. Return a runnable plan using honeycomb_tube."
    )
    error = ""
    for _ in range(2):
        try:
            refined = _invoke(
                llm, "manager", evidence, SimulationPlan, instruction, json.dumps(example)
            )
            candidate = refined.honeycomb_tube
            if (
                refined.action != "run"
                or refined.target_property != plan.target_property
                or candidate is None
                or candidate.chirality != tube.chirality
                or candidate.wall_elements != tube.wall_elements
                or any(getattr(refined, key) != value for key, value in constraints.items())
            ):
                raise ValueError(
                    "The refinement changed a fixed user condition or the material identity"
                )
            capacity = tube_loading_estimate(candidate, interface)
            if candidate.fluid_molecules > capacity["loading_at_1_25_g_cm3"]:
                raise ValueError(
                    f"Loading {candidate.fluid_molecules} exceeds the supported estimate for "
                    f"the proposed tube. Geometry and loading estimates: {capacity}. "
                    "Use the reported loading if it matches the selected chirality and length."
                )
            return refined.model_copy(update={"reason": composed_plan_reason(refined, interface)})
        except (OutputParserException, ValueError) as exc:
            error = str(exc)[-1500:]
            evidence["validation_feedback"] = error
    raise ValueError("The literature-refined proposal did not pass physical validation: " + error)


def compile_molecular(
    plan: SimulationPlan,
    request: str,
    lammps: str,
    llm: LLM,
    *,
    execution: Execution | None,
    smoke: bool,
    interface: WallInterface | None = None,
) -> TaskConfig:
    catalog = load_catalog()
    species = catalog.species.get(plan.species)
    if species is None or plan.phase not in species.phases:
        elements = species.elements if species else plan.elements
        if not elements:
            identity = _invoke(
                llm,
                "manager",
                {"request": request, "species": plan.species},
                MolecularIdentity,
                "Give the molecular formula of the requested pure species without changing the identity. "
                "Use case-sensitive element symbols and their integer counts.",
                '{"formula":"C2H6O"}',
            )
            elements = list(Formula(identity.formula).count())
        if not elements or not all(atomic_numbers.get(element, 0) for element in elements):
            raise ValueError(
                "Provide the unregistered molecule identity and its chemical elements for discovery"
            )
        return TaskConfig.model_validate(
            {
                "name": f"{plan.species}_{plan.target_property}",
                "goal": request,
                "structure": {"elements": elements, "research_species": plan.species},
                "potential": {
                    "kind": "auto",
                    "discover": "repositories",
                    "phase": plan.phase,
                    "target_property": plan.target_property,
                },
                "protocol": {
                    "temperature_k": plan.temperature_k,
                    "pressure_bar": plan.pressure_bar or 0,
                },
                "execution": execution or Execution(),
                "llm": llm,
            }
        )
    geometry = MOLECULAR_GEOMETRIES[plan.method]
    if geometry == "bulk" and (not plan.pressure_bar or plan.pressure_bar < 0):
        raise ValueError("Bulk transport requires a positive target pressure")
    if geometry == "bulk" and plan.phase != "gas" and plan.equilibration_ensemble != "npt":
        raise ValueError("Dense bulk transport equilibrates the box under the specified pressure")
    if (
        geometry == "bulk"
        and plan.phase == "gas"
        and plan.pressure_bar > 5
        and plan.equilibration_ensemble != "npt"
    ):
        raise ValueError("Non-dilute gas requires density equilibration at the specified pressure")
    density = target_density(
        catalog, plan.species, plan.phase, plan.temperature_k, plan.pressure_bar, plan.density_g_cm3
    )
    if execution is None:
        solver = shutil.which(lammps)
        if not solver:
            raise ValueError("LAMMPS executable not found; use --lammps to specify it")
        execution = Execution(command=[solver], timeout_seconds=1200 if smoke else 86400)
    elif not smoke and execution.production_walltime:
        h, m, s = map(int, execution.production_walltime.split(":"))
        execution = execution.model_copy(
            update={
                "walltime": execution.production_walltime,
                "timeout_seconds": h * 3600 + m * 60 + s,
            }
        )
    target_frames = {"bulk": 1200, "slab": 100, "graphene_slit": 500, "cnt": 500}
    stride = max(1, ceil(plan.production_steps / target_frames[geometry]))
    thermo = max(1, ceil(plan.production_steps / 2000))
    wall = (
        list(plan.honeycomb_tube.wall_elements)
        if plan.honeycomb_tube
        else ["C"]
        if geometry in {"graphene_slit", "cnt"}
        else []
    )
    elements = [*species.elements, *wall]
    potential = {"kind": "molecular", "target_property": plan.target_property}
    if interface:
        potential.update(candidate_id=interface.fluid_model, wall_interface=interface)
    return TaskConfig.model_validate(
        {
            "name": f"{plan.species}_{plan.target_property}",
            "goal": request,
            "structure": {
                "elements": elements,
                "molecular": {
                    "species": plan.species,
                    "phase": plan.phase,
                    "geometry": geometry,
                    "repeat": plan.repeat,
                    "density_g_cm3": density,
                    "vacuum_angstrom": plan.vacuum_angstrom,
                    "slit_width_angstrom": plan.slit_width_angstrom,
                    "slit_model": plan.slit_model,
                    "cnt_model": plan.cnt_model,
                    "honeycomb_tube": plan.honeycomb_tube,
                },
            },
            "potential": potential,
            "protocol": {
                "ensemble": plan.ensemble,
                "equilibration_ensemble": plan.equilibration_ensemble,
                "timestep_fs": plan.timestep_fs,
                "temperature_k": plan.temperature_k,
                "pressure_bar": plan.pressure_bar or 0,
                "steps": plan.production_steps,
                "equilibration_steps": plan.equilibration_steps,
                "thermo_every": thermo,
                "dump_every": stride,
                "minimize": plan.phase != "gas",
                "sampling_mode": "smoke" if smoke else "production",
            },
            "execution": execution.model_dump(),
            "llm": llm,
        }
    )


def task_from_request(
    request: str,
    lammps: str = "lmp",
    *,
    on_plan=None,
    configured: TaskConfig | None = None,
    execution: Execution | None = None,
    smoke: bool = False,
) -> TaskConfig:
    plan, evidence, llm = manager_request(
        request, configured=configured, execution=execution, smoke=smoke
    )
    return task_from_plan(
        plan,
        evidence,
        llm,
        lammps,
        on_plan=on_plan,
        configured=configured,
        execution=execution,
        smoke=smoke,
    )


def compile_configured(
    plan: SimulationPlan, configured: TaskConfig | None, request: str, llm: LLM
) -> TaskConfig:
    if configured is None:
        raise ValueError("This plan requires a task YAML supplied via --system")
    p = configured.protocol
    expected_pressure = (
        None if p.ensemble == "nvt" and plan.pressure_bar is None else p.pressure_bar
    )
    if (p.temperature_k, expected_pressure) != (plan.temperature_k, plan.pressure_bar):
        raise ValueError("Requested conditions conflict with the supplied system protocol")
    if plan.method == "equilibrated_md":
        valid = p.ensemble != "relax"
    else:
        valid = p.ensemble == "relax" and p.elastic == (plan.target_property == "elastic_constants")
    if not valid:
        raise ValueError("The configured protocol cannot produce the requested observable")
    return TaskConfig.model_validate({**configured.model_dump(), "goal": request, "llm": llm})


def compile_generated(
    plan: SimulationPlan,
    request: str,
    lammps: str,
    llm: LLM,
    *,
    execution: Execution | None = None,
    smoke: bool = False,
) -> TaskConfig:
    water = plan.system == "water"
    symbol = plan.element
    elements = (
        ["O", "H"]
        if water
        else list(plan.elements or ())
        if plan.system == "ordered_crystal"
        else [symbol]
    )
    if not all(atomic_numbers.get(element, 0) for element in elements):
        raise ValueError(f"Manager proposed an invalid element list: {elements!r}")
    if not water and (
        plan.phase not in {"fcc", "bcc", "sc", "diamond", "b2", "rocksalt", "zincblende"}
        or plan.repeat > 6
    ):
        raise ValueError(
            "The automatic crystal builder needs a supported cubic prototype and repeat 2–6"
        )
    if symbol == "Au" and plan.phase != "fcc":
        raise ValueError("Proposed Au phase contradicts the supplied bulk phase evidence")
    dynamic = plan.method == "equilibrated_md"
    elastic = plan.target_property == "elastic_constants"
    if water and not dynamic:
        raise ValueError("The molecular water runner produces RDF and MSD from a liquid trajectory")
    if not dynamic and (plan.temperature_k != 0 or plan.pressure_bar != 0):
        raise ValueError("The automatic static method requires 0 K and zero external pressure")
    if dynamic and (
        plan.temperature_k <= 0
        or (
            "npt" not in {plan.ensemble, plan.equilibration_ensemble}
            and plan.pressure_bar not in {None, 0}
        )
    ):
        raise ValueError(
            "Trajectory production requires a positive temperature; use NPT to impose pressure"
        )
    if dynamic and water and plan.target_property == "msd" and not smoke and plan.ensemble != "nvt":
        raise ValueError(
            "Production water MSD requires fixed-volume NVT sampling after equilibration"
        )
    if execution is None:
        solver = shutil.which(lammps)
        if not solver:
            raise ValueError("LAMMPS executable not found; use --lammps to specify it")
        execution = Execution(
            command=[solver],
            poll_seconds=2,
            max_watch_seconds=1800 if dynamic else 600,
            timeout_seconds=(1200 if smoke else 86400) if dynamic else 180,
        )
    elif dynamic and not smoke and execution.production_walltime:
        h, m, s = map(int, execution.production_walltime.split(":"))
        execution = execution.model_copy(
            update={
                "walltime": execution.production_walltime,
                "timeout_seconds": h * 3600 + m * 60 + s,
            }
        )
    structure = (
        {"elements": ["O", "H"], "water_repeat": plan.repeat}
        if water
        else {
            "elements": elements,
            "builder": "auto",
            "crystal": {
                "lattice": plan.phase,
                "a_angstrom": plan.initial_a_angstrom,
                "repeat": [plan.repeat] * 3,
            },
        }
    )
    diagnostic_property = (
        "rdf" if plan.target_property == "custom_trajectory" else plan.target_property
    )
    potential = (
        {
            "kind": "water",
            "water_model": plan.water_model or "auto",
            "target_property": diagnostic_property,
        }
        if water
        else {"kind": "auto", "discover": "repositories", "target_property": diagnostic_property}
    )
    protocol = {
        "ensemble": plan.ensemble if dynamic else "relax",
        "temperature_k": plan.temperature_k,
        "pressure_bar": plan.pressure_bar if plan.pressure_bar is not None else 0,
        "elastic": elastic,
    }
    validation = {"relax_pressure_tolerance_bar": 5.0}
    if dynamic:
        frames = 100 if smoke else 300
        stride = ceil(plan.production_steps / frames)
        protocol.update(
            timestep_fs=plan.timestep_fs,
            equilibration_steps=plan.equilibration_steps,
            equilibration_ensemble=plan.equilibration_ensemble,
            sampling_mode="smoke" if smoke else "production",
            steps=plan.production_steps,
            dump_every=stride,
            thermo_every=stride,
        )
        validation.update(
            trajectory_backend="auto",
            species_resolved=water,
            rdf_max_angstrom=8 if water else 5,
            max_analysis_frames=60 if smoke else frames + 1,
        )
    material = "water" if water else "".join(elements).lower()
    suffix = "elastic" if elastic else "lattice" if not dynamic else plan.target_property
    return TaskConfig.model_validate(
        {
            "name": f"{material}_{suffix}",
            "goal": request,
            "structure": structure,
            "potential": potential,
            "protocol": protocol,
            "validation": validation,
            "execution": execution.model_dump(),
            "llm": llm,
        }
    )
