from __future__ import annotations

import shutil
from hashlib import sha256 as hash_bytes
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, build_opener

import yaml
from pydantic import Field, model_validator

from . import nist_discovery, repository_discovery
from .artifacts import atomic_json, read_json, sha256, verify_files
from .config import PotentialStyle, StrictModel, TaskConfig
from .water import WATER_MODELS


class CatalogEntry(StrictModel):
    id: str = Field(min_length=1)
    kind: PotentialStyle
    elements: list[str]
    units: Literal["metal"]
    application: str | None = Field(default=None, min_length=1)
    phases: list[str] = Field(default_factory=list)
    target_properties: list[str] = Field(default_factory=list)
    atom_fraction_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    source: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    temperature_k: tuple[float, float]
    pressure_bar: tuple[float, float] | None = None
    model: str | None = None
    model_url: str | None = None
    fit_notes: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def valid(self):
        if bool(self.model) == bool(self.model_url):
            raise ValueError("Provide either a local model path or an HTTPS model_url")
        if self.application is None and not (self.phases and self.target_properties):
            raise ValueError("General catalog entries require phases and target_properties")
        if (
            self.application is None
            and len(self.elements) > 1
            and set(self.atom_fraction_ranges) != set(self.elements)
        ):
            raise ValueError(
                "General alloy entries must declare an atom-fraction range for every element"
            )
        if any(
            e not in self.elements or not 0 <= lo <= hi <= 1
            for e, (lo, hi) in self.atom_fraction_ranges.items()
        ):
            raise ValueError("Invalid catalog atom-fraction range")
        if self.model_url:
            check_https(self.model_url)
        if not 0 <= self.temperature_k[0] <= self.temperature_k[1]:
            raise ValueError("Invalid catalog temperature range")
        if self.pressure_bar and self.pressure_bar[0] > self.pressure_bar[1]:
            raise ValueError("Invalid catalog pressure range")
        return self


class Catalog(StrictModel):
    entries: list[CatalogEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def valid(self):
        if len({entry.id for entry in self.entries}) != len(self.entries):
            raise ValueError("Catalog IDs must be unique")
        return self


def check_https(url: str):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None:
        raise ValueError("Model URLs must use HTTPS without embedded credentials")
    return parsed


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_model(url: str, target: Path) -> None:
    check_https(url)
    opener = build_opener(HTTPSHandler(), HTTPSRedirects())
    limit = 256 * 1024**2
    with opener.open(url, timeout=30) as response, target.open("wb") as output:
        total = 0
        while block := response.read(1024**2):
            total += len(block)
            if total > limit:
                raise ValueError("Model download exceeds 256 MiB. Use a local model artifact.")
            output.write(block)


def select_entry(
    cfg: TaskConfig,
    catalog_path: Path,
    choose: Callable[[list[dict]], dict] | None = None,
    *,
    element_counts: dict[str, int] | None = None,
) -> tuple[CatalogEntry, dict]:
    raw_catalog = catalog_path.read_bytes()
    try:
        catalog = Catalog.model_validate(yaml.safe_load(raw_catalog))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid potential catalog YAML: {catalog_path}") from error
    matches = []
    styles = cfg.potential.allowed_kinds if cfg.potential.kind == "auto" else [cfg.potential.kind]
    scope = cfg.potential_scope()
    application = cfg.potential.application
    total = sum(element_counts.values()) if element_counts else 0
    fractions = {e: n / total for e, n in element_counts.items()} if total else {}
    for entry in catalog.entries:
        if entry.kind not in styles or entry.elements != cfg.structure.elements:
            continue
        if application is not None:
            if entry.application != application:
                continue
        elif (
            entry.application is not None
            or scope["phase"] not in entry.phases
            or scope["target_property"] not in entry.target_properties
        ):
            continue
        if entry.atom_fraction_ranges and any(
            e not in fractions or not lo - 1e-12 <= fractions[e] <= hi + 1e-12
            for e, (lo, hi) in entry.atom_fraction_ranges.items()
        ):
            continue
        if entry.kind == "eam" and len(entry.elements) != 1:
            continue
        if not entry.temperature_k[0] <= cfg.protocol.temperature_k <= entry.temperature_k[1]:
            continue
        if cfg.protocol.ensemble in {"npt", "relax"} and (
            entry.pressure_bar is None
            or not entry.pressure_bar[0] <= cfg.protocol.pressure_bar <= entry.pressure_bar[1]
        ):
            continue
        matches.append(entry)
    matching_ids = [entry.id for entry in matches]
    if cfg.potential.candidate_id:
        matches = [entry for entry in matches if entry.id == cfg.potential.candidate_id]
    evidence = {
        "catalog_sha256": hash_bytes(raw_catalog).hexdigest(),
        "compatible_ids": matching_ids,
        "scope": {"application": application, **scope, "atom_fractions": fractions},
    }
    if not matches:
        raise ValueError(
            f"No compatible catalog potential; compatible IDs: {matching_ids}. "
            "Check chemistry/type order, composition, style, phase/property, and temperature/pressure; "
            "add a verified potential or configure an explicit model. No fallback will run."
        )
    if len(matches) > 1 and choose is not None:
        if len(matches) > 12:
            raise ValueError("Curate the potential shortlist to at most 12 candidates")
        if len({entry.model_sha256.lower() for entry in matches}) != len(matches):
            raise ValueError("Potential shortlist contains duplicate artifacts; use unique hashes")
        candidates = [
            e.model_dump(mode="json", exclude={"model", "model_url", "model_sha256"})
            for e in matches
        ]
        decision = choose(candidates)
        evidence.update(candidates=candidates, decision=decision)
        matches = [e for e in matches if e.id == decision["choice"]]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one compatible catalog potential; compatible IDs: {matching_ids}. "
            "Check application, type mapping and conditions, or specify candidate_id."
        )
    return matches[0], evidence


def model_filename(source: str) -> str:
    # Keep the full setfl suffix in the manifest and in the generated LAMMPS input.
    for eam_suffix in (".eam.alloy", ".eam.fs"):
        if source.endswith(eam_suffix):
            return "model" + eam_suffix
    suffix = Path(source).suffix
    if suffix and not suffix[1:].isalnum():
        raise ValueError("Model extension must be alphanumeric")
    return "model" + suffix


def retrieve_discovered(cfg: TaskConfig, folder: Path, choose, record: Path) -> dict:
    if choose is None:
        raise ValueError("Live NIST selection requires the potential Agent's decision function")
    discovery_file = folder / "potential_discovery.json"
    element = cfg.structure.elements[0]
    scope = cfg.potential_scope()
    if discovery_file.exists():
        snapshot = read_json(discovery_file)
        candidates = snapshot["shortlist"]
    else:
        details, candidates = nist_discovery.discover(
            element, scope["phase"], scope["target_property"]
        )
        snapshot = {**details, "element": element, **scope, "shortlist": candidates}
        atomic_json(discovery_file, snapshot)
    if not candidates:
        raise ValueError(
            "No NIST candidate with evidence for the requested phase; see potential_discovery.json"
        )
    decision = choose([{**c, "elements": [element]} for c in candidates])
    match = next((c for c in candidates if c["id"] == decision["choice"]), None)
    if not match:
        raise ValueError("The chosen ID is not in the live NIST shortlist")
    data = nist_discovery.fetch(match["url"], 12_000_000)
    nist_discovery.validate_setfl(data, element)
    digest = hash_bytes(data).hexdigest()
    if match.get("discovery_model_sha256", digest) != digest:
        raise ValueError("NIST model bytes changed after candidate screening")
    name = "model.eam.alloy"
    (folder / name).write_bytes(data)
    metadata = {
        **match,
        "elements": [element],
        "units": "metal",
        "model_sha256": digest,
        "checksum_basis": "Recorded from the HTTPS download, not a publisher-provided checksum",
    }
    atomic_json(folder / "potential_metadata.json", metadata)
    result = {
        "kind": "eam/alloy",
        "mode": "discovered",
        "candidate_id": match["id"],
        "source": match["source"],
        "filename": name,
        "model_sha256": digest,
        "candidates": candidates,
        "decision": decision,
        "tool": {"name": "NIST Interatomic Potentials Repository live search", "transfer": "HTTPS"},
        "files": {
            key: sha256(folder / key)
            for key in (name, "potential_metadata.json", discovery_file.name)
        },
    }
    verify_files(folder, {name: digest})
    atomic_json(record, result)
    return result


def retrieve_potential(cfg: TaskConfig, folder: Path, choose=None, *, element_counts=None) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "potential_selection.json"
    if record.exists():
        selection = read_json(record)
        verify_files(folder, selection["files"])
        return selection

    potential = cfg.potential
    if potential.discover == "nist":
        return retrieve_discovered(cfg, folder, choose, record)
    if potential.discover == "repositories":
        if choose is None:
            raise ValueError("Live repository selection requires the potential Agent")
        return repository_discovery.retrieve(cfg, folder, choose, record)
    if potential.kind == "water":
        model_id = potential.water_model
        candidates = [{"id": key, "kind": "water", **value} for key, value in WATER_MODELS.items()]
        decision = None
        if model_id == "auto":
            if choose is None:
                raise ValueError("Automatic water model selection requires the potential Agent")
            decision = choose(candidates)
            model_id = decision["choice"]
        model = WATER_MODELS[model_id]
        name = "potential_metadata.json"
        atomic_json(folder / name, {"model": model_id, "units": "real", **model})
        result = {
            "kind": "water",
            "mode": "parameterized",
            "candidate_id": model_id,
            "water_model": model_id,
            "source": model["source"],
            "filename": None,
            "decision": decision,
            "tool": {"name": "LAMMPS rigid water model parameters"},
            "files": {name: sha256(folder / name)},
        }
        atomic_json(record, result)
        return result
    result = {
        "kind": potential.kind,
        "mode": "demo",
        "filename": None,
        "files": {},
        "tool": {"name": "MDGraph built-in potential"},
    }
    if potential.kind != "lj_demo":
        url = None
        if potential.catalog:
            catalog_path = Path(potential.catalog)
            entry, evidence = select_entry(cfg, catalog_path, choose, element_counts=element_counts)
            metadata = entry.model_dump(mode="json")
            model = (catalog_path.parent / entry.model).resolve() if entry.model else None
            url = entry.model_url
            result.update(
                mode="catalog",
                kind=entry.kind,
                candidate_id=entry.id,
                source=entry.source,
                **evidence,
            )
        else:
            model = Path(potential.model)
            metadata = read_json(Path(potential.metadata))
            result.update(mode="explicit", source=metadata.get("source"))
        if metadata.get("elements") != cfg.structure.elements or metadata.get("units") != "metal":
            raise ValueError("Potential metadata elements/type order or units mismatch")
        expected = metadata.get("model_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Potential metadata requires model_sha256")
        name = model_filename(urlsplit(url).path if url else model.name)
        target = folder / name
        temporary = folder / (name + ".part")
        try:
            if url:
                download_model(url, temporary)
            else:
                shutil.copyfile(model, temporary)
            digest = sha256(temporary)
            if digest != expected.lower():
                raise ValueError("Potential SHA256 differs from its declared metadata")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        result.update(filename=name, model_sha256=digest)
        result["tool"] = {
            "name": "MDGraph potential catalog and SHA256"
            if potential.catalog
            else "MDGraph explicit potential and SHA256",
            "transfer": "HTTPS" if url else "local file",
        }
        atomic_json(folder / "potential_metadata.json", metadata)
        files = [name, "potential_metadata.json"]
        if potential.plugin:
            shutil.copyfile(potential.plugin, folder / "plugin.so")
            files.append("plugin.so")
        result["files"] = {name: sha256(folder / name) for name in files}
    atomic_json(record, result)
    return result
