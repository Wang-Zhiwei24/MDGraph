"""Live potential discovery. Repository metadata proposes choices; only validated data reach LAMMPS."""

from __future__ import annotations

import io
import json
import lzma
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ase.data import atomic_numbers

from . import nist_discovery as nist
from .artifacts import atomic_json, read_json

STYLES = {"eam", "eam/alloy", "eam/fs", "sw", "tersoff"}
NIST_SUFFIXES = {
    "eam": (".eam", ".funcfl"),
    "eam/alloy": (".eam.alloy", ".alloy", ".eamalloy", ".set", ".setfl", ".eam"),
    "eam/fs": (".eam.fs", ".fs", ".eam", ".setfl"),
    "sw": (".sw",),
    "tersoff": (".tersoff",),
}
KIM_QUERY = "https://query.openkim.org/api"
KIM_ID = re.compile(r"[A-Za-z0-9_]+__(?:MO|SM)_[0-9]{12}_[0-9]{3}")
DATA_LIMIT = 12_000_000
ARCHIVE_LIMIT = 24_000_000


def repository_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.fragment:
        raise ValueError("Expected a repository HTTPS URL")
    if parts.netloc in {"www.ctcms.nist.gov", "ctcms.nist.gov"}:
        return nist.checked_url(url)
    if parts.netloc == "query.openkim.org" and parts.path == "/api":
        return url
    if (
        parts.netloc in {"openkim.org", "www.openkim.org", "s3.openkim.org"}
        and ".." not in parts.path.split("/")
        and not parts.query
        and re.fullmatch(r"/(?:download|archives)/[A-Za-z0-9_/.-]+\.txz", parts.path)
    ):
        return url
    raise ValueError("URL is outside the supported potential repositories")


class RepositoryRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, repository_url(newurl))


def fetch(url: str, limit: int) -> bytes:
    # The cluster may export an unrelated local browser proxy in .bashrc.
    opener = build_opener(ProxyHandler({}), RepositoryRedirects())
    request = Request(repository_url(url), headers={"User-Agent": "MDGraph repository client"})
    with opener.open(request, timeout=22) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Repository resource exceeds the size limit")
    return payload


def short_text(value, size=450):
    return " ".join(str(value or "").split())[:size]


def nist_candidates(elements, styles, phase, target_property):
    slug = "-".join(sorted(elements))
    index = f"{nist.ORIGIN}/potentials/system/{slug}/"
    raw = fetch(index, 2_000_000)
    parser = nist.IndexParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    candidates = {}
    for item in parser.items:
        label = short_text(" ".join(item["text"]), 1800)
        style = re.match(r"LAMMPS pair_style (eam(?:/alloy|/fs)?|sw|tersoff)(?=\s|$)", label)
        if (
            not style
            or style[1] not in styles
            or re.search(r"\b(?:superseded|retracted)\b", label, re.I)
        ):
            continue
        kind = style[1]
        links = []
        for href in item["links"]:
            try:
                links.append(nist.checked_url(urljoin(index, href)))
            except ValueError:
                continue
        sources = sorted(
            {u for u in links if "/entry/" in u and re.search(r"--LAMMPS--ipr\d+\.html$", u)}
        )
        files = sorted(
            {u for u in links if "/Download/" in u and u.lower().endswith(NIST_SUFFIXES[kind])}
        )
        if len(sources) != 1 or len(files) != 1:
            continue
        key = sources[0].rsplit("/", 1)[-1].removesuffix(".html")
        candidates[key] = {
            "id": f"nist:{key}",
            "repository": "NIST",
            "kind": kind,
            "source": sources[0],
            "download": files[0],
            "elements": elements,
            "notes": short_text(label),
        }
    # Reuse computed phase/elastic evidence for the elemental cubic cases when it exists.
    if len(elements) == 1 and phase in {"fcc", "bcc"}:

        def attach(row):
            element = elements[0]
            prefix = "elastic" if target_property == "elastic_constants" else "crystal"
            url = row["source"].removesuffix(".html") + f"/{prefix}.{element}.csv"
            try:
                data = fetch(url, 2_000_000)
                evidence = (
                    nist.elastic_evidence(data, element, phase)
                    if prefix == "elastic"
                    else nist.crystal_evidence(data, phase)
                )
            except (OSError, ValueError):
                evidence = None
            return {**row, "computed_evidence": evidence, "evidence_url": url if evidence else None}

        with ThreadPoolExecutor(max_workers=6) as pool:
            ordered = list(pool.map(attach, list(candidates.values())[:16]))
        ordered.sort(key=lambda row: not bool(row["computed_evidence"]))
    else:
        ordered = list(candidates.values())
    return {
        "url": index,
        "response_sha256": sha256(raw).hexdigest(),
        "matched": len(candidates),
    }, ordered[:5]


def kim_candidates(elements, styles, phase, target_property, *, inventory=False):
    query = {
        "type": {"$in": ["mo", "sm"]},
        "species": {"$all": elements},
        "approved": True,
        "latest": True,
    }
    fields = dict.fromkeys(
        (
            "kimcode",
            "species",
            "potential-type",
            "model-driver",
            "description",
            "title",
            "doi",
            "source-citations",
            "kimid-typecode",
        ),
        1,
    )
    url = (
        KIM_QUERY
        + "?"
        + urlencode(
            {
                "database": "obj",
                "query": json.dumps(query, separators=(",", ":")),
                "fields": json.dumps(fields, separators=(",", ":")),
                "limit": 45,
            }
        )
    )
    payload = fetch(url, 2_500_000)
    records = json.loads(payload)
    if not isinstance(records, list):
        raise ValueError("OpenKIM returned an invalid model query response")
    results = []
    for row in records:
        key = row.get("kimcode", "")
        species = row.get("species") or []
        if not KIM_ID.fullmatch(key) or not set(elements) <= set(species):
            continue
        # A Dynamo model holds a raw LAMMPS-readable table. The KIM compiler/driver
        # and arbitrary simulator-model commands are not used by this adapter.
        family = row.get("potential-type")
        driver = row.get("model-driver", "")
        native_eam = family == "eam" and driver.startswith("EAM_Dynamo__MD_")
        kinds = (
            sorted(
                styles & {"eam/alloy", "eam/fs"}
                | (styles & {"eam"} if len(elements) == 1 else set())
            )
            if native_eam
            else []
        )
        if not (inventory or kinds):
            continue
        citations = [c.get("doi") for c in row.get("source-citations", ()) if c.get("doi")]
        results.append(
            {
                "id": f"openkim:{key}",
                "repository": "OpenKIM",
                "kind": ", ".join(kinds)
                if kinds
                else f"{family or 'unclassified'} (metadata only)",
                "native_styles": kinds,
                "source": f"https://openkim.org/id/{key}",
                "download": f"https://openkim.org/download/{key}.txz",
                "elements": species,
                "doi": row.get("doi"),
                "paper_dois": citations[:3],
                "evidence": short_text(row.get("description"), 650),
            }
        )
    # Exact-species matches and explicit phase evidence take priority over the query limit.
    phase_terms = {
        "b2": ("b2", "cscl"),
        "rocksalt": ("rocksalt", "rock salt", "b1"),
        "zincblende": ("zincblende", "zinc blende", "b3"),
    }.get(phase, (phase,))
    results.sort(
        key=lambda c: (
            not any(
                re.search(r"\b" + re.escape(term) + r"\b", c["evidence"].lower())
                for term in phase_terms
            ),
            len(set(c["elements"]) - set(elements)),
        )
    )
    return {
        "url": url,
        "response_sha256": sha256(payload).hexdigest(),
        "records": len(records),
        "matched": len(results),
        "native_adapter": "OpenKIM Dynamo EAM parameter data",
    }, results[:5]


def discover(elements, phase, target_property, allowed_kinds, *, inventory=False):
    """Build a repository shortlist constrained by composition and solver styles."""
    if not elements or not set(elements) <= set(atomic_numbers):
        raise ValueError("Invalid elements for repository discovery")
    styles = set(allowed_kinds) or STYLES
    readers = {"NIST": nist_candidates, "OpenKIM": kim_candidates}
    audits, candidates = {}, []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {
            name: pool.submit(
                reader,
                elements,
                styles,
                phase,
                target_property,
                **({"inventory": inventory} if name == "OpenKIM" else {}),
            )
            for name, reader in readers.items()
        }
        for name, future in pending.items():
            try:
                details, found = future.result()
                audits[name] = {"status": "searched", **details}
                candidates.extend(found)
            except (OSError, ValueError) as exc:
                audits[name] = {"status": "unavailable", "error": short_text(exc)}
    return {
        "elements": elements,
        "phase": phase,
        "target_property": target_property,
        "repositories": audits,
        "shortlist": candidates,
    }


def validate_eam(data: bytes, kind: str, elements: list[str]) -> None:
    lines = data.decode("ascii").splitlines()
    try:
        if kind == "eam":
            if len(elements) != 1 or int(lines[1].split()[0]) != atomic_numbers[elements[0]]:
                raise ValueError("Wrong funcfl species header")
            grid = lines[2].split()
            values = " ".join(lines[3:]).split()
            headers, blocks = [elements[0]], 0
        else:
            number, *headers = lines[3].split()
            if (
                int(number) != len(headers)
                or not set(elements) <= set(headers)
                or len(set(headers)) != len(headers)
            ):
                raise ValueError("Native table does not contain all requested species")
            grid = lines[4].split()
            values = " ".join(lines[5:]).split()
            blocks = len(headers)
        nrho, nr = int(grid[0]), int(grid[2])
        if (
            len(grid) != 5
            or not (4 <= nrho <= 1_000_000 and 4 <= nr <= 1_000_000)
            or not all(math.isfinite(float(grid[i])) and float(grid[i]) > 0 for i in (1, 3, 4))
        ):
            raise ValueError("Invalid EAM grid")
        if not blocks:
            expected = nrho + 2 * nr
        else:
            length = 4 + nrho + nr * (blocks if kind == "eam/fs" else 1)
            expected = blocks * length + nr * blocks * (blocks + 1) // 2
            for i, symbol in enumerate(headers):
                offset = i * length
                atomic_number = int(values[offset])
                # Some setfl writers use zero as the atomic-number placeholder.
                if symbol not in atomic_numbers or atomic_number not in {0, atomic_numbers[symbol]}:
                    raise ValueError("Invalid EAM species mapping")
            values = [
                value for i, value in enumerate(values) if i >= blocks * length or i % length != 3
            ]
            expected -= blocks
        if len(values) != expected or not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Wrong EAM table shape or non-finite values")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid {kind} parameter data: {exc}") from exc


def kim_parameter(data: bytes, kinds: list[str], elements: list[str]) -> tuple[str, bytes]:
    with lzma.LZMAFile(io.BytesIO(data)) as uncompressed:
        content = uncompressed.read(ARCHIVE_LIMIT + 1)
    if len(content) > ARCHIVE_LIMIT:
        raise ValueError("Decompressed OpenKIM archive exceeds the limit")
    plausible = []
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if (
                not item.isfile()
                or not 0 < item.size <= DATA_LIMIT
                or name.is_absolute()
                or ".." in name.parts
            ):
                continue
            if not name.name.lower().endswith(
                (
                    ".eam.alloy",
                    ".eam.fs",
                    ".eam",
                    ".alloy",
                    ".fs",
                    ".set",
                    ".setfl",
                    ".funcfl",
                    ".dat",
                )
            ):
                continue
            with archive.extractfile(item) as stream:
                candidate = stream.read()
            valid_kinds = []
            for kind in kinds:
                try:
                    validate_eam(candidate, kind, elements)
                except (ValueError, UnicodeError):
                    continue
                valid_kinds.append(kind)
            if valid_kinds:
                # Single-species setfl has identical layout under eam/alloy and eam/fs.
                kind = (
                    "eam/fs"
                    if name.name.endswith((".fs", ".eam.fs")) and "eam/fs" in valid_kinds
                    else valid_kinds[0]
                )
                plausible.append((kind, candidate))
    if len(plausible) != 1:
        raise ValueError(
            "OpenKIM archive does not contain one unambiguous supported native parameter file"
        )
    return plausible[0]


def native_parameter(row, elements):
    payload = fetch(row["download"], DATA_LIMIT)
    if row["repository"] == "OpenKIM":
        try:
            return kim_parameter(payload, row["native_styles"], elements)
        except (lzma.LZMAError, tarfile.TarError, EOFError) as exc:
            raise ValueError(f"Invalid OpenKIM archive: {exc}") from exc
    kind = row["kind"]
    if kind.startswith("eam"):
        validate_eam(payload, kind, elements)
    else:
        # The simulator verifies the complete numeric grammar and all required triplets.
        payload.decode("ascii")
    return kind, payload


def preflight(cfg, folder, kind, data):
    from .prepare import atomic_potential_setup

    if cfg.execution.backend == "ssh":
        raise ValueError(
            "Live repository preflight must run on the host that has the configured LAMMPS executable"
        )
    name = "model." + kind.replace("/", ".")
    with tempfile.TemporaryDirectory(prefix="mdgraph_potential_") as tmp:
        trial = Path(tmp)
        shutil.copyfile(folder / "structure.data", trial / "structure.data")
        (trial / name).write_bytes(data)
        setup = atomic_potential_setup(cfg, name, kind)
        script = setup + (
            "thermo_style custom step atoms pe press fmax\n"
            "run 0\nvariable check_energy equal pe\nvariable check_pressure equal press\n"
            'print "MDGRAPH_PROBE ${check_energy} ${check_pressure}"\n'
        )
        (trial / "probe.in").write_text(script, encoding="ascii")
        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            in {"PATH", "LD_LIBRARY_PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
        }
        env["OMP_NUM_THREADS"] = "1"
        try:
            result = subprocess.run(
                [*cfg.execution.command, "-in", "probe.in", "-log", "none"],
                cwd=trial,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=45,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("LAMMPS run 0 exceeded 45 seconds") from exc
        marker = re.search(r"^MDGRAPH_PROBE\s+(\S+)\s+(\S+)\s*$", result.stdout, re.M)
        if result.returncode or not marker:
            err = next(
                (line for line in result.stdout.splitlines() if line.startswith("ERROR")),
                result.stderr,
            )
            raise ValueError(
                "LAMMPS run 0 failed: " + short_text(err or f"exit {result.returncode}")
            )
        energy, pressure = map(float, marker.groups())
        if not all(map(math.isfinite, (energy, pressure))):
            raise ValueError("LAMMPS run 0 returned non-finite energy or pressure")
    return name, {
        "passed": True,
        "command": "LAMMPS run 0 on the built structure",
        "energy_ev": energy,
        "pressure_bar": pressure,
    }


def discovery_snapshot(cfg, folder, *, inventory=False):
    path = folder / "potential_discovery.json"
    if path.is_file():
        return read_json(path)
    scope = cfg.potential_scope()
    data = discover(
        cfg.structure.elements,
        scope["phase"],
        scope["target_property"],
        cfg.potential.allowed_kinds,
        inventory=inventory,
    )
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(path, data)
    return data


def retrieve(cfg, folder, choose, record):
    """Validate downloaded candidates before saving the chosen potential."""
    sketch = folder / "potential_discovery.json"
    elements = cfg.structure.elements
    snapshot = discovery_snapshot(cfg, folder)
    available = snapshot["shortlist"][:]
    failures = []
    proof = folder / "potential_verification.json"
    if not available:
        states = ", ".join(
            f"{key}: {row['status']}" for key, row in snapshot["repositories"].items()
        )
        raise ValueError(
            f"No executable repository candidate for {elements} ({states}). See {sketch.name}."
        )
    for _ in range(len(available)):
        decision = choose(
            [
                *available,
                {
                    "id": "none",
                    "kind": "decline",
                    "notes": "No candidate has adequate applicability evidence",
                },
            ],
            failures,
        )
        if decision["choice"] == "none":
            raise ValueError(f"Potential Agent found no suitable model: {decision['reason']}")
        selected = next((c for c in available if c["id"] == decision["choice"]), None)
        if selected is None:
            raise ValueError("Potential Agent did not choose a current repository candidate")
        try:
            kind, data = native_parameter(selected, elements)
            if cfg.potential.allowed_kinds and kind not in cfg.potential.allowed_kinds:
                raise ValueError(f"Selected download is {kind}, which the task disallows")
            filename, check = preflight(cfg, folder, kind, data)
        except (OSError, ValueError) as err:
            failures.append({"id": selected["id"], "reason": short_text(err), "decision": decision})
            atomic_json(proof, {"rejected": failures})
            available.remove(selected)
            continue
        digest = sha256(data).hexdigest()
        (folder / filename).write_bytes(data)
        metadata = {
            **selected,
            "kind": kind,
            "type_order": elements,
            "units": "metal",
            "model_sha256": digest,
            "preflight": check,
        }
        atomic_json(folder / "potential_metadata.json", metadata)
        atomic_json(proof, {"rejected": failures, "selected": metadata})
        files = (filename, "potential_metadata.json", sketch.name, proof.name)
        chosen = {
            "kind": kind,
            "mode": "live_repositories",
            "candidate_id": selected["id"],
            "source": selected["source"],
            "filename": filename,
            "model_sha256": digest,
            "decision": decision,
            "candidates": snapshot["shortlist"],
            "tool": {"name": "NIST and OpenKIM live repository search"},
            "files": {name: sha256((folder / name).read_bytes()).hexdigest() for name in files},
        }
        atomic_json(record, chosen)
        return chosen
    raise ValueError(f"All LLM-selected candidates failed validation. See {proof.name}.")
