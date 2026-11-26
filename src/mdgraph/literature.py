"""Live literature discovery and evidence-checked wall/fluid parameters."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import monotonic
from urllib.parse import quote
from xml.etree import ElementTree

from ase.data import atomic_names, atomic_numbers
from langchain_core.exceptions import OutputParserException
from pydantic import Field, ValidationError

from .agents import _invoke
from .config import LLM, HoneycombTube, StrictModel, WallInterface

CROSSREF = "https://api.crossref.org/works"
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMC_BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml"
PMC_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
OSTI = "https://www.osti.gov"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(
        str.maketrans({"−": "-", "–": "-", "—": "-"})
    )
    return " ".join(text.split())


def request_data(
    url: str, *, params=None, maximum=8_000_000, accept: str | None = None, total_seconds: int = 35
) -> bytes:
    import httpx

    headers = {"User-Agent": "MDGraph literature research client"}
    if accept:
        headers["Accept"] = accept
    deadline = monotonic() + total_seconds
    with httpx.Client(timeout=httpx.Timeout(20, connect=6), trust_env=False) as client:
        with client.stream("GET", url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                if monotonic() > deadline:
                    raise TimeoutError("Literature download exceeded its time budget")
                data.extend(chunk)
                if len(data) > maximum:
                    raise ValueError("Literature response exceeded the size budget")
            return bytes(data)


def search_index(query: str, provider: str) -> list[dict]:
    if provider == "Europe PMC":
        data = json.loads(
            request_data(
                f"{EUROPE_PMC}/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": 7,
                    "resultType": "core",
                },
            )
        )
        rows = data["resultList"]["result"]
        return [
            {
                "title": row["title"],
                "doi": row["doi"].lower(),
                "provider": provider,
                "pmcid": row.get("pmcid"),
                "abstract": normalize(row.get("abstractText") or "")[:1200],
            }
            for row in rows
            if row.get("doi") and row.get("title")
        ]
    data = json.loads(
        request_data(
            CROSSREF,
            params={"query.bibliographic": query, "rows": 12, "filter": "type:journal-article"},
        )
    )
    return [
        {
            "title": row["title"][0],
            "doi": row["DOI"].lower(),
            "provider": provider,
            "abstract": normalize(row.get("abstract") or "")[:1200],
        }
        for row in data["message"]["items"]
        if row.get("title") and row.get("DOI")
    ]


def lookup(queries: list[str]) -> tuple[list[dict], list[dict]]:
    records = {}
    calls = [(query, provider) for query in queries[:3] for provider in ("Crossref", "Europe PMC")]
    log = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [
            (query, provider, executor.submit(search_index, query, provider))
            for query, provider in calls
        ]
        for query, provider, job in jobs:
            try:
                rows = job.result()
                for row in rows:
                    previous = records.setdefault(row["doi"], row)
                    if row.get("pmcid"):
                        previous["pmcid"] = row["pmcid"]
                log.append(
                    {
                        "provider": provider,
                        "query": query,
                        "doi_results": [row["doi"] for row in rows],
                    }
                )
            except Exception as exc:
                log.append({"provider": provider, "query": query, "error": str(exc)[:250]})
    return list(records.values()), log


def identify_source(doi: str) -> dict:
    if not re.fullmatch(r"10\.\d{4,9}/[^\s]+", doi):
        raise ValueError("A cached primary document needs a valid DOI")
    item = json.loads(request_data(f"{CROSSREF}/{quote(doi, safe='/')}"))["message"]
    if item["DOI"].lower() != doi:
        raise ValueError("Crossref returned a different DOI")
    return {"doi": doi, "title": item["title"][0], "provider": "Crossref DOI record"}


def search_osti(
    query: str, title_query: str | None = None, *, exact_doi: bool = False
) -> list[dict]:
    searches = (
        [("title", title_query), ("q", query)]
        if title_query
        else [("doi" if exact_doi else "q", query)]
    )
    for field, text in searches:
        raw = request_data(
            f"{OSTI}/api/v1/records", params={field: text, "rows": 8}, accept="application/json"
        )
        rows = [
            {
                "doi": row["doi"].lower(),
                "title": row["title"],
                "provider": "OSTI",
                "osti_id": str(row["osti_id"]),
                "publication_type": row.get("product_type"),
                "abstract": normalize(row.get("description") or "")[:1200],
            }
            for row in json.loads(raw)
            if row.get("doi")
            and row.get("title")
            and row.get("osti_id")
            and (not exact_doi or row["doi"].lower() == query.lower())
            and any(link.get("rel") == "fulltext" for link in row.get("links", []))
        ]
        if rows:
            journals = {
                normalize(row["title"]).casefold().rstrip(".")
                for row in rows
                if row["publication_type"] == "Journal Article"
            }
            return [
                row
                for row in rows
                if row["publication_type"] == "Journal Article"
                or normalize(row["title"]).casefold().rstrip(".") not in journals
            ]
    return []


def read_osti_article(record: dict) -> dict | None:
    import httpx

    identifier = record.get("osti_id", "")
    command = shutil.which("pdftotext")
    if not command or not re.fullmatch(r"[0-9]{1,12}", identifier):
        return None
    url = f"{OSTI}/servlets/purl/{identifier}"
    try:
        raw = request_data(url, maximum=12_000_000, total_seconds=75)
        if not raw.startswith(b"%PDF-"):
            return None
        result = subprocess.run(
            [command, "-enc", "UTF-8", "-nopgbrk", "-", "-"],
            input=raw,
            capture_output=True,
            timeout=25,
            check=True,
        )
        extracted = result.stdout.decode("utf-8", errors="replace")
        # Preserve reading order across PDF columns and join discretionary word breaks.
        text = normalize(re.sub(r"(?<=\w)\u00ad\s*(?=\w)", "", extracted))
        return (
            {
                **record,
                "provider": "OSTI full text",
                "url": url,
                "sha256": sha256(raw).hexdigest(),
                "text": text,
            }
            if text
            else None
        )
    except (httpx.HTTPError, OSError, ValueError, subprocess.SubprocessError):
        return None


def read_bioc(pmcid: str, doi: str | None = None) -> dict | None:
    try:
        raw = request_data(f"{PMC_BIOC}/{pmcid}/unicode")
        root = ElementTree.fromstring(raw)
        passages = root.findall(".//passage")
        front = next((p for p in passages if p.findtext("infon[@key='type']") == "front"), None)
        if front is None or front.findtext("infon[@key='article-id_pmc']") != pmcid:
            return None
        source_doi = front.findtext("infon[@key='article-id_doi']", "").lower()
        if not source_doi or (doi and source_doi != doi):
            return None
        sections = [
            p.findtext("text", "")
            for p in passages
            if p.findtext("infon[@key='section_type']") not in {"REF", "SUPPL"}
        ]
        return {
            "pmcid": pmcid,
            "doi": source_doi,
            "title": front.findtext("text", ""),
            "provider": "PMC BioC",
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            "sha256": sha256(raw).hexdigest(),
            "text": normalize(" ".join(sections)),
        }
    except Exception:
        return None


def search_open_archive(query: str) -> list[dict]:
    data = json.loads(
        request_data(
            PMC_SEARCH,
            params={
                "db": "pmc",
                "retmax": 2,
                "sort": "relevance",
                "retmode": "json",
                "term": query,
            },
        )
    )
    return [
        doc
        for identifier in data["esearchresult"]["idlist"]
        if identifier.isdigit()
        if (doc := read_bioc(f"PMC{identifier}")) and doc["text"] and doc["title"]
    ]


def read_open_article(record: dict) -> dict | None:
    pmcid = record.get("pmcid")
    if not pmcid or not re.fullmatch(r"PMC[0-9]+", pmcid):
        return None
    doc = read_bioc(pmcid, record["doi"])
    if doc and doc["text"]:
        return doc
    try:
        raw = request_data(f"{EUROPE_PMC}/{pmcid}/fullTextXML")
        root = ElementTree.fromstring(raw)
        sections = ["".join(p.itertext()) for p in root.findall("./body//p")]
        tables = [" ".join(t.itertext()) for t in root.findall("./body//table-wrap")]
        if not sections:
            return None
        return {
            **record,
            "url": f"https://europepmc.org/articles/{pmcid}",
            "sha256": sha256(raw).hexdigest(),
            "text": normalize(" ".join(sections + tables)),
        }
    except Exception:
        return None


def public_source_documents() -> list[dict]:
    """Read optional open full-text documents independently cached by the operator."""
    source = os.environ.get("MDGRAPH_LITERATURE_DOCUMENTS")
    if not source:
        source = "literature_documents.json"
    path = Path(source)
    if not path.is_file():
        return []
    docs = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for doc in docs:
        if "pdf" in doc:
            pdf = (path.parent / doc["pdf"]).resolve()
            cmd = shutil.which("pdftotext")
            if not cmd or not pdf.is_file() or pdf.stat().st_size > 12_000_000:
                raise ValueError(
                    "A cached source PDF needs pdftotext and a readable, bounded document"
                )
            result = subprocess.run(
                [cmd, "-layout", str(pdf), "-"], capture_output=True, timeout=30, check=True
            )
            text = normalize(result.stdout.decode("utf-8", errors="replace"))
            digest = sha256(pdf.read_bytes()).hexdigest()
        else:
            text = normalize(doc["text"])
            digest = sha256(text.encode("utf-8")).hexdigest()
        output.append(
            {
                "doi": doc["doi"].lower(),
                "title": doc["title"],
                "url": doc["url"],
                "text": text,
                "sha256": digest,
                "provider": "cached public primary source",
            }
        )
    return output


def excerpts(text: str) -> str:
    if len(text) <= 6500:
        return text
    anchors = list(
        re.finditer(
            r"Lennard.Jones|SPC/E|wall.{0,12}water|water.{0,12}wall|partial charges|"
            r"electrostatic interactions|equilibrat|cutoff",
            text,
            re.I,
        )
    )
    if not anchors:
        return text[:5500]
    windows = []
    for match in anchors[:32]:
        lo, hi = max(0, match.start() - 420), min(len(text), match.end() + 700)
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = hi
        else:
            windows.append([lo, hi])
    return " […] ".join(text[a:b] for a, b in windows)[:6500]


class LiteratureSelection(StrictModel):
    candidates: list[WallInterface] = Field(max_length=3)
    preferred_doi: str | None = None
    reason: str = Field(min_length=10, max_length=1600)


def find_number(value: float, text: str) -> bool:
    numbers = re.findall(r"(?<![\w.])[-+]?(?:\d+\.\d+|\.\d+|\d+)(?!\w|\.\d)", text)
    return any(math.isclose(value, float(item), abs_tol=1e-6, rel_tol=0) for item in numbers)


def verify_interface(candidate: WallInterface, tube: HoneycombTube, document: dict) -> list[str]:
    if {site.element for site in candidate.sites} != set(tube.wall_elements):
        raise ValueError("The extracted parameters do not cover the wall composition")
    quotes = [normalize(line) for line in candidate.evidence_quotes]
    if any(len(line) < 22 or line not in document["text"] for line in quotes):
        raise ValueError("An evidence quote is missing from the retrieved full text")
    joined = " ".join(quotes)
    if not re.search(r"lennard.jones|\bLJ\b", joined, flags=re.I):
        raise ValueError("The source quote must identify the Lennard-Jones model")
    for site in candidate.sites:
        name = atomic_names[atomic_numbers[site.element]]
        if not (
            re.search(rf"\b{re.escape(name)}\b", joined, re.I)
            or re.search(rf"\b{site.element}\b", joined)
        ):
            raise ValueError(f"The evidence does not identify the {site.element} wall type")
        if not find_number(site.epsilon_kcal_mol, joined) or not re.search(r"kcal", joined, re.I):
            raise ValueError(
                f"The epsilon value and kcal/mol units for {site.element} need a source quote"
            )
        sigmas = site.sigma_mixing_inputs_angstrom or (site.sigma_angstrom,)
        if not all(find_number(value, joined) for value in sigmas):
            raise ValueError(f"The sigma values for {site.element} need source quotes")
        if len(sigmas) == 2 and not re.search(r"Lorentz.Berthelot|arithmetic mean", joined, re.I):
            rule = re.search(
                r"[^.]{0,130}(?:Lorentz.Berthelot|arithmetic mean)[^.]{0,80}",
                document["text"],
                re.I,
            )
            if rule is None:
                raise ValueError("Mixed sigma requires evidence for the combining rule")
            snippet = rule.group().strip()
            if snippet not in quotes:
                quotes.append(snippet)
        if site.charge_e and not find_number(site.charge_e, joined):
            raise ValueError(f"The charge of {site.element} needs a source quote")
    return quotes


def resolve_interface(
    request: str, tube: HoneycombTube, queries: list[str], llm: LLM
) -> tuple[WallInterface, dict]:
    matches, log = lookup(queries)
    cached = public_source_documents()
    with ThreadPoolExecutor(max_workers=4) as executor:
        open_docs = list(
            executor.map(
                read_open_article, [record for record in matches if record.get("pmcid")][:5]
            )
        )
    indexed = {row["doi"] for row in matches}
    documents = {doc["doi"]: doc for doc in cached if doc["doi"] in indexed}
    for doc in cached:
        if doc["doi"] not in documents:
            try:
                match = identify_source(doc["doi"])
            except Exception as exc:
                log.append({"provider": "Crossref", "doi": doc["doi"], "error": str(exc)[:250]})
                continue
            matches.append(match)
            log.append(
                {"provider": "Crossref", "doi_results": [match["doi"]], "query": "DOI verification"}
            )
            documents[doc["doi"]] = doc
    for doc in open_docs:
        if doc:
            documents.setdefault(doc["doi"], doc)
    if not documents:
        raise ValueError(
            "No matching primary full text was retrieved. The wall force field remains unresolved."
        )
    evidence = {
        "request": request,
        "recipe": tube.model_dump(mode="json"),
        "documents": [
            {
                **{k: v for k, v in doc.items() if k not in {"text", "abstract"}},
                "excerpts": excerpts(doc["text"]),
            }
            for doc in list(documents.values())[:4]
        ],
    }
    instruction = (
        "Compare directly sourced rigid wall plus water Lennard-Jones candidates for the exact requested wall "
        "composition and confinement. Prefer matching water models, tube and observable. Use only the primary "
        "full-text excerpts. All sites use cross-interactions to water oxygen in kcal/mol and Angstrom. "
        "If sigma is derived by Lorentz–Berthelot, include the two source diameters in "
        "sigma_mixing_inputs_angstrom; the program computes the mean, so leave sigma_angstrom null if "
        "it is derived. If the material paragraph explicitly keeps sigma constant, "
        "follow the same document back to the reference wall and oxygen LJ diameters and the stated rule; "
        "a literal cross-sigma value then need not be printed. Do not mix numbers from different papers. "
        "If epsilon is only given in kJ/mol or would require another derivation, "
        "omit that candidate. For each model include up to five exact contiguous quotes in evidence_quotes "
        "covering the chemical elements, numeric coefficients, units, rule for sigma, and electrostatics. "
        "Keep quotes verbatim and short. Use lj_only with zero wall charges for a model describing wall–water "
        "interactions as LJ and electrostatics between the water molecules only. Explain applicability limits "
        "in at most three short sentences. Return empty candidates if evidence is incomplete. "
        "Only give a preferred DOI that matches one candidate."
    )
    example = (
        '{"candidates":[{"source_doi":"10.1234/example","fluid_model":"spce",'
        '"fluid_lj_element":"O","sites":[{"element":"C","charge_e":0,"epsilon_kcal_mol":0.1,'
        '"sigma_angstrom":3.2,"sigma_mixing_inputs_angstrom":null}],"evidence_quotes":["Exact source passage"],'
        '"scope":"Fixed pore matched to the cited water model.","wall_electrostatics":"lj_only"}],'
        '"preferred_doi":"10.1234/example","reason":"Explain the comparison."}'
    )
    valid = {}
    rejected = {}
    for attempt in range(2):
        try:
            decision = _invoke(
                llm,
                "potential",
                {**evidence, "rejected": rejected},
                LiteratureSelection,
                instruction,
                example,
            )
        except OutputParserException as exc:
            message = str(exc)
            rejected["structured_output"] = message[-1500:]
            continue
        if not decision.candidates:
            rejected["missing_evidence"] = decision.reason
        for candidate in decision.candidates:
            try:
                doc = documents.get(candidate.source_doi.lower())
                if not doc:
                    raise ValueError("The DOI is missing from the retrieved full text")
                checked_quotes = verify_interface(candidate, tube, doc)
                valid[candidate.source_doi] = candidate.model_copy(
                    update={"evidence_quotes": checked_quotes}
                )
            except (ValueError, KeyError, ValidationError) as exc:
                rejected[candidate.source_doi] = str(exc)
        if valid:
            selected = valid.get(decision.preferred_doi) or next(iter(valid.values()))
            metadata = [
                {k: v for k, v in doc.items() if k not in {"text", "abstract"}}
                for doc in documents.values()
            ]
            return selected, {
                "searches": log,
                "matches": matches,
                "documents": metadata,
                "candidates": [row.model_dump(mode="json") for row in valid.values()],
                "rejected": rejected,
                "selected": selected.source_doi,
                "reason": decision.reason,
                "planning_evidence": excerpts(documents[selected.source_doi.lower()]["text"]),
            }
    raise ValueError("No complete wall force field passed source verification: " + str(rejected))
