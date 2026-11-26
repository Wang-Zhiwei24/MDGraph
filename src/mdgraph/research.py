"""Research and document unfamiliar simulation systems before compiling a run."""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, time
from typing import Literal

from langchain_core.exceptions import OutputParserException
from pydantic import ConfigDict, Field, ValidationError

from .agents import _invoke
from .artifacts import atomic_json
from .config import LLM, StrictModel
from .literature import (
    excerpts,
    identify_source,
    lookup,
    normalize,
    public_source_documents,
    read_open_article,
    read_osti_article,
    search_index,
    search_open_archive,
    search_osti,
)

MAX_SEARCH_ROUNDS = 3
REQUIRED_EVIDENCE = frozenset({"system", "force_field", "analysis"})


class SearchTerms(StrictModel):
    material: str = Field(min_length=2, max_length=60)
    geometry: str = Field(min_length=2, max_length=60)
    medium: str | None = Field(default=None, min_length=2, max_length=50)

    def query(self) -> str:
        return " ".join(term for term in (self.material, self.medium, self.geometry) if term)

    def title_query(self) -> str:
        # The broader query still retains these qualifiers and the review checks the exact geometry.
        material = normalize(
            re.sub(r"\([^()]*\)|\b(?:surface|slabs?|plates?)\b", "", self.material, flags=re.I)
        )
        geometry = normalize(
            re.sub(r"\b(?:planar|parallel|periodic)\b", "", self.geometry, flags=re.I)
        )
        return f"{material or self.material} {geometry or self.geometry}"


class ResearchRequest(StrictModel):
    model_config = ConfigDict(extra="ignore")

    system: str = Field(min_length=2, max_length=240)
    geometry: str = Field(min_length=2, max_length=240)
    observable: str = Field(min_length=2, max_length=180)
    queries: list[str] = Field(min_length=1, max_length=3)
    search_terms: SearchTerms | None = None


class SourceClaim(StrictModel):
    doi: str = Field(min_length=7, max_length=180)
    quote: str = Field(min_length=20, max_length=300)
    application: str = Field(min_length=5, max_length=350)
    supports: list[
        Literal["system", "force_field", "substrate_model", "medium_model", "analysis"]
    ] = Field(default_factory=list, max_length=3)


class ResearchProposal(StrictModel):
    model: str = Field(min_length=10, max_length=850)
    force_field: str = Field(min_length=10, max_length=850)
    simulation: str = Field(min_length=10, max_length=850)
    analysis: str = Field(min_length=10, max_length=850)
    limitations: list[str] = Field(max_length=6)
    evidence: list[SourceClaim] = Field(max_length=6)
    next_queries: list[str] = Field(default_factory=list, max_length=3)


@dataclass(frozen=True)
class ResearchDraft:
    record: dict
    path: Path
    reused: bool = False


def recent_research(request: str, root: Path) -> ResearchDraft | None:
    """Reuse a recent completed review for the exact same user request."""
    from .artifacts import read_json

    cutoff = time() - 24 * 3600
    for path in sorted(root.glob("*_research_*/research_plan.json"), reverse=True):
        try:
            if path.stat().st_mtime < cutoff:
                continue
            record = read_json(path)
        except (OSError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and record.get("status") == "research_only"
            and record.get("request") == request
            and record.get("search_stop")
            in {"verified_evidence_sufficient", "search_budget_exhausted", "no_new_queries"}
            and record.get("proposal")
            and not record.get("author_clarification")
        ):
            return ResearchDraft(record, path.resolve(), reused=True)
    return None


def search_round(
    queries: list[str], title_query: str | None = None, *, on_leads=None
) -> tuple[list[dict], list[dict], list[dict]]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        indexed_results = executor.submit(lookup, queries)
        open_results = executor.submit(search_open_archive, queries[0])
        osti_results = executor.submit(search_osti, queries[0], title_query)
        if on_leads:
            for future in as_completed((indexed_results, open_results, osti_results)):
                if future.exception() is None:
                    leads = future.result()[0] if future is indexed_results else future.result()
                    if leads:
                        on_leads(leads)
                        break
        matches, searches = indexed_results.result()
        documents = []
        for provider, future in (("PMC BioC", open_results), ("OSTI", osti_results)):
            try:
                rows = future.result()
                searches.append(
                    {
                        "provider": provider,
                        "query": queries[0],
                        "doi_results": [row["doi"] for row in rows],
                    }
                )
                if provider == "OSTI" and title_query:
                    searches[-1]["title_query"] = title_query
                if provider == "PMC BioC":
                    documents.extend(rows)
                    matches.extend(
                        {key: value for key, value in row.items() if key != "text"} for row in rows
                    )
                else:
                    matches.extend(rows)
            except Exception as error:
                searches.append(
                    {"provider": provider, "query": queries[0], "error": str(error)[:250]}
                )
    return matches, documents, searches


def fetch_full_text(record: dict) -> dict | None:
    if record.get("osti_id"):
        document = read_osti_article(record)
        if document:
            return document
    if not record.get("pmcid") and not record.get("osti_id"):
        try:
            osti_match = search_osti(record["doi"], exact_doi=True)
        except Exception:
            osti_match = []
        if osti_match:
            document = read_osti_article(osti_match[0])
            if document:
                return document
    if not record.get("pmcid"):
        linked = search_index(f'DOI:"{record["doi"]}"', "Europe PMC")
        record = next(
            (row for row in linked if row["doi"] == record["doi"] and row.get("pmcid")), record
        )
    return read_open_article(record)


def search_keywords(text: str) -> set[str]:
    return {
        word.rstrip("s")
        for word in re.findall(r"[a-z]{3,}", text.lower())
        if word
        not in {"and", "the", "with", "inside", "between", "requested", "material", "geometry"}
    }


def research_system(
    request: str,
    target: ResearchRequest,
    conditions: dict,
    llm: LLM,
    *,
    context: dict | None = None,
    progress=None,
    on_preview=None,
) -> dict:
    """Retrieve and verify primary evidence while publishing provisional findings."""
    from openai import OpenAIError

    anchor = target.search_terms.query() if target.search_terms else ""
    initial_queries = [anchor, *target.queries]
    queries = list(
        dict.fromkeys(query for query in map(normalize, initial_queries) if 8 <= len(query) <= 240)
    )[:3]
    if not queries:
        raise ValueError("Research requires at least one usable bibliographic search query")

    terms = target.search_terms
    material = search_keywords(terms.material if terms else target.system)
    geometry = search_keywords(terms.geometry if terms else target.geometry)
    medium = search_keywords(terms.medium or "") if terms else set()
    matches, documents, searches = {}, {}, []
    attempts = Counter()
    searched_queries, rounds = set(), []
    proposal = None
    verified = []
    best_coverage = (False, 0, 0)
    stop_reason = "search_budget_exhausted"
    reviewed_inputs = set()

    def rank(row):
        title = search_keywords(row["title"])
        abstract = search_keywords(row.get("abstract", ""))
        return (
            bool(title & material),
            bool(title & geometry),
            bool(title & medium),
            len(title & (material | geometry | medium)),
            len(abstract & (material | geometry | medium)),
            row.get("publication_type") == "Journal Article",
            bool(row.get("osti_id") or row.get("pmcid")),
        )

    def record(reason, issues, *, leads=None):
        return {
            "status": "research_only",
            "request": request,
            "target": target.model_dump(),
            "author_clarification": context,
            "explicit_conditions": conditions,
            "searches": list(searches),
            "search_rounds": [dict(item) for item in rounds],
            "search_stop": reason,
            "matches": [
                {k: v for k, v in row.items() if k != "abstract"}
                for row in sorted(
                    leads if leads is not None else matches.values(), key=rank, reverse=True
                )
            ],
            "full_text": [
                {k: v for k, v in doc.items() if k not in {"text", "abstract"}}
                for doc in documents.values()
            ],
            "proposal": proposal.model_dump(exclude={"evidence", "next_queries"})
            if proposal
            else None,
            "verified_evidence": verified,
            "issues": issues,
        }

    cached = []
    for doc in public_source_documents():
        if search_keywords(doc["title"]) & material:
            cached.append(doc)

    for index in range(1, MAX_SEARCH_ROUNDS + 1):
        if progress:
            progress(
                f"Research {index}/{MAX_SEARCH_ROUNDS}: searching literature and retrieving full text"
            )
        searched_queries.update(query.casefold() for query in queries)
        title_query = terms.title_query() if index == 1 and terms else None
        started = monotonic()
        if index == 1 and on_preview:

            def publish_leads(leads):
                bibliography = [{"doi": row["doi"], "title": row["title"]} for row in leads]
                on_preview(
                    record(
                        "in_progress",
                        ["First literature leads. Search and evidence checks continue."],
                        leads=bibliography,
                    )
                )

            rows, archive_docs, trace = search_round(queries, title_query, on_leads=publish_leads)
        else:
            rows, archive_docs, trace = search_round(queries, title_query)
        round_log = {
            "round": index,
            "queries": queries,
            "search_seconds": round(monotonic() - started, 2),
        }
        rounds.append(round_log)
        searches.extend(trace)
        for row in rows:
            existing = matches.setdefault(row["doi"], row)
            for field in ("osti_id", "pmcid", "abstract", "publication_type"):
                if row.get(field) and not existing.get(field):
                    existing[field] = row[field]
        for doc in archive_docs:
            documents.setdefault(doc["doi"], doc)
        for doc in cached:
            if doc["doi"] not in matches:
                try:
                    matches[doc["doi"]] = identify_source(doc["doi"])
                except Exception as error:
                    searches.append(
                        {
                            "provider": "Crossref DOI record",
                            "doi": doc["doi"],
                            "error": str(error)[:250],
                        }
                    )
                    continue
                searches.append({"provider": "Crossref DOI record", "doi_results": [doc["doi"]]})
            documents.setdefault(doc["doi"], doc)

        candidates = sorted(matches.values(), key=rank, reverse=True)
        if index == 1 and on_preview:
            on_preview(
                record(
                    "in_progress",
                    ["Search results are leads. Full-text and force-field checks are underway."],
                )
            )
        started = monotonic()
        unread = [
            row for row in candidates if attempts[row["doi"]] < 2 and row["doi"] not in documents
        ]
        unread.sort(key=lambda row: attempts[row["doi"]])
        with ThreadPoolExecutor(max_workers=3) as executor:
            jobs = [(item, executor.submit(fetch_full_text, item)) for item in unread[:3]]
            for item, future in jobs:
                attempts[item["doi"]] += 1
                try:
                    article = future.result()
                except Exception as error:
                    searches.append(
                        {"provider": "full text", "doi": item["doi"], "error": str(error)[:250]}
                    )
                else:
                    if article:
                        documents[article["doi"]] = article
        ranked_docs = sorted(documents.values(), key=rank, reverse=True)[:4]
        round_log["considered_full_text"] = [doc["doi"] for doc in ranked_docs]
        round_log["full_text_seconds"] = round(monotonic() - started, 2)
        evidence_key = tuple(
            sorted((doc["doi"], doc.get("sha256", doc["text"])) for doc in ranked_docs)
        )
        # Another review can establish new claims only when the source text changes.
        refinements = []
        started = monotonic()
        reviewed = None
        try:
            if evidence_key in reviewed_inputs:
                round_log["review_skipped"] = "same_full_text_already_reviewed"
            else:
                if progress:
                    model = llm.settings_for("potential")["model"]
                    progress(
                        f"Research {index}/{MAX_SEARCH_ROUNDS}: checking evidence with {model}"
                    )
                reviewed = review_sources(
                    request,
                    target,
                    conditions,
                    llm,
                    context,
                    ranked_docs,
                    candidates[:6],
                    sorted(searched_queries),
                )
                reviewed_inputs.add(evidence_key)
        except (OutputParserException, OpenAIError, RuntimeError, ValueError) as error:
            round_log["review_error"] = type(error).__name__
            if isinstance(error.__cause__, ValidationError):
                round_log["validation_errors"] = error.__cause__.errors(
                    include_url=False, include_input=False, include_context=False
                )
        round_log["review_seconds"] = round(monotonic() - started, 2)
        if reviewed is not None:
            refinements = reviewed.next_queries
            current = []
            for claim in reviewed.evidence:
                source = next((doc for doc in ranked_docs if doc["doi"] == claim.doi.lower()), None)
                if source and normalize(claim.quote) in source["text"]:
                    current.append(
                        {
                            **claim.model_dump(),
                            "doi": source["doi"],
                            "title": source["title"],
                            "url": source["url"],
                        }
                    )
                else:
                    round_log.setdefault("unverified_dois", []).append(claim.doi)
            by_doi = {
                doi: {
                    item for claim in current if claim["doi"] == doi for item in claim["supports"]
                }
                for doi in {claim["doi"] for claim in current}
            }
            for roles in by_doi.values():
                if {"substrate_model", "medium_model"} <= roles:
                    roles.add("force_field")
            covered = REQUIRED_EVIDENCE & {role for roles in by_doi.values() for role in roles}
            matched = any({"system", "force_field"} <= roles for roles in by_doi.values())
            score = (matched, len(covered), len(current))
            if proposal is None or score >= best_coverage:
                proposal, verified = reviewed, current
                best_coverage = score
            round_log["verified_coverage"] = sorted(covered)
            round_log["matched_system_and_force_field"] = matched
            if matched and REQUIRED_EVIDENCE <= covered:
                stop_reason = "verified_evidence_sufficient"
                break
            if on_preview:
                on_preview(
                    record(
                        "in_progress", ["Evidence review continues. This draft is for review only."]
                    )
                )

        seed = anchor or initial_queries[1]
        proposed = [
            *refinements,
            f"{seed} molecular dynamics",
            f"{seed} force field",
            f"{seed} {target.observable}",
            f"{seed} simulation parameters",
        ]
        if anchor:
            proposed = [
                query
                if material <= search_keywords(query) and geometry & search_keywords(query)
                else f"{anchor} {query}"
                for query in proposed
            ]
        queries = list(
            dict.fromkeys(
                q
                for q in map(normalize, proposed)
                if 8 <= len(q) <= 240 and q.casefold() not in searched_queries
            )
        )[:2]
        if not queries:
            stop_reason = "no_new_queries"
            break

    issues = []
    if stop_reason != "verified_evidence_sufficient":
        issues.append(
            "Search ended before primary evidence verified the system, force field and analysis."
        )
    if not documents:
        issues.append("No primary full text was retrieved.")
    return record(stop_reason, issues)


def review_sources(
    request, target, conditions, llm, context, documents, candidates, search_history
):
    evidence = {
        "request": request,
        "target": target.model_dump(),
        "explicit_conditions": conditions,
        "author_clarification": context,
        "search_history": search_history,
        "full_text": [
            {
                "doi": doc["doi"],
                "title": doc["title"],
                "methods_excerpts": excerpts(doc["text"])[:3600],
            }
            for doc in documents
        ],
        "bibliographic_leads": [{"doi": row["doi"], "title": row["title"]} for row in candidates],
    }
    instruction = (
        "Draft a reviewable MD study for the exact request. Preserve the material, geometry, observable and explicit conditions. "
        "Give a feasible geometry/structure recipe, force-field and topology plan, equilibration/production protocol, "
        "and a mathematical definition of the requested analysis. Distinguish literature-reported settings from "
        "recommended settings and list missing artifacts or tests. Separate a matched force field from a hypothesis. "
        "For slit mass density, sum full molecular masses in each bin and divide by the in-plane area times bin "
        "width, then average over frames. Distinguish center-of-mass coordinates from oxygen-site binning. "
        "When the user fixes the slit width, keep the normal length fixed during sampling. Explain loading and "
        "distinguish geometric from accessible or dynamic widths. "
        "Return four short strings of up to 450 characters each, at most four brief limitations, and up to "
        "four evidence objects. Cite provided full text DOIs and supply an exact contiguous quote of "
        "20–250 characters for each claim. In supports use system for the exact requested material and geometry "
        "together, force_field for explicit named substrate-fluid parameters or matched models, and analysis "
        "for a reported analysis method or explicit sampling of the positions required for the requested observable. "
        "When model names are reported separately for a single simulation, quote the exact phrases as "
        "substrate_model and medium_model. A saved atomic configuration trajectory "
        "supports analysis feasibility; it does not prove that the target observable was reported. "
        "Identify any inference from a related observable. Emit separate passages when a single passage cannot "
        "support all those claims. A different wall morphology or fluid is insufficient for system. "
        "A paper on a nearby system or a review can be mentioned with supports empty. "
        "If critical evidence is missing, give next_queries with one or two materially different short searches: "
        "use accurate synonyms, distinctive geometry and missing force-field or analysis terms. "
        "When no matched force field is known, avoid narrowing the search to a speculative named model. "
        "Queries must add information to search_history. Preserve all explicit conditions and the author's corrections. "
        "Titles are discovery leads. "
        "Do not claim that a structure, topology, parameter file or solver input has been prepared or validated."
    )
    example = (
        '{"model":"Geometry and assumptions","force_field":"Candidate and evidence limits",'
        '"simulation":"Physical stages and checks","analysis":"Observable and normalization",'
        '"limitations":["Missing force field artifact"],'
        '"evidence":[{"doi":"returned DOI","quote":"exact passage from source","application":"scope of claim",'
        '"supports":["system"]}],"next_queries":["focused follow-up search"]}'
    )
    try:
        return _invoke(llm, "potential", evidence, ResearchProposal, instruction, example)
    except OutputParserException as error:
        return _invoke(
            llm,
            "potential",
            {**evidence, "validation_error": str(error)[-1500:]},
            ResearchProposal,
            instruction + " Correct the response format using the example.",
            example,
        )


def save_research(record: dict, directory: Path, llm: LLM) -> ResearchDraft:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.resolve() / "research_plan.json"
    atomic_json(path, record)
    from .slit_preview import PreviewDecision, build_preview, tool_inventory

    tools = tool_inventory()
    if (
        tools
        and record["verified_evidence"]
        and record.get("search_stop", "verified_evidence_sufficient")
        == "verified_evidence_sufficient"
    ):
        from openai import OpenAIError

        evidence = {
            key: record[key]
            for key in ("request", "target", "explicit_conditions", "proposal", "verified_evidence")
        }
        evidence["tools"] = {
            **tools,
            "scope": "CelluloseForge crystal slabs of cellulose/chitin, ASE assembly of explicit water slit",
            "recipe_schema": PreviewDecision.model_json_schema(),
        }
        instruction = (
            "Choose the installed geometry preview tool only when it preserves the material, phase, requested "
            "geometry and each explicit condition, and verified evidence supports the crystalline face. "
            "The slab is periodic across x and y with complete periodic chains and repeated along the z normal. "
            "The width is the geometric gap between facing outermost atom planes, not a fitted effective height. "
            "Do not choose this slit for tube or bulk requests. Return recipe null whenever the tool does not fit. "
            "Silently changing the requested material or a consequential width definition is invalid. "
            "Use an orthorhombic cell, dimensions suited to interfacial sampling, and moderate atom counts. "
            "Use the literal fluid string water. Keep reason under 220 characters. Use exactly the recipe keys "
            "in the example with no additional fields. This tool prepares coordinates only."
        )
        example = (
            '{"recipe":{"polymer":"cellulose","allomorph":"I-beta","miller_face":[1,1,0],'
            '"layers":4,"chains_per_layer":8,"degree_polymerization":8,"width_angstrom":20,'
            '"fluid":"water","reason":"Explain the face, dimensions and width definition."}} or {"recipe":null}'
        )
        try:
            try:
                decision = _invoke(
                    llm, "structure", evidence, PreviewDecision, instruction, example
                )
            except OutputParserException as error:
                if isinstance(error.__cause__, ValidationError):
                    feedback = error.__cause__.errors(
                        include_url=False, include_input=False, include_context=False
                    )
                else:
                    feedback = "Return only the JSON object matching the example."
                decision = _invoke(
                    llm,
                    "structure",
                    {**evidence, "validation_error": feedback},
                    PreviewDecision,
                    instruction + " Correct the schema errors in the last response.",
                    example,
                )
            if decision.recipe:
                recipe = decision.recipe
                width = record["explicit_conditions"].get(
                    "slit_width_angstrom", recipe.width_angstrom
                )
                if recipe.width_angstrom != width:
                    raise ValueError("The preview would change the requested slit width")
                record["structure_recipe"] = recipe.model_dump(mode="json")
                record["structure_preview"] = build_preview(recipe, directory.resolve() / "preview")
        except (OutputParserException, OpenAIError, OSError, ValueError) as error:
            if isinstance(error.__cause__, ValidationError):
                record["preview_validation_errors"] = error.__cause__.errors(
                    include_url=False, include_input=False, include_context=False
                )
            detail = (
                ""
                if isinstance(error, (OutputParserException, OpenAIError))
                else f": {str(error)[:180]}"
            )
            record["issues"].append(f"Preview unavailable ({type(error).__name__}){detail}")
        atomic_json(path, record)
    return ResearchDraft(record, path)


def describe_research(draft: ResearchDraft) -> str:
    record = draft.record
    lines = [
        f"Research draft for {record['target']['system']}. A runnable input is pending validation."
    ]
    if draft.reused:
        lines.append("Reused the recent review for this exact request.")
    if record.get("search_stop"):
        result = {
            "verified_evidence_sufficient": "verified evidence covers system, force field and analysis",
            "search_budget_exhausted": "search budget reached with open evidence gaps",
            "no_new_queries": "no unused queries remained",
            "in_progress": "provisional evidence; verification continues",
        }
        lines.append(
            f"Literature search ({len(record['search_rounds'])} rounds): {result[record['search_stop']]}."
        )
    if record["proposal"]:
        for key, label in (
            ("model", "Model"),
            ("force_field", "Force field"),
            ("simulation", "Protocol"),
            ("analysis", "Analysis"),
        ):
            lines.append(f"{label} — {record['proposal'][key]}")
        lines.extend(f"Open item — {item}" for item in record["proposal"]["limitations"])
    lines.extend(
        f"DOI {claim['doi']} — {claim['application']}" for claim in record["verified_evidence"]
    )
    if record.get("search_stop") == "in_progress" and not record["verified_evidence"]:
        lines.extend(
            f"Literature lead {row['doi']} — {row['title']}"
            for row in record.get("matches", [])[:3]
        )
    if "structure_preview" in record:
        model = record["structure_preview"]
        lines.append(
            f"ASE coordinates checked: {model['width_angstrom']:g} Å geometric gap, "
            f"{model['solid_atoms']} wall atoms, {model['water_molecules']} water molecules."
        )
        lines.append(f"STRUCTURE_FILE={model['coordinate_file']}")
    lines.extend(record["issues"])
    lines.append(f"RESEARCH_FILE={draft.path}")
    return "\n".join(lines)
