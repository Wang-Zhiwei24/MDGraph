import json
from argparse import Namespace

import pytest
from ase import Atoms
from ase.io import read
from langchain_core.output_parsers import PydanticOutputParser

from mdgraph.artifacts import atomic_json, read_json
from mdgraph.chat import chat
from mdgraph.cli import natural_run
from mdgraph.config import LLM
from mdgraph.natural import ManagerTurn, ResearchContinuation, ResearchRecovery, manager_request
from mdgraph.research import ResearchProposal, ResearchRequest, fetch_full_text, research_system
from mdgraph.slit_preview import pack_slit


def test_review_reuses_recent_exact_request_without_calling_manager(tmp_path, monkeypatch):
    path = tmp_path / "20261126_093000_research_example" / "research_plan.json"
    path.parent.mkdir()
    request = "Calculate the wall-normal density of water in a silica slit"
    atomic_json(
        path,
        {
            "status": "research_only",
            "request": request,
            "proposal": {"model": "Silica slit"},
            "search_stop": "search_budget_exhausted",
        },
    )
    monkeypatch.setattr(
        "mdgraph.natural.manager_request", lambda *a, **k: pytest.fail("Repeated model call")
    )
    result = natural_run(request, Namespace(system=None, runs_root=tmp_path), review=True)
    assert result.path == path.resolve() and result.reused


def test_research_publishes_previews_and_reuses_same_full_text_review(monkeypatch):
    target = ResearchRequest(
        system="silica and water",
        geometry="slit",
        observable="density",
        queries=["silica slit water density"],
    )
    paper = {
        "doi": "10.1234/slit",
        "title": "Water in silica slits",
        "url": "https://example.org/slit",
        "text": "A silica slit pore was filled with water.",
    }
    reviews, previews = [], []

    def review(*args):
        assert len(previews) == 1
        reviews.append(args)
        return ResearchProposal(
            model="Silica slit with water",
            force_field="Force field requires more evidence",
            simulation="Fixed channel sampling",
            analysis="Bin water along the normal",
            evidence=[],
            limitations=[],
        )

    monkeypatch.setattr("mdgraph.research.public_source_documents", lambda: [])
    monkeypatch.setattr(
        "mdgraph.research.search_round", lambda *args, **kwargs: ([paper], [paper], [])
    )
    monkeypatch.setattr("mdgraph.research.review_sources", review)
    result = research_system("Water density", target, {}, LLM(), on_preview=previews.append)
    assert len(reviews) == 1
    assert previews[0]["proposal"] is None
    assert previews[1]["proposal"]["model"] == "Silica slit with water"
    assert result["search_rounds"][1]["review_skipped"] == "same_full_text_already_reviewed"
    assert result["search_stop"] != "verified_evidence_sufficient"


def test_new_confinement_search_is_logged_and_cannot_submit(tmp_path, monkeypatch, capsys):
    request = "计算受限于纤维素二维通道中水的法向密度分布，受限高度为2纳米"
    target = ResearchRequest(
        system="cellulose and water",
        geometry="slit",
        observable="wall-normal mass density",
        queries=["cellulose water slit molecular dynamics"],
    )
    turns = iter([request, "/plan", "/confirm", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(turns))
    monkeypatch.setattr(
        "mdgraph.natural._invoke", lambda *args: ManagerTurn(action="research", research=target)
    )
    monkeypatch.setattr("mdgraph.chat.launch_draft", lambda *args: pytest.fail("Unexpected job"))
    monkeypatch.setattr(
        "mdgraph.research.lookup", lambda queries: ([], [{"query": queries[0], "doi_results": []}])
    )
    monkeypatch.setattr("mdgraph.research.search_open_archive", lambda _: [])
    monkeypatch.setattr("mdgraph.research.search_osti", lambda *_: [])

    def no_llm(*args):
        raise ValueError("No credentials in fixture")

    monkeypatch.setattr("mdgraph.research._invoke", no_llm)
    chat(
        Namespace(
            run=None,
            system=None,
            execution=None,
            lammps=None,
            smoke=False,
            review=True,
            runs_root=tmp_path,
        )
    )
    assert "requires validated inputs before it can be submitted" in capsys.readouterr().out
    draft = read_json(next(tmp_path.glob("*/research_plan.json")))
    assert draft["status"] == "research_only"
    assert draft["explicit_conditions"] == {"slit_width_angstrom": 20}
    assert draft["searches"][0]["query"] == target.queries[0]
    assert not list(tmp_path.rglob("run_manifest.json"))


def test_slit_preview_keeps_geometric_width_and_counts_water(tmp_path):
    slab = Atoms("C2", positions=[(1, 1, 2), (5, 8, 7)], cell=[25.2, 26.3, 30], pbc=True)
    record = pack_slit(slab, 20, tmp_path)
    system = read(tmp_path / "initial.extxyz")
    lower, upper = record["surface_planes_angstrom"]
    assert upper - lower == pytest.approx(20)
    assert len(system) == 2 + 3 * record["water_molecules"]
    assert list(system.arrays["fluid_molecule_id"][:5]) == [0, 0, 1, 1, 1]
    assert record["initial_accessible_density_g_cm3"] == pytest.approx(0.997, abs=0.006)


def test_search_continues_after_irrelevant_full_text_and_stops_on_verified_methods(monkeypatch):
    target = ResearchRequest(
        system="water between amorphous silica slabs",
        geometry="planar slit",
        observable="wall-normal mass density",
        queries=["silica liquid interface"],
        search_terms={"material": "amorphous silica", "medium": "water", "geometry": "slit pores"},
    )
    unrelated = {
        "doi": "10.1234/cylinder",
        "title": "Water in a cylindrical silica nanopore",
        "url": "https://example.org/cylinder",
        "text": "Water was confined in a cylindrical silica pore.",
    }
    methods = (
        "Liquid water was confined in amorphous silica slit pores. "
        "The Gulmen-Thompson surface model was paired with SPC/E water. "
        "Water positions were saved every 2 ps for wall-normal density analysis."
    )
    matched = {
        "doi": "10.1234/slit",
        "title": "Water in amorphous silica slit pores",
        "url": "https://example.org/slit",
        "text": methods,
    }
    queries_seen, opened = [], []

    def search(queries, _):
        queries_seen.append(queries)
        if len(queries_seen) == 1:
            return [], [unrelated], []
        return [matched], [unrelated], []

    def read_source(item):
        opened.append(item["doi"])
        return matched

    def review(request, target, conditions, llm, context, docs, candidates, search_history):
        if docs[0]["doi"] == unrelated["doi"]:
            claims = [
                {
                    "doi": unrelated["doi"],
                    "quote": "This passage does not appear in the retrieved source.",
                    "application": "Insufficient match",
                    "supports": ["system", "force_field", "analysis"],
                }
            ]
        else:
            claims = [
                {
                    "doi": matched["doi"],
                    "quote": quote.strip(),
                    "application": "Direct method evidence",
                    "supports": [role],
                }
                for quote, role in zip(
                    methods.split(". ")[:3], ("system", "force_field", "analysis")
                )
            ]
        return ResearchProposal(
            model="Amorphous silica slit pore.",
            force_field="Gulmen-Thompson and SPC/E.",
            simulation="Constrained NVT run at 298 K.",
            analysis="Bin the molecular centers.",
            evidence=claims,
            limitations=[],
            next_queries=["silica slit hydroxylation"],
        )

    monkeypatch.setattr("mdgraph.research.search_round", search)
    monkeypatch.setattr("mdgraph.research.fetch_full_text", read_source)
    monkeypatch.setattr("mdgraph.research.review_sources", review)
    monkeypatch.setattr("mdgraph.research.public_source_documents", lambda: [])
    result = research_system("water density normal to the slabs", target, {}, LLM())

    assert queries_seen[0][0] == "amorphous silica water slit pores"
    assert "amorphous silica" in queries_seen[1][0] and "slit" in queries_seen[1][0]
    assert opened == ["10.1234/slit"]
    assert len(queries_seen) == 2
    assert result["search_rounds"][0]["unverified_dois"] == ["10.1234/cylinder"]
    assert result["search_stop"] == "verified_evidence_sufficient"
    assert {row["doi"] for row in result["verified_evidence"]} == {"10.1234/slit"}


def test_research_revision_keeps_author_constraints(tmp_path, monkeypatch):
    initial = "计算二维二氧化硅壁面里水的法相密度，通道高度为2纳米"
    clarification = "我指的是两块平行的无定形二氧化硅平板"
    inputs = iter([initial, clarification, "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    def manager(llm, role, evidence, schema, *_):
        follow_up = evidence["request"] == clarification
        research = ResearchRequest(
            system="water and amorphous silica" if follow_up else "water and silica",
            geometry="planar slit",
            observable="wall-normal density",
            queries=["silica water confinement"],
        )
        if follow_up:
            assert schema is ResearchContinuation
            assert evidence["original_goal"] == initial
            assert evidence["prior_explicit_conditions"] == {"slit_width_angstrom": 20}
            return ResearchContinuation(research=research)
        return ManagerTurn(action="research", research=research)

    def fake_research(request, target, conditions, llm, *, context=None, progress=None):
        return {
            "status": "research_only",
            "request": request,
            "target": target.model_dump(),
            "explicit_conditions": conditions,
            "proposal": None,
            "verified_evidence": [],
            "issues": [],
        }

    monkeypatch.setattr("mdgraph.natural._invoke", manager)
    monkeypatch.setattr("mdgraph.cli.research_system", fake_research)
    chat(
        Namespace(
            run=None,
            system=None,
            execution=None,
            lammps=None,
            smoke=False,
            review=True,
            runs_root=tmp_path,
        )
    )
    records = [read_json(p) for p in tmp_path.glob("*/research_plan.json")]
    revised = next(record for record in records if clarification in record["request"])
    assert initial in revised["request"]
    assert revised["explicit_conditions"] == {"slit_width_angstrom": 20}
    assert revised["target"]["observable"] == "wall-normal density"


def test_full_text_can_follow_a_crossref_doi_to_osti(monkeypatch):
    hit = {"doi": "10.1234/slit", "title": "A planar slit", "provider": "Crossref"}
    archived = {**hit, "osti_id": "100"}
    full_text = {**archived, "text": "Primary methods for the slit."}

    def osti(query, *, exact_doi):
        assert query == hit["doi"] and exact_doi
        return [archived]

    monkeypatch.setattr("mdgraph.research.search_osti", osti)
    monkeypatch.setattr(
        "mdgraph.research.read_osti_article",
        lambda record: full_text if record == archived else None,
    )
    monkeypatch.setattr(
        "mdgraph.research.search_index", lambda *_: pytest.fail("Full text was already found")
    )
    assert fetch_full_text(hit) == full_text


SILICA_REQUEST = "计算受限于二维二氧化硅壁面的水的法相密度分布"
SILICA_RESEARCH = {
    "system": "water confined between two-dimensional silica surfaces",
    "geometry": "planar slit",
    "observable": "wall-normal water density",
    "queries": ["silica walls confined water density"],
}


def test_chat_records_and_repairs_non_executable_reason_alias(tmp_path, monkeypatch, capsys):
    inputs = iter([SILICA_REQUEST, "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    def invoke(llm, role, evidence, schema, *_):
        assert evidence["request"] == SILICA_REQUEST and schema is ManagerTurn
        return PydanticOutputParser(pydantic_object=schema).parse(
            json.dumps(
                {
                    "action": "research",
                    "research": SILICA_RESEARCH,
                    "reason": "Research the wall and water model.",
                }
            )
        )

    def research(request, target, conditions, llm, *, context=None, progress=None):
        assert request == SILICA_REQUEST
        return {
            "status": "research_only",
            "request": request,
            "target": target.model_dump(),
            "explicit_conditions": conditions,
            "proposal": None,
            "verified_evidence": [],
            "issues": [],
        }

    monkeypatch.setattr("mdgraph.natural._invoke", invoke)
    monkeypatch.setattr("mdgraph.cli.research_system", research)
    chat(
        Namespace(
            run=None,
            system=None,
            execution=None,
            lammps=None,
            smoke=False,
            review=True,
            runs_root=tmp_path,
        )
    )
    found = read_json(next(tmp_path.glob("*/manager_validation.json")))
    assert found["attempts"][0]["errors"][0]["field"] == "$.reason"
    assert found["attempts"][0]["errors"][0]["type"] == "extra_forbidden"
    assert found["attempts"][0]["handling"] == "reason_to_message"
    assert found["final_action"] == "research"
    assert "MANAGER_VALIDATION_FILE=" in capsys.readouterr().err
    assert read_json(next(tmp_path.glob("*/research_plan.json")))["request"] == SILICA_REQUEST


def test_manager_retries_nested_fields_then_recovers_without_runnable_plan(monkeypatch):
    events, schemas = [], []

    def invoke(llm, role, evidence, schema, *_):
        schemas.append(schema)
        assert evidence["request"] == SILICA_REQUEST
        if schema is ResearchRecovery:
            return schema(
                action="research", research=ResearchRequest.model_validate(SILICA_RESEARCH)
            )
        research = {**SILICA_RESEARCH, "search_terms": {"material": "silica", "geometry": 12}}
        return PydanticOutputParser(pydantic_object=schema).parse(
            json.dumps({"action": "research", "research": research})
        )

    monkeypatch.setattr("mdgraph.natural._invoke", invoke)
    decision, _, _ = manager_request(SILICA_REQUEST, conversation={}, validation_events=events)
    assert schemas == [ManagerTurn, ManagerTurn, ResearchRecovery]
    assert [event["errors"][0]["field"] for event in events[:2]] == [
        "$.research.search_terms.geometry"
    ] * 2
    assert (decision.action, decision.plan) == ("research", None)


def test_chat_persists_field_errors_when_all_attempts_are_rejected(tmp_path, monkeypatch, capsys):
    inputs = iter([SILICA_REQUEST, "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    def invoke(llm, role, evidence, schema, *_):
        return PydanticOutputParser(pydantic_object=schema).parse(
            '{"action":"run","plan":null,"message":"raw response marker"}'
        )

    monkeypatch.setattr("mdgraph.natural._invoke", invoke)
    monkeypatch.setattr(
        "mdgraph.chat.launch_draft", lambda *_: pytest.fail("An invalid run cannot be submitted")
    )
    chat(
        Namespace(
            run=None,
            system=None,
            execution=None,
            lammps=None,
            smoke=False,
            review=True,
            runs_root=tmp_path,
        )
    )
    path = next(tmp_path.glob("*/manager_validation.json"))
    log = read_json(path)
    assert log["final_action"] is None
    assert [event["errors"][0]["field"] for event in log["attempts"]] == [
        "$.plan",
        "$.plan",
        "$.action",
    ]
    assert "raw response marker" not in path.read_text(encoding="utf-8")
    assert "$.action" in capsys.readouterr().err
    assert not list(tmp_path.rglob("run_manifest.json"))
