import json

import httpx
import pytest
import yaml

from mdgraph.agents import AgentAdvice, AgentDecision, _invoke, decide
from mdgraph.artifacts import atomic_json, read_json, sha256
from mdgraph.cli import create_run, drive
from mdgraph.config import TaskConfig
from mdgraph.natural import default_llm


@pytest.mark.parametrize(
    "role, model, effort, token_limit",
    [
        ("manager", "deepseek-v4-pro", "high", 16384),
        ("potential", "deepseek-flash", "none", 2048),
    ],
)
def test_deepseek_thinking_settings_reach_chat_completions(
    monkeypatch, role, model, effort, token_limit
):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("MDGRAPH_API_KEY", "test-key")
    llm = default_llm()
    llm = llm.model_validate(
        {
            **llm.model_dump(),
            "roles": {
                "potential": {"model": "deepseek-flash", "reasoning_effort": "none"},
            },
        }
    )
    requests = []

    def send(client, request, **kwargs):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"summary":"Validated reply"}',
                            "reasoning_content": "Fixture reasoning, not part of the answer.",
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    answer = _invoke(llm, role, {}, AgentAdvice, "", '{"summary":"Brief finding"}')
    assert answer == AgentAdvice(summary="Validated reply")
    ((path, payload),) = requests
    assert path == "/chat/completions"
    assert payload["model"] == model
    assert payload["thinking"] == {"type": "enabled" if effort != "none" else "disabled"}
    assert payload.get("reasoning_effort", "none") == effort
    assert payload["max_tokens"] == token_limit
    assert "max_completion_tokens" not in payload
    assert payload["response_format"] == {"type": "json_object"}


def task(tmp_path, *, duplicate=False, corrupt=False):
    artifact = tmp_path / "cu.yace"
    artifact.write_bytes(b"test artifact only")
    entry = {
        "id": "cu-test",
        "kind": "pace",
        "elements": ["Cu"],
        "units": "metal",
        "application": "fcc",
        "temperature_k": [200, 400],
        "source": "test fixture",
        "model": artifact.name,
        "model_sha256": "0" * 64 if corrupt else sha256(artifact),
    }
    catalog = tmp_path / "catalog.yaml"
    entries = [entry, {**entry, "id": "second"}] if duplicate else [entry]
    catalog.write_text(yaml.safe_dump({"entries": entries}), encoding="utf-8")
    return TaskConfig.model_validate(
        {
            "name": "cu_test",
            "structure": {
                "elements": ["Cu"],
                "crystal": {"lattice": "fcc", "a_angstrom": 3.615, "repeat": [2, 2, 2]},
            },
            "potential": {"kind": "pace", "catalog": str(catalog), "application": "fcc"},
        }
    )


def test_supervisor_dispatches_preparation_and_resumes_same_job(tmp_path, monkeypatch):
    class Backend:
        submissions = 0

        def submit(self, directory, run_id):
            self.submissions += 1
            assert "pair_coeff * * model.yace Cu" in (directory / "in.lammps").read_text()
            return {"job_id": "fixture", "backend": "test"}

        def poll(self, directory, receipt):
            return {"status": "running"}

    backend = Backend()
    monkeypatch.setattr("mdgraph.graph.get_backend", lambda execution: backend)
    root = create_run(task(tmp_path), tmp_path / "runs")
    prepared = drive(root, prepare_only=True)
    assert prepared["status"] == "prepared" and backend.submissions == 0
    assert read_json(root / "status.json")["next"] == ["operator"]
    result = drive(root)
    assert result["status"] == "submitted"
    assert result["structure"]["atoms"] == 32
    assert result["potential"]["candidate_id"] == "cu-test"
    assert result["receipt"]["job_id"] == "fixture"
    assert read_json(root / "status.json")["next"] == ["monitor"]
    assert drive(root)["receipt"]["job_id"] == "fixture"
    assert backend.submissions == 1


def test_ambiguous_and_corrupt_potentials_stop_before_submission(tmp_path, monkeypatch):
    def forbidden(execution):
        raise AssertionError("Invalid potential selection must not initialize the runner")

    monkeypatch.setattr("mdgraph.graph.get_backend", forbidden)
    for options, expected in (
        ({"duplicate": True}, "compatible IDs"),
        ({"corrupt": True}, "SHA256"),
    ):
        root = create_run(task(tmp_path, **options), tmp_path / "runs")
        result = drive(root)
        assert result["status"] == "failed"
        assert expected in result["errors"][0]
        assert not (root / "attempt_00" / "in.lammps").exists()


def test_version_1_graph_remains_available(tmp_path, monkeypatch):
    cfg = TaskConfig.model_validate(
        {
            "name": "old",
            "structure": {"demo_argon": True, "elements": ["Ar"]},
            "potential": {"kind": "lj_demo"},
        }
    )
    root = create_run(cfg, tmp_path)
    manifest = read_json(root / "run_manifest.json")
    atomic_json(root / "run_manifest.json", {**manifest, "schema": 1})

    class Backend:
        def submit(self, directory, run_id):
            return {"job_id": "legacy"}

        def poll(self, directory, receipt):
            return {"status": "running"}

    monkeypatch.setattr("mdgraph.graph.get_backend", lambda execution: Backend())
    result = drive(root)
    assert result["receipt"]["job_id"] == "legacy"
    assert "designer" in result["advice"]
    assert "next_agent" not in result


def test_decisions_change_actual_input_and_resume_without_reselecting(tmp_path, monkeypatch):
    cfg = task(tmp_path)
    path = tmp_path / "catalog.yaml"
    catalog = yaml.safe_load(path.read_text())
    second = tmp_path / "alternative.yace"
    second.write_bytes(b"genuinely different fixture")
    catalog["entries"].append(
        {
            **catalog["entries"][0],
            "id": "alternative",
            "model": second.name,
            "model_sha256": sha256(second),
        }
    )
    catalog["entries"].append(
        {**catalog["entries"][0], "id": "wrong_temp", "temperature_k": [400, 500]}
    )
    path.write_text(yaml.safe_dump(catalog))
    data = cfg.model_dump()
    data["llm"] = {
        "enabled": True,
        "decisions": True,
        "model": "shared",
        "roles": {"potential": {"model": "potential-model"}},
    }
    cfg = TaskConfig.model_validate(data)
    assert cfg.llm.settings_for("potential")["model"] == "potential-model"
    assert cfg.llm.settings_for("operator")["model"] == "shared"
    calls = []

    def choose(cfg, role, evidence, choices, **options):
        calls.append(role)
        value = {
            "manager": "proceed",
            "structure": "wrap",
            "potential": "alternative",
            "designer": "standard",
            "operator": "stop",
        }[role]
        comparisons = []
        if role == "potential":
            assert set(choices) == {"cu-test", "alternative"}
            comparisons = [{"candidate_id": k, "assessment": "fixture comparison"} for k in choices]
        return {
            "choice": value,
            "reason": "fixture reason",
            "comparisons": comparisons,
            "mode": "llm",
        }

    monkeypatch.setattr("mdgraph.graph.decide", choose)
    monkeypatch.setattr("mdgraph.graph.advise", lambda *args: {"summary": "fixture"})
    monkeypatch.setattr(
        "mdgraph.graph.get_backend", lambda _: pytest.fail("operator stop must prevent submission")
    )
    root = create_run(cfg, tmp_path / "runs")
    state = drive(root, prepare_only=True)
    attempt = root / "attempt_00"
    assert state["potential"]["candidate_id"] == "alternative"
    assert state["structure"]["wrapped"]
    assert (attempt / "model.yace").read_bytes() == second.read_bytes()
    assert read_json(attempt / "potential_decision.json")["choice"] == "alternative"
    result = drive(root)
    assert result["status"] == "failed"
    assert calls == ["manager", "structure", "potential", "designer", "operator"]


@pytest.mark.parametrize(
    "answer",
    [
        {"choice": "invented", "reason": "fixture"},
        {
            "choice": "a",
            "reason": "fixture",
            "comparisons": [{"candidate_id": "a", "assessment": "only one"}],
        },
    ],
)
def test_invalid_llm_decision_is_rejected(tmp_path, monkeypatch, answer):
    cfg = task(tmp_path)
    cfg.llm.enabled = cfg.llm.decisions = True
    monkeypatch.setattr("mdgraph.agents._invoke", lambda *_: AgentDecision.model_validate(answer))
    with pytest.raises(ValueError):
        decide(cfg, "potential", {}, {"a": "one", "b": "two"}, compare_candidates=True)
