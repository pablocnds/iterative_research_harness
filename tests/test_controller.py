from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from httpx import Request
from openai import APITimeoutError

import irh.agent
from irh.config import ExternalToolConfig
from irh.controller import ActiveTimeBudget, Controller, Supervisor
from irh.store import Store
from irh.types import IterationDefinition, ProjectCreate


@pytest.fixture(autouse=True)
def fake_token_count(monkeypatch) -> None:
    async def count(_self, _request):
        return 10

    monkeypatch.setattr(irh.agent.AgentSession, "_count_tokens", count)


class FakeCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


@pytest.mark.asyncio
async def test_active_time_budget_does_not_expire_while_paused() -> None:
    budget = ActiveTimeBudget(0.08)
    expiry = asyncio.create_task(budget.wait_until_exhausted())
    await asyncio.sleep(0.03)
    budget.pause()
    paused_remaining = budget.remaining()
    await asyncio.sleep(0.10)
    assert not expiry.done()
    assert budget.remaining() == pytest.approx(paused_remaining, abs=0.01)
    budget.resume()
    await asyncio.wait_for(expiry, timeout=0.2)
    assert budget.remaining() == 0


class FakeCompletions:
    upper_calls = 0

    async def create(self, **request):
        names = {tool["function"]["name"] for tool in request.get("tools", [])}
        calls = []
        if "decide" in names:
            self.__class__.upper_calls += 1
            if self.__class__.upper_calls == 1:
                arguments = {
                    "action": "iterate",
                    "reason": "run the smallest baseline",
                    "definition": {
                        "subgoal": "write one measured result",
                        "hypothesis": "the result can be recorded",
                        "introduction": "deterministic test",
                        "parent": None,
                        "inputs": [],
                    },
                }
            else:
                arguments = {"action": "stop", "reason": "evidence reviewed", "stop_reason": "achieved"}
            calls = [FakeCall("decision", "decide", arguments)]
        elif "submit_results" in names:
            calls = [
                FakeCall("write", "write_file", {"path": "result.txt", "content": "measured"}),
                FakeCall(
                    "paper",
                    "write_file",
                    {
                        "path": "paper.md",
                        "content": "# Study\n## Methodology\nMeasured one result.\n## Results\nOne result.\n## Conclusions\nThe baseline works.\n",
                    },
                ),
                FakeCall(
                    "submit",
                    "submit_results",
                    {
                        "outcome": "achieved",
                        "artifacts": ["result.txt"],
                    },
                ),
            ]
        message = SimpleNamespace(content=None, tool_calls=calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeOpenAI:
    latest_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs):
        self.__class__.latest_kwargs = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions())


class InterventionCompletions:
    async def create(self, **_request):
        call = FakeCall(
            "manual",
            "manual__request_manual_intervention",
            {"reason": "permission denied", "requested_action": "grant access to the dataset"},
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class InterventionOpenAI:
    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(completions=InterventionCompletions())


class FailingCompletions:
    async def create(self, **_request):
        raise APITimeoutError(request=Request("POST", "http://model/v1/chat/completions"))


class FailingOpenAI:
    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(completions=FailingCompletions())


def test_controller_uses_private_frozen_execution_configuration(config) -> None:
    frozen_tool = ExternalToolConfig.model_validate(
        {
            "id": "example",
            "configuration_id": "example@frozen",
            "transport": {"type": "stdio", "command": "/bin/true"},
            "roles": {"upper": [], "iteration": []},
        }
    )
    live_tool = frozen_tool.model_copy(update={"configuration_id": "example@live"})
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="snapshot", goal="remain stable", tools=["example"]),
        {"enabled_tools": ["example"], "audit_only": "visible"},
        600,
        execution_snapshot={
            "base_prompts": {"common": "frozen common"},
            "tools": [frozen_tool.model_dump(mode="json")],
            "skills": {"upper": [{"name": "method", "text": "frozen skill"}], "iteration": []},
        },
    )
    controller = Controller(config.model_copy(update={"tools": [live_tool]}), store, project["id"])

    assert controller._tool_configs()[0].configuration_id == "example@frozen"
    assert controller._base_prompts()["common"] == "frozen common"
    assert "frozen skill" in controller._skills("upper")
    assert store.get_project(project["id"])["snapshot"] == {
        "enabled_tools": ["example"],
        "audit_only": "visible",
    }
    store.close()


@pytest.mark.asyncio
async def test_supervisor_logs_unexpected_controller_failure(config, caplog) -> None:
    store = Store(config.data_root)
    supervisor = Supervisor(config, store)

    async def fail() -> None:
        raise RuntimeError("unexpected controller failure")

    task = asyncio.create_task(fail())
    await asyncio.gather(task, return_exceptions=True)
    supervisor.tasks["project"] = task
    with caplog.at_level("ERROR", logger="irh.controller"):
        supervisor._finished("project", task)

    assert "Controller task failed for project project" in caplog.text
    assert "unexpected controller failure" in caplog.text
    assert "project" not in supervisor.tasks
    store.close()


@pytest.mark.asyncio
async def test_complete_research_loop_with_hashed_artifact(config, monkeypatch) -> None:
    FakeCompletions.upper_calls = 0
    monkeypatch.setattr(irh.agent, "AsyncOpenAI", FakeOpenAI)
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"),
        {"enabled_tools": [], "skills": {"upper": [], "iteration": []}},
        600,
    )
    await Controller(config, store, project["id"]).run()
    final = store.get_project(project["id"])
    assert final["phase"] == "complete"
    assert store.get_iteration(project["id"], 1)["status"] == "achieved"
    artifact = next(item for item in store.artifacts(project["id"]) if item["path"] == "result.txt")
    assert artifact["path"] == "result.txt"
    assert len(artifact["sha256"]) == 64
    assert FakeOpenAI.latest_kwargs["max_retries"] == 0
    assert FakeOpenAI.latest_kwargs["timeout"].read == 3600
    events = store.events(project["id"], limit=100)
    starts = [event for event in events if event["kind"] == "model_request_started"]
    finishes = [event for event in events if event["kind"] == "model_request_finished"]
    assert len(starts) == len(finishes) == 3
    assert all(event["data"]["ok"] for event in finishes)
    assert sum(event["data"]["input_tokens"] for event in finishes) == 30
    assert sum(event["data"]["output_tokens"] for event in finishes) == 15
    review_session = store.private_dir(project["id"]) / "sessions" / "upper-review-1.json"
    assert review_session.is_file()
    store.close()


@pytest.mark.asyncio
async def test_manual_intervention_tool_pauses_the_controller(config, monkeypatch) -> None:
    monkeypatch.setattr(irh.agent, "AsyncOpenAI", InterventionOpenAI)
    manual = ExternalToolConfig.model_validate(
        {
            "id": "manual",
            "configuration_id": "manual-intervention-tool@0.1.0",
            "transport": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "manual_intervention_tool.server"],
                "env": {"PYTHONPATH": str(Path("/home/omega/llms/manual_intervention_tool/src"))},
            },
            "roles": {"upper": ["request_manual_intervention"], "iteration": []},
            "pause_on_calls": ["request_manual_intervention"],
        }
    )
    configured = config.model_copy(update={"tools": [manual]})
    store = Store(configured.data_root)
    project = store.create_project(
        ProjectCreate(name="blocked", goal="need operator"),
        {"enabled_tools": ["manual"], "skills": {"upper": [], "iteration": []}},
        600,
    )
    await Controller(configured, store, project["id"]).run()
    paused = store.get_project(project["id"])
    assert paused["phase"] == "paused"
    assert paused["resume_phase"] == "planning"
    assert any(event["kind"] == "operator_intervention_required" for event in store.events(project["id"]))
    store.close()


@pytest.mark.asyncio
async def test_model_request_failure_pauses_for_intervention(config, monkeypatch) -> None:
    monkeypatch.setattr(irh.agent, "AsyncOpenAI", FailingOpenAI)
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="model failure", goal="pause safely"),
        {"enabled_tools": [], "skills": {"upper": [], "iteration": []}},
        600,
    )
    await Controller(config, store, project["id"]).run()
    paused = store.get_project(project["id"])
    assert paused["phase"] == "paused"
    assert paused["resume_phase"] == "planning"
    assert paused["pause_reason"] == "model requires operator intervention"
    events = store.events(project["id"])
    intervention = next(event for event in events if event["kind"] == "operator_intervention_required")
    assert intervention["data"]["source"] == "model"
    assert intervention["data"]["detail"]["type"] == "APITimeoutError"
    model_finished = next(event for event in events if event["kind"] == "model_request_finished")
    assert model_finished["data"]["ok"] is False
    assert model_finished["data"]["error_type"] == "APITimeoutError"
    store.close()


@pytest.mark.asyncio
async def test_review_sees_only_publication_view(config, monkeypatch) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="evaluate evidence"),
        {"enabled_tools": [], "skills": {"upper": [], "iteration": []}},
        600,
    )
    definition = IterationDefinition(
        subgoal="short subgoal", hypothesis="hypothesis", introduction="x" * 20_000, parent=None
    )
    store.define_iteration(project["id"], definition, 60, 120)
    store.finish_iteration(project["id"], 1, "timeout", None, "time limit")
    store.record_note(project["id"], "n" * 20_000)
    captured = {}

    async def capture_phase(
        self, role, seconds, workspace, scratch, system_prompt, prompt, session_path, **kwargs
    ):
        captured.update(role=role, prompt=prompt, session_path=session_path, **kwargs)
        return {}

    monkeypatch.setattr(Controller, "_phase", capture_phase)
    await Controller(config, store, project["id"])._upper()
    packet = json.loads(captured["prompt"])
    assert captured["role"] == "upper"
    assert captured["session_path"].name == "upper-review-1.json"
    assert len(captured["prompt"]) < 2_000
    assert packet["latest"]["status"] == "timeout"
    assert captured["mode"] == "paper_review"
    view = captured["project_view"]
    assert view.name == "publications"
    assert (view / packet["latest"]["manifest"]).is_file()
    assert (view / "0001/notice.md").is_file()
    assert not (view / "iterations").exists()
    assert json.loads((view / packet["review_files"]["notes"]).read_text())[0]["text"] == "n" * 20_000
    store.close()
