from __future__ import annotations

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
from irh.controller import Controller
from irh.store import Store
from irh.types import ProjectCreate


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
                    "submit",
                    "submit_results",
                    {
                        "outcome": "achieved",
                        "results": "recorded one result",
                        "conclusions": "the baseline works",
                        "metrics": {"count": 1},
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
    artifact = store.artifacts(project["id"])[0]
    assert artifact["path"] == "result.txt"
    assert len(artifact["sha256"]) == 64
    assert FakeOpenAI.latest_kwargs["max_retries"] == 0
    assert FakeOpenAI.latest_kwargs["timeout"].read == 3600
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
    store.close()
