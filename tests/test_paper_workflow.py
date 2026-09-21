from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from irh.agent import AgentSession
from irh.controller import Controller
from irh.errors import ConflictError
from irh.internal_tools import phase_tools
from irh.sandbox import Sandbox
from irh.store import Store
from irh.tooling import AgentTool
from irh.types import Decision, IterationReport, ProjectCreate, ToolResult

PAPER = "# Study\n## Methodology\nChecked existing evidence.\n## Results\nIncomplete.\n## Conclusions\nMore work is needed.\n"


@pytest.mark.asyncio
async def test_writeup_tools_only_read_existing_files_and_write_paper(config, tmp_path: Path) -> None:
    store = Store(config.data_root)
    project = store.create_project(ProjectCreate(name="study", goal="measure"), {}, 600)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_text("existing evidence")
    sandbox = Sandbox(
        "iteration", workspace, tmp_path / "scratch", store.project_dir(project["id"]), config.sandbox
    )
    submitted: list[IterationReport] = []
    tools = phase_tools(
        "iteration",
        sandbox,
        store,
        project["id"],
        time.monotonic() + 60,
        submit=submitted.append,
        mode="writeup",
    )
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"read_file", "time_remaining", "list_files", "write_paper", "submit_results"}
    root_listing = await by_name["list_files"].handler({})
    assert root_listing.structured["entries"] == ["result.txt"]
    assert root_listing.structured["next_offset"] is None
    assert (await by_name["read_file"].handler({"path": "result.txt"})).text == "existing evidence"
    (workspace / "long.txt").write_text("x" * 12_000)
    excerpt = await by_name["read_file"].handler({"path": "long.txt"})
    assert len(excerpt.text) < 8_100
    assert "offset_chars=8000" in excerpt.text
    tail = await by_name["read_file"].handler({"path": "long.txt", "offset_chars": 8000})
    assert "characters 8000–12000 of 12000" in tail.text
    await by_name["write_paper"].handler({"content": PAPER})
    await by_name["submit_results"].handler({"outcome": "partial", "artifacts": ["result.txt"]})
    assert submitted[0].paper == "paper.md"
    with pytest.raises(ConflictError, match="Results heading"):
        await by_name["write_paper"].handler({"content": "## Methodology\n## Conclusions\n"})
        await by_name["submit_results"].handler({"outcome": "partial"})
    store.close()


@pytest.mark.asyncio
async def test_review_can_only_read_publications(config, tmp_path: Path) -> None:
    store = Store(config.data_root)
    project = store.create_project(ProjectCreate(name="study", goal="measure"), {}, 600)
    root = store.project_dir(project["id"])
    (root / "iterations" / "secret.txt").write_text("raw workspace")
    (root / "publications" / "paper.md").write_text(PAPER)
    sandbox = Sandbox(
        "upper", tmp_path / "scratch", tmp_path / "scratch", root / "publications", config.sandbox
    )
    tools = phase_tools(
        "upper",
        sandbox,
        store,
        project["id"],
        time.monotonic() + 60,
        choose=lambda _decision: None,
        mode="paper_review",
    )
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"read_file", "time_remaining", "record_note", "decide"}
    assert "Methodology" in (await by_name["read_file"].handler({"path": "paper.md"})).text
    with pytest.raises(ConflictError, match="contained relative path"):
        await by_name["read_file"].handler({"path": "../iterations/secret.txt"})
    store.close()


@pytest.mark.asyncio
async def test_timeout_writeup_reuses_session_then_reviews_only_publication(config, monkeypatch) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"),
        {"enabled_tools": [], "skills": {"upper": [], "iteration": []}},
        600,
    )
    seen: list[tuple[str, str, Path]] = []

    async def phase(
        self,
        role,
        seconds,
        workspace,
        scratch,
        system_prompt,
        prompt,
        session_path,
        choose=None,
        submit=None,
        mode="normal",
        project_view=None,
    ):
        seen.append((role, mode, session_path))
        if role == "upper" and mode == "normal":
            choose.value = Decision.model_validate(
                {
                    "action": "iterate",
                    "reason": "small test",
                    "definition": {
                        "subgoal": "measure",
                        "hypothesis": "measurable",
                        "introduction": "test",
                        "parent": None,
                    },
                }
            )
        elif role == "iteration" and mode == "normal":
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps({"messages": [{"role": "user", "content": "prior work"}]}))
            (workspace / "result.txt").write_text("partial result")
            raise TimeoutError
        elif mode == "writeup":
            assert session_path == seen[-2][2]
            assert json.loads(session_path.read_text())["messages"][0]["content"] == "prior work"
            (workspace / "paper.md").write_text(PAPER)
            submit.value = IterationReport(outcome="partial", artifacts=["result.txt"])
        else:
            assert mode == "paper_review"
            assert project_view.name == "publications"
            assert not (project_view / "iterations").exists()
            assert (project_view / "0001/evidence/paper.md").is_file()
            assert (project_view / "0001/evidence/result.txt").is_file()
            choose.value = Decision(action="stop", reason="reviewed partial work", stop_reason="achieved")
        return {}

    monkeypatch.setattr(Controller, "_phase", phase)
    await Controller(config, store, project["id"]).run()
    assert [mode for _, mode, _ in seen] == ["normal", "normal", "writeup", "paper_review"]
    assert store.get_iteration(project["id"], 1)["status"] == "timeout"
    assert store.get_iteration(project["id"], 1)["report"]["outcome"] == "partial"
    assert {item["path"] for item in store.artifacts(project["id"])} == {"paper.md", "result.txt"}
    store.close()


@pytest.mark.asyncio
async def test_writeup_agent_keeps_prior_messages_but_only_new_tools(
    config, tmp_path: Path, monkeypatch
) -> None:
    store = Store(config.data_root)
    project = store.create_project(ProjectCreate(name="study", goal="measure"), {}, 600)
    session_path = tmp_path / "iteration-1.json"
    session_path.write_text(json.dumps({"messages": [{"role": "user", "content": "prior experiment"}]}))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    finished = False
    seen = {}

    async def submit(_arguments):
        nonlocal finished
        finished = True
        return ToolResult(text="done")

    session = AgentSession(
        config,
        store,
        project["id"],
        "iteration",
        workspace,
        session_path,
        "write-up only",
        [AgentTool("submit_results", "Submit", {"type": "object", "properties": {}}, submit)],
        asyncio.Event(),
        lambda: finished,
        report_only=True,
    )

    async def count(_request):
        return 10

    async def complete(**request):
        seen.update(request)
        call = SimpleNamespace(
            id="submit",
            function=SimpleNamespace(name="submit_results", arguments="{}"),
            model_dump=lambda mode="json": {
                "id": "submit",
                "type": "function",
                "function": {"name": "submit_results", "arguments": "{}"},
            },
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]
        )

    monkeypatch.setattr(session, "_count_tokens", count)
    monkeypatch.setattr(session, "_complete", complete)
    await session.run("execution ended; write paper")
    assert [message["content"] for message in seen["messages"] if message["role"] == "user"] == [
        "prior experiment",
        "execution ended; write paper",
    ]
    assert [tool["function"]["name"] for tool in seen["tools"]] == ["submit_results"]
    store.close()
