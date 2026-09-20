from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import irh.agent
from irh.agent import AgentSession
from irh.store import Store
from irh.tooling import AgentTool
from irh.types import ProjectCreate, ToolResult


def make_session(config, tmp_path: Path, *, role: str = "iteration") -> tuple[AgentSession, Store]:
    store = Store(config.data_root)
    project = store.create_project(ProjectCreate(name="study", goal="test"), {}, 600)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession(
        config,
        store,
        project["id"],
        role,
        workspace,
        tmp_path / "session.json",
        "system prompt",
        [],
        asyncio.Event(),
        lambda: False,
    )
    return session, store


@pytest.mark.asyncio
async def test_parent_cancellation_reaps_in_flight_model_request() -> None:
    session = object.__new__(AgentSession)
    session.abort = asyncio.Event()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def request() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    parent = asyncio.create_task(session._await_or_abort(request()))
    await started.wait()
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert finished.is_set()


@pytest.mark.asyncio
async def test_model_token_counter_receives_complete_chat_request(config, tmp_path, monkeypatch) -> None:
    session, store = make_session(config, tmp_path)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"input_tokens": 1234})

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(irh.agent, "AsyncClient", client_factory)
    session.messages.append({"role": "user", "content": "hello"})
    tool = {"type": "function", "function": {"name": "example", "parameters": {"type": "object"}}}
    request = session._agent_request([tool])
    request["extra_body"] = {"custom_option": True}
    assert await session._count_tokens(request) == 1234
    assert seen["path"] == "/v1/chat/completions/input_tokens"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]
    assert seen["body"]["tools"] == [tool]
    assert seen["body"]["custom_option"] is True
    assert "extra_body" not in seen["body"]
    store.close()


def test_tool_result_is_single_copy_and_large_result_is_inspectable(config, tmp_path) -> None:
    session, store = make_session(config, tmp_path)
    structured = {"state": "running", "log_tail": "short"}
    result = ToolResult(text=json.dumps(structured), structured=structured)
    assert json.loads(session._tool_payload(result)) == structured

    large = "output " * 4000
    preview = json.loads(session._tool_payload(ToolResult(text=large)))
    assert preview["truncated"] is True
    assert preview["characters"] == len(large)
    assert len(preview["preview"]) < len(large)
    assert preview["shell_path"].startswith("/workspace/tool-results/")
    assert (session.workspace / preview["full_result_path"]).read_text() == large
    store.close()


@pytest.mark.asyncio
async def test_compaction_uses_existing_96k_threshold_with_exact_count(config, tmp_path, monkeypatch) -> None:
    model = config.model.model_copy(update={"context_window": 131_072, "max_output_tokens": 16_384})
    configured = config.model_copy(update={"model": model})
    session, store = make_session(configured, tmp_path)
    session.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "recent"}]
    counts = iter((98_303, 98_304, 1000, 1000))
    requests = []

    async def count(request):
        requests.append(request)
        return next(counts)

    async def complete(**_request):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])

    monkeypatch.setattr(session, "_count_tokens", count)
    monkeypatch.setattr(session, "_complete", complete)
    await session.compact_if_needed([])
    assert session.usage["compactions"] == 0
    await session.compact_if_needed([])
    assert session.usage["compactions"] == 1
    assert session.messages[0]["content"] == "Prior-session summary:\nsummary"
    assert requests[0]["messages"][0]["content"] == "system prompt"
    store.close()


@pytest.mark.asyncio
async def test_iteration_resume_does_not_append_unchanged_definition(config, tmp_path, monkeypatch) -> None:
    session, store = make_session(config, tmp_path)
    session.messages = [{"role": "user", "content": "same definition"}]
    done = False
    seen = {}

    async def finish(_arguments):
        nonlocal done
        done = True
        return ToolResult(text="done")

    session.terminal = lambda: done
    session.tools = {
        "finish": AgentTool("finish", "Finish", {"type": "object", "properties": {}}, finish)
    }

    async def count(_request):
        return 10

    async def complete(**request):
        seen["messages"] = request["messages"]
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="finish", arguments="{}"),
            model_dump=lambda mode="json": {
                "id": "call-1",
                "type": "function",
                "function": {"name": "finish", "arguments": "{}"},
            },
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])

    monkeypatch.setattr(session, "_count_tokens", count)
    monkeypatch.setattr(session, "_complete", complete)
    await session.run("same definition")
    assert sum(m.get("content") == "same definition" for m in seen["messages"]) == 1
    store.close()
