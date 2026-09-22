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


def test_resume_reuses_the_persisted_system_prompt(config, tmp_path) -> None:
    session, store = make_session(config, tmp_path)
    session.messages.append({"role": "user", "content": "existing history"})
    session.save()

    resumed = AgentSession(
        config,
        store,
        session.project_id,
        "iteration",
        session.workspace,
        session.session_path,
        "replacement prompt",
        [],
        asyncio.Event(),
        lambda: False,
    )

    assert resumed.system_prompt == "system prompt"
    assert resumed._agent_request([])["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "existing history"},
    ]
    store.close()


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
    model = config.model.model_copy(update={"extra_body": {"cache_prompt": False}})
    session, store = make_session(config.model_copy(update={"model": model}), tmp_path)
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
    request["extra_body"]["custom_option"] = True
    assert await session._count_tokens(request) == 1234
    assert seen["path"] == "/v1/chat/completions/input_tokens"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]
    assert seen["body"]["tools"] == [tool]
    assert seen["body"]["cache_prompt"] is True
    assert seen["body"]["custom_option"] is True
    assert "extra_body" not in seen["body"]
    store.close()


@pytest.mark.asyncio
async def test_cache_usage_is_recorded_and_unexpected_reprocessing_warns(config, tmp_path) -> None:
    session, store = make_session(config, tmp_path)
    responses = iter(
        (
            SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=10,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                )
            ),
            SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=130,
                    completion_tokens=5,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=10),
                )
            ),
        )
    )

    class Completions:
        async def create(self, **_request):
            return next(responses)

    session.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    await session._complete("agent", model="test-model", messages=[])
    await session._complete("agent", model="test-model", messages=[])

    events = store.events(session.project_id)
    finishes = [event for event in events if event["kind"] == "model_request_finished"]
    assert finishes[0]["data"]["cache_expected"] is False
    assert finishes[0]["data"]["cache_reset_reason"] == "new_session"
    assert finishes[1]["data"]["cache_expected"] is True
    assert finishes[1]["data"]["cached_input_tokens"] == 10
    assert finishes[1]["data"]["evaluated_input_tokens"] == 120
    warning = next(event for event in events if event["kind"] == "model_prompt_cache_warning")
    assert warning["data"]["reason"] == "cached_prefix_smaller_than_expected"
    assert warning["data"]["expected_prefix_tokens"] == 100
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


def test_compaction_never_starts_with_an_orphaned_tool_result(config, tmp_path) -> None:
    session, store = make_session(config, tmp_path)
    session.messages = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "old result"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "recent"}, {"id": "recent-2"}],
        },
        {"role": "tool", "tool_call_id": "recent", "content": "first result"},
        {"role": "tool", "tool_call_id": "recent-2", "content": "second result"},
        {"role": "assistant", "content": "continuing"},
    ]

    cut = session._compaction_cut()
    assert session.messages[cut]["role"] == "assistant"
    assert session.messages[cut]["tool_calls"][0]["id"] == "recent"
    assert session.messages[cut - 1]["role"] == "tool"
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
    session.tools = {"finish": AgentTool("finish", "Finish", {"type": "object", "properties": {}}, finish)}

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
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]
        )

    monkeypatch.setattr(session, "_count_tokens", count)
    monkeypatch.setattr(session, "_complete", complete)
    await session.run("same definition")
    assert sum(m.get("content") == "same definition" for m in seen["messages"]) == 1
    store.close()


def test_resume_closes_interrupted_tool_calls_without_guessing_outcome(config, tmp_path) -> None:
    session, store = make_session(config, tmp_path)
    session.messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "done", "type": "function", "function": {"name": "first", "arguments": "{}"}},
                {"id": "unknown", "type": "function", "function": {"name": "second", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "done", "content": "completed"},
    ]
    session._close_interrupted_tool_calls()
    assert [item["tool_call_id"] for item in session.messages if item["role"] == "tool"] == [
        "done",
        "unknown",
    ]
    recovered = json.loads(session.messages[-1]["content"])
    assert recovered["is_error"] is True
    assert "outcome and side effects are unknown" in recovered["error"]
    session._close_interrupted_tool_calls()
    assert len(session.messages) == 3
    store.close()
