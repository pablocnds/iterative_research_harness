from __future__ import annotations

import asyncio
import io
import json
import zipfile
from types import SimpleNamespace

import httpx
import pytest

import irh.agent
from irh.agent import AgentSession
from irh.api import create_app
from irh.controller import Controller, ExecutionManager
from irh.errors import ConflictError, CooperativePause
from irh.publications import PublicationManager
from irh.store import Store
from irh.tooling import AgentTool
from irh.types import (
    Decision,
    IterationDefinition,
    IterationReport,
    ProjectCreate,
    PublicationCreate,
    SteeringMessage,
    ToolResult,
)
from irh.workspace import WorkspaceManager

PAPER = "# Study\n## Methodology\nMeasured baseline.\n## Results\nPartial evidence.\n## Conclusions\nMore work needed.\n"


def project_store(config):
    store = Store(config.data_root)
    project = store.create_project(ProjectCreate(name="study", goal="measure"), {}, 600)
    return store, project["id"]


def session_for(config, store, project_id, tmp_path, tools=(), pause=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentSession(
        config,
        store,
        project_id,
        "iteration",
        workspace,
        tmp_path / "session.json",
        "immutable",
        list(tools),
        asyncio.Event(),
        lambda: False,
        pause_requested=pause,
    )


def finalized_source(config, store, project_id):
    definition = IterationDefinition(
        subgoal="measure", hypothesis="test", introduction="context", parent=None
    )
    number = store.define_iteration(project_id, definition, 60, 120)
    workspace = WorkspaceManager(store).stage(project_id, number, [])
    store.start_attempt(project_id, number)
    (workspace / "paper.md").write_text(PAPER)
    (workspace / "data.txt").write_text("evidence")
    Controller(config, store, project_id)._finish_with_paper(
        number, IterationReport(outcome="partial", artifacts=["data.txt"]), "partial", None
    )
    return number


def test_steering_receipt_survives_crash_before_ack(config, tmp_path, monkeypatch):
    store, project_id = project_store(config)
    queued = store.enqueue_message(
        project_id, SteeringMessage(text="Change baseline", actor="supervisor", idempotency_key="request-1")
    )
    assert (
        store.enqueue_message(
            project_id,
            SteeringMessage(text="Change baseline", actor="supervisor", idempotency_key="request-1"),
        )["id"]
        == queued["id"]
    )
    with pytest.raises(ConflictError):
        store.enqueue_message(project_id, SteeringMessage(text="different", idempotency_key="request-1"))
    session = session_for(config, store, project_id, tmp_path)
    original = store.acknowledge_message

    def crash(*_args):
        raise RuntimeError("crash before SQLite acknowledgement")

    monkeypatch.setattr(store, "acknowledge_message", crash)
    with pytest.raises(RuntimeError):
        session.deliver_messages()
    monkeypatch.setattr(store, "acknowledge_message", original)
    resumed = session_for(config, store, project_id, tmp_path)
    resumed.deliver_messages()
    assert len(resumed.messages) == 1
    assert resumed.system_prompt == "immutable"
    assert store.messages(project_id)[0]["delivered_at"]
    store.close()


@pytest.mark.asyncio
async def test_pause_finishes_current_tool_and_skips_remaining_batch(config, tmp_path, monkeypatch):
    store, project_id = project_store(config)
    pause = asyncio.Event()
    started, finish = asyncio.Event(), asyncio.Event()
    invoked = []

    async def slow(_args):
        started.set()
        await finish.wait()
        invoked.append("slow")
        return ToolResult(text="saved operation result")

    async def later(_args):
        invoked.append("later")
        return ToolResult(text="should not execute")

    tools = [
        AgentTool("slow", "slow", {"type": "object"}, slow),
        AgentTool("later", "later", {"type": "object"}, later),
    ]
    session = session_for(config, store, project_id, tmp_path, tools, pause)

    async def count(_request):
        return 10

    async def complete(**_request):
        calls = []
        for name in ("slow", "later"):
            payload = {"id": name, "type": "function", "function": {"name": name, "arguments": "{}"}}
            calls.append(
                SimpleNamespace(
                    id=name,
                    function=SimpleNamespace(name=name, arguments="{}"),
                    model_dump=lambda mode="json", payload=payload: payload,
                )
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=calls))]
        )

    monkeypatch.setattr(session, "_count_tokens", count)
    monkeypatch.setattr(session, "_complete", complete)
    task = asyncio.create_task(session.run("work"))
    await asyncio.wait_for(started.wait(), 2)
    pause.set()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(CooperativePause):
        await asyncio.wait_for(task, 2)
    saved = json.loads(session.session_path.read_text())["messages"]
    assert saved[-2]["content"] == "saved operation result"
    assert "Not executed" in saved[-1]["content"]
    assert invoked == ["slow"]
    store.close()


@pytest.mark.asyncio
async def test_one_iteration_includes_review_and_defers_revision(config, monkeypatch):
    store, project_id = project_store(config)
    phases = []

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
        **_kwargs,
    ):
        phases.append(role)
        number = store.get_project(project_id)["current_iteration"]
        if role == "upper":
            choose.value = Decision(
                action="revise" if number else "iterate",
                reason="More evidence",
                definition=IterationDefinition(
                    subgoal="measure", hypothesis="test", introduction="context", parent=number or None
                ),
            )
        else:
            (workspace / "paper.md").write_text(PAPER)
            submit.value = IterationReport(outcome="partial")
        return {}

    monkeypatch.setattr(Controller, "_phase", phase)
    await Controller(config, store, project_id).run(max_iterations=1)
    assert phases == ["upper", "iteration", "upper"]
    assert store.get_project(project_id)["phase"] == "ready"
    assert store.get_project(project_id)["current_iteration"] == 1
    assert store.get_iteration(project_id, 1)["review"]["action"] == "revise"
    run = store.latest_run(project_id)
    assert run["completion_reason"] == "iteration_limit_reached"
    assert run["iterations_completed"] == 1
    assert not (store.runtime_dir(project_id) / "iteration/workspace").exists()
    # A new bounded execution consumes the saved proposal and runs its revision.
    await Controller(config, store, project_id).run(max_iterations=1)
    assert store.get_project(project_id)["current_iteration"] == 2
    assert store.latest_run(project_id)["id"] != run["id"]
    assert store.latest_run(project_id)["iterations_completed"] == 1
    store.close()


@pytest.mark.asyncio
async def test_pause_idempotence_provenance_and_run_resume(config):
    store, project_id = project_store(config)
    run = store.start_run(project_id, False, 1, "supervisor")
    request = store.request_pause(project_id, "supervisor", "Inspect evidence")
    store.pause(project_id, "Inspect evidence")
    manager = ExecutionManager(config, store)
    repeated = await manager.pause(project_id, "supervisor", "Repeated request")
    assert repeated["pause_request"]["id"] == request["id"]
    assert repeated["pause_request"]["actor"] == "supervisor"
    assert store.get_project(project_id)["resume_phase"] == "planning"
    with pytest.raises(ConflictError, match="preserves"):
        store.start_run(project_id, True, 2, "operator")
    assert store.start_run(project_id, True, None, "operator")["id"] == run["id"]
    store.resume(project_id)
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_paused", [False, True])
async def test_operator_hold_supersedes_automatic_pause_and_blocks_agent_resume(config, already_paused):
    store, project_id = project_store(config)
    run = store.start_run(project_id, False, 1, "supervisor")
    automatic = store.request_pause(project_id, "supervisor", "Routine inspection")
    if already_paused:
        store.pause(project_id, "Routine inspection")
    manager = ExecutionManager(config, store)
    hold = await manager.pause(project_id, "operator", "Wait for my instructions")
    assert hold["pause_request"]["id"] != automatic["id"]
    assert hold["pause_request"]["actor"] == "operator"
    assert hold["pause_request"]["reason"] == "Wait for my instructions"
    assert hold["state"] == "paused"
    for actor in ("operator", "supervisor", "harness", "daemon"):
        repeated = await manager.pause(project_id, actor, "Another pause")
        assert repeated["pause_request"]["id"] == hold["pause_request"]["id"]

    reopened = Store(config.data_root)
    store.close()
    manager = ExecutionManager(config, reopened)
    with pytest.raises(ConflictError, match="held by the operator"):
        await manager.start(project_id, resume=True, actor="supervisor")
    assert not manager.tasks
    assert reopened.latest_run(project_id)["state"] == "paused"
    assert reopened.get_project(project_id)["phase"] == "paused"
    assert reopened.start_run(project_id, True, None, "operator")["id"] == run["id"]
    reopened.resume(project_id)
    # After operator resume, a fresh automatic pause is no longer a user hold.
    fresh = reopened.request_pause(project_id, "supervisor", "New routine check")
    assert fresh["actor"] == "supervisor"
    reopened.pause(project_id, "New routine check")
    assert reopened.start_run(project_id, True, None, "supervisor")["id"] == run["id"]
    reopened.close()


@pytest.mark.asyncio
async def test_operator_hold_without_previous_run_rejects_resume_via_api(config):
    store, project_id = project_store(config)
    manager = ExecutionManager(config, store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(config, store, manager)), base_url="http://test"
    ) as client:
        response = await client.post(f"/v1/projects/{project_id}/pause", json={"actor": "operator"})
        assert response.status_code == 202
        blocked = await client.post(
            f"/v1/projects/{project_id}/run", json={"resume": True, "actor": "supervisor"}
        )
        assert blocked.status_code == 409
        assert store.latest_run(project_id) is None
        assert not manager.tasks
    store.close()


def test_reminders_coalesce_and_preserve_run_clock(config, tmp_path):
    store, project_id = project_store(config)
    store.start_run(project_id, False, 1, "supervisor")
    store.tick_run(project_id, 1800, 1800)
    store.tick_run(project_id, 1800, 1800)
    messages = store.messages(project_id, "iteration")
    assert len(messages) == 1
    assert "60.0 min" in messages[0]["text"]
    session_for(config, store, project_id, tmp_path).deliver_messages()
    assert store.messages(project_id)[0]["delivered_at"]
    store.pause(project_id, "inspect")
    store.start_run(project_id, True, None, "supervisor")
    store.resume(project_id)
    store.tick_run(project_id, 1799, 1800)
    assert not store.messages(project_id, "iteration")
    store.tick_run(project_id, 1, 1800)
    assert len(store.messages(project_id, "iteration")) == 1
    store.close()


@pytest.mark.asyncio
async def test_publication_api_exports_self_contained_paper_and_provenance(config):
    store, project_id = project_store(config)
    finalized_source(config, store, project_id)
    manager = ExecutionManager(config, store)
    transport = httpx.ASGITransport(app=create_app(config, store, manager))
    async with httpx.AsyncClient(transport=transport, base_url="http://irh") as client:
        response = await client.post(
            f"/v1/projects/{project_id}/publications",
            json={
                "title": "Combined findings",
                "paper": PAPER,
                "sources": [{"project_id": project_id, "iteration": 1}],
                "supplements": {"table.csv": "method,result\nbaseline,partial\n"},
            },
        )
        assert response.status_code == 201, response.text
        publication = response.json()
        assert publication["sources"][0]["artifacts"][0]["sha256"]
        publication_id = publication["id"]
        exported = await client.get(f"/v1/projects/{project_id}/publications/{publication_id}/export")
        assert exported.status_code == 200
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            assert set(archive.namelist()) == {"paper.md", "manifest.json", "supplements/table.csv"}
            assert archive.read("paper.md").decode() == PAPER
        status = (await client.get(f"/v1/projects/{project_id}")).json()
        assert status["control"]["state"] == "idle"
        assert any(a["owner_kind"] == "publication" and a["iteration"] is None for a in status["artifacts"])
        queued = await client.post(
            f"/v1/projects/{project_id}/messages", json={"text": "Prioritize baseline"}
        )
        assert queued.status_code == 201
        pause = await client.post(f"/v1/projects/{project_id}/pause", json={"actor": "supervisor"})
        assert pause.json()["state"] == "paused"
        invalid = await client.post(f"/v1/projects/{project_id}/run", json={"max_iterations": True})
        assert invalid.status_code == 400
    # Immutable publication artifacts survive a database reopen.
    store.close()
    reopened = Store(config.data_root)
    assert PublicationManager(reopened).get(project_id, publication_id) == publication
    assert len(reopened.artifacts(project_id)) == 4
    reopened.close()


def test_publication_rejects_escape_and_accepts_completed_projects(config):
    store, project_id = project_store(config)
    finalized_source(config, store, project_id)
    store.apply_decision(project_id, Decision(action="stop", reason="done", stop_reason="achieved"), 60, 120)
    manager = PublicationManager(store)
    value = PublicationCreate(
        title="Synthesis", paper=PAPER, sources=[{"project_id": project_id, "iteration": 1}]
    )
    with pytest.raises(ConflictError):
        manager.create(project_id, value.model_copy(update={"supplements": {"../escape": "bad"}}))
    manifest = manager.create(project_id, value)
    assert manifest["sources"][0]["iteration"] == 1
    with pytest.raises(ConflictError, match="complete"):
        store.enqueue_message(project_id, SteeringMessage(text="too late"))
    store.close()


@pytest.mark.asyncio
async def test_pause_during_model_request_waits_and_saves_response(config, monkeypatch):
    store, project_id = project_store(config)
    started, finish = asyncio.Event(), asyncio.Event()

    async def complete(**_request):
        started.set()
        await finish.wait()
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Completed reasoning", tool_calls=None))
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    def client(**_kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))

    async def count(_self, _request):
        return 10

    monkeypatch.setattr(irh.agent, "AsyncOpenAI", client)
    monkeypatch.setattr(AgentSession, "_count_tokens", count)
    manager = ExecutionManager(config, store)
    await manager.start(project_id, max_iterations=1)
    task = manager.tasks[project_id]
    await asyncio.wait_for(started.wait(), 2)
    requested = await manager.pause(project_id, "supervisor", "Inspect")
    assert requested["state"] == "waiting_to_pause"
    assert requested["current_operation"]["kind"] == "model_request_started"
    assert not task.done()
    finish.set()
    await asyncio.wait_for(task, 2)
    status = store.control_status(project_id)
    assert status["state"] == "paused"
    assert status["pause_request"]["acknowledged_at"]
    saved = json.loads((store.private_dir(project_id) / "sessions/upper.json").read_text())
    assert any(m.get("content") == "Completed reasoning" for m in saved["messages"])
    assert not any(e["kind"] == "interrupted_tool_calls_closed" for e in store.events(project_id))
    store.close()


@pytest.mark.asyncio
async def test_pause_racing_with_submission_finalizes_before_ack(config, monkeypatch):
    store, project_id = project_store(config)
    controller = Controller(config, store, project_id)

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
        **_kwargs,
    ):
        if choose is not None:
            choose.value = Decision(
                action="iterate",
                reason="baseline",
                definition=IterationDefinition(
                    subgoal="measure", hypothesis="test", introduction="context", parent=None
                ),
            )
        else:
            (workspace / "paper.md").write_text(PAPER)
            submit.value = IterationReport(outcome="partial")
            store.save_phase_result(project_id, session_path.stem, "submission", submit.value)
            store.request_pause(project_id, "supervisor", "Inspect submission")
            controller.stop()
        return {}

    monkeypatch.setattr(Controller, "_phase", phase)
    await controller.run(max_iterations=1)
    assert store.get_project(project_id)["resume_phase"] == "review"
    assert store.get_iteration(project_id, 1)["status"] == "partial"
    assert store.artifacts(project_id)[0]["path"] == "paper.md"
    assert store.phase_result(project_id, "iteration-1") is None
    store.close()


@pytest.mark.asyncio
async def test_accepted_submission_recovers_without_new_model_call(config, monkeypatch):
    store, project_id = project_store(config)
    definition = IterationDefinition(
        subgoal="measure", hypothesis="test", introduction="context", parent=None
    )
    store.define_iteration(project_id, definition, 60, 120)
    workspace = WorkspaceManager(store).stage(project_id, 1, [])
    store.start_attempt(project_id, 1)
    store.start_run(project_id, False, 1, "supervisor")
    store.event(project_id, "controller_started", {})
    (workspace / "paper.md").write_text(PAPER)
    store.save_phase_result(project_id, "iteration-1", "submission", IterationReport(outcome="partial"))
    # Even an exhausted budget cannot discard an already accepted paper.
    store.charge(project_id, "iteration", 60)
    store.close()
    store = Store(config.data_root)
    assert store.recover_after_restart() == [project_id]
    assert store.get_project(project_id)["resume_phase"] == "executing"
    store.start_run(project_id, True, None, "supervisor")
    store.resume(project_id)

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Accepted submissions must not invoke the agent again")

    monkeypatch.setattr(Controller, "_phase", forbidden)
    await Controller(config, store, project_id)._iterate()
    assert store.get_iteration(project_id, 1)["status"] == "partial"
    assert store.get_project(project_id)["phase"] == "review"
    store.close()


def test_artifact_migration_retains_legacy_registry(config):
    store, project_id = project_store(config)
    with store.transaction() as db:
        db.execute("DROP TABLE artifacts")
        db.execute("""CREATE TABLE artifacts (project_id TEXT NOT NULL, iteration INTEGER NOT NULL,
            path TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(project_id,iteration,path))""")
        db.execute("INSERT INTO artifacts VALUES(?,1,'paper.md',?,12,'legacy')", (project_id, "a" * 64))
    store.close()
    migrated = Store(config.data_root)
    assert migrated.artifacts(project_id) == [
        {
            "iteration": 1,
            "owner_kind": "iteration",
            "owner_id": "1",
            "path": "paper.md",
            "sha256": "a" * 64,
            "bytes": 12,
            "created_at": "legacy",
        }
    ]
    migrated.close()


def test_reminders_can_be_disabled_and_roles_remain_pending(config, tmp_path):
    store, project_id = project_store(config)
    store.start_run(project_id, False, None, "supervisor")
    store.tick_run(project_id, 7200, 0)
    assert not store.messages(project_id)
    queued = store.enqueue_message(project_id, SteeringMessage(text="Review carefully", target="upper"))
    session_for(config, store, project_id, tmp_path).deliver_messages()
    assert not store.messages(project_id)[0]["delivered_at"]
    assert store.messages(project_id, "upper")[0]["id"] == queued["id"]
    store.close()


def test_reminder_cadence_change_and_artifact_immutability(config):
    store, project_id = project_store(config)
    store.start_run(project_id, False, 1, "supervisor")
    store.tick_run(project_id, 1800, 1800)
    store.tick_run(project_id, 900, 900)
    assert "45.0 min" in store.messages(project_id)[0]["text"]
    store.register_artifact(project_id, 1, "result.txt", "a" * 64, 10)
    store.register_artifact(project_id, 1, "result.txt", "a" * 64, 10)
    with pytest.raises(ConflictError, match="immutable"):
        store.register_artifact(project_id, 1, "result.txt", "b" * 64, 10)
    assert store.artifacts(project_id)[0]["sha256"] == "a" * 64
    store.close()


def test_recovery_handles_run_created_before_controller_start(config):
    store, project_id = project_store(config)
    run = store.start_run(project_id, False, 1, "supervisor")
    store.close()
    restarted = Store(config.data_root)
    assert restarted.recover_after_restart() == [project_id]
    assert restarted.latest_run(project_id)["id"] == run["id"]
    assert restarted.latest_run(project_id)["state"] == "paused"
    assert restarted.get_project(project_id)["resume_phase"] == "planning"
    restarted.close()
