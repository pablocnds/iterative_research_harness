from __future__ import annotations

from typing import Any

import pytest
from mcp import Client

from irh.runpod_mcp import RunpodAPI, build_server


class FakeAPI:
    session_id = "bound-session"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.state = "queued"

    def call(self, method: str, path: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, path, value))
        return {"job_id": "job-1", "state": self.state}

    def prepare_inputs(self, inputs: list[str]) -> list[str]:
        return inputs

    def publish_outputs(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload


@pytest.mark.asyncio
async def test_runpod_facade_binds_session_and_exposes_native_operations() -> None:
    api = FakeAPI()
    async with Client(build_server(api)) as client:  # type: ignore[arg-type]
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert names == {"run", "status", "wait", "cancel", "replace", "host_status"}
        result = await client.call_tool("run", {"command": "python train.py", "inputs": ["train.py"]})
        assert result.is_error is False
    method, path, value = api.calls[-1]
    assert (method, path) == ("POST", "/runs")
    assert value is not None and value["sessionId"] == "bound-session"


@pytest.mark.asyncio
async def test_runpod_wait_marks_only_unmetered_infrastructure_waits() -> None:
    api = FakeAPI()
    messages: list[str | None] = []

    async def progress(_progress: float, _total: float | None, message: str | None) -> None:
        messages.append(message)

    async with Client(build_server(api)) as client:  # type: ignore[arg-type]
        await client.call_tool(
            "wait",
            {"job_id": "job-1", "timeout_seconds": 0},
            progress_callback=progress,
        )
        assert messages == ["irh:budget-pause", "irh:budget-resume"]
        messages.clear()
        api.state = "running"
        await client.call_tool(
            "wait",
            {"job_id": "job-1", "timeout_seconds": 0},
            progress_callback=progress,
        )
        assert messages == []


def test_runpod_facade_freezes_inputs_and_publishes_collected_outputs(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    workspace.mkdir()
    input_root.mkdir()
    (workspace / "train.py").write_text("print('train')")
    api = RunpodAPI(
        str(tmp_path / "unused.sock"),
        "project",
        "session",
        input_root,
        output_root,
        workspace,
        "iteration",
    )
    assert api.prepare_inputs(["train.py"]) == ["train.py"]
    (workspace / "train.py").write_text("changed after submission")
    assert (input_root / "train.py").read_text() == "print('train')"
    collected = output_root / "session" / "outputs" / "metrics.json"
    collected.parent.mkdir(parents=True)
    collected.write_text('{"score":1}')
    result = api.publish_outputs({"state": "succeeded"})
    assert result["workspace_outputs"] == ["gpu-outputs/outputs/metrics.json"]
    assert (workspace / "gpu-outputs" / "outputs" / "metrics.json").read_text() == '{"score":1}'


def test_runpod_facade_rejects_symlinked_workspace_ancestors(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    outside = tmp_path / "outside"
    for path in (workspace, input_root, output_root, outside):
        path.mkdir()
    (outside / "secret").write_text("private")
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    api = RunpodAPI(
        str(tmp_path / "unused.sock"),
        "project",
        "session",
        input_root,
        output_root,
        workspace,
        "iteration",
    )

    with pytest.raises(ValueError, match="escapes its root|contains a symlink"):
        api.prepare_inputs(["link/secret"])

    collected = output_root / "session" / "sub" / "result.txt"
    collected.parent.mkdir(parents=True)
    collected.write_text("remote result")
    (workspace / "gpu-outputs").mkdir()
    (workspace / "gpu-outputs" / "sub").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes its root|contains a symlink"):
        api.publish_outputs({"state": "succeeded"})
    assert not (outside / "result.txt").exists()
