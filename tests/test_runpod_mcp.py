from __future__ import annotations

from typing import Any

import pytest
from mcp import Client

from irh.runpod_mcp import RunpodAPI, build_server


class FakeAPI:
    session_id = "bound-session"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def call(self, method: str, path: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, path, value))
        return {"job_id": "job-1", "state": "queued"}

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
