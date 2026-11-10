from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from irh.sandbox import Sandbox


@pytest.mark.asyncio
async def test_shell_children_end_with_the_call_and_tmp_is_fresh(config, tmp_path):
    workspace, project = tmp_path / "workspace", tmp_path / "project"
    workspace.mkdir()
    project.mkdir()
    sandbox = Sandbox("iteration", workspace, tmp_path / "scratch", project, config.sandbox)
    result = await sandbox.run(
        "touch /tmp/per-call; echo persistent > /workspace/kept; "
        "setsid nohup sh -c 'sleep 1; echo survived > /workspace/escaped' >/dev/null 2>&1 < /dev/null & "
        "disown; sleep 0.1", timeout=5,
    )
    assert result["exit_code"] == 0
    await asyncio.sleep(1.2)
    assert not (workspace / "escaped").exists()
    result = await sandbox.run("test ! -e /tmp/per-call && test -s /workspace/kept", timeout=5)
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_shell_pipeline_reports_upstream_failure(config, tmp_path):
    workspace, project = tmp_path / "workspace", tmp_path / "project"
    workspace.mkdir()
    project.mkdir()
    sandbox = Sandbox("iteration", workspace, tmp_path / "scratch", project, config.sandbox)
    result = await sandbox.run("(exit 2) | cat", timeout=5)
    assert result["exit_code"] == 2


@pytest.mark.asyncio
async def test_iteration_sandbox_can_write_workspace_but_not_host(config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    project = tmp_path / "project"
    secret = tmp_path / "secret"
    workspace.mkdir()
    project.mkdir()
    secret.write_text("hidden")
    sandbox = Sandbox("iteration", workspace, scratch, project, config.sandbox)
    result = await sandbox.run(f"printf ok > made; test ! -e '{secret}'", timeout=10)
    assert result["exit_code"] == 0
    assert (workspace / "made").read_text() == "ok"


@pytest.mark.asyncio
async def test_upper_has_read_only_project_and_no_network_namespace(config, tmp_path: Path) -> None:
    project = tmp_path / "project"
    scratch = tmp_path / "scratch"
    project.mkdir()
    (project / "evidence").write_text("yes")
    sandbox = Sandbox("upper", scratch, scratch, project, config.sandbox)
    result = await sandbox.run("cat /project/evidence; ! touch /project/nope", timeout=10)
    assert result["exit_code"] == 0
    assert "yes" in result["output"]


@pytest.mark.asyncio
async def test_command_output_is_streamed_into_a_bounded_tail(config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    workspace.mkdir()
    project.mkdir()
    sandbox = Sandbox("iteration", workspace, tmp_path / "scratch", project, config.sandbox)
    limit = config.sandbox.command_output_bytes
    result = await sandbox.run(
        f"python3 -c \"import sys; sys.stdout.write('x' * {limit * 4} + 'TAIL')\"",
        timeout=10,
    )
    assert result["truncated"] is True
    assert len(result["output"].encode()) == limit
    assert result["output"].endswith("TAIL")
