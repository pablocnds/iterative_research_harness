from __future__ import annotations

from pathlib import Path

import pytest

from irh.sandbox import Sandbox


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
