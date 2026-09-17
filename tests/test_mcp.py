from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from irh.config import ExternalToolConfig
from irh.errors import InterventionRequired
from irh.store import Store
from irh.tooling import MCPHub
from irh.types import ProjectCreate


@pytest.mark.asyncio
async def test_manual_intervention_is_a_direct_tool_and_pauses(config) -> None:
    root = Path("/home/omega/llms/manual_intervention_tool/src")
    tool = ExternalToolConfig.model_validate(
        {
            "id": "manual",
            "configuration_id": "manual-intervention-tool@0.1.0",
            "transport": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "manual_intervention_tool.server"],
                "env": {"PYTHONPATH": str(root)},
            },
            "roles": {"upper": ["request_manual_intervention"], "iteration": ["request_manual_intervention"]},
            "pause_on_calls": ["request_manual_intervention"],
        }
    )
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"), {"enabled_tools": [], "skills": {}}, 600
    )
    abort = asyncio.Event()
    async with MCPHub([tool], "iteration", {"project_id": project["id"]}, store, project["id"], abort) as hub:
        exposed = {item.name: item for item in hub.model_tools}
        assert set(exposed) == {"manual__request_manual_intervention"}
        with pytest.raises(InterventionRequired) as raised:
            await exposed["manual__request_manual_intervention"].handler(
                {"reason": "permission denied", "requested_action": "grant read access"}
            )
        assert raised.value.detail["requested"] is True
    store.close()
