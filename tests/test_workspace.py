from __future__ import annotations

from pathlib import Path

import pytest

from irh.errors import ConflictError
from irh.store import Store
from irh.types import InputSelection, IterationReport, ProjectCreate
from irh.workspace import WorkspaceManager


def test_resources_stage_archive_and_hash(config, tmp_path: Path) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"), {"enabled_tools": [], "skills": {}}, 600
    )
    source = tmp_path / "source.txt"
    source.write_text("input")
    workspaces = WorkspaceManager(store)
    workspaces.add_resource(project["id"], source, "dataset.txt")
    workspace = workspaces.stage(
        project["id"], 1, [InputSelection(source="resources/dataset.txt", destination="input.txt")]
    )
    assert (workspace / "input.txt").read_text() == "input"
    (workspace / "result.txt").write_text("evidence")
    archive = workspaces.archive(project["id"], 1)
    artifacts = workspaces.register_report_artifacts(
        project["id"],
        1,
        archive,
        IterationReport(outcome="achieved", results="done", conclusions="done", artifacts=["result.txt"]),
    )
    assert artifacts[0]["sha256"] == "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
    store.close()


def test_traversal_is_rejected(config, tmp_path: Path) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"), {"enabled_tools": [], "skills": {}}, 600
    )
    with pytest.raises(ConflictError):
        WorkspaceManager(store).stage(
            project["id"], 1, [InputSelection(source="../outside", destination="input")]
        )
    store.close()
