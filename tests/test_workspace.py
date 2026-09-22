from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from irh.errors import ConflictError
from irh.store import Store
from irh.types import InputSelection, IterationReport, ProjectCreate
from irh.workspace import MAX_PAPER_BYTES, WorkspaceManager, report_files


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
    (workspace / "unpublished.txt").write_text("private")
    (workspace / "paper.md").write_text(
        "# Study\n## Methodology\nMeasured.\n## Results\nOne result.\n## Conclusions\nDone.\n"
    )
    report = IterationReport(outcome="achieved", artifacts=["result.txt"])
    publication = workspaces.publish_report(project["id"], 1, workspace, report)
    assert (publication / "evidence/paper.md").is_file()
    assert (publication / "evidence/result.txt").is_file()
    assert not (publication / "evidence/unpublished.txt").exists()
    archive = workspaces.archive(project["id"], 1)
    artifacts = workspaces.register_report_artifacts(
        project["id"],
        1,
        archive,
        report,
    )
    result = next(item for item in artifacts if item["path"] == "result.txt")
    assert result["sha256"] == "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
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


def test_report_accepts_numbered_required_sections(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "paper.md").write_text(
        "# Study\n## 2. Methodology\nMethod.\n## 4 Results\nResult.\n## 6) Conclusions\nConclusion.\n"
    )
    assert report_files(workspace, IterationReport(outcome="partial")) == ["paper.md"]


def test_report_package_limits_are_enforced(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "paper.md").write_bytes(b"x" * (MAX_PAPER_BYTES + 1))
    with pytest.raises(ConflictError, match="4 MiB"):
        report_files(workspace, IterationReport(outcome="partial"))
    with pytest.raises(ValidationError, match="at most 64 items"):
        IterationReport(outcome="partial", artifacts=[f"result-{index}" for index in range(65)])


def test_resource_source_symlink_is_rejected(config, tmp_path: Path) -> None:
    store = Store(config.data_root)
    project = store.create_project(ProjectCreate(name="study", goal="measure"), {}, 600)
    source = tmp_path / "source.txt"
    source.write_text("input")
    link = tmp_path / "link.txt"
    link.symlink_to(source)
    with pytest.raises(ConflictError, match="non-symlink"):
        WorkspaceManager(store).add_resource(project["id"], link, "dataset.txt")
    store.close()
