from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from .errors import ConflictError
from .store import Store
from .types import InputSelection, IterationReport


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ConflictError(f"Path must be a contained relative path: {value}")
    return Path(*path.parts)


def contained(root: Path, relative: str) -> Path:
    target = root / safe_relative(relative)
    resolved_root = root.resolve()
    # strict=False checks containment before a destination exists.
    if not target.resolve(strict=False).is_relative_to(resolved_root):
        raise ConflictError(f"Path escapes its root: {relative}")
    return target


def reject_symlinks(root: Path, target: Path) -> None:
    current = root
    for part in target.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ConflictError(f"Symlinks are not accepted: {target.relative_to(root)}")


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def manifest_tree(path: Path) -> list[dict[str, object]]:
    base = path if path.is_dir() else path.parent
    files = [path] if path.is_file() else sorted(child for child in path.rglob("*") if child.is_file())
    result: list[dict[str, object]] = []
    for child in files:
        if child.is_symlink():
            raise ConflictError(f"Resource contains a symlink: {child}")
        digest, size = hash_file(child)
        result.append({"path": str(child.relative_to(base)), "sha256": digest, "bytes": size})
    return result


class WorkspaceManager:
    def __init__(self, store: Store):
        self.store = store

    def add_resource(self, project_id: str, source: Path, name: str) -> dict[str, object]:
        source = source.resolve()
        if not source.exists() or source.is_symlink():
            raise ConflictError("Resource must be an existing non-symlink file or directory")
        destination = contained(self.store.project_dir(project_id) / "resources", name)
        if destination.exists():
            raise ConflictError(f"Resource already exists: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for child in source.rglob("*"):
                if child.is_symlink():
                    raise ConflictError(f"Resource contains a symlink: {child.relative_to(source)}")
            shutil.copytree(source, destination, symlinks=False)
        elif source.is_file():
            shutil.copy2(source, destination)
        else:
            raise ConflictError("Special files cannot be resources")
        manifest = manifest_tree(destination)
        manifest_path = destination.parent / f".{destination.name}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        self.store.event(
            project_id,
            "resource_added",
            {
                "name": name,
                "source": str(source),
                "files": len(manifest),
                "bytes": sum(int(x["bytes"]) for x in manifest),
            },
        )
        return {"name": name, "manifest": manifest}

    def _copy_reflink(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["/bin/cp", "-a", "--reflink=auto", "--no-preserve=ownership", str(source), str(destination)],
            check=True,
        )

    def stage(self, project_id: str, number: int, inputs: Iterable[InputSelection]) -> Path:
        runtime = self.store.runtime_dir(project_id)
        workspace = runtime / "iteration" / "workspace"
        scratch = runtime / "iteration" / "scratch"
        if workspace.exists():
            shutil.rmtree(workspace)
        if scratch.exists():
            shutil.rmtree(scratch)
        workspace.mkdir(parents=True, mode=0o770)
        scratch.mkdir(parents=True, mode=0o770)
        project = self.store.project_dir(project_id)
        destinations: list[Path] = []
        staged: list[dict[str, str]] = []
        for selection in inputs:
            source = contained(project, selection.source)
            reject_symlinks(project, source)
            if not source.exists() or not (source.is_file() or source.is_dir()):
                raise ConflictError(f"Selected input does not exist: {selection.source}")
            destination = contained(workspace, selection.destination)
            for prior in destinations:
                if (
                    destination == prior
                    or destination.is_relative_to(prior)
                    or prior.is_relative_to(destination)
                ):
                    raise ConflictError("Iteration input destinations overlap")
            destinations.append(destination)
            self._copy_reflink(source, destination)
            staged.append({"source": selection.source, "destination": selection.destination})
        self.store.event(project_id, "workspace_staged", {"iteration": number, "inputs": staged})
        return workspace

    def archive(self, project_id: str, number: int) -> Path:
        workspace = self.store.runtime_dir(project_id) / "iteration" / "workspace"
        destination = self.store.project_dir(project_id) / "iterations" / f"{number:04d}" / "workspace"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            suffix = 1
            while destination.with_name(f"workspace.attempt-{suffix}").exists():
                suffix += 1
            destination = destination.with_name(f"workspace.attempt-{suffix}")
        if workspace.exists():
            os.replace(workspace, destination)
        else:
            destination.mkdir(mode=0o770)
        return destination

    def register_report_artifacts(
        self, project_id: str, number: int, archive: Path, report: IterationReport | None
    ) -> list[dict[str, object]]:
        if report is None:
            return []
        registered: list[dict[str, object]] = []
        for relative in report.artifacts:
            target = contained(archive, relative)
            reject_symlinks(archive, target)
            metadata = target.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ConflictError(f"Artifact is not a regular file: {relative}")
            digest, size = hash_file(target)
            self.store.register_artifact(project_id, number, relative, digest, size)
            registered.append({"path": relative, "sha256": digest, "bytes": size})
        return registered
