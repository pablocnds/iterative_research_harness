from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from .errors import ConflictError
from .store import Store
from .types import InputSelection, IterationReport

PAPER_SECTIONS = ("Methodology", "Results", "Conclusions")
MAX_PAPER_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_REPORT_BYTES = 8 * 1024 * 1024 * 1024


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


def report_files(workspace: Path, report: IterationReport) -> list[str]:
    paths = [report.paper, *report.artifacts]
    if len(paths) != len(set(paths)):
        raise ConflictError("Paper and supplementary artifact paths must be distinct")
    total_size = 0
    for relative in paths:
        target = contained(workspace, relative)
        reject_symlinks(workspace, target)
        if not target.is_file() or not stat.S_ISREG(target.stat().st_mode):
            raise ConflictError(f"Report file is not a regular file: {relative}")
        size = target.stat().st_size
        if relative == report.paper and size > MAX_PAPER_BYTES:
            raise ConflictError("paper.md exceeds the 4 MiB limit")
        if relative != report.paper and size > MAX_ARTIFACT_BYTES:
            raise ConflictError(f"Supplementary artifact exceeds the 2 GiB limit: {relative}")
        total_size += size
    if total_size > MAX_REPORT_BYTES:
        raise ConflictError("Paper package exceeds the 8 GiB total limit")
    paper = (workspace / report.paper).read_text(errors="replace")
    for section in PAPER_SECTIONS:
        if not re.search(
            rf"^#{{1,6}}\s+(?:\d+(?:\.\d+)*[.)]?\s+)?{section}\s*$",
            paper,
            flags=re.IGNORECASE | re.MULTILINE,
        ):
            raise ConflictError(f"paper.md needs a {section} heading")
    return paths


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
        if source.is_symlink():
            raise ConflictError("Resource must be an existing non-symlink file or directory")
        source = source.resolve()
        if not source.exists():
            raise ConflictError("Resource must be an existing non-symlink file or directory")
        destination = contained(self.store.project_dir(project_id) / "resources", name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        staged = staging_root / "resource"
        try:
            if source.is_dir():
                for child in source.rglob("*"):
                    if child.is_symlink():
                        raise ConflictError(f"Resource contains a symlink: {child.relative_to(source)}")
                shutil.copytree(source, staged, symlinks=False)
            elif source.is_file():
                shutil.copy2(source, staged)
            else:
                raise ConflictError("Special files cannot be resources")
            manifest = manifest_tree(staged)
            manifest_path = destination.parent / f".{destination.name}.manifest.json"
            with self.store.resource_lock:
                if destination.exists() or manifest_path.exists():
                    raise ConflictError(f"Resource already exists: {name}")
                os.replace(staged, destination)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)
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
        if not workspace.exists():
            if destination.is_dir():
                return destination
            raise ConflictError("Iteration workspace is missing and has not been archived")
        if destination.exists():
            suffix = 1
            while destination.with_name(f"workspace.attempt-{suffix}").exists():
                suffix += 1
            destination = destination.with_name(f"workspace.attempt-{suffix}")
        os.replace(workspace, destination)
        return destination

    def register_report_artifacts(
        self, project_id: str, number: int, archive: Path, report: IterationReport | None
    ) -> list[dict[str, object]]:
        if report is None:
            return []
        registered: list[dict[str, object]] = []
        for relative in report_files(archive, report):
            target = contained(archive, relative)
            digest, size = hash_file(target)
            self.store.register_artifact(project_id, number, relative, digest, size)
            registered.append({"path": relative, "sha256": digest, "bytes": size})
        return registered

    def publish_report(self, project_id: str, number: int, archive: Path, report: IterationReport) -> Path:
        """Expose only the paper and chosen supplements to the upper agent."""
        files = report_files(archive, report)
        manifest: dict[str, object] = {"paper": None, "supplementary": []}
        for relative in files:
            source = contained(archive, relative)
            digest, size = hash_file(source)
            entry = {"path": f"evidence/{relative}", "sha256": digest, "bytes": size}
            if relative == report.paper:
                manifest["paper"] = entry
            else:
                assert isinstance(manifest["supplementary"], list)
                manifest["supplementary"].append(entry)
        root = self.store.project_dir(project_id) / "publications"
        root.mkdir(parents=True, exist_ok=True, mode=0o770)
        publication = root / f"{number:04d}"
        if publication.exists():
            existing = publication / "manifest.json"
            if (
                existing.is_file()
                and json.loads(existing.read_text()) == manifest
                and all(
                    (publication / "evidence" / relative).is_file()
                    and hash_file(publication / "evidence" / relative)
                    == hash_file(contained(archive, relative))
                    for relative in files
                )
            ):
                return publication
            raise ConflictError(f"Publication already differs for iteration {number}")
        staging = Path(tempfile.mkdtemp(prefix=f".{number:04d}-", dir=root))
        try:
            staging.chmod(0o770)
            evidence = staging / "evidence"
            evidence.mkdir(mode=0o770)
            for relative in files:
                self._copy_reflink(contained(archive, relative), contained(evidence, relative))
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(staging, publication)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return publication
