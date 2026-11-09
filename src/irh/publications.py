"""Supervisor-authored synthesis papers, using the common artifact registry."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from .control_store import now
from .errors import ConflictError, NotFoundError
from .store import Store
from .types import PublicationCreate
from .workspace import contained, hash_file, reject_symlinks


class PublicationManager:
    def __init__(self, store: Store):
        self.store = store

    def list(self, project_id: str) -> list[dict]:
        self.store.get_project(project_id)
        with self.store.lock:
            return [
                json.loads(row[0])
                for row in self.store.db.execute(
                    "SELECT manifest_json FROM publications WHERE project_id=? ORDER BY created_at",
                    (project_id,),
                )
            ]

    def get(self, project_id: str, publication_id: str) -> dict:
        for value in self.list(project_id):
            if value["id"] == publication_id:
                return value
        raise NotFoundError("Unknown publication")

    def file(self, project_id: str, publication_id: str, path: str) -> Path:
        manifest = self.get(project_id, publication_id)
        if path not in {item["path"] for item in manifest["files"]} | {"manifest.json"}:
            raise NotFoundError("Unknown publication file")
        root = self.store.project_dir(project_id) / "syntheses" / publication_id
        target = contained(root, path)
        reject_symlinks(root, target)
        return target

    def create(self, project_id: str, value: PublicationCreate) -> dict:
        self.store.get_project(project_id)
        if value.supersedes:
            self.get(project_id, value.supersedes)
        sources = []
        seen = set()
        for source in value.sources:
            key = (source.project_id, source.iteration)
            if key in seen:
                continue
            seen.add(key)
            iteration = self.store.get_iteration(*key)
            if not iteration["report"] or iteration["status"] in ("running", "writeup", "finalizing"):
                raise ConflictError("Publication sources must have finalized reports")
            artifacts = [
                a
                for a in self.store.artifacts(source.project_id)
                if a["owner_kind"] == "iteration" and a["iteration"] == source.iteration
            ]
            sources.append({**source.model_dump(), "artifacts": artifacts})
        content = {"paper.md": value.paper}
        if len(value.paper.encode()) > 4 * 1024 * 1024:
            raise ValueError("Publication paper exceeds 4 MiB")
        for name, text in value.supplements.items():
            # All supplements live in a separate subtree; no reserved-file collisions.
            normalized = (
                contained(self.store.project_dir(project_id), name)
                .relative_to(self.store.project_dir(project_id))
                .as_posix()
            )
            path = f"supplements/{normalized}"
            if path in content:
                raise ValueError("Supplement names must be distinct after normalization")
            for previous in content:
                if Path(previous) in Path(path).parents or Path(path) in Path(previous).parents:
                    raise ValueError("Supplement paths cannot overlap")
            content[path] = text
        if sum(len(text.encode()) for text in content.values()) > 8 * 1024 * 1024:
            raise ValueError("Publication text package exceeds 8 MiB")
        publication_id = uuid.uuid4().hex
        root = self.store.project_dir(project_id) / "syntheses"
        root.mkdir(exist_ok=True, mode=0o770)
        staging = Path(tempfile.mkdtemp(prefix=".publication-", dir=root))
        destination = root / publication_id
        try:
            files = []
            for path, text in content.items():
                target = contained(staging, path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
                digest, size = hash_file(target)
                files.append({"path": path, "sha256": digest, "bytes": size})
            manifest = {
                "id": publication_id,
                "project_id": project_id,
                "kind": "publication",
                "title": value.title,
                "actor": value.actor,
                "created_at": now(),
                "supersedes": value.supersedes,
                "sources": sources,
                "files": files,
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(staging, destination)
            # Files are visible before registration. A crash can leave an unreferenced
            # directory, but never a registered partial publication.
            with self.store.transaction() as db:
                db.execute(
                    "INSERT INTO publications VALUES(?,?,?,?)",
                    (publication_id, project_id, json.dumps(manifest), manifest["created_at"]),
                )
                for item in files:
                    self.store._register_artifact(
                        db,
                        project_id,
                        "publication",
                        publication_id,
                        None,
                        item["path"],
                        item["sha256"],
                        item["bytes"],
                    )
                self.store._event(
                    db, project_id, "publication_created", {"id": publication_id, "title": value.title}
                )
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)
