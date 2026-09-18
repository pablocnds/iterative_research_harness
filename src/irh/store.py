from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ConflictError, NotFoundError
from .types import Decision, IterationDefinition, IterationReport, ProjectCreate


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def decode(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


class Store:
    """SQLite is the only authoritative state and event store."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o770)
        self.db = sqlite3.connect(root / "state.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            PRAGMA busy_timeout=5000;
            CREATE TABLE IF NOT EXISTS projects (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, goal TEXT NOT NULL, guidelines TEXT NOT NULL,
              phase TEXT NOT NULL, resume_phase TEXT, current_iteration INTEGER NOT NULL DEFAULT 0,
              spent_seconds REAL NOT NULL DEFAULT 0, budget_seconds REAL NOT NULL,
              created_at TEXT NOT NULL, conclusion TEXT, pause_reason TEXT, snapshot_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS iterations (
              project_id TEXT NOT NULL REFERENCES projects(id), number INTEGER NOT NULL,
              status TEXT NOT NULL, definition_json TEXT NOT NULL, report_json TEXT, review_json TEXT,
              error TEXT, active_seconds REAL NOT NULL DEFAULT 0, resume_allowed INTEGER NOT NULL DEFAULT 0,
              current_attempt INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY(project_id, number)
            );
            CREATE TABLE IF NOT EXISTS attempts (
              project_id TEXT NOT NULL, iteration INTEGER NOT NULL, attempt INTEGER NOT NULL,
              status TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT, error TEXT,
              PRIMARY KEY(project_id, iteration, attempt),
              FOREIGN KEY(project_id, iteration) REFERENCES iterations(project_id, number)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              project_id TEXT NOT NULL, iteration INTEGER NOT NULL, path TEXT NOT NULL,
              sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, created_at TEXT NOT NULL,
              PRIMARY KEY(project_id, iteration, path)
            );
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
              at TEXT NOT NULL, kind TEXT NOT NULL, data_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
              at TEXT NOT NULL, text TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
              BEGIN SELECT RAISE(ABORT, 'append-only events'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
              BEGIN SELECT RAISE(ABORT, 'append-only events'); END;
            """
        )
        self.db.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock, self.db:
            yield self.db

    @staticmethod
    def _event(db: sqlite3.Connection, project_id: str, kind: str, data: dict[str, Any]) -> None:
        db.execute(
            "INSERT INTO events(project_id,at,kind,data_json) VALUES(?,?,?,?)",
            (project_id, utcnow(), kind, json.dumps(data, separators=(",", ":"))),
        )

    def event(self, project_id: str, kind: str, data: dict[str, Any]) -> None:
        with self.transaction() as db:
            self._event(db, project_id, kind, data)

    def project_dir(self, project_id: str) -> Path:
        self.get_project(project_id)
        return self.root / "projects" / project_id

    def runtime_dir(self, project_id: str) -> Path:
        self.get_project(project_id)
        return self.root / "runtime" / project_id

    def private_dir(self, project_id: str) -> Path:
        self.get_project(project_id)
        return self.root / "controller" / project_id

    def create_project(
        self, request: ProjectCreate, snapshot: dict[str, Any], default_budget: float
    ) -> dict[str, Any]:
        project_id = str(uuid.uuid4())
        created_at = utcnow()
        budget = request.budget_seconds or default_budget
        with self.transaction() as db:
            db.execute(
                """INSERT INTO projects(id,name,goal,guidelines,phase,budget_seconds,created_at,snapshot_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    project_id,
                    request.name,
                    request.goal,
                    request.guidelines,
                    "planning",
                    budget,
                    created_at,
                    json.dumps(snapshot, separators=(",", ":")),
                ),
            )
            self._event(db, project_id, "project_created", {"name": request.name, "budget_seconds": budget})
        directory = self.root / "projects" / project_id
        for child in ("resources", "iterations"):
            (directory / child).mkdir(parents=True, exist_ok=True, mode=0o770)
        (self.root / "controller" / project_id / "sessions").mkdir(parents=True, exist_ok=True, mode=0o770)
        (directory / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        return self.get_project(project_id)

    @staticmethod
    def _project(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["snapshot"] = decode(value.pop("snapshot_json"))
        return value

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.lock:
            row = self.db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Unknown project: {project_id}")
        return self._project(row)

    def list_projects(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM projects ORDER BY created_at").fetchall()
        return [self._project(row) for row in rows]

    def get_iteration(self, project_id: str, number: int) -> dict[str, Any]:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM iterations WHERE project_id=? AND number=?", (project_id, number)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Unknown iteration: {number}")
        value = dict(row)
        for name in ("definition_json", "report_json", "review_json"):
            value[name.removesuffix("_json")] = decode(value.pop(name))
        value["resume_allowed"] = bool(value["resume_allowed"])
        return value

    def history(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        with self.lock:
            rows = self.db.execute(
                "SELECT number FROM iterations WHERE project_id=? ORDER BY number", (project_id,)
            ).fetchall()
        return [self.get_iteration(project_id, row["number"]) for row in rows]

    def define_iteration(self, project_id: str, definition: IterationDefinition, max_seconds: float) -> int:
        project = self.get_project(project_id)
        if project["phase"] not in ("planning", "review"):
            raise ConflictError(f"Cannot define an iteration during {project['phase']}")
        seconds = definition.time_limit_seconds or max_seconds
        if seconds > max_seconds:
            raise ConflictError("Iteration time limit exceeds configured ceiling")
        if definition.parent is not None:
            self.get_iteration(project_id, definition.parent)
        number = project["current_iteration"] + 1
        payload = definition.model_dump(mode="json")
        payload["time_limit_seconds"] = seconds
        with self.transaction() as db:
            db.execute(
                "INSERT INTO iterations(project_id,number,status,definition_json) VALUES(?,?,?,?)",
                (project_id, number, "defined", json.dumps(payload, separators=(",", ":"))),
            )
            db.execute(
                "UPDATE projects SET current_iteration=?,phase='executing',resume_phase=NULL,pause_reason=NULL WHERE id=?",
                (number, project_id),
            )
            self._event(db, project_id, "iteration_defined", {"iteration": number, "definition": payload})
        return number

    def start_attempt(self, project_id: str, number: int) -> int:
        iteration = self.get_iteration(project_id, number)
        if iteration["status"] not in ("defined", "running"):
            raise ConflictError(f"Cannot execute iteration in {iteration['status']}")
        attempt = iteration["current_attempt"] or 1
        with self.transaction() as db:
            if not iteration["current_attempt"]:
                db.execute(
                    "INSERT INTO attempts(project_id,iteration,attempt,status,started_at) VALUES(?,?,?,?,?)",
                    (project_id, number, attempt, "running", utcnow()),
                )
            db.execute(
                "UPDATE iterations SET status='running',current_attempt=?,resume_allowed=0 WHERE project_id=? AND number=?",
                (attempt, project_id, number),
            )
            db.execute(
                "UPDATE projects SET phase='executing',resume_phase=NULL,pause_reason=NULL WHERE id=?",
                (project_id,),
            )
            self._event(db, project_id, "attempt_started", {"iteration": number, "attempt": attempt})
        return attempt

    def charge(self, project_id: str, role: str, seconds: float) -> None:
        if seconds <= 0:
            return
        project = self.get_project(project_id)
        with self.transaction() as db:
            db.execute("UPDATE projects SET spent_seconds=spent_seconds+? WHERE id=?", (seconds, project_id))
            if role == "iteration" and project["current_iteration"]:
                db.execute(
                    "UPDATE iterations SET active_seconds=active_seconds+? WHERE project_id=? AND number=?",
                    (seconds, project_id, project["current_iteration"]),
                )

    def finish_iteration(
        self,
        project_id: str,
        number: int,
        status: str,
        report: IterationReport | None,
        error: str | None,
    ) -> None:
        iteration = self.get_iteration(project_id, number)
        attempt = iteration["current_attempt"]
        with self.transaction() as db:
            db.execute(
                """UPDATE iterations SET status=?,report_json=?,error=?,resume_allowed=0
                   WHERE project_id=? AND number=?""",
                (
                    status,
                    json.dumps(report.model_dump(mode="json"), separators=(",", ":")) if report else None,
                    error,
                    project_id,
                    number,
                ),
            )
            if attempt:
                db.execute(
                    """UPDATE attempts SET status=?,ended_at=?,error=?
                       WHERE project_id=? AND iteration=? AND attempt=?""",
                    (status, utcnow(), error, project_id, number, attempt),
                )
            db.execute("UPDATE projects SET phase='review' WHERE id=?", (project_id,))
            self._event(
                db,
                project_id,
                "iteration_finished",
                {"iteration": number, "attempt": attempt, "status": status, "error": error},
            )

    def apply_decision(self, project_id: str, decision: Decision, max_iteration_seconds: float) -> int | None:
        project = self.get_project(project_id)
        if project["current_iteration"]:
            with self.transaction() as db:
                db.execute(
                    "UPDATE iterations SET review_json=? WHERE project_id=? AND number=?",
                    (
                        json.dumps(decision.model_dump(mode="json"), separators=(",", ":")),
                        project_id,
                        project["current_iteration"],
                    ),
                )
                self._event(
                    db,
                    project_id,
                    "upper_decision",
                    {"iteration": project["current_iteration"], **decision.model_dump(mode="json")},
                )
        if decision.action == "stop":
            with self.transaction() as db:
                db.execute(
                    "UPDATE projects SET phase='complete',conclusion=?,resume_phase=NULL WHERE id=?",
                    (decision.reason, project_id),
                )
                self._event(db, project_id, "project_completed", decision.model_dump(mode="json"))
            return None
        assert decision.definition is not None
        return self.define_iteration(project_id, decision.definition, max_iteration_seconds)

    def pause(self, project_id: str, reason: str, *, resume_phase: str | None = None) -> None:
        project = self.get_project(project_id)
        if project["phase"] == "complete":
            return
        target = resume_phase or project["phase"]
        with self.transaction() as db:
            db.execute(
                "UPDATE projects SET phase='paused',resume_phase=?,pause_reason=? WHERE id=?",
                (target, reason[:2000], project_id),
            )
            if target == "executing" and project["current_iteration"]:
                db.execute(
                    "UPDATE iterations SET resume_allowed=1 WHERE project_id=? AND number=?",
                    (project_id, project["current_iteration"]),
                )
            self._event(db, project_id, "project_paused", {"reason": reason, "resume_phase": target})

    def resume(self, project_id: str) -> None:
        project = self.get_project(project_id)
        if project["phase"] != "paused" or not project["resume_phase"]:
            raise ConflictError("Project is not resumable")
        with self.transaction() as db:
            db.execute(
                "UPDATE projects SET phase=resume_phase,resume_phase=NULL,pause_reason=NULL WHERE id=?",
                (project_id,),
            )
            self._event(db, project_id, "project_resumed", {})

    def record_note(self, project_id: str, text: str) -> None:
        self.get_project(project_id)
        with self.transaction() as db:
            db.execute("INSERT INTO notes(project_id,at,text) VALUES(?,?,?)", (project_id, utcnow(), text))
            self._event(db, project_id, "note_recorded", {"text": text})

    def notes(self, project_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT seq,at,text FROM notes WHERE project_id=? ORDER BY seq", (project_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def register_artifact(self, project_id: str, iteration: int, path: str, sha256: str, size: int) -> None:
        with self.transaction() as db:
            db.execute(
                """INSERT OR REPLACE INTO artifacts(project_id,iteration,path,sha256,bytes,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (project_id, iteration, path, sha256, size, utcnow()),
            )
            self._event(
                db,
                project_id,
                "artifact_registered",
                {"iteration": iteration, "path": path, "sha256": sha256, "bytes": size},
            )

    def artifacts(self, project_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT iteration,path,sha256,bytes,created_at FROM artifacts WHERE project_id=? ORDER BY iteration,path",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def events(self, project_id: str, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        self.get_project(project_id)
        limit = min(max(limit, 1), 1000)
        with self.lock:
            rows = self.db.execute(
                """SELECT seq,at,kind,data_json FROM events
                   WHERE project_id=? AND seq>? ORDER BY seq LIMIT ?""",
                (project_id, after, limit),
            ).fetchall()
        return [
            {"seq": row["seq"], "at": row["at"], "kind": row["kind"], "data": decode(row["data_json"])}
            for row in rows
        ]

    def model_activity(self, project_id: str) -> dict[str, Any]:
        self.get_project(project_id)
        with self.lock:
            latest = self.db.execute(
                """SELECT seq,at,kind,data_json FROM events
                   WHERE project_id=? AND kind IN ('model_request_started','model_request_finished')
                   ORDER BY seq DESC LIMIT 1""",
                (project_id,),
            ).fetchone()
            latest_finished = self.db.execute(
                """SELECT seq,at,kind,data_json FROM events
                   WHERE project_id=? AND kind='model_request_finished'
                   ORDER BY seq DESC LIMIT 1""",
                (project_id,),
            ).fetchone()

        def event(row: sqlite3.Row | None) -> dict[str, Any] | None:
            if row is None:
                return None
            return {
                "seq": row["seq"],
                "at": row["at"],
                "kind": row["kind"],
                "data": decode(row["data_json"]),
            }

        current = event(latest) if latest is not None and latest["kind"] == "model_request_started" else None
        return {
            "state": "running" if current is not None else "idle",
            "current_request": current,
            "latest_request": event(latest_finished),
        }

    def attempts(self, project_id: str, iteration: int) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM attempts WHERE project_id=? AND iteration=? ORDER BY attempt",
                (project_id, iteration),
            ).fetchall()
        return [dict(row) for row in rows]

    def recover_after_restart(self) -> list[str]:
        """Pause work that was active when the daemon disappeared."""
        recovered: list[str] = []
        for project in self.list_projects():
            if project["phase"] not in ("planning", "review", "executing"):
                continue
            # A newly created planning project has no running controller yet.
            with self.lock:
                row = self.db.execute(
                    """SELECT
                       COALESCE(MAX(CASE WHEN kind='controller_started' THEN seq END),0) AS started,
                       COALESCE(MAX(CASE WHEN kind='controller_stopped' THEN seq END),0) AS stopped
                       FROM events WHERE project_id=?""",
                    (project["id"],),
                ).fetchone()
            latest_started, latest_stopped = row["started"], row["stopped"]
            if latest_started <= latest_stopped:
                continue
            resume_phase = project["phase"]
            if resume_phase == "executing" and project["current_iteration"]:
                iteration = self.get_iteration(project["id"], project["current_iteration"])
                if iteration["status"] == "running":
                    self.finish_iteration(
                        project["id"],
                        project["current_iteration"],
                        "interrupted",
                        None,
                        "Harness daemon stopped without completing the active agent phase",
                    )
                    resume_phase = "review"
            self.pause(project["id"], "Recovered after an unclean daemon stop", resume_phase=resume_phase)
            recovered.append(project["id"])
        return recovered

    def close(self) -> None:
        with self.lock:
            self.db.close()
