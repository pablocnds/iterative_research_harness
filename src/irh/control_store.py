"""Durable execution controls. Mixed into the daemon's single authoritative Store."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from .errors import ConflictError
from .types import SteeringMessage


def now() -> str:
    return datetime.now(UTC).isoformat()


class ControlStore:
    def _init_controls(self) -> None:
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
            state TEXT NOT NULL, actor TEXT NOT NULL, max_iterations INTEGER,
            first_iteration INTEGER NOT NULL, created_at TEXT NOT NULL,
            ended_at TEXT, completion_reason TEXT,
            running_seconds REAL NOT NULL DEFAULT 0, next_reminder REAL NOT NULL DEFAULT 0,
            reminder_interval REAL NOT NULL DEFAULT 0
          );
          CREATE TABLE IF NOT EXISTS messages (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
            project_id TEXT NOT NULL REFERENCES projects(id), text TEXT NOT NULL,
            actor TEXT NOT NULL, target TEXT NOT NULL, kind TEXT NOT NULL,
            created_at TEXT NOT NULL, delivered_at TEXT, session TEXT,
            idempotency_key TEXT, UNIQUE(project_id, idempotency_key)
          );
          CREATE TABLE IF NOT EXISTS pause_requests (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
            actor TEXT NOT NULL, reason TEXT NOT NULL, requested_at TEXT NOT NULL,
            acknowledged_at TEXT
          );
          CREATE TABLE IF NOT EXISTS pending_decisions (
            project_id TEXT PRIMARY KEY REFERENCES projects(id), decision_json TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS phase_results (
            project_id TEXT NOT NULL, session TEXT NOT NULL, kind TEXT NOT NULL,
            value_json TEXT NOT NULL, PRIMARY KEY(project_id,session)
          );
          CREATE TABLE IF NOT EXISTS publications (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
            manifest_json TEXT NOT NULL, created_at TEXT NOT NULL
          );
        """)
        if "reminder_interval" not in {r["name"] for r in self.db.execute("PRAGMA table_info(runs)")}:
            self.db.execute("ALTER TABLE runs ADD COLUMN reminder_interval REAL NOT NULL DEFAULT 0")
            self.db.commit()
        if "owner_kind" not in {r["name"] for r in self.db.execute("PRAGMA table_info(artifacts)")}:
            self.db.executescript("""
              BEGIN;
              ALTER TABLE artifacts RENAME TO legacy_artifacts;
              CREATE TABLE artifacts (
                project_id TEXT NOT NULL, iteration INTEGER, owner_kind TEXT NOT NULL,
                owner_id TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
                bytes INTEGER NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(project_id,owner_kind,owner_id,path)
              );
              INSERT INTO artifacts SELECT project_id,iteration,'iteration',CAST(iteration AS TEXT),
                path,sha256,bytes,created_at FROM legacy_artifacts;
              DROP TABLE legacy_artifacts;
              COMMIT;
            """)

    def enqueue_message(self, project_id: str, value: SteeringMessage, kind: str = "steering") -> dict:
        project = self.get_project(project_id)
        if project["phase"] == "complete":
            raise ConflictError("Project is complete")
        with self.transaction() as db:
            if value.idempotency_key:
                row = db.execute(
                    "SELECT * FROM messages WHERE project_id=? AND idempotency_key=?",
                    (project_id, value.idempotency_key),
                ).fetchone()
                if row:
                    if any(row[k] != getattr(value, k) for k in ("text", "actor", "target")):
                        raise ConflictError("Idempotency key already used for a different message")
                    return dict(row)
            message_id = uuid.uuid4().hex
            db.execute(
                """INSERT INTO messages
                (id,project_id,text,actor,target,kind,created_at,idempotency_key) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    message_id,
                    project_id,
                    value.text,
                    value.actor,
                    value.target,
                    kind,
                    now(),
                    value.idempotency_key,
                ),
            )
            self._event(
                db,
                project_id,
                "message_queued",
                {"id": message_id, "kind": kind, "actor": value.actor, "target": value.target},
            )
        return next(m for m in self.messages(project_id) if m["id"] == message_id)

    def messages(self, project_id: str, role: str | None = None) -> list[dict]:
        self.get_project(project_id)
        with self.lock:
            if role:
                rows = self.db.execute(
                    """SELECT * FROM messages WHERE project_id=?
                    AND delivered_at IS NULL AND target IN ('next',?) ORDER BY seq""",
                    (project_id, role),
                )
            else:
                rows = self.db.execute(
                    "SELECT * FROM messages WHERE project_id=? ORDER BY seq", (project_id,)
                )
            return [dict(row) for row in rows]

    def acknowledge_message(self, project_id: str, message_id: str, session: str) -> None:
        with self.transaction() as db:
            changed = db.execute(
                """UPDATE messages SET delivered_at=?,session=?
                WHERE project_id=? AND id=? AND delivered_at IS NULL""",
                (now(), session, project_id, message_id),
            ).rowcount
            if changed:
                self._event(db, project_id, "message_delivered", {"id": message_id, "session": session})

    def request_pause(self, project_id: str, actor: str, reason: str) -> dict:
        project = self.get_project(project_id)
        if project["phase"] == "complete":
            raise ConflictError("Project is complete")
        with self.transaction() as db:
            existing = db.execute(
                """SELECT * FROM pause_requests WHERE project_id=?
                ORDER BY requested_at DESC LIMIT 1""",
                (project_id,),
            ).fetchone()
            if existing and (not existing["acknowledged_at"] or project["phase"] == "paused"):
                return dict(existing)
            value = {
                "id": uuid.uuid4().hex,
                "project_id": project_id,
                "actor": actor,
                "reason": reason,
                "requested_at": now(),
                "acknowledged_at": None,
            }
            db.execute(
                "INSERT INTO pause_requests VALUES(:id,:project_id,:actor,:reason,:requested_at,NULL)", value
            )
            self._event(db, project_id, "pause_requested", value)
        return value

    def control_status(self, project_id: str) -> dict:
        project = self.get_project(project_id)
        with self.lock:
            pause = self.db.execute(
                "SELECT * FROM pause_requests WHERE project_id=? ORDER BY requested_at DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            pending = self.db.execute(
                "SELECT decision_json FROM pending_decisions WHERE project_id=?", (project_id,)
            ).fetchone()
            operation = self.db.execute(
                """SELECT kind,at,data_json FROM events WHERE project_id=? AND kind IN
                ('tool_started','tool_finished','model_request_started','model_request_finished')
                ORDER BY seq DESC LIMIT 1""",
                (project_id,),
            ).fetchone()
        state = "paused" if project["phase"] == "paused" else "idle"
        run = self.latest_run(project_id)
        if run and run["state"] == "running":
            state = "running"
        if pause and not pause["acknowledged_at"] and project["phase"] != "complete":
            state = "waiting_to_pause"
        return {
            "state": state,
            "pause_request": dict(pause) if pause else None,
            "run": run,
            "pending_decision": json.loads(pending[0]) if pending else None,
            "current_operation": (
                {
                    "kind": operation["kind"],
                    "started_at": operation["at"],
                    **json.loads(operation["data_json"]),
                }
                if operation
                and operation["kind"].endswith("started")
                and state in ("running", "waiting_to_pause")
                else None
            ),
        }

    def acknowledge_pause(self, project_id: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE pause_requests SET acknowledged_at=? WHERE project_id=? AND acknowledged_at IS NULL",
                (now(), project_id),
            )

    def start_run(self, project_id: str, resume: bool, max_iterations: int | None, actor: str) -> dict:
        previous = self.latest_run(project_id)
        if resume and previous and previous["state"] == "paused":
            if max_iterations is not None and max_iterations != previous["max_iterations"]:
                raise ConflictError("Resume preserves the existing run limit")
            with self.transaction() as db:
                db.execute("UPDATE runs SET state='running' WHERE id=?", (previous["id"],))
            return self.latest_run(project_id)
        project = self.get_project(project_id)
        first = project["current_iteration"] + 1
        if project["current_iteration"]:
            latest = self.get_iteration(project_id, project["current_iteration"])
            if latest["review"] is None:
                first = latest["number"]
        with self.transaction() as db:
            db.execute(
                """INSERT INTO runs
                (id,project_id,state,actor,max_iterations,first_iteration,created_at) VALUES(?,?,'running',?,?,?,?)""",
                (uuid.uuid4().hex, project_id, actor, max_iterations, first, now()),
            )
            self._event(db, project_id, "run_started", {"max_iterations": max_iterations, "actor": actor})
        return self.latest_run(project_id)

    def latest_run(self, project_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM runs WHERE project_id=? ORDER BY rowid DESC LIMIT 1", (project_id,)
            ).fetchone()
            if not row:
                return None
            value = dict(row)
            value["iterations_completed"] = self.db.execute(
                """SELECT COUNT(*) FROM iterations
                WHERE project_id=? AND number>=? AND review_json IS NOT NULL""",
                (project_id, value["first_iteration"]),
            ).fetchone()[0]
            return value

    def end_run(self, project_id: str, state: str, reason: str) -> None:
        run = self.latest_run(project_id)
        if not run or run["state"] != "running":
            return
        with self.transaction() as db:
            db.execute(
                "UPDATE runs SET state=?,completion_reason=?,ended_at=? WHERE id=?",
                (state, reason, None if state == "paused" else now(), run["id"]),
            )
            self._event(
                db, project_id, "run_stopped", {"run_id": run["id"], "state": state, "reason": reason}
            )

    def tick_run(self, project_id: str, seconds: float, interval: float) -> None:
        run = self.latest_run(project_id)
        if not run:
            return
        elapsed = run["running_seconds"] + seconds
        next_reminder = run["next_reminder"]
        if interval != run["reminder_interval"]:
            next_reminder = (int(run["running_seconds"] / interval) + 1) * interval if interval else 0
        due = (
            run["state"] == "running"
            and self.get_project(project_id)["phase"] != "complete"
            and interval > 0
            and elapsed >= (next_reminder or interval)
        )
        with self.transaction() as db:
            db.execute(
                "UPDATE runs SET running_seconds=?,reminder_interval=?,next_reminder=? WHERE id=?",
                (
                    elapsed,
                    interval,
                    next_reminder,
                    run["id"],
                ),
            )
        if due:
            project = self.get_project(project_id)
            progress = "No completed tool recorded."
            with self.lock:
                row = self.db.execute(
                    """SELECT data_json FROM events WHERE project_id=? AND kind='tool_finished'
                    ORDER BY seq DESC LIMIT 1""",
                    (project_id,),
                ).fetchone()
            if row:
                data = json.loads(row[0])
                progress = f"Last completed tool: {data['tool']} (ok={data['ok']})."
            text = (
                f"Time reminder: running {elapsed / 60:.1f} min; charged active time "
                f"{project['spent_seconds'] / 60:.1f} min; project budget remaining "
                f"{max(0, project['budget_seconds'] - project['spent_seconds']) / 60:.1f} min. "
                f"Iteration {project['current_iteration']}, phase {project['phase']}. {progress}"
            )
            # Coalesce a backlog while one long synchronous operation is running.
            with self.transaction() as db:
                db.execute(
                    "DELETE FROM messages WHERE project_id=? AND kind='reminder' AND delivered_at IS NULL",
                    (project_id,),
                )
                message_id = uuid.uuid4().hex
                db.execute(
                    """INSERT INTO messages
                    (id,project_id,text,actor,target,kind,created_at)
                    VALUES(?,?,?,'harness','next','reminder',?)""",
                    (message_id, project_id, text, now()),
                )
                db.execute(
                    "UPDATE runs SET next_reminder=? WHERE id=?",
                    ((int(elapsed / interval) + 1) * interval, run["id"]),
                )
                self._event(
                    db,
                    project_id,
                    "message_queued",
                    {"id": message_id, "kind": "reminder", "actor": "harness", "target": "next"},
                )

    def phase_result(self, project_id: str, session: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM phase_results WHERE project_id=? AND session=?", (project_id, session)
            ).fetchone()
        return {"kind": row["kind"], "value": json.loads(row["value_json"])} if row else None

    def save_phase_result(self, project_id: str, session: str, kind: str, value: Any) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO phase_results VALUES(?,?,?,?)",
                (project_id, session, kind, value.model_dump_json()),
            )

    def clear_phase_result(self, project_id: str, session: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM phase_results WHERE project_id=? AND session=?", (project_id, session))
