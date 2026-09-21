from __future__ import annotations

import sqlite3

import pytest

from irh.store import Store
from irh.types import Decision, IterationReport, ProjectCreate


def test_lifecycle_distinguishes_attempt_and_hashes_evidence(config, tmp_path) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"), {"enabled_tools": [], "skills": {}}, 600
    )
    decision = Decision.model_validate(
        {
            "action": "iterate",
            "reason": "baseline",
            "definition": {
                "subgoal": "measure one thing",
                "hypothesis": "it is measurable",
                "introduction": "small test",
                "parent": None,
                "inputs": [],
            },
        }
    )
    number = store.apply_decision(project["id"], decision, 60, 120)
    assert number == 1
    assert store.get_iteration(project["id"], 1)["definition"]["time_limit_seconds"] == 60
    assert store.start_attempt(project["id"], 1) == 1
    report = IterationReport(outcome="achieved", artifacts=[])
    store.finish_iteration(project["id"], 1, "achieved", report, None)
    assert store.get_project(project["id"])["phase"] == "review"
    assert store.attempts(project["id"], 1)[0]["status"] == "achieved"
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.db.execute("DELETE FROM events")
    store.close()


def test_revise_parent_requirement_is_exposed_in_the_tool_schema() -> None:
    schema = Decision.model_json_schema()
    parent = schema["$defs"]["IterationDefinition"]["properties"]["parent"]
    assert "Always include this key" in parent["description"]
    assert "parent" in schema["$defs"]["IterationDefinition"]["required"]


def test_pause_and_explicit_resume(config) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"), {"enabled_tools": [], "skills": {}}, 600
    )
    store.pause(project["id"], "operator")
    assert store.get_project(project["id"])["phase"] == "paused"
    store.resume(project["id"])
    assert store.get_project(project["id"])["phase"] == "planning"
    store.close()


def test_model_activity_tracks_current_and_latest_request(config) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"), {"enabled_tools": [], "skills": {}}, 600
    )
    store.event(
        project["id"],
        "model_request_started",
        {"role": "upper", "request_id": "request-1", "request_kind": "agent"},
    )
    running = store.model_activity(project["id"])
    assert running["state"] == "running"
    assert running["current_request"]["data"]["request_id"] == "request-1"
    store.event(
        project["id"],
        "model_request_finished",
        {"role": "upper", "request_id": "request-1", "request_kind": "agent", "ok": True},
    )
    idle = store.model_activity(project["id"])
    assert idle["state"] == "idle"
    assert idle["current_request"] is None
    assert idle["latest_request"]["data"]["request_id"] == "request-1"
    store.close()


def test_unclean_active_controller_is_paused_on_daemon_restart(config) -> None:
    store = Store(config.data_root)
    project = store.create_project(
        ProjectCreate(name="study", goal="measure"), {"enabled_tools": [], "skills": {}}, 600
    )
    store.event(project["id"], "controller_started", {"resume": False})
    assert store.recover_after_restart() == [project["id"]]
    recovered = store.get_project(project["id"])
    assert recovered["phase"] == "paused"
    assert recovered["resume_phase"] == "planning"
    store.close()


def test_unclean_iteration_resumes_in_same_session_for_writeup(config) -> None:
    store = Store(config.data_root)
    project = store.create_project(ProjectCreate(name="study", goal="measure"), {}, 600)
    store.define_iteration(
        project["id"],
        Decision.model_validate(
            {
                "action": "iterate",
                "reason": "test",
                "definition": {
                    "subgoal": "measure",
                    "hypothesis": "testable",
                    "introduction": "test",
                    "parent": None,
                },
            }
        ).definition,
        60,
        120,
    )
    store.start_attempt(project["id"], 1)
    store.event(project["id"], "controller_started", {"resume": False})
    assert store.recover_after_restart() == [project["id"]]
    recovered = store.get_project(project["id"])
    assert (recovered["phase"], recovered["resume_phase"]) == ("paused", "writeup")
    assert store.get_iteration(project["id"], 1)["status"] == "writeup"
    store.close()


def test_existing_database_gains_writeup_counter(config) -> None:
    config.data_root.mkdir(parents=True)
    old = sqlite3.connect(config.data_root / "state.sqlite3")
    old.execute(
        """CREATE TABLE iterations (
        project_id TEXT NOT NULL, number INTEGER NOT NULL, status TEXT NOT NULL,
        definition_json TEXT NOT NULL, report_json TEXT, review_json TEXT,
        error TEXT, active_seconds REAL NOT NULL DEFAULT 0,
        resume_allowed INTEGER NOT NULL DEFAULT 0, current_attempt INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(project_id, number))"""
    )
    old.commit()
    old.close()
    store = Store(config.data_root)
    columns = {row["name"] for row in store.db.execute("PRAGMA table_info(iterations)")}
    assert "writeup_seconds" in columns
    store.close()
