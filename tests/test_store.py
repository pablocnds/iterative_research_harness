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
                "inputs": [],
            },
        }
    )
    number = store.apply_decision(project["id"], decision, 120)
    assert number == 1
    assert store.start_attempt(project["id"], 1) == 1
    report = IterationReport(
        outcome="achieved", results="one", conclusions="measured", metrics={"value": 1}, artifacts=[]
    )
    store.finish_iteration(project["id"], 1, "achieved", report, None)
    assert store.get_project(project["id"])["phase"] == "review"
    assert store.attempts(project["id"], 1)[0]["status"] == "achieved"
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.db.execute("DELETE FROM events")
    store.close()


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
