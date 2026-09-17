from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from irh.api import create_app
from irh.controller import Supervisor
from irh.store import Store


def test_api_creates_and_reads_project(config, tmp_path: Path) -> None:
    store = Store(config.data_root)
    supervisor = Supervisor(config, store)
    skill = tmp_path / "methods.md"
    skill.write_text("Use a fixed seed.")
    source = tmp_path / "input.txt"
    source.write_text("data")
    with TestClient(create_app(config, store, supervisor)) as client:
        response = client.post(
            "/v1/projects",
            json={
                "name": "api study",
                "goal": "exercise API",
                "upper_skills": [str(skill)],
            },
        )
        assert response.status_code == 201
        project = response.json()
        assert project["snapshot"]["skills"]["upper"][0]["text"] == "Use a fixed seed."
        project_id = project["id"]
        added = client.post(
            f"/v1/projects/{project_id}/resources",
            json={"source": str(source), "name": "input.txt"},
        )
        assert added.status_code == 201
        status = client.get(f"/v1/projects/{project_id}")
        assert status.status_code == 200
        assert status.json()["project"]["name"] == "api study"
        events = client.get(f"/v1/projects/{project_id}/events?limit=2")
        assert len(events.json()) == 2
    store.close()
