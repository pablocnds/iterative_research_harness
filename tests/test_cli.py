from __future__ import annotations

import pytest

import irh.cli


def test_pause_wait_timeout_leaves_request_pending(monkeypatch, capsys):
    calls = []

    class FakeAPI:
        def __init__(self, _socket):
            pass

        def call(self, method, path, value=None):
            calls.append((method, path, value))
            return {"state": "waiting_to_pause"}

    clock = iter([0, 2])
    monkeypatch.setattr(irh.cli, "API", FakeAPI)
    monkeypatch.setattr(irh.cli.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr("sys.argv", ["irh", "pause", "project", "--wait", "--wait-timeout", "1"])
    with pytest.raises(SystemExit) as exited:
        irh.cli.main()
    assert exited.value.code == 1
    assert "request remains pending" in capsys.readouterr().err
    assert calls == [
        ("POST", "/v1/projects/project/pause", {"actor": "operator", "reason": "Pause requested"})
    ]


def test_publication_cli_uploads_local_text_and_source_references(monkeypatch, tmp_path, capsys):
    paper = tmp_path / "paper.md"
    paper.write_text("# Combined findings\nExisting evidence.")
    table = tmp_path / "table.csv"
    table.write_text("method,result\nbaseline,partial\n")
    calls = []

    class FakeAPI:
        def __init__(self, _socket):
            pass

        def call(self, method, path, value=None):
            calls.append((method, path, value))
            return {"id": "publication"}

    monkeypatch.setattr(irh.cli, "API", FakeAPI)
    monkeypatch.setattr(
        "sys.argv",
        [
            "irh",
            "publication-create",
            "project",
            "--title",
            "Combined findings",
            "--paper",
            str(paper),
            "--source",
            "project:1",
            "--source",
            "other:2",
            "--supplement",
            f"table.csv={table}",
        ],
    )
    irh.cli.main()
    method, path, value = calls[0]
    assert method == "POST" and path == "/v1/projects/project/publications"
    assert value["paper"] == paper.read_text()
    assert value["supplements"] == {"table.csv": table.read_text()}
    assert value["sources"] == [
        {"project_id": "project", "iteration": 1},
        {"project_id": "other", "iteration": 2},
    ]
    assert "publication" in capsys.readouterr().out
