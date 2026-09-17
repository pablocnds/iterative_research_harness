from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .sandbox import Sandbox
from .store import Store
from .tooling import AgentTool
from .types import Decision, IterationReport, Role, ToolResult
from .workspace import contained, reject_symlinks

EMPTY_OBJECT = {"type": "object", "properties": {}, "additionalProperties": False}


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def phase_tools(
    role: Role,
    sandbox: Sandbox,
    store: Store,
    project_id: str,
    deadline: float,
    choose: Callable[[Decision], None] | None = None,
    submit: Callable[[IterationReport], None] | None = None,
) -> list[AgentTool]:
    read_root = store.project_dir(project_id) if role == "upper" else sandbox.workspace

    async def read_file(arguments: dict[str, Any]) -> ToolResult:
        target = contained(read_root, arguments["path"])
        reject_symlinks(read_root, target)
        if not target.is_file():
            raise ValueError("Path is not a regular file")
        if target.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("File exceeds 8 MiB; use run_command to inspect a bounded range")
        return ToolResult(text=target.read_text(errors="replace"))

    async def run_command(arguments: dict[str, Any]) -> ToolResult:
        timeout = arguments.get("timeout_seconds")
        remaining = max(0.1, deadline - time.monotonic())
        if timeout is None:
            timeout = min(remaining, 3600)
        else:
            timeout = min(float(timeout), remaining)
        value = await sandbox.run(arguments["command"], timeout=timeout)
        return ToolResult(text=json.dumps(value, separators=(",", ":")), structured=value)

    async def time_remaining(_arguments: dict[str, Any]) -> ToolResult:
        value = {"seconds": max(0, deadline - time.monotonic())}
        return ToolResult(text=json.dumps(value), structured=value)

    tools = [
        AgentTool(
            "read_file",
            "Read one UTF-8 text file from your permitted files. Paths are relative to your project view.",
            _object({"path": {"type": "string", "minLength": 1}}, ["path"]),
            read_file,
        ),
        AgentTool(
            "run_command",
            "Run a foreground shell command in the isolated sandbox. Large output is truncated; save evidence to files.",
            _object(
                {
                    "command": {"type": "string", "minLength": 1},
                    "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
                },
                ["command"],
            ),
            run_command,
        ),
        AgentTool("time_remaining", "Return seconds remaining in this phase.", EMPTY_OBJECT, time_remaining),
    ]

    if role == "iteration":

        async def write_file(arguments: dict[str, Any]) -> ToolResult:
            target = contained(sandbox.workspace, arguments["path"])
            reject_symlinks(sandbox.workspace, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(arguments["content"])
            return ToolResult(text="File written")

        async def edit_file(arguments: dict[str, Any]) -> ToolResult:
            target = contained(sandbox.workspace, arguments["path"])
            reject_symlinks(sandbox.workspace, target)
            if not target.is_file():
                raise ValueError("Path is not a regular file")
            content = target.read_text()
            old = arguments["old_text"]
            count = content.count(old)
            if count != 1:
                raise ValueError(f"old_text must match exactly once; found {count}")
            target.write_text(content.replace(old, arguments["new_text"], 1))
            return ToolResult(text="File edited")

        async def submit_results(arguments: dict[str, Any]) -> ToolResult:
            assert submit is not None
            report = IterationReport.model_validate(arguments)
            for path in report.artifacts:
                target = contained(sandbox.workspace, path)
                reject_symlinks(sandbox.workspace, target)
                if not target.is_file():
                    raise ValueError(f"Artifact is not a regular file: {path}")
            submit(report)
            return ToolResult(text="Results accepted; this iteration phase is complete")

        tools.extend(
            [
                AgentTool(
                    "write_file",
                    "Write a UTF-8 file in the iteration workspace, creating parent directories.",
                    _object(
                        {"path": {"type": "string", "minLength": 1}, "content": {"type": "string"}},
                        ["path", "content"],
                    ),
                    write_file,
                ),
                AgentTool(
                    "edit_file",
                    "Replace one exact occurrence in a UTF-8 workspace file.",
                    _object(
                        {
                            "path": {"type": "string", "minLength": 1},
                            "old_text": {"type": "string", "minLength": 1},
                            "new_text": {"type": "string"},
                        },
                        ["path", "old_text", "new_text"],
                    ),
                    edit_file,
                ),
                AgentTool(
                    "submit_results",
                    "End the iteration with structured results, conclusions, measurements, and evidence paths.",
                    IterationReport.model_json_schema(),
                    submit_results,
                ),
            ]
        )
    else:

        async def record_note(arguments: dict[str, Any]) -> ToolResult:
            store.record_note(project_id, arguments["text"])
            return ToolResult(text="Note recorded")

        async def decide(arguments: dict[str, Any]) -> ToolResult:
            assert choose is not None
            decision = Decision.model_validate(arguments)
            choose(decision)
            return ToolResult(text="Decision accepted; this upper-agent phase is complete")

        tools.extend(
            [
                AgentTool(
                    "record_note",
                    "Append a durable research or planning note to the project.",
                    _object({"text": {"type": "string", "minLength": 1}}, ["text"]),
                    record_note,
                ),
                AgentTool(
                    "decide",
                    "End this phase by defining the next iteration, requesting a child revision, or concluding the project.",
                    Decision.model_json_schema(),
                    decide,
                ),
            ]
        )
    return tools
