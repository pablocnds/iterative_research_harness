from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from .sandbox import Sandbox
from .store import Store
from .tooling import AgentTool
from .types import Decision, IterationReport, Role, ToolResult
from .workspace import contained, reject_symlinks, report_files

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
    remaining_seconds: float | Callable[[], float],
    choose: Callable[[Decision], None] | None = None,
    submit: Callable[[IterationReport], None] | None = None,
    mode: Literal["normal", "writeup", "paper_review"] = "normal",
) -> list[AgentTool]:
    read_root = sandbox.project if role == "upper" else sandbox.workspace

    def remaining() -> float:
        if callable(remaining_seconds):
            return max(0.0, remaining_seconds())
        return max(0.0, remaining_seconds - time.monotonic())

    async def read_file(arguments: dict[str, Any]) -> ToolResult:
        target = contained(read_root, arguments["path"])
        reject_symlinks(read_root, target)
        if not target.is_file():
            raise ValueError("Path is not a regular file")
        if target.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("File exceeds the 8 MiB read limit")
        content = target.read_text(errors="replace")
        offset = min(arguments.get("offset_chars", 0), len(content))
        limit = arguments.get("max_chars", 8_000)
        end = min(len(content), offset + limit)
        excerpt = content[offset:end]
        if offset or end < len(content):
            excerpt += f"\n[characters {offset}–{end} of {len(content)}; use offset_chars={end} to continue]"
        return ToolResult(text=excerpt)

    async def run_command(arguments: dict[str, Any]) -> ToolResult:
        timeout = arguments.get("timeout_seconds")
        seconds_left = max(0.1, remaining())
        if timeout is None:
            timeout = min(seconds_left, 3600)
        else:
            timeout = min(float(timeout), seconds_left)
        value = await sandbox.run(arguments["command"], timeout=timeout)
        return ToolResult(text=json.dumps(value, separators=(",", ":")), structured=value)

    async def time_remaining(_arguments: dict[str, Any]) -> ToolResult:
        value = {"seconds": remaining()}
        return ToolResult(text=json.dumps(value), structured=value)

    tools = [
        AgentTool(
            "read_file",
            "Read a bounded section of a UTF-8 file from your permitted files. Paths are relative to your project view.",
            _object(
                {
                    "path": {"type": "string", "minLength": 1},
                    "offset_chars": {"type": "integer", "minimum": 0},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 10_000},
                },
                ["path"],
            ),
            read_file,
        ),
        AgentTool("time_remaining", "Return seconds remaining in this phase.", EMPTY_OBJECT, time_remaining),
    ]

    if mode == "normal":
        tools.append(
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
            )
        )

    if role == "iteration":

        async def list_files(arguments: dict[str, Any]) -> ToolResult:
            directory = (
                contained(sandbox.workspace, arguments["directory"])
                if arguments.get("directory")
                else sandbox.workspace
            )
            reject_symlinks(sandbox.workspace, directory)
            if not directory.is_dir():
                raise ValueError("Path is not a directory")
            entries = sorted(
                child.name + ("/" if child.is_dir() else "")
                for child in directory.iterdir()
                if not child.is_symlink() and (child.is_file() or child.is_dir())
            )
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit", 100)
            value = {
                "entries": entries[offset : offset + limit],
                "total": len(entries),
                "next_offset": offset + limit if offset + limit < len(entries) else None,
            }
            return ToolResult(text=json.dumps(value, separators=(",", ":")), structured=value)

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
            report_files(sandbox.workspace, report)
            submit(report)
            return ToolResult(text="Paper and supplementary files accepted; this iteration is complete")

        if mode == "normal":
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
                ]
            )
        else:
            tools.append(
                AgentTool(
                    "list_files",
                    "List immediate files and directories in the existing workspace; omit directory for its root.",
                    _object(
                        {
                            "directory": {"type": "string", "minLength": 1},
                            "offset": {"type": "integer", "minimum": 0},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        }
                    ),
                    list_files,
                )
            )

            async def write_paper(arguments: dict[str, Any]) -> ToolResult:
                target = sandbox.workspace / "paper.md"
                reject_symlinks(sandbox.workspace, target)
                target.write_text(arguments["content"])
                return ToolResult(text="paper.md written")

            tools.append(
                AgentTool(
                    "write_paper",
                    "Write or replace only paper.md in the existing iteration workspace.",
                    _object({"content": {"type": "string"}}, ["content"]),
                    write_paper,
                )
            )
        tools.append(
            AgentTool(
                "submit_results",
                "Submit paper.md and selected supplementary file paths; the paper contains methodology, results, and conclusions.",
                IterationReport.model_json_schema(),
                submit_results,
            )
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
                    "End this phase by defining the next iteration, requesting a child revision, or concluding the project. Always include definition.parent: the reviewed iteration number for action='revise', or null for action='iterate'.",
                    Decision.model_json_schema(),
                    decide,
                ),
            ]
        )
    return tools
