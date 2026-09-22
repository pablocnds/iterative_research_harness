from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import __version__
from .config import HarnessConfig
from .controller import (
    COMMON_PROMPT,
    ITERATION_PROMPT,
    REVIEW_PROMPT,
    UPPER_PROMPT,
    WRITEUP_PROMPT,
    Supervisor,
)
from .errors import ConflictError, HarnessError, NotFoundError
from .store import Store
from .types import ProjectCreate, ProjectRun
from .workspace import WorkspaceManager


def json_response(value: object, status: int = 200) -> JSONResponse:
    return JSONResponse(value, status_code=status)


async def body(request: Request) -> dict[str, Any]:
    value = await request.json()
    if not isinstance(value, dict):
        raise TypeError("JSON body must be an object")
    return value


def source_identity() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[2]
    release = root / "release.json"
    if release.is_file():
        value = json.loads(release.read_text())
        return {
            "version": str(value.get("version", __version__)),
            "git_revision": value.get("git_revision"),
            "source_sha256": value.get("source_sha256"),
        }
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return {
        "version": __version__,
        "git_revision": result.stdout.strip() if result.returncode == 0 else None,
        "source_sha256": None,
    }


def read_skills(paths: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for value in paths:
        supplied = Path(value)
        if supplied.is_symlink():
            raise ValueError(f"Skill must be a regular non-symlink file: {value}")
        path = supplied.resolve()
        if not path.is_file():
            raise ValueError(f"Skill must be a regular non-symlink file: {value}")
        if path.stat().st_size > 128 * 1024:
            raise ValueError(f"Skill exceeds 128 KiB: {value}")
        text = path.read_text()
        result.append(
            {
                "name": path.name,
                "source": str(path),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
            }
        )
    return result


def model_usage(store: Store, project_id: str, controller_active: bool) -> dict[str, Any]:
    def count(usage: dict[str, Any], name: str) -> int:
        value = usage.get(name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    sessions: list[dict[str, Any]] = []
    directory = store.private_dir(project_id) / "sessions"
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        iteration_match = re.fullmatch(r"iteration-(\d+)", path.stem)
        review_match = re.fullmatch(r"upper-review-(\d+)", path.stem)
        if path.stem != "upper" and iteration_match is None and review_match is None:
            continue
        try:
            usage = json.loads(path.read_text()).get("usage", {})
        except (OSError, json.JSONDecodeError):
            continue

        input_tokens = count(usage, "input_tokens")
        output_tokens = count(usage, "output_tokens")
        sessions.append(
            {
                "session": path.stem,
                "role": "iteration" if iteration_match is not None else "upper",
                "iteration": int(match.group(1)) if (match := iteration_match or review_match) else None,
                "model": usage.get("model") if isinstance(usage.get("model"), str) else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "compactions": count(usage, "compactions"),
            }
        )
    totals = {
        "input_tokens": sum(item["input_tokens"] for item in sessions),
        "output_tokens": sum(item["output_tokens"] for item in sessions),
        "total_tokens": sum(item["total_tokens"] for item in sessions),
        "compactions": sum(item["compactions"] for item in sessions),
    }
    activity = store.model_activity(project_id)
    if activity["state"] == "running" and not controller_active:
        activity["state"] = "interrupted"
    return {"totals": totals, "sessions": sessions, "activity": activity}


def create_app(
    config: HarnessConfig,
    store: Store,
    supervisor: Supervisor,
    restart_callback: Any = None,
) -> Starlette:
    workspaces = WorkspaceManager(store)

    async def health(_request: Request) -> JSONResponse:
        active = [key for key, task in supervisor.tasks.items() if not task.done()]
        return json_response({"ok": True, "version": __version__, "active_projects": active})

    async def create_project(request: Request) -> JSONResponse:
        value = ProjectCreate.model_validate(await body(request))
        tools = [config.tool(tool_id) for tool_id in value.tools]
        prompts = {
            "common": COMMON_PROMPT,
            "upper": UPPER_PROMPT,
            "iteration": ITERATION_PROMPT,
            "writeup": WRITEUP_PROMPT,
            "review": REVIEW_PROMPT,
        }
        skills = {
            "upper": read_skills(value.upper_skills),
            "iteration": read_skills(value.iteration_skills),
        }
        snapshot = {
            "source": source_identity(),
            "base_prompts": prompts,
            "enabled_tools": value.tools,
            "tool_configurations": [
                {"id": tool.id, "configuration_id": tool.configuration_id} for tool in tools
            ],
            "skills": skills,
        }
        execution_snapshot = {
            "base_prompts": prompts,
            "tools": [tool.model_dump(mode="json") for tool in tools],
            "skills": skills,
        }
        project = store.create_project(
            value,
            snapshot,
            config.project_seconds,
            execution_snapshot=execution_snapshot,
        )
        return json_response(project, 201)

    async def list_projects(_request: Request) -> JSONResponse:
        return json_response(store.list_projects())

    async def project_status(request: Request) -> JSONResponse:
        project_id = request.path_params["project_id"]
        task = supervisor.tasks.get(project_id)
        return json_response(
            {
                "project": store.get_project(project_id),
                "iterations": store.history(project_id),
                "artifacts": store.artifacts(project_id),
                "model_usage": model_usage(store, project_id, task is not None and not task.done()),
            }
        )

    async def add_resource(request: Request) -> JSONResponse:
        value = await body(request)
        if not isinstance(value.get("source"), str) or not isinstance(value.get("name"), str):
            raise TypeError("source and name are required strings")
        project_id = request.path_params["project_id"]
        async with supervisor.lock:
            task = supervisor.tasks.get(project_id)
            if task is not None and not task.done():
                raise ConflictError("Pause the project before adding resources")
            result = await __import__("asyncio").to_thread(
                workspaces.add_resource,
                project_id,
                Path(value["source"]),
                value["name"],
            )
        return json_response(result, 201)

    async def run_project(request: Request) -> JSONResponse:
        value = ProjectRun.model_validate(await body(request))
        result = await supervisor.start(request.path_params["project_id"], value.resume)
        return json_response(result, 202)

    async def pause_project(request: Request) -> JSONResponse:
        return json_response(await supervisor.pause(request.path_params["project_id"]), 202)

    async def events(request: Request) -> JSONResponse:
        after = int(request.query_params.get("after", "0"))
        limit = int(request.query_params.get("limit", "200"))
        return json_response(store.events(request.path_params["project_id"], after, limit))

    async def artifacts(request: Request) -> JSONResponse:
        return json_response(store.artifacts(request.path_params["project_id"]))

    async def restart(_request: Request) -> JSONResponse:
        active = [key for key, task in supervisor.tasks.items() if not task.done()]
        if active:
            raise ConflictError(f"Pause the active project before updating: {active[0]}")
        if restart_callback is None:
            raise ConflictError("Daemon restart is unavailable in this process")
        restart_callback()
        return json_response({"accepted": True, "action": "daemon_restart"}, 202)

    async def exception_handler(_request: Request, error: Exception) -> JSONResponse:
        if isinstance(error, NotFoundError):
            return json_response({"error": "not_found", "message": str(error)}, 404)
        if isinstance(error, ConflictError):
            return json_response({"error": "conflict", "message": str(error)}, 409)
        if isinstance(error, (ValidationError, ValueError, TypeError, HarnessError, json.JSONDecodeError)):
            return json_response({"error": "invalid_request", "message": str(error)}, 400)
        return json_response({"error": "internal_error", "message": "Internal daemon error"}, 500)

    return Starlette(
        routes=[
            Route("/v1/health", health, methods=["GET"]),
            Route("/v1/projects", create_project, methods=["POST"]),
            Route("/v1/projects", list_projects, methods=["GET"]),
            Route("/v1/projects/{project_id}", project_status, methods=["GET"]),
            Route("/v1/projects/{project_id}/resources", add_resource, methods=["POST"]),
            Route("/v1/projects/{project_id}/run", run_project, methods=["POST"]),
            Route("/v1/projects/{project_id}/pause", pause_project, methods=["POST"]),
            Route("/v1/projects/{project_id}/events", events, methods=["GET"]),
            Route("/v1/projects/{project_id}/artifacts", artifacts, methods=["GET"]),
            Route("/v1/admin/restart", restart, methods=["POST"]),
        ],
        exception_handlers={
            NotFoundError: exception_handler,
            ConflictError: exception_handler,
            ValidationError: exception_handler,
            ValueError: exception_handler,
            TypeError: exception_handler,
            HarnessError: exception_handler,
            Exception: exception_handler,
        },
    )
