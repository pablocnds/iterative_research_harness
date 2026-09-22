"""Thin MCP facade over the standalone Runpod tool's Unix-socket API."""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import shutil
import socket
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlencode

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from .tooling import BUDGET_PAUSE_MESSAGE, BUDGET_RESUME_MESSAGE

UNMETERED_WAIT_STATES = {"queued", "acquiring", "staging", "launching", "syncing"}


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str):
        super().__init__("localhost", timeout=310)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


def relative_path(value: str) -> Path:
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or not parsed.parts or any(part in ("", ".", "..") for part in parsed.parts):
        raise ValueError(f"Path must be relative and contained: {value}")
    return Path(*parsed.parts)


def contained_no_symlinks(root: Path, relative: Path, description: str) -> Path:
    """Resolve an agent-selected path without letting a parent symlink escape its root."""
    target = root / relative
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError(f"{description} escapes its root: {relative}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{description} contains a symlink: {relative}")
    return target


class RunpodAPI:
    def __init__(
        self,
        socket_path: str,
        project_id: str,
        session_id: str,
        input_root: Path,
        output_root: Path,
        workspace: Path,
        role: str,
    ):
        self.socket_path = socket_path
        self.project_id = project_id
        self.session_id = session_id
        self.input_root = input_root.resolve()
        self.output_root = output_root.resolve()
        self.workspace = workspace.resolve()
        self.role = role

    def call(self, method: str, path: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
        connection = UnixHTTPConnection(self.socket_path)
        data = json.dumps(value or {}, separators=(",", ":")) if method == "POST" else None
        connection.request(method, path, body=data, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read().decode()
        connection.close()
        payload = json.loads(raw)
        if response.status >= 400:
            incident = payload.get("incident")
            if incident:
                raise RuntimeError(
                    f"operator_intervention_required: {json.dumps(incident, separators=(',', ':'))}"
                )
            raise ValueError(payload.get("message") or payload.get("error") or f"HTTP {response.status}")
        if not isinstance(payload, dict):
            raise TypeError("Runpod service returned a non-object response")
        return payload

    def ensure_budget(self, max_seconds: float, session_seconds: float) -> None:
        status = self.call("GET", "/status")
        projects = status.get("projects", {})
        sessions = status.get("sessions", {})
        if self.project_id not in projects:
            self.call("POST", "/projects", {"projectId": self.project_id, "maxGpuSeconds": max_seconds})
        elif projects[self.project_id].get("max_seconds") != max_seconds:
            raise RuntimeError("Configured Runpod project maximum differs from its durable value")
        if self.session_id not in sessions:
            self.call(
                "POST",
                "/budgets",
                {"projectId": self.project_id, "sessionId": self.session_id, "seconds": session_seconds},
            )
        elif sessions[self.session_id].get("limit_seconds") != session_seconds:
            raise RuntimeError("Configured Runpod session allowance differs from its durable value")

    def prepare_inputs(self, inputs: list[str]) -> list[str]:
        """Freeze agent-writable inputs in the service-owned input root before submission."""
        prepared: list[str] = []
        for value in inputs:
            relative = relative_path(value)
            source = contained_no_symlinks(self.workspace, relative, "Input")
            if not source.exists():
                raise ValueError(f"Input is missing: {value}")
            if source.is_dir() and any(child.is_symlink() for child in source.rglob("*")):
                raise ValueError(f"Input directory contains a symlink: {value}")
            destination = self.input_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.irh-{uuid.uuid4().hex}")
            if source.is_dir():
                shutil.copytree(source, temporary)
            elif source.is_file():
                shutil.copy2(source, temporary)
            else:
                raise ValueError(f"Input is not a regular file or directory: {value}")
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            os.replace(temporary, destination)
            prepared.append(str(relative))
        return prepared

    def publish_outputs(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.role != "iteration":
            return payload
        source = self.output_root / self.session_id
        if not source.is_dir():
            return payload
        destination = self.workspace / "gpu-outputs"
        destination.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ValueError("Workspace output directory is symlinked")
        published: list[str] = []
        for child in sorted(source.rglob("*")):
            if child.is_symlink() or not child.is_file():
                continue
            relative = child.relative_to(source)
            target = contained_no_symlinks(destination, relative, "Workspace output")
            target.parent.mkdir(parents=True, exist_ok=True)
            target = contained_no_symlinks(destination, relative, "Workspace output")
            temporary = target.with_name(f".{target.name}.irh-{uuid.uuid4().hex}")
            subprocess.run(
                ["/bin/cp", "--reflink=auto", "--preserve=mode,timestamps", str(child), str(temporary)],
                check=True,
            )
            contained_no_symlinks(destination, relative, "Workspace output")
            os.replace(temporary, target)
            published.append(str(Path("gpu-outputs") / relative))
        if published:
            payload = {**payload, "workspace_outputs": published}
        return payload


def build_server(api: RunpodAPI) -> MCPServer:
    server = MCPServer(
        "runpod-agent-tool",
        version="0.3.0",
        instructions=(
            "Submit asynchronous GPU jobs with explicit inputs and outputs. Use wait rather than repeated status calls. "
            "Bundle related setup, validation, downloads, and useful work into substantial jobs; avoid short probes."
        ),
    )

    @server.tool(structured_output=True)
    def run(
        command: str,
        inputs: list[str] | None = None,
        setup_command: str | None = None,
        outputs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start a substantial asynchronous GPU job and return its handle. Bundle related work; avoid short probes."""
        return api.call(
            "POST",
            "/runs",
            {
                "sessionId": api.session_id,
                "command": command,
                "inputs": api.prepare_inputs(inputs or []),
                "setupCommand": setup_command,
                "outputs": outputs,
            },
        )

    @server.tool(structured_output=True)
    def status(job_id: str | None = None) -> dict[str, Any]:
        """Inspect one job, or overall service/session status when job_id is omitted."""
        result = api.call("GET", f"/runs/{job_id}" if job_id else "/status")
        return api.publish_outputs(result) if job_id else result

    @server.tool(structured_output=True)
    async def wait(job_id: str, timeout_seconds: float = 60, ctx: Context | None = None) -> dict[str, Any]:
        """Wait efficiently for a metering-state change or completion, for at most 300 seconds."""
        bounded = min(max(timeout_seconds, 0), 300)
        initial = await asyncio.to_thread(api.call, "GET", f"/runs/{job_id}")
        unmetered = initial.get("state") in UNMETERED_WAIT_STATES
        if unmetered and ctx is not None:
            await ctx.report_progress(0, None, BUDGET_PAUSE_MESSAGE)
        try:
            result = await asyncio.to_thread(
                api.call,
                "GET",
                f"/runs/{job_id}/wait?{urlencode({'timeout': bounded})}",
            )
        finally:
            if unmetered and ctx is not None:
                await ctx.report_progress(1, 1, BUDGET_RESUME_MESSAGE)
        return api.publish_outputs(result)

    @server.tool(structured_output=True)
    def cancel(job_id: str) -> dict[str, Any]:
        """Cancel the selected job and its remote process tree."""
        return api.call("POST", f"/runs/{job_id}/cancel", {})

    @server.tool(structured_output=True)
    def replace(
        job_id: str,
        command: str,
        inputs: list[str] | None = None,
        setup_command: str | None = None,
        outputs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Kill a current job and launch a replacement in the same retained Pod workspace."""
        return api.call(
            "POST",
            f"/runs/{job_id}/replace",
            {
                "command": command,
                "inputs": api.prepare_inputs(inputs or []),
                "setupCommand": setup_command,
                "outputs": outputs,
            },
        )

    @server.tool(structured_output=True)
    def host_status() -> dict[str, Any]:
        """Host-only monitor: return budgets, active work, retained Pod, and durable incident state."""
        return api.call("GET", "/status")

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose runpod-agent-tool through MCP")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--max-gpu-seconds", required=True, type=float)
    parser.add_argument("--session-seconds", type=float)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--role", required=True, choices=("upper", "iteration"))
    args = parser.parse_args()
    if not Path(args.socket).is_socket():
        raise SystemExit(f"Runpod tool socket is unavailable: {args.socket}")
    session_id = args.session_id or args.project_id
    api = RunpodAPI(
        args.socket,
        args.project_id,
        session_id,
        Path(args.input_root),
        Path(args.output_root),
        Path(args.workspace),
        args.role,
    )
    api.ensure_budget(args.max_gpu_seconds, args.session_seconds or args.max_gpu_seconds)
    build_server(api).run()


if __name__ == "__main__":
    main()
