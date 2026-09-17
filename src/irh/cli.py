from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SOCKET = "/var/lib/iterative-research-harness/daemon.sock"


class API:
    def __init__(self, socket_path: str):
        self.client = httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path), base_url="http://irh", timeout=60
        )

    def call(self, method: str, path: str, value: object | None = None) -> Any:
        response = self.client.request(method, path, json=value)
        payload = response.json()
        if response.is_error:
            raise RuntimeError(payload.get("message", json.dumps(payload)))
        return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Control the iterative research harness daemon")
    result.add_argument("--socket", default=os.environ.get("IRH_SOCKET", DEFAULT_SOCKET))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("health")
    commands.add_parser("restart-daemon")
    commands.add_parser("list")
    create = commands.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--goal", required=True)
    create.add_argument("--guidelines-file")
    create.add_argument("--budget-hours", type=float)
    create.add_argument("--tool", action="append", default=[])
    create.add_argument("--upper-skill", action="append", default=[])
    create.add_argument("--iteration-skill", action="append", default=[])
    status = commands.add_parser("status")
    status.add_argument("project_id")
    run = commands.add_parser("run")
    run.add_argument("project_id")
    run.add_argument("--resume", action="store_true")
    pause = commands.add_parser("pause")
    pause.add_argument("project_id")
    resource = commands.add_parser("add-resource")
    resource.add_argument("project_id")
    resource.add_argument("source")
    resource.add_argument("--name", required=True)
    events = commands.add_parser("events")
    events.add_argument("project_id")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=200)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("project_id")
    return result


def main() -> None:
    args = parser().parse_args()
    api = API(args.socket)
    try:
        if args.command == "health":
            value = api.call("GET", "/v1/health")
        elif args.command == "restart-daemon":
            value = api.call("POST", "/v1/admin/restart", {})
        elif args.command == "list":
            value = api.call("GET", "/v1/projects")
        elif args.command == "create":
            guidelines = Path(args.guidelines_file).read_text() if args.guidelines_file else ""
            value = api.call(
                "POST",
                "/v1/projects",
                {
                    "name": args.name,
                    "goal": args.goal,
                    "guidelines": guidelines,
                    "budget_seconds": args.budget_hours * 3600 if args.budget_hours else None,
                    "tools": args.tool,
                    "upper_skills": args.upper_skill,
                    "iteration_skills": args.iteration_skill,
                },
            )
        elif args.command == "status":
            value = api.call("GET", f"/v1/projects/{args.project_id}")
        elif args.command == "run":
            value = api.call("POST", f"/v1/projects/{args.project_id}/run", {"resume": args.resume})
        elif args.command == "pause":
            value = api.call("POST", f"/v1/projects/{args.project_id}/pause", {})
        elif args.command == "add-resource":
            value = api.call(
                "POST",
                f"/v1/projects/{args.project_id}/resources",
                {"source": str(Path(args.source).resolve()), "name": args.name},
            )
        elif args.command == "events":
            value = api.call(
                "GET", f"/v1/projects/{args.project_id}/events?after={args.after}&limit={args.limit}"
            )
        elif args.command == "artifacts":
            value = api.call("GET", f"/v1/projects/{args.project_id}/artifacts")
        else:  # pragma: no cover
            raise AssertionError(args.command)
        print(json.dumps(value, indent=2))
    except (OSError, httpx.HTTPError, RuntimeError) as error:
        print(f"irh: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
